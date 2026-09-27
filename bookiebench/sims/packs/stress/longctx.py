"""Long context: embed a world in 4k-32k tokens of real filler documents, with the world's sentences scattered.

Filler comes from local HF dataset copies: CNN/DailyMail validation articles (news; realcoh uses the test split),
WikiQA answer sentences grouped by Wikipedia article (encyclopedic text), and 20 Newsgroups test posts (forum/email).
The prelude becomes an archive of documents with the world's prelude units inserted between them in their original
order; each evidence step is preceded by one more filler document (the same for all `next_evidence` alternatives of
that step, so it carries no information). Filler documents that mention any of the world's names are skipped.
Lengths are measured with the Qwen3.5 tokenizer on the full final state (`core.state_text`).
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from bookiebench.sims.core import state_text

from .common import map_evidence, start, transform_rng, units

HF = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "datasets")
FILES = {
    "news": f"{HF}/abisee___cnn_dailymail/3.0.0/0.0.0/96df5e686bee6baa90b8bee7c28b81fa3fa6223d/"
            "cnn_dailymail-validation.arrow",
    "wiki": f"{HF}/microsoft___wiki_qa/default/0.0.0/3f104672b5de699878fe7907afc486f0de325eb5/wiki_qa-train.arrow",
    "forum": f"{HF}/SetFit___20_newsgroups/default/0.0.0/f1b91292074e7cfb69be58b642d583ec262f30ed/"
             "20_newsgroups-test.arrow",
}
TOKENIZER = "Qwen/Qwen3.5-2B-Base"
TARGETS = [4096, 8192, 16384, 32768]
MAX_DOC_TOKENS = 1800
FORUM_GROUPS = ("comp.", "sci.", "rec.", "misc.forsale")  # leave out the politics / religion groups
FILL_FACTOR = 0.975  # joining documents costs a few tokens; aim slightly low so lengths land near the target
EV_SHARE = 0.12  # share of the filler budget placed before evidence steps


@lru_cache(maxsize=1)
def tokenizer():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def n_tokens(text: str) -> int:
    return len(tokenizer()(text, add_special_tokens=False)["input_ids"])


def _read(path):
    import pyarrow as pa
    with open(path, "rb") as f:
        return pa.ipc.open_stream(f).read_all()


def _clean(t):
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


@lru_cache(maxsize=1)
def corpus(max_per_source: int = 6000):
    """[(source, title, text, n_tokens)] filler documents (deterministic order)."""
    docs = []
    tb = _read(FILES["news"])
    for a in tb.column("article").to_pylist()[:max_per_source]:
        a = re.sub(r"^.{0,120}?\(CNN\)\s*(--\s*)?", "", a)
        a = re.sub(r"^By \. .*? \. (PUBLISHED: .*? \. \| \. UPDATED: .*? \. )?", "", a)
        docs.append(("news", "News article", _clean(a)))
    tb = _read(FILES["wiki"])
    groups = {}
    for title, ans in zip(tb.column("document_title").to_pylist(), tb.column("answer").to_pylist()):
        groups.setdefault(title, [])
        if ans not in groups[title]:
            groups[title].append(ans)
    for title, sents in list(groups.items())[:max_per_source]:
        if len(sents) >= 3:
            docs.append(("wiki", f"Encyclopedia entry: {title}", _clean(" ".join(s.replace(" .", ".").replace(" ,", ",")
                                                                               for s in sents))))
    tb = _read(FILES["forum"])
    for t, lab in list(zip(tb.column("text").to_pylist(), tb.column("label_text").to_pylist()))[:max_per_source]:
        t = _clean(t)
        if len(t) > 400 and lab.startswith(FORUM_GROUPS):
            docs.append(("forum", "Forum post", t))
    tok = tokenizer()
    lens = [len(x) for x in tok([d[2] for d in docs], add_special_tokens=False)["input_ids"]]
    return [(s, ti, tx, n) for (s, ti, tx), n in zip(docs, lens) if n >= 60]


def _truncate(text, n_tok):
    tok = tokenizer()
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tok]
    t = tok.decode(ids)
    cut = max(t.rfind(". "), t.rfind(".\n"))
    return (t[: cut + 1] if cut > len(t) // 2 else t).strip()


def _names(inst):
    from .llm import required_entities
    txt = inst["prelude"] + " " + " ".join(st["evidence"] for st in inst["steps"])
    return [n for n in required_entities(txt) if len(n) >= 4]


def _pick_docs(rng, budget, avoid, docs, used):
    out, tot = [], 0
    order = rng.permutation(len(docs))
    for i in order:
        if tot >= budget:
            break
        i = int(i)
        if i in used:
            continue
        s, title, text, n = docs[i]
        if any(a in text for a in avoid):
            continue
        if n > MAX_DOC_TOKENS:
            text, n = _truncate(text, MAX_DOC_TOKENS), MAX_DOC_TOKENS
        if tot + n > budget:
            rest = budget - tot
            if rest < 80:
                break
            text, n = _truncate(text, rest), rest
        used.add(i)
        out.append((s, title, text))
        tot += n
    return out


INTRO = ("The material below is a mixed collection of documents: news articles, encyclopedia entries and forum posts. "
         "Scattered among them, as separate paragraphs, are the notes that describe the situation the questions are "
         "about. Everything else is unrelated.")


def long_context(inst, target: int, seed: int = 0):
    name = f"long_{target // 1024}k"
    rng = transform_rng(inst, name, seed)
    out = start(inst, name, target_tokens=target)
    docs = corpus()
    base = n_tokens(state_text(inst)) + n_tokens(INTRO) + 40
    budget = max(0, int((target - base) * FILL_FACTOR))
    T = len(inst["steps"])
    ev_budget = int(budget * EV_SHARE) // max(T, 1) if T > 1 else 0
    if ev_budget < 120:
        ev_budget = 0  # too many steps for per-step filler at this length: everything goes into the prelude
    pre_budget = budget - ev_budget * T
    avoid = _names(inst)
    used: set = set()
    pre_docs = _pick_docs(rng, pre_budget, avoid, docs, used)
    ev_docs = [_pick_docs(rng, ev_budget, avoid, docs, used) if ev_budget >= 80 else [] for _ in range(T)]
    # scatter the prelude units (in order) between the documents
    us = units(inst["prelude"])
    slots = sorted(int(x) for x in rng.integers(0, len(pre_docs) + 1, size=len(us)))
    if us and slots:
        slots[0] = min(slots[0], max(0, len(pre_docs) // 3))  # the opening unit is not pushed to the very end
    blocks, j, counter = [INTRO], 0, 0
    for d in range(len(pre_docs) + 1):
        while j < len(us) and slots[j] == d:
            blocks.append(us[j].get("prefix", "") + us[j]["text"])
            j += 1
        if d < len(pre_docs):
            counter += 1
            s, title, text = pre_docs[d]
            blocks.append(f"--- {title} ---\n{text}")
    ev_blocks = []
    for k in range(T):
        parts = []
        for s, title, text in ev_docs[k]:
            counter += 1
            parts.append(f"--- {title} ---\n{text}")
        ev_blocks.append("\n\n".join(parts))

    def wrap(t, k):
        return (ev_blocks[k] + "\n\n" + t) if ev_blocks[k] else t

    map_evidence(out, wrap)
    out["prelude"] = "\n\n".join(blocks)
    n_final = n_tokens(state_text(out))
    for _ in range(4):  # top up (truncation and skipped documents leave the first pass a little short)
        missing = int((target - n_final) * FILL_FACTOR)
        if missing < 80:
            break
        extra = _pick_docs(rng, missing, avoid, docs, used)
        if not extra:
            break
        for s_, title, text in extra:
            counter += 1
            pos = int(rng.integers(1, len(blocks) + 1))
            blocks.insert(pos, f"--- {title} ---\n{text}")
        out["prelude"] = "\n\n".join(blocks)
        n_final = n_tokens(state_text(out))
    out["meta"]["stress"].update({"n_tokens": n_final, "n_tokens_world": base - n_tokens(INTRO) - 40,
                                  "n_docs": counter, "sources": sorted({d[0] for d in pre_docs}),
                                  "unit_positions": slots, "tokenizer": TOKENIZER})
    return out
