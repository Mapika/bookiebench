"""Drive a local LLM (vLLM, in-process, no server) over the paraphrase / translation / framing jobs of `llm.py`.

    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=. python \
        -m bookiebench.sims.packs.stress.llm_run --n 300 --langs en,de,es,hu,pt,zh --framing

writes data/v2/stress/{paraphrase,lang_<xx>,framing}/<family>.jsonl and logs/stress_llm_stats.json.
Every segment is checked with `llm.check_text` (or the options/framing checks); failures are retried with a new
sampling seed up to --attempts times, and an instance with any segment still failing is rejected.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from bookiebench.sims.core import read_jsonl, validate, write_jsonl

from . import llm as L

MODEL = "Qwen/Qwen3.8-27B-FP8"
CHUNK = 16
DEBUG = os.environ.get("STRESS_LLM_DEBUG")


def load_sources(dirs, n):
    out = {}
    for d in dirs:
        for p in sorted(Path(d).glob("*.jsonl")):
            out[p.stem] = read_jsonl(p)[:n]
            for i in out[p.stem]:
                i.setdefault("meta", {})["source_file"] = str(p)  # copied into every output's meta
    return out


class Runner:
    def __init__(self, model, max_len=16384, mem=0.85):
        from vllm import LLM
        self.llm = LLM(model=model, max_model_len=max_len, gpu_memory_utilization=mem, enable_prefix_caching=True,
                       language_model_only=True)
        self.tok = self.llm.get_tokenizer()

    def __call__(self, convs, seed, temperature=0.7):
        from vllm import SamplingParams
        sps = []
        for i, c in enumerate(convs):
            n_in = sum(len(m["content"]) for m in c) // 3
            sps.append(SamplingParams(temperature=temperature, top_p=0.9, max_tokens=min(6000, 400 + 2 * n_in),
                                      seed=seed * 100003 + i))
        outs = self.llm.chat(convs, sps, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
        return [o.outputs[0].text for o in outs]


class Job:
    """A rewrite job over one or more source items. build(items) -> messages; parse(output, n) -> n values or None;
    check(src, value) -> reasons. Items that fail are retried (alone, as a smaller list) with a new sampling seed."""

    def __init__(self, srcs, build, check, parse=None, is_list=False):
        self.srcs, self.build, self.check = list(srcs), build, check
        self.parse = parse or ((lambda o, n: L.parse_list(o, n)) if is_list else (lambda o, n: [L.clean_output(o)]))
        self.done: dict = {}
        self.last = None  # (pending item indices, raw output, failure reasons) of the previous attempt

    def pending(self):
        return [i for i in range(len(self.srcs)) if i not in self.done]


FEEDBACK = True
EXPLAIN = {
    "numbers": "the numbers changed: every number of the original must appear exactly as written, and no number may "
               "be added (do not turn digits into words or words into digits)",
    "number_words": "a numeral word (two, twice, half, ...) was added; do not add any",
    "negation": "the negations changed: keep every negation (not, no, never, neither, nor, none, nobody, false, "
                "wrong, without, absent, missing, fails, ...) one for one, with the same logical effect",
    "entity": "a name or invented term was changed, dropped or transliterated; copy it exactly",
    "quoted": "a word in quotes was changed; keep quoted words exactly",
    "option": "an option phrase was not kept (or not translated as in the glossary)",
    "question": "every item must be a question ending with '?'",
    "json": "the JSON structure changed; keep the same keys and value types",
    "length": "the length changed too much; do not drop or add content",
    "not_translated": "part of the text was not translated",
    "unchanged": "the text was returned unchanged; rephrase it",
    "weekday": "a weekday was not translated correctly",
    "format": "the output format was wrong; return exactly what was asked for (a JSON array when asked)",
    "complement_numbers": "a probability was not replaced by its complement (100 minus it) or a count changed",
    "partial_complement": "some probabilities were not complemented; complement all of them",
    "no_added_negation": "the complementary outcome must be stated explicitly (e.g. 'does not ...')",
    "roundtrip_numbers": "the original numbers were not restored",
    "options_format": "return a JSON object with the same keys, each a list with the same number of options",
    "options_not_distinct": "options of the same variable must be distinct",
    "options_numbers": "digits in the options must be kept",
    "options_entity": "names in the options must be kept exactly",
}


def feedback(reasons):
    kinds = sorted({r.split(":")[0] for r in reasons})
    extra = sorted({r.split(":", 1)[1] for r in reasons if ":" in r})
    msg = "That violates the rules: " + "; ".join(EXPLAIN.get(k, k) for k in kinds) + "."
    if extra:
        msg += " Affected: " + ", ".join(extra[:6]) + "."
    return msg + " Please redo the whole task, following every rule, and output only the result."


def run_jobs(runner, jobs, attempts, stats, tag):
    """jobs: {key: Job}. Returns {key: [values]} for jobs whose items were all accepted."""
    for att in range(attempts):
        todo = [(k, j, j.pending()) for k, j in jobs.items() if j.pending()]
        if not todo:
            break
        t0 = time.time()
        convs = []
        for _, j, idx in todo:
            m = j.build([j.srcs[i] for i in idx])
            if FEEDBACK and j.last and j.last[0] == tuple(idx):  # retry in the same conversation, saying what failed
                m = m + [{"role": "assistant", "content": j.last[1]}, {"role": "user", "content": feedback(j.last[2])}]
            convs.append(m)
        outs = runner(convs, seed=att + 1)
        n_ok = 0
        for (k, j, idx), o in zip(todo, outs):
            seg = k[-1].rstrip("0123456789")
            vals = j.parse(o, len(idx))
            if vals is None:
                j.last = (tuple(idx), o, ["format"])
                stats[tag]["reasons"][f"{seg}:format"] += len(idx)
                stats[tag]["attempts"][seg] += len(idx)
                if att == attempts - 1:
                    stats[tag]["examples"].append({"key": list(k), "reasons": ["format"], "output": o[:600]})
                continue
            fails = []
            for i, v in zip(idx, vals):
                stats[tag]["attempts"][seg] += 1
                r = j.check(j.srcs[i], v)
                fails += r
                if not r:
                    j.done[i] = v
                    stats[tag]["pass_at"][f"{seg}@{att + 1}"] += 1
                    n_ok += 1
                else:
                    for x in r:
                        stats[tag]["reasons"][f"{seg}:{x.split(':')[0]}"] += 1
                    if DEBUG:
                        with open(DEBUG, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"tag": tag, "att": att, "key": list(k) + [i], "reasons": r,
                                                "src": j.srcs[i], "out": v}, ensure_ascii=False, default=str) + "\n")
                    if att == attempts - 1:
                        stats[tag]["examples"].append({"key": list(k) + [i], "reasons": r, "src": j.srcs[i][:400]
                                                       if isinstance(j.srcs[i], str) else None,
                                                       "output": str(v)[:600]})
            j.last = (tuple(j.pending()), o, fails) if tuple(j.pending()) == tuple(idx) else None
        print(f"[{tag}] attempt {att + 1}: {sum(len(t[2]) for t in todo)} items in {len(todo)} prompts, {n_ok} "
              f"accepted, {time.time() - t0:.0f}s", flush=True)
    return {k: [j.done[i] for i in range(len(j.srcs))] for k, j in jobs.items() if not j.pending()}


def rewrite(runner, sources, lang, attempts, stats):
    tag = "paraphrase" if lang == "en" else f"lang_{lang}"
    insts = {i["id"]: i for fam in sources.values() for i in fam}
    opt_tr = {}
    if lang != "en":
        jobs = {}
        for iid, i in insts.items():
            src = L.options_by_var(i)
            jobs[(iid, "opts")] = Job([src], lambda it, i=i: L.prompt_options(it[0], lang, i["prelude"]),
                                      lambda s, v: L.check_options(s, v, lang),
                                      parse=lambda o, n: [L.parse_json(o)])
        got = run_jobs(runner, jobs, attempts, stats, tag)
        opt_tr = {k[0]: v[0] for k, v in got.items()}
    jobs = {}
    for iid, i in insts.items():
        if lang != "en" and iid not in opt_tr:
            continue
        gl = L.glossary(i, opt_tr[iid]) if lang != "en" else None
        opts = L.all_options(i)
        ctx = {"options": opts, "opt_map": gl or {}, "exempt": L.option_exempt_words(i)}
        pctx = dict(ctx, need_change=True)
        jobs[(iid, "prelude")] = Job([i["prelude"]], lambda it, gl=gl, opts=opts: L.prompt_text(it[0], lang, opts, gl),
                                     lambda s, v, c=pctx: L.check_text(s, v, lang, c))
        evs = L.evidence_texts(i)
        for c in range(0, len(evs), CHUNK):
            jobs[(iid, f"ev{c // CHUNK}")] = Job(
                evs[c:c + CHUNK], lambda it, gl=gl, opts=opts: L.prompt_list(it, lang, opts, gl, "observation sentences"),
                lambda s, v, c=ctx: L.check_text(s, v, lang, c), is_list=True)
        qs = [q["text"] for q in i["queries"]]
        for c in range(0, len(qs), CHUNK):
            jobs[(iid, f"q{c // CHUNK}")] = Job(
                qs[c:c + CHUNK], lambda it, gl=gl, opts=opts: L.prompt_list(it, lang, opts, gl, "questions", True),
                lambda s, v, c=ctx: L.check_text(s, v, lang, c, is_question=True), is_list=True)
    got = run_jobs(runner, jobs, attempts, stats, tag)
    by = defaultdict(dict)
    for (iid, seg), v in got.items():
        by[iid][seg] = v
    out = defaultdict(list)
    for iid, i in insts.items():
        segs = by.get(iid, {})
        evs = L.evidence_texts(i)
        n_ev, n_q = (len(evs) + CHUNK - 1) // CHUNK, (len(i["queries"]) + CHUNK - 1) // CHUNK
        need = ["prelude"] + [f"ev{c}" for c in range(n_ev)] + [f"q{c}" for c in range(n_q)]
        fam = i["family"]
        stats[tag]["n_src"][fam] += 1
        if (lang != "en" and iid not in opt_tr) or any(s not in segs for s in need):
            stats[tag]["rejected"][fam] += 1
            continue
        ev_new = sum((segs[f"ev{c}"] for c in range(n_ev)), [])
        q_new = sum((segs[f"q{c}"] for c in range(n_q)), [])
        try:
            o = L.assemble(i, tag, segs["prelude"][0], dict(zip(evs, ev_new)), q_new,
                           opt_tr=opt_tr.get(iid), info={"lang": lang, "model": MODEL})
            validate(o)
        except Exception as e:  # noqa: BLE001
            stats[tag]["rejected"][fam] += 1
            stats[tag]["reasons"][f"assemble:{type(e).__name__}"] += 1
            stats[tag]["examples"].append({"key": [iid, "assemble"], "reasons": [str(e)[:200]]})
            continue
        stats[tag]["accepted"][fam] += 1
        out[fam].append(o)
    return tag, out


def framing(runner, sources, attempts, stats):
    """Loss framing, one prelude unit at a time. A unit is reframed only if all of its probability numbers are
    complemented (counts and times unchanged) and a rewrite back to the positive frame restores its exact numbers;
    otherwise the unit keeps its original wording. An instance is kept if at least one unit was reframed."""
    from .common import join_units, units
    tag = "framing"
    insts = {i["id"]: i for fam in sources.values() for i in fam if L.frame_eligible(i)}
    for fam, lst in sources.items():
        stats[tag]["n_src"][fam] += len(lst)
        stats[tag]["eligible"][fam] += sum(L.frame_eligible(i) for i in lst)

    def fcheck(src, t):
        if t.strip() == src.strip():
            return []  # declined (multi-alternative statement): the unit keeps its wording
        ok, n_comp = L.complement_match(src, t)
        n_prob = sum(k in ("pct", "dec") for _, k in L.prob_numbers(src))
        r = [] if ok else ["complement_numbers"]
        if ok and n_comp != n_prob:
            r.append("partial_complement")
        r += [x for x in L.check_text(src, t, "en", {}) if x.split(":")[0] in ("entity", "quoted", "length")]
        if L._count(L.NEG_EN, t) <= L._count(L.NEG_EN, src):
            r.append("no_added_negation")  # a loss frame names the complementary outcome ("does not ...")
        return r

    jobs = {}
    for iid, i in insts.items():
        for j, u in enumerate(units(i["prelude"])):
            if any(k in ("pct", "dec") for _, k in L.prob_numbers(u["text"])) and not u["text"].lstrip().startswith("-") \
                    and not re.search(r"percentage point|\bthe rest\b|\bremaining\b|\bremainder\b|times as|lower|higher",
                                      u["text"]):  # relative / residual statements have no simple complement
                jobs[(iid, f"u{j}")] = Job([u["text"]], lambda it: L.prompt_frame(it[0]), fcheck)
    got = run_jobs(runner, jobs, attempts, stats, tag)
    ujobs = {k: Job([v[0]], lambda it: L.prompt_unframe(it[0]),
                    lambda s, t, orig=jobs[k].srcs[0]: [] if L.number_tokens(t) == L.number_tokens(orig)
                    else ["roundtrip_numbers"]) for k, v in got.items()}
    back = run_jobs(runner, ujobs, attempts, stats, tag)
    out = defaultdict(list)
    for iid, i in insts.items():
        us = units(i["prelude"])
        done = []
        for j, u in enumerate(us):
            k = (iid, f"u{j}")
            if k in got and k in back and got[k][0].strip() != u["text"].strip():
                u["text"] = got[k][0]
                done.append(j)
        fam = i["family"]
        stats[tag]["units_eligible"][fam] += sum(1 for k in jobs if k[0] == iid)
        stats[tag]["units_reframed"][fam] += len(done)
        if not done:
            stats[tag]["rejected"][fam] += 1
            continue
        o = L.assemble(i, tag, join_units(us), {}, [q["text"] for q in i["queries"]],
                       info={"frame": "loss", "reframed_units": done, "n_units": len(us), "model": MODEL})
        validate(o)
        stats[tag]["accepted"][fam] += 1
        out[fam].append(o)
    return tag, out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=["data/test", "data/heldout"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--langs", default="en,de,es,hu,pt,zh")
    ap.add_argument("--framing", action="store_true")
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--out", default="data/v2/stress")
    ap.add_argument("--stats", default="logs/stress_llm_stats.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--resume", action="store_true", help="only process sources without an accepted output yet")
    args = ap.parse_args(argv)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "run on GPU 2 only (CUDA_VISIBLE_DEVICES=2)"
    model = args.model
    if model is None:
        from huggingface_hub import snapshot_download
        model = snapshot_download(MODEL, local_files_only=True)
    sources = load_sources(args.sources, args.n)
    if args.families:
        sources = {k: v for k, v in sources.items() if k in args.families}
    runner = Runner(model)
    stats_all = {}
    if Path(args.stats).exists():
        stats_all = json.loads(Path(args.stats).read_text())

    def new_stats():
        return {"n_src": Counter(), "accepted": Counter(), "rejected": Counter(), "eligible": Counter(),
                "units_eligible": Counter(), "units_reframed": Counter(),
                "attempts": Counter(), "pass_at": Counter(), "reasons": Counter(), "examples": []}

    tasks = [("rw", l) for l in args.langs.split(",") if l] + ([("frame", None)] if args.framing else [])
    for kind, lang in tasks:
        t0 = time.time()
        stats = defaultdict(new_stats)
        tag0 = "framing" if kind == "frame" else ("paraphrase" if lang == "en" else f"lang_{lang}")
        have = {}
        srcs = sources
        if args.resume:  # second pass (with feedback retries) over the instances the first pass rejected
            for fam in sources:
                p = Path(args.out) / tag0 / f"{fam}.jsonl"
                have[fam] = read_jsonl(p) if p.exists() else []
            done_ids = {o["meta"]["source_id"] for lst in have.values() for o in lst}
            srcs = {fam: [i for i in lst if i["id"] not in done_ids] for fam, lst in sources.items()}
        tag, out = rewrite(runner, srcs, lang, args.attempts, stats) if kind == "rw" else \
            framing(runner, srcs, args.attempts, stats)
        for fam in sources:
            lst = have.get(fam, []) + out.get(fam, [])
            order = {i["id"]: n for n, i in enumerate(sources[fam])}
            lst.sort(key=lambda o: order[o["meta"]["source_id"]])
            if lst:
                write_jsonl(Path(args.out) / tag / f"{fam}.jsonl", lst)
        st = stats[tag]
        st["examples"] = st["examples"][:40]
        st["seconds"] = round(time.time() - t0)
        key = tag + ("#repair" if args.resume else "")
        stats_all[key] = {k: (dict(v) if isinstance(v, Counter) else v) for k, v in st.items()}
        if args.resume:
            stats_all[key]["final_accepted"] = {fam: len(have.get(fam, [])) + len(out.get(fam, [])) for fam in sources}
        Path(args.stats).write_text(json.dumps(stats_all, indent=1, ensure_ascii=False))
        tot, acc = sum(st["n_src"].values()), sum(st["accepted"].values())
        print(f"== {tag}: {acc}/{tot} accepted ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
