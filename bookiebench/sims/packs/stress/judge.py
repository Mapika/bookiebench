"""Semantic spot check of the LLM rewrites with an LLM judge (same local model, temperature 0).

    CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=. python \
        -m bookiebench.sims.packs.stress.judge --out data/release/stress --n 100

For each LLM transform, a deterministic sample of `--n` accepted instances is compared with its source instance:
  prelude    source vs rewritten prelude (framing: "is B logically equivalent to A")
  evidence   every realised evidence step, as numbered pairs
  queries    every query text, as numbered pairs (translations also show the option glossary)
The judge answers SAME / DIFFERENT per pair. Reported: the disagreement rate per segment and per instance (any pair
DIFFERENT). As a sensitivity control, the same prelude pairs are judged again with one number in the rewritten prelude
changed (a known-different pair); the detection rate on those says how much a low disagreement rate is worth.
Writes logs/stress_judge_<tag>.json (summary + every DIFFERENT verdict with its reason).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bookiebench.sims.core import iter_jsonl, write_jsonl

from . import llm as L

TRANSFORMS = ["paraphrase", "lang_de", "lang_es", "lang_hu", "lang_pt", "lang_zh", "framing"]
SYS = "You are a strict, careful fact checker. You compare meanings, not wording."


def _pair_prompt(a, b, lang, framing=False):
    if framing:
        task = ("Text B was produced from Text A by restating some probabilities as the probability of the "
                "complementary outcome (for example \"35% have a cough\" -> \"65% do not have a cough\"). Decide whether "
                "Text B describes exactly the same probability model as Text A: every stated probability must be "
                "correct after the restatement, nothing added or lost.")
    else:
        task = (f"Text B is a {'paraphrase' if lang == 'en' else 'translation into ' + L.LANGS[lang]} of Text A. "
                "Decide whether B states exactly the same facts as A: the same numbers attached to the same things, "
                "the same names, the same logical meaning (including every negation, 'and'/'or', conditions), "
                "nothing added and nothing left out. Differences in wording or style do not matter.")
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"{task}\n\nText A:\n{a}\n\nText B:\n{b}\n\nAnswer with one line: SAME or "
                                        f"DIFFERENT, followed by ' - ' and a short reason."}]


def _list_prompt(pairs, lang, what, glossary=None):
    rel = "paraphrase" if lang == "en" else "translation into " + L.LANGS[lang]
    items = "\n".join(f"{i + 1}. A: {a}\n   B: {b}" for i, (a, b) in enumerate(pairs))
    gl = (f"\nThe answer options were translated with this glossary: {json.dumps(glossary, ensure_ascii=False)}\n"
          if glossary else "")
    return [{"role": "system", "content": SYS},
            {"role": "user", "content":
                f"Each B below is a {rel} of the {what} A before it. For each pair decide whether B has exactly the "
                f"same meaning as A: same numbers, names and outcomes, same logic (every negation, 'and'/'or', "
                f"'if'), nothing added or left out. Wording and style do not matter.{gl}\n\n{items}\n\nReturn a JSON "
                f"array with one object per pair, in order: {{\"verdict\": \"SAME\" or \"DIFFERENT\", \"reason\": "
                f"short}}. Output only the JSON array."}]


def _verdict(text):
    t = L.clean_output(text).strip().upper()
    if t.startswith("SAME"):
        return "SAME"
    if t.startswith("DIFFERENT"):
        return "DIFFERENT"
    return "SAME" if "SAME" in t[:30] and "DIFFERENT" not in t[:30] else ("DIFFERENT" if "DIFFERENT" in t else None)


def _corrupt_number(text, rng):
    ms = list(re.finditer(r"\d+", text))
    if not ms:
        return None
    m = ms[int(rng.integers(len(ms)))]
    v = int(m.group(0))
    new = str(v + int(rng.choice([-3, -2, 2, 3])) if v > 5 else v + 2)
    return text[:m.start()] + new + text[m.end():]


def sample(out_dir, tag, n):
    items = []
    for f in sorted((Path(out_dir) / tag).glob("*.jsonl")):
        items += list(iter_jsonl(f))
    items.sort(key=lambda o: zlib.crc32(o["id"].encode()))
    return items[:n]


def annotate(runner, out_dir, tags, filter_tags=(), source=None, field="judge", model_name=None):
    """Judge EVERY instance of `tags`; store meta.stress.judge = {"different": bool, "prelude": verdict,
    "evidence_different": [...], "queries_different": [...]} and rewrite the files. Instances of `filter_tags`
    judged DIFFERENT are dropped (they then show up as rejected in the manifest). Returns per-tag counts."""
    res = {}
    for tag in tags:
        files = sorted((Path(out_dir) / tag).glob("*.jsonl"))
        insts = [(f, o) for f in files for o in iter_jsonl(f)]
        if not insts:
            continue
        framing = tag == "framing"
        convs, keys = [], []
        for n, (f, o) in enumerate(insts):
            lang = o["meta"]["stress"].get("lang", "en")
            s_ = source(o)
            convs.append(_pair_prompt(s_["prelude"], o["prelude"], lang, framing))
            keys.append((n, "prelude", None))
            if framing:
                continue
            gl = None
            if "option_map" in o["meta"]["stress"]:
                gl = {k: v for m in o["meta"]["stress"]["option_map"].values() for k, v in m.items()}
            ev = [(x["evidence"], y["evidence"]) for x, y in zip(s_["steps"], o["steps"])]
            convs.append(_list_prompt(ev, lang, "observation", gl))
            keys.append((n, "evidence", len(ev)))
            qs = [(x["text"], y["text"]) for x, y in zip(s_["queries"], o["queries"])]
            convs.append(_list_prompt(qs, lang, "question", gl))
            keys.append((n, "queries", len(qs)))
        outs = runner(convs, seed=0, temperature=0.0)
        J = defaultdict(lambda: {"different": False, "prelude": None, "evidence_different": [],
                                 "queries_different": [], "unparsed": 0})
        for (n, kind, k), out in zip(keys, outs):
            j = J[n]
            if k is None:
                v = _verdict(out)
                j["prelude"] = v
                if v == "DIFFERENT":
                    j["different"] = True
                    j["prelude_reason"] = L.clean_output(out)[:300]
                elif v is None:
                    j["unparsed"] += 1
                continue
            vals = L.parse_json(out)
            if not isinstance(vals, list) or len(vals) != k:
                j["unparsed"] += k
                continue
            for i, x in enumerate(vals):
                v = str(x.get("verdict", "")).upper() if isinstance(x, dict) else ""
                if v.startswith("DIFF"):
                    j[f"{kind}_different"].append(i)
                    j["different"] = True
        by_file = defaultdict(list)
        c = Counter()
        for n, (f, o) in enumerate(insts):
            o["meta"]["stress"][field] = dict(J[n], model=model_name or MODEL_NAME)
            c["different" if J[n]["different"] else "same"] += 1
            if tag in filter_tags and (J[n]["different"] or J[n]["unparsed"]):
                c["dropped"] += 1
                continue
            by_file[f].append(o)
        for f in files:
            write_jsonl(f, by_file.get(f, []))
        res[tag] = dict(c, n=len(insts))
        print("annotate", tag, dict(c, n=len(insts)), flush=True)
    return res


MODEL_NAME = "Qwen/Qwen3.8-27B-FP8"


class HttpRunner:
    """OpenAI-compatible chat endpoint as a second, independent judge (e.g. a DeepSeek model served by the user).
    At most `concurrency` requests in flight (the server is shared)."""

    def __init__(self, base_url, model, concurrency=32, max_tokens=2048, timeout=600):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.conc, self.max_tokens, self.timeout = model, concurrency, max_tokens, timeout

    def _one(self, conv, temperature):
        import urllib.request
        body = json.dumps({"model": self.model, "messages": conv, "temperature": temperature,
                           "max_tokens": self.max_tokens}).encode()
        for attempt in range(4):
            try:
                req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())["choices"][0]["message"].get("content") or ""
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return ""
        return ""

    def __call__(self, convs, seed=0, temperature=0.0):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.conc) as ex:
            return list(ex.map(lambda c: self._one(c, temperature), convs))


def _flip_negation(text, lang):
    """Remove the first negation cue (a planted meaning flip); None if the text has none."""
    pat = L.NEG.get(lang, L.NEG_EN)
    m = re.search(pat, text, flags=re.I)
    if not m:
        return None
    out = (text[:m.start()] + text[m.end():]).replace("  ", " ")
    return out if out != text else None


def controls(runner, out_dir, tags, n, source, logs):
    """Planted subtle errors on a sample of `n` accepted instances per transform:
    prelude_number (one number changed), query_negation (one negation removed from one question, inside the full
    question list), evidence_error (one number changed, else one negation removed, in one observation).
    Reports the detection rate per control type and the false-DIFFERENT rate on the untouched items of those lists."""
    out = {}
    for tag in tags:
        insts = sample(out_dir, tag, n)
        if not insts:
            continue
        rng = np.random.default_rng(zlib.crc32(("ctl" + tag).encode()))
        framing = tag == "framing"
        convs, keys = [], []
        for o in insts:
            lang = o["meta"]["stress"].get("lang", "en")
            s_ = source(o)
            bad = _corrupt_number(o["prelude"], rng)
            if bad:
                convs.append(_pair_prompt(s_["prelude"], bad, lang, framing))
                keys.append(("prelude_number", None, None))
            if framing:
                bad = _flip_negation(o["prelude"], lang)
                if bad:
                    convs.append(_pair_prompt(s_["prelude"], bad, lang, True))
                    keys.append(("prelude_negation", None, None))
                continue
            gl = None
            if "option_map" in o["meta"]["stress"]:
                gl = {k: v for m in o["meta"]["stress"]["option_map"].values() for k, v in m.items()}
            qs = [[x["text"], y["text"]] for x, y in zip(s_["queries"], o["queries"])]
            cand = [i for i, (_, b) in enumerate(qs) if _flip_negation(b, lang)]
            if cand:
                i = int(rng.choice(cand))
                qs[i][1] = _flip_negation(qs[i][1], lang)
                convs.append(_list_prompt([tuple(x) for x in qs], lang, "question", gl))
                keys.append(("query_negation", i, len(qs)))
            ev = [[x["evidence"], y["evidence"]] for x, y in zip(s_["steps"], o["steps"])]
            cand = [i for i, (_, b) in enumerate(ev) if re.search(r"\d", b) or _flip_negation(b, lang)]
            if cand:
                i = int(rng.choice(cand))
                ev[i][1] = _corrupt_number(ev[i][1], rng) if re.search(r"\d", ev[i][1]) else _flip_negation(ev[i][1], lang)
                convs.append(_list_prompt([tuple(x) for x in ev], lang, "observation", gl))
                keys.append(("evidence_error", i, len(ev)))
        outs = runner(convs, seed=0, temperature=0.0)
        det, fp = defaultdict(Counter), Counter()
        for (kind, i, k), o in zip(keys, outs):
            if i is None:
                det[kind]["detected" if _verdict(o) == "DIFFERENT" else "missed"] += 1
                continue
            vals = L.parse_json(o)
            if not isinstance(vals, list) or len(vals) != k:
                det[kind]["unparsed"] += 1
                continue
            vv = [str(x.get("verdict", "")).upper().startswith("DIFF") if isinstance(x, dict) else False for x in vals]
            det[kind]["detected" if vv[i] else "missed"] += 1
            fp["different"] += sum(v for j, v in enumerate(vv) if j != i)
            fp["items"] += k - 1
        row = {k: {"n": sum(c.values()), "detection": round(c["detected"] / max(1, sum(c.values())), 3), **c}
               for k, c in det.items()}
        row["untouched_false_different"] = round(fp["different"] / max(1, fp["items"]), 4)
        out[tag] = row
        print("controls", tag, json.dumps(row), flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/release/stress")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--transforms", nargs="*", default=TRANSFORMS)
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--annotate", action="store_true", help="judge every instance and store meta.stress.<field>")
    ap.add_argument("--controls", action="store_true", help="planted subtle-error controls on --n per transform")
    ap.add_argument("--backend", default="qwen", choices=["qwen", "http"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--field", default="judge")
    ap.add_argument("--filter", nargs="*", default=["framing"], help="with --annotate: drop DIFFERENT instances")
    a = ap.parse_args(argv)
    if a.backend == "http":
        runner, model_name = HttpRunner(a.base_url, a.model_id, a.concurrency), a.model_id
    else:
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "run on GPU 2 only"
        from huggingface_hub import snapshot_download

        from .llm_run import MODEL, Runner
        runner, model_name = Runner(snapshot_download(MODEL, local_files_only=True)), MODEL_NAME
    src_cache = {}

    def source(o):
        sf = o["meta"].get("source_file") or f"data/{o['meta']['source_split']}/{o['family']}.jsonl"
        if sf not in src_cache:
            src_cache[sf] = {d["id"]: d for d in iter_jsonl(sf)}
        return src_cache[sf][o["meta"]["source_id"]]

    if a.controls:
        res = controls(runner, a.out, a.transforms, a.n, source, a.logs)
        Path(a.logs, f"stress_judge_controls_{a.field}.json").write_text(json.dumps(res, indent=1))
        return
    if a.annotate:
        res = annotate(runner, a.out, a.transforms, set(a.filter), source, field=a.field, model_name=model_name)
        Path(a.logs, f"stress_judge_annotate_{a.field}.json").write_text(json.dumps(res, indent=1))
        return
    summary = {}
    for tag in a.transforms:
        insts = sample(a.out, tag, a.n)
        if not insts:
            continue
        lang = insts[0]["meta"]["stress"].get("lang", "en")
        framing = tag == "framing"
        rng = np.random.default_rng(zlib.crc32(tag.encode()))
        convs, keys = [], []
        for o in insts:
            s = source(o)
            convs.append(_pair_prompt(s["prelude"], o["prelude"], lang, framing))
            keys.append((o["id"], "prelude", None))
            bad = _corrupt_number(o["prelude"], rng)
            if bad is not None:
                convs.append(_pair_prompt(s["prelude"], bad, lang, framing))
                keys.append((o["id"], "control", None))
            if framing:
                continue
            gl = None
            if "option_map" in o["meta"]["stress"]:
                gl = {k: v for m in o["meta"]["stress"]["option_map"].values() for k, v in m.items()}
            ev = [(x["evidence"], y["evidence"]) for x, y in zip(s["steps"], o["steps"])]
            convs.append(_list_prompt(ev, lang, "observation", gl))
            keys.append((o["id"], "evidence", len(ev)))
            qs = [(x["text"], y["text"]) for x, y in zip(s["queries"], o["queries"])]
            convs.append(_list_prompt(qs, lang, "question", gl))
            keys.append((o["id"], "queries", len(qs)))
        outs = runner(convs, seed=0, temperature=0.0)
        seg = defaultdict(Counter)
        per_inst = defaultdict(bool)
        diffs = []
        for (iid, kind, n), out in zip(keys, outs):
            if n is None:
                v = _verdict(out)
                seg[kind][v or "unparsed"] += 1
                if kind == "prelude" and v == "DIFFERENT":
                    per_inst[iid] = True
                    diffs.append({"id": iid, "segment": kind, "reason": L.clean_output(out)[:400]})
                elif kind == "prelude":
                    per_inst[iid] |= False
                continue
            vals = L.parse_json(out)
            if not isinstance(vals, list) or len(vals) != n:
                seg[kind]["unparsed"] += n
                continue
            for i, x in enumerate(vals):
                v = str((x or {}).get("verdict", "")).upper() if isinstance(x, dict) else ""
                v = "SAME" if v.startswith("SAME") else ("DIFFERENT" if v.startswith("DIFF") else "unparsed")
                seg[kind][v] += 1
                if v == "DIFFERENT":
                    per_inst[iid] = True
                    diffs.append({"id": iid, "segment": kind, "item": i, "reason": str(x.get("reason"))[:300]})
        n_inst = len(insts)
        row = {"n_instances": n_inst, "lang": lang,
               "instance_disagreement": round(sum(per_inst[o["id"]] for o in insts) / n_inst, 4)}
        for kind, c in seg.items():
            tot = sum(c.values())
            key = "control_detection" if kind == "control" else f"{kind}_disagreement"
            row[key] = round(c["DIFFERENT"] / max(1, tot), 4)
            row[f"{kind}_counts"] = dict(c)
        summary[tag] = row
        Path(a.logs, f"stress_judge_{tag}.json").write_text(json.dumps({"summary": row, "different": diffs},
                                                                      indent=1, ensure_ascii=False))
        print(tag, json.dumps(row), flush=True)
    Path(a.logs, "stress_judge_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
