"""Build realcoh v2 (BookieBench v2 pack `realcoh`, level R): data/v2/realcoh/<source>.jsonl + manifest.json.

    python -m bookiebench.realcoh.build_v2                     # 25 sources x 200 states, seed 0 (internal)
    python -m bookiebench.realcoh.build_v2 --release           # + data/release/realcoh (licence-aware)
    python -m bookiebench.realcoh.build_v2 rebuild --release data/release/realcoh --out DIR   # text from HF cache
    python -m bookiebench.realcoh.build_v2 --sources ledgar,sports --n 50 --out /tmp/rc

Deterministic in (seed, source, idx). Offline (HF cache only), CPU only. The fineweb-edu sources scan the sorted
sample files until they have enough candidates (a few minutes the first time in a process).

Instance format = realcoh v1 (SPEC "Realcoh notes"): one step with "joint": null, "gold" with only the labelled
variables, "paraphrases" {qid: [text, ...]} (runners ask them as variants "para:<i>"). New in v2:
- tags "level": "R", "pack": "realcoh", "family": <source>, "transform": "none"; meta: source, source_ref
  (dataset, revision, file, row/id), in_decider_train, likely_pretraining (bookiebench/realcoh/provenance.py)
- linked question sets (see bookiebench/realcoh/queries_v2.py): marginals, coarse-option sums, cumulative/monotone
  thresholds, conjunctions (2- and 3-way), negations, negated conjunctions, disjunctions, within-variable "either",
  Bayes-linked conditionals in both directions P(A|B), P(B|A) with P(A), P(B), P(A&B) all asked, and P(not A|B);
  each query has "rel" (and "link" for a Bayes pair). Every relation is a constraint over the product outcome space,
  so `dutch` sees all of them with the existing metrics.
- "variables" carry the extra spec fields "coarse" (a deterministic coarsening of that variable) and "cum"
  (threshold clauses of an ordinal variable); `meta.gold_coarse` holds gold for coarse variables.
- forecast_news: the linked-forecast family (events A, B, their conjunction, disjunction, conditionals, negations and
  a monotone "when" variable over the same real, dated article).
- Release (--release): `keep` sources ship full text; `ids_only` sources ship evidence=null + meta.evidence_sha256 and
  are reconstructed with `rebuild`, which rebuilds the source and checks ids, source_ref and text hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from pathlib import Path

from .queries_v2 import make_queries_v2
from .provenance import DROPPED, PROVENANCE
from .sources_v2 import SOURCES_V2

MAX_CELLS = 256
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")


def mask_emails(text):
    """Every state text, in every source, has e-mail addresses replaced by [EMAIL] (user decision, 2026-09-27)."""
    return EMAIL.sub("[EMAIL]", text)


def instance_variables(variables):
    out = []
    for v in variables:
        d = {"name": v["name"], "options": list(v["options"])}
        if v.get("coarse"):
            c = v["coarse"]
            d["coarse"] = {"name": c["name"], "options": list(c["options"]), "map": list(c["map"])}
        if v.get("cum"):
            d["ordinal"] = True
        out.append(d)
    return out


def build_source(name, n=200, seed=0):
    fn, variables, prelude, opts = SOURCES_V2[name]
    cells = 1
    for v in variables:
        cells *= len(v["options"])
    assert 2 <= len(variables) <= 4 and cells <= MAX_CELLS, (name, cells)
    rng = random.Random(f"v2:{seed}:{name}")
    states = fn(rng, n)
    assert len(states) == n, (name, len(states))
    prov = PROVENANCE[name]
    out = []
    for i, (text, gold, ref) in enumerate(states):
        text = mask_emails(text)
        qrng = random.Random(f"v2:{seed}:{name}:{i}")
        qs, para = make_queries_v2(qrng, variables, opts.get("templates"), opts.get("forecast", False))
        gold = {k: int(g) for k, g in gold.items() if g is not None}
        gc = {}
        for v in variables:
            if v.get("coarse") and v["name"] in gold:
                gc[v["coarse"]["name"]] = v["coarse"]["map"][gold[v["name"]]]
        out.append({
            "id": f"realcoh2-{name}-{i:06d}", "family": name, "split": "realcoh2",
            "level": "R", "pack": "realcoh", "transform": "none",
            "variables": instance_variables(variables),
            "steps": [{"evidence": text, "joint": None}],
            "prelude": prelude, "gold": gold, "queries": qs, "paraphrases": para,
            "meta": {"source": name, "version": 2, "gold_coarse": gc, "source_ref": ref,
                     "in_decider_train": prov["in_decider_train"], "likely_pretraining": prov["likely_pretraining"]},
        })
    return out


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def release_rows(name, insts):
    """The release version of a source: full text for `keep`, ids + source_ref + text hash + questions for `ids_only`."""
    mode = PROVENANCE[name]["release_mode"]
    if mode == "keep":
        return insts
    assert mode == "ids_only", (name, mode)
    out = []
    for inst in insts:
        r = json.loads(json.dumps(inst))
        r["steps"][0]["evidence"] = None
        r["meta"]["evidence_sha256"] = text_sha(inst["steps"][0]["evidence"])
        r["meta"]["release_mode"] = "ids_only"
        out.append(r)
    return out


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_release(release_dir, built):
    """built: {source: instances}. Writes <release_dir>/<source>.jsonl, manifest.json and NOTICE.md."""
    os.makedirs(release_dir, exist_ok=True)
    mpath = os.path.join(release_dir, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {"sources": {}}
    for name, insts in built.items():
        rows = release_rows(name, insts)
        path = os.path.join(release_dir, f"{name}.jsonl")
        _write(path, rows)
        prov = PROVENANCE[name]
        man["sources"][name] = dict(n=len(rows), release_mode=prov["release_mode"], license=prov["license"],
                                    dataset=prov["dataset"], in_decider_train=prov["in_decider_train"],
                                    likely_pretraining=prov["likely_pretraining"],
                                    sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
    man["dropped"] = DROPPED
    man["text_masking"] = "e-mail addresses -> [EMAIL] in every state text (all sources)"
    from .sources import PINNED
    man["revisions"] = {PROVENANCE[k]["dataset"].split(" (")[0]: PINNED[PROVENANCE[k]["dataset"].split(" (")[0]]
                        for k in man["sources"]}
    man["rebuild"] = "python -m bookiebench.realcoh.build_v2 rebuild --release " + release_dir + " --out <dir>"
    with open(mpath, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    with open(os.path.join(release_dir, "NOTICE.md"), "w") as f:
        f.write("# realcoh v2 release: sources, licences and notices\n\n"
                "`ids_only` sources ship without their text; rebuild it from the Hugging Face datasets with\n"
                "`python -m bookiebench.realcoh.build_v2 rebuild --release <this dir> --out <dir>` (needs the datasets in the local HF cache,\n"
                "at the revisions pinned in manifest.json `revisions`: `huggingface-cli download <repo> --repo-type dataset --revision <rev>`).\n\n"
                "E-mail addresses in all state texts are masked as `[EMAIL]`, in every source (`rebuild` applies the same masking).\n\n"
                "| source | dataset | licence | release mode | in decider training | likely in pretraining |\n|---|---|---|---|---|---|\n")
        for name in sorted(man["sources"]):
            p = PROVENANCE[name]
            f.write(f"| {name} | {p['dataset']} | {p['license']} | {p['release_mode']} | {p['in_decider_train']} | {p['likely_pretraining']} |\n")
        for name in sorted(man["sources"]):
            if PROVENANCE[name].get("notice"):
                f.write(f"\n**{name}**: {PROVENANCE[name]['notice']}\n")


def rebuild(release_dir, out_dir, sources=None):
    """Fill the text of ids_only release files from the HF cache: rebuild the source deterministically, check the ids
    and the text hashes, and write complete instances. Returns {source: n}."""
    man = json.load(open(os.path.join(release_dir, "manifest.json")))
    os.makedirs(out_dir, exist_ok=True)
    done = {}
    for name in sources or sorted(man["sources"]):
        rows = [json.loads(l) for l in open(os.path.join(release_dir, f"{name}.jsonl"))]
        if all(r["steps"][0]["evidence"] is not None for r in rows):
            _write(os.path.join(out_dir, f"{name}.jsonl"), rows); done[name] = len(rows); continue
        full = {r["id"]: r for r in build_source(name, man.get("n", len(rows)), man.get("seed", 0))}
        out = []
        for r in rows:
            src = full.get(r["id"])
            if src is None or src["meta"]["source_ref"] != r["meta"]["source_ref"]:
                raise RuntimeError(f"{r['id']}: not reproduced by the local HF cache (source_ref differs)")
            text = src["steps"][0]["evidence"]
            if text_sha(text) != r["meta"]["evidence_sha256"]:
                raise RuntimeError(f"{r['id']}: rebuilt text hash differs (different dataset revision?)")
            r = json.loads(json.dumps(r))
            r["steps"][0]["evidence"] = text
            r["meta"].pop("evidence_sha256", None); r["meta"].pop("release_mode", None)   # = the internal build
            out.append(r)
        _write(os.path.join(out_dir, f"{name}.jsonl"), out)
        done[name] = len(out)
    return done


def summarize(name, insts):
    from collections import Counter
    rel = Counter(q["rel"] for i in insts for q in i["queries"])
    gold = Counter(k for i in insts for k in i["gold"])
    return {"n": len(insts), "queries": sum(len(i["queries"]) for i in insts),
            "queries_per_state": round(sum(len(i["queries"]) for i in insts) / max(1, len(insts)), 2),
            "variables": [v["name"] for v in insts[0]["variables"]] if insts else [],
            "cells": int(__import__("numpy").prod([len(v["options"]) for v in insts[0]["variables"]])) if insts else 0,
            "gold": dict(gold), "gold_frac": {k: round(c / len(insts), 3) for k, c in gold.items()}, "rel": dict(rel)}


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[2] / "data"
    if argv[:1] == ["rebuild"]:
        ap = argparse.ArgumentParser(description="rebuild the text of an ids_only release from the HF cache")
        ap.add_argument("--release", default=str(root / "release" / "realcoh"))
        ap.add_argument("--out", required=True)
        ap.add_argument("--sources", default=None)
        a = ap.parse_args(argv[1:])
        for k, v in rebuild(a.release, a.out, a.sources.split(",") if a.sources else None).items():
            print(f"{k}: {v} states rebuilt")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(root / "v2" / "realcoh"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=",".join(SOURCES_V2))
    ap.add_argument("--release", nargs="?", const=str(root / "release" / "realcoh"), default=None,
                    help="also write the licence-aware release (default dir data/release/realcoh)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    mpath = os.path.join(a.out, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest.setdefault("sources", {})
    for k in list(manifest["sources"]):
        if k not in SOURCES_V2:
            del manifest["sources"][k]
    built = {}
    for s in a.sources.split(","):
        insts = build_source(s, a.n, a.seed)
        built[s] = insts
        path = os.path.join(a.out, f"{s}.jsonl")
        _write(path, insts)
        info = summarize(s, insts)
        info["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        info.update({k: PROVENANCE[s][k] for k in ("dataset", "license", "release_mode", "in_decider_train", "decider_note",
                                                   "likely_pretraining", "pretraining_note")})
        manifest["sources"][s] = info
        print(f"{s}: {info['n']} states, {info['queries_per_state']} queries/state, {info['cells']} cells, gold {info['gold']}", flush=True)
    manifest.update(seed=a.seed, n=a.n, version=2, dropped=DROPPED)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    if a.release:
        man_r = os.path.join(a.release, "manifest.json")
        write_release(a.release, built)
        m = json.load(open(man_r)); m.update(seed=a.seed, n=a.n, version=2)
        json.dump(m, open(man_r, "w"), indent=1, sort_keys=True)
        print(f"release written to {a.release}")


if __name__ == "__main__":
    main()
