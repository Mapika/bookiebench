"""Paired manifest of a stress output directory.

    PYTHONPATH=. python -m bookiebench.sims.packs.stress.manifest --out data/release/stress \
        --sources data/release/test data/release/heldout

writes <out>/manifest.json:
  transforms[<transform>][<family>] = {"source_file", "n_source", "n_kept", "kept_source_ids", "rejected_source_ids"}
  families[<family>]["kept_by_all"]  = source ids kept by EVERY transform (for one fully paired table)
LLM transforms reject instances that fail their checks, so a stressed-vs-unstressed comparison is only unbiased on
`kept_source_ids` (score the unstressed source instances restricted to the same ids). For `scaling`, the kept id is
the composite's first sub-world (`meta.source_id`); the other parts are listed in `meta.stress.source_ids`.
LLM transforms also have `kept_judge_clean_source_ids`: kept instances that an LLM judge (same model, temperature 0)
found equivalent to the source in every segment (a stricter, judge-dependent subset; framing is already filtered).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bookiebench.sims.core import iter_jsonl


# Transforms built but excluded from the v1 release (user decision). Their judge-filtered outputs are kept in
# logs/stress_excluded_v1/<tag>/ and the unfiltered ones in logs/stress_unfiltered/<tag>/.
EXCLUDED_V1 = {
    "lang_hu": "Excluded from v1: only 48.6% of source items survived the LLM judge (<50%), and a hand check of 20 "
               "judge-clean items found several meaning errors (e.g. 'batch' -> 'food', clubs -> spades, "
               "'Otherwise' -> 'possibly').",
    "framing": "Excluded from v1: only 26.6% of source items survived the round-trip + judge filter (<50%), and the "
               "judge detected only 73% of planted single-number changes on framed preludes.",
}


def build(out_dir, source_dirs, n=300):
    out_dir = Path(out_dir)
    sources = {}
    for d in source_dirs:
        for p in sorted(Path(d).glob("*.jsonl")):
            ids = []
            for j, inst in enumerate(iter_jsonl(p)):
                if j >= n:
                    break
                ids.append(inst["id"])
            sources[p.stem] = {"file": str(p), "ids": ids}
    man = {"description": __doc__.split("\n\n")[2].strip(), "n_per_family": n,
           "sources": {f: v["file"] for f, v in sources.items()}, "transforms": {}, "families": {}}
    man["excluded_v1"] = EXCLUDED_V1
    for tdir in sorted(p for p in out_dir.iterdir() if p.is_dir() and p.name not in EXCLUDED_V1):
        rows = {}
        for fam, src in sources.items():
            f = tdir / f"{fam}.jsonl"
            objs = [(o["meta"]["source_id"], o["meta"]["stress"].get("judge")) for o in iter_jsonl(f)] if f.exists() \
                else []
            kept = [i for i, _ in objs]
            ks = set(kept)
            judged = {i: j for i, j in objs if j is not None}
            rows[fam] = {"source_file": src["file"], "n_source": len(src["ids"]), "n_kept": len(kept),
                         "kept_source_ids": [i for i in src["ids"] if i in ks],
                         "rejected_source_ids": [i for i in src["ids"] if i not in ks]}
            if judged:  # LLM transforms: the subset the LLM judge found fully equivalent to the source
                rows[fam]["kept_judge_clean_source_ids"] = [i for i in src["ids"] if i in judged
                                                            and not judged[i]["different"]]
            assert len(ks) == len(kept) and ks <= set(src["ids"]), (tdir.name, fam)
        man["transforms"][tdir.name] = rows
    for fam, src in sources.items():
        keep = set(src["ids"])
        for rows in man["transforms"].values():
            keep &= set(rows[fam]["kept_source_ids"])
        man["families"][fam] = {"n_source": len(src["ids"]), "kept_by_all": [i for i in src["ids"] if i in keep]}
    man["summary"] = {t: {"kept": sum(r["n_kept"] for r in rows.values()),
                          "source": sum(r["n_source"] for r in rows.values()),
                          **({"judge_clean": sum(len(r.get("kept_judge_clean_source_ids", [])) for r in rows.values())}
                             if any("kept_judge_clean_source_ids" in r for r in rows.values()) else {})}
                      for t, rows in man["transforms"].items()}
    (out_dir / "manifest.json").write_text(json.dumps(man, indent=1))
    return man


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/release/stress")
    ap.add_argument("--sources", nargs="*", default=["data/release/test", "data/release/heldout"])
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args(argv)
    man = build(a.out, a.sources, a.n)
    for t, s in man["summary"].items():
        print(f"{t:16s} {s['kept']}/{s['source']}")


if __name__ == "__main__":
    main()
