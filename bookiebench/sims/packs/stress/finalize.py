"""Judge-based final filtering of the LLM transforms, judge agreement, and the paired manifest.

    PYTHONPATH=. python -m bookiebench.sims.packs.stress.finalize --require judge [judge2]

For every LLM transform in --out, an instance is kept only if EVERY judge field in --require exists and is clean
(no DIFFERENT verdict and no unparsed verdict on any segment: prelude, evidence, queries). The unfiltered files are
backed up once to --backup (default logs/stress_unfiltered/) and filtering always starts from that backup, so the rule
can be tightened later (e.g. once a second judge has annotated the backup) without losing items. With two judge
fields present, per-transform Cohen's kappa (instance level: clean vs not) is written to logs/stress_judge_kappa.json.
Finally the manifest is rebuilt.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bookiebench.sims.core import iter_jsonl, write_jsonl

from . import manifest
from .llm import zh_double_negation

LLM_TRANSFORMS = ["paraphrase", "lang_de", "lang_es", "lang_hu", "lang_pt", "lang_zh", "framing"]


def clean(j) -> bool:
    return j is not None and not j.get("different") and not j.get("unparsed")


_SRC = {}


def _source(o):
    sf = o["meta"]["source_file"]
    if sf not in _SRC:
        _SRC[sf] = {d["id"]: d for d in iter_jsonl(sf)}
    return _SRC[sf][o["meta"]["source_id"]]


def programmatic_ok(tag, o) -> bool:
    """Deterministic post-checks that the judges miss (see llm.zh_double_negation)."""
    if tag == "lang_zh":
        src = _source(o)
        return not any(zh_double_negation(a["text"], b["text"]) for a, b in zip(src["queries"], o["queries"]))
    return True


def kappa(a, b) -> float:
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/release/stress")
    ap.add_argument("--backup", default="logs/stress_unfiltered")
    ap.add_argument("--require", nargs="+", default=["judge"])
    ap.add_argument("--sources", nargs="*", default=["data/release/test", "data/release/heldout"])
    a = ap.parse_args(argv)
    out, bak = Path(a.out), Path(a.backup)
    report = {"require": a.require, "transforms": {}}
    for tag in LLM_TRANSFORMS:
        if tag in manifest.EXCLUDED_V1 or not (out / tag).exists():
            continue
        if not (bak / tag).exists():
            shutil.copytree(out / tag, bak / tag)
        row = {"before": 0, "kept": 0, "programmatic_dropped": 0, "families": {}}
        pairs = []
        for f in sorted((bak / tag).glob("*.jsonl")):
            insts = list(iter_jsonl(f))
            keep = [o for o in insts if all(clean(o["meta"]["stress"].get(k)) for k in a.require)]
            n0 = len(keep)
            keep = [o for o in keep if programmatic_ok(tag, o)]
            row["programmatic_dropped"] += n0 - len(keep)
            for o in insts:
                js = [o["meta"]["stress"].get(k) for k in a.require]
                if len(js) >= 2 and all(j is not None for j in js):
                    pairs.append((clean(js[0]), clean(js[1])))
            write_jsonl(out / tag / f.name, keep)
            row["families"][f.stem] = {"before": len(insts), "kept": len(keep)}
            row["before"] += len(insts)
            row["kept"] += len(keep)
        if pairs:
            row["kappa"] = round(kappa([p[0] for p in pairs], [p[1] for p in pairs]), 4)
            row["clean_rate"] = {a.require[0]: round(sum(p[0] for p in pairs) / len(pairs), 4),
                                 a.require[1]: round(sum(p[1] for p in pairs) / len(pairs), 4)}
        report["transforms"][tag] = row
        print(tag, {k: v for k, v in row.items() if k != "families"}, flush=True)
    Path("logs/stress_finalize.json").write_text(json.dumps(report, indent=1))
    man = manifest.build(out, a.sources)
    for t, s in man["summary"].items():
        frac = s["kept"] / max(1, s["source"])
        print(f"{t:16s} {s['kept']}/{s['source']} ({100 * frac:.1f}%)" + ("  <50% KEPT" if frac < 0.5 else ""))


if __name__ == "__main__":
    main()
