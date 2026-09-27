"""CLI: python -m bookiebench.metrics.compare results/* [--data data] [--train data/train] [--ref DIR ...] [--recompute]

Cross-model markdown tables (overall, then one per split, then informative coherence), read from each directory's
report.json (a directory without one, or every directory with --recompute, is evaluated against --data).

Reference rows are always included, computed internally on the same splits:
- `ref:uniform_joint`  uniform over joint cells (needs nothing)
- `ref:indep_joint`    product of train-mean marginals, fitted from --train (skipped with a note if it is missing)
- `xref:<dir name>`    extra reference prediction directories given with --ref (e.g. review/shortcuts/results/
                       tfidf_joint); always evaluated fresh and never written to. Built-in refs also cover their splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import METRIC_COLUMNS, build_report, evaluate, markdown_table
from .loading import load_instances
from .refs import fit_indep, ref_predictions
from .report import run

COH_COLUMNS = ["n", "skill", "sens", "dutch", "dutch_per_bet", "dutch@0.005", "dutch_frac_exploitable",
               "optperm", "evperm", "mart", "para"]


def load_reports(dirs, data="data", recompute=False) -> list[dict]:
    reps = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        rj = d / "report.json"
        if rj.exists() and not recompute:
            with open(rj) as f:
                reps.append(json.load(f))
        elif any(d.glob("*.jsonl")):
            reps.append(run(d, data))
    return reps


def reference_reports(splits, data="data", train="data/train", extra=(), n_per_file=1500,
                      builtin=True) -> tuple[list[dict], list]:
    """Built-in refs (on `splits` plus every split the extra refs cover) and extra refs labelled `xref:<dir>`."""
    notes = []
    ext = []
    for d in extra:
        d = Path(d)
        if not any(d.glob("*.jsonl")):
            notes.append(f"ref {d}: no predictions")
            continue
        rep = run(d, data, write=False)
        rep["model"] = f"xref:{d.name}"
        ext.append(rep)
    reps = []
    splits = set(splits) | {s for r in ext for s in r.get("by_split", {})}
    instances = load_instances(data, splits) if splits and builtin else {}
    if instances:
        for name, table in (("uniform_joint", None), ("indep_joint", fit_indep(train, n_per_file))):
            if name == "indep_joint" and table is None:
                notes.append(f"ref:indep_joint skipped: no train data at {train}")
                continue
            recs, info = evaluate(instances, ref_predictions(instances, name, table), splits=splits)
            rep = build_report(recs)
            rep["model"] = f"ref:{name}"
            rep["info"] = info
            reps.append(rep)
    return reps + ext, notes


def render(reps: list[dict], notes=()) -> str:
    cols = ["n"] + METRIC_COLUMNS
    parts = ["## overall", "", markdown_table([(r.get("model", "?"), r["overall"]) for r in reps], cols, "model"), ""]
    splits = sorted({s for r in reps for s in r.get("by_split", {})})
    for s in splits:
        rows = [(r.get("model", "?"), r["by_split"][s]) for r in reps if s in r.get("by_split", {})]
        parts += [f"## split: {s}", "", markdown_table(rows, cols, "model"), ""]
    parts.insert(1, "_overall rows pool whatever splits each model was run on; compare models per split below._")
    for label, rows in [("overall", [(r.get("model", "?"), r["overall"]) for r in reps])] + [
            (f"split: {s}", [(r.get("model", "?"), r["by_split"][s]) for r in reps if s in r.get("by_split", {})])
            for s in splits]:
        inf = [(m, v) for m, v in rows if v.get("coh_valid")]
        parts += [f"## informative coherence, {label} (only models with skill > 0)", "",
                  markdown_table(inf, COH_COLUMNS, "model"), ""]
    parts += [f"_note: {n}_" for n in notes]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results/<model> directories")
    ap.add_argument("--data", default="data")
    ap.add_argument("--train", default="data/train", help="train data for ref:indep_joint")
    ap.add_argument("--ref", action="append", default=[], help="extra reference prediction directory (repeatable)")
    ap.add_argument("--no-refs", action="store_true", help="omit the built-in reference rows")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args(argv)
    reps = load_reports(a.results, a.data, a.recompute)
    if not reps:
        print("no reports found", file=sys.stderr)
        return 1
    notes = []
    if not a.no_refs or a.ref:
        splits = sorted({s for r in reps for s in r.get("by_split", {})})
        refs, notes = reference_reports(splits, a.data, a.train, a.ref, builtin=not a.no_refs)
        reps = refs + reps
    print(render(reps, notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
