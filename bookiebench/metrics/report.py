"""CLI: python -m bookiebench.metrics.report results/<model> [--data data]

Prints markdown tables (overall, per split, per family) and writes results/<model>/report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import build_report, evaluate, markdown_table
from .exact import exact_source
from .loading import load_instances, load_predictions, prediction_splits


def run(results_dir: str | Path, data: str | Path = "data", write: bool = True) -> dict:
    results_dir = Path(results_dir)
    preds, model = load_predictions(results_dir)
    instances = load_instances(data, prediction_splits(results_dir))
    recs, info = evaluate(instances, preds)
    rep = build_report(recs)
    rep["model"] = model
    rep["info"] = {**info, "exact_source": exact_source(), "data": str(data)}
    if write and results_dir.is_dir():
        with open(results_dir / "report.json", "w") as f:
            json.dump(rep, f, indent=2)
    return rep


def render(rep: dict) -> str:
    rows = [("overall", rep["overall"])]
    rows += [(f"split:{k}", v) for k, v in rep["by_split"].items()]
    parts = [f"## {rep.get('model', '?')}", "", markdown_table(rows), ""]
    if rep["by_family"]:
        parts += ["### per family", "", markdown_table(list(rep["by_family"].items()), label="split/family"), ""]
    i = rep.get("info", {})
    parts.append(f"_evaluated {i.get('n_evaluated')} dataset instances of splits {i.get('splits')}; "
                 f"{i.get('n_missing_inst')} without a prediction (scored as unanswered); "
                 f"{i.get('n_pred_without_instance')} prediction ids without an instance; "
                 f"exact answers from {i.get('exact_source')}_")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="results/<model> directory (or a single predictions .jsonl)")
    ap.add_argument("--data", default="data", help="instance directory (default: data)")
    ap.add_argument("--no-write", action="store_true", help="do not write report.json")
    a = ap.parse_args(argv)
    rep = run(a.results, a.data, write=not a.no_write)
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
