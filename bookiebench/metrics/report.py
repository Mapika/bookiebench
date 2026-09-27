"""CLI: python -m bookiebench.metrics.report results/<model> [--data data] [--train DIR] [--skill-gate G] [--sens-gate G]

Prints markdown tables (overall, per group from the sims manifest, per split, per family) and writes
results/<model>/report.json. `skill_prior` uses label/template priors fitted on --train (default <data>/train).
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path

from .core import SENS_GATE, SKILL_GATE, build_report, evaluate, markdown_table
from .exact import exact_source
from .loading import default_train, load_groups, load_instances, load_predictions, prediction_splits
from .refs import fit_prior


@lru_cache(maxsize=4)
def _prior(train: str):
    return fit_prior(train)


def run(results_dir: str | Path, data: str | Path = "data", write: bool = True, train: str | Path | None = None,
        skill_gate: float = SKILL_GATE, sens_gate: float = SENS_GATE, prior=True) -> dict:
    results_dir = Path(results_dir)
    preds, model = load_predictions(results_dir)
    instances = load_instances(data, prediction_splits(results_dir))
    train = Path(train) if train else default_train(data)
    tables = _prior(str(train)) if prior is True else (prior or None)
    groups = load_groups(data)
    recs, info = evaluate(instances, preds, prior=tables, groups=groups)
    rep = build_report(recs, skill_gate, sens_gate)
    rep["model"] = model
    rep["info"] = {**info, "exact_source": exact_source(), "data": str(data),
                   "prior_train": str(train) if tables is not None else None, "groups": groups is not None}
    if write and results_dir.is_dir():
        with open(results_dir / "report.json", "w") as f:
            json.dump(rep, f, indent=2)
    return rep


def render(rep: dict) -> str:
    rows = [("overall", rep["overall"])]
    rows += [(f"group:{k}", v) for k, v in rep.get("by_group", {}).items()]
    rows += [(f"split:{k}", v) for k, v in rep["by_split"].items()]
    parts = [f"## {rep.get('model', '?')}", "", markdown_table(rows), ""]
    if rep["by_family"]:
        parts += ["### per family", "", markdown_table(list(rep["by_family"].items()), label="split/family"), ""]
    i = rep.get("info", {})
    parts.append(f"_evaluated {i.get('n_evaluated')} dataset instances of splits {i.get('splits')}; "
                 f"{i.get('n_missing_inst')} without a prediction (scored as unanswered); "
                 f"{i.get('n_pred_without_instance')} prediction ids without an instance; "
                 f"exact answers from {i.get('exact_source')}; prior fitted on {i.get('prior_train')}; "
                 f"gates {rep.get('gates')}_")
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", help="results/<model> directory (or a single predictions .jsonl)")
    ap.add_argument("--data", default="data", help="instance directory (default: data)")
    ap.add_argument("--train", default=None, help="train data for skill_prior (default <data>/train)")
    ap.add_argument("--skill-gate", type=float, default=SKILL_GATE)
    ap.add_argument("--sens-gate", type=float, default=SENS_GATE)
    ap.add_argument("--no-write", action="store_true", help="do not write report.json")
    a = ap.parse_args(argv)
    rep = run(a.results, a.data, write=not a.no_write, train=a.train, skill_gate=a.skill_gate,
              sens_gate=a.sens_gate)
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
