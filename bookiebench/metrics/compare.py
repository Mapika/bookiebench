"""CLI: python -m bookiebench.metrics.compare results/* [--data data] [--train DIR] [--ref DIR ...] [--recompute]
                                     [--skill-gate G] [--sens-gate G]

Cross-model markdown tables: overall, one per manifest group (read at runtime from <data>/sims_manifest.json: in_family,
in_family_v2, prior, surface_transfer, new_mechanics, realcoh, stress, ...), one per split, then informative coherence
(only rows with coh_valid: skill_prior >= 0.05, or skill where no prior was fitted, and sens >= 0.05). Reports are read from each directory's report.json; a directory without one, a report
without the current gates / groups, or every directory with --recompute, is evaluated against --data.

Reference rows are always included, computed internally on the same splits:
- `ref:uniform_joint`  uniform over joint cells (needs nothing)
- `ref:indep_joint`    product of train-mean marginals keyed by option text (fitted from --train)
- `ref:prior`          per family, the better of `ref:label_prior` and `ref:template_uj` (train-fitted label and
                       question-template priors); new_mechanics families get uniform_joint. Both components are
                       listed too. `skill_prior` in every row is measured against min(uniform_joint, label, template).
- `xref:<dir name>`    extra reference prediction directories given with --ref; evaluated fresh, never written to.
Train-fitted refs are skipped with a note when --train does not exist.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import GATE_ON, METRIC_COLUMNS, SENS_GATE, SKILL_GATE, build_report, evaluate, markdown_table
from .loading import default_train, load_groups, load_instances
from .refs import fit_indep, ref_predictions
from .report import _prior, run

COH_COLUMNS = ["n", "skill", "skill_prior", "sens", "dutch", "dutch_per_bet", "dutch@0.005",
               "dutch_frac_exploitable", "optperm", "evperm", "mart", "para"]


def load_reports(dirs, data="data", recompute=False, train=None, skill_gate=SKILL_GATE,
                 sens_gate=SENS_GATE) -> list[dict]:
    reps = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        rj = d / "report.json"
        rep = None
        if rj.exists() and not recompute:
            with open(rj) as f:
                rep = json.load(f)
            stale = (rep.get("gates") != {"skill": skill_gate, "sens": sens_gate, "on": GATE_ON}
                     or "by_group" not in rep or "skill_prior" not in rep.get("overall", {}))
            if stale:
                rep = None
        if rep is None and any(d.glob("*.jsonl")):
            rep = run(d, data, train=train, skill_gate=skill_gate, sens_gate=sens_gate)
        if rep is not None:
            reps.append(rep)
    return reps


def _fam_kl(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["split"], r["family"])].extend(r["kl_marg"] + r["kl_bin"])
    return {k: (float(np.mean(v)) if v else np.inf) for k, v in by.items()}


def reference_reports(splits, data="data", train=None, extra=(), n_per_file=1500, builtin=True,
                      skill_gate=SKILL_GATE, sens_gate=SENS_GATE) -> tuple[list[dict], list]:
    """Built-in refs (on `splits` plus every split the extra refs cover) and extra refs labelled `xref:<dir>`."""
    notes = []
    ext = []
    train = Path(train) if train else default_train(data)
    for d in extra:
        d = Path(d)
        if not any(d.glob("*.jsonl")):
            notes.append(f"ref {d}: no predictions")
            continue
        rep = run(d, data, write=False, train=train, skill_gate=skill_gate, sens_gate=sens_gate)
        rep["model"] = f"xref:{d.name}"
        ext.append(rep)
    reps = []
    splits = set(splits) | {s for r in ext for s in r.get("by_split", {})}
    instances = load_instances(data, splits) if splits and builtin else {}
    if not instances:
        return reps + ext, notes
    groups = load_groups(data)
    prior = _prior(str(train))
    if prior is None:
        notes.append(f"ref:indep_joint, ref:prior and skill_prior skipped: no train data at {train}")

    def one(name, table):
        recs, info = evaluate(instances, ref_predictions(instances, name, table, groups), splits=splits,
                              prior=prior, groups=groups)
        return recs, info

    def report(label, recs, info):
        rep = build_report(recs, skill_gate, sens_gate)
        rep["model"] = label
        rep["info"] = info
        return rep

    recs, info = one("uniform_joint", None)
    reps.append(report("ref:uniform_joint", recs, info))
    if prior is not None:
        recs, info = one("indep_joint", fit_indep(train, n_per_file))
        reps.append(report("ref:indep_joint", recs, info))
        lab, info = one("label_prior", prior)
        tmp, _ = one("template_uj", prior)
        kl_l, kl_t = _fam_kl(lab), _fam_kl(tmp)
        best = [r for r in lab if kl_l[(r["split"], r["family"])] <= kl_t[(r["split"], r["family"])]]
        best += [r for r in tmp if kl_t[(r["split"], r["family"])] < kl_l[(r["split"], r["family"])]]
        reps.append(report("ref:prior", best, info))
        reps.append(report("ref:label_prior", lab, info))
        reps.append(report("ref:template_uj", tmp, info))
    return reps + ext, notes


def render(reps: list[dict], notes=()) -> str:
    cols = ["n"] + METRIC_COLUMNS
    parts = ["## overall", "_overall rows pool whatever splits each model was run on; compare models per group or "
             "split below._", "", markdown_table([(r.get("model", "?"), r["overall"]) for r in reps], cols, "model"),
             ""]
    sections = []
    groups = sorted({g for r in reps for g in r.get("by_group", {})})
    for g in groups:
        sections.append((f"group: {g}", [(r.get("model", "?"), (r.get("by_group") or {})[g]) for r in reps
                                         if g in (r.get("by_group") or {})]))
    splits = sorted({s for r in reps for s in r.get("by_split", {})})
    for s in splits:
        sections.append((f"split: {s}", [(r.get("model", "?"), r["by_split"][s]) for r in reps
                                         if s in r.get("by_split", {})]))
    for label, rows in sections:
        parts += [f"## {label}", "", markdown_table(rows, cols, "model"), ""]
    gates = next((r.get("gates") for r in reps if r.get("gates")), {"skill": SKILL_GATE, "sens": SENS_GATE})
    parts += [f"_coherence gate (coh_valid, dutch_inf, informative-coherence tables): skill_prior >= {gates['skill']} "
              f"(skill where no prior tables were fitted) and sens >= {gates['sens']} (skipped where sens is "
              f"undefined). skill_prior is measured against the best of uniform_joint, label_prior and "
              f"template_uj fitted on train._", ""]
    for label, rows in [("overall", [(r.get("model", "?"), r["overall"]) for r in reps])] + sections:
        inf = [(m, v) for m, v in rows if v.get("coh_valid")]
        parts += [f"## informative coherence, {label} (only rows with skill_prior >= {gates['skill']} and "
                  f"sens >= {gates['sens']})", "", markdown_table(inf, COH_COLUMNS, "model"), ""]
    parts += [f"_note: {n}_" for n in notes]
    return "\n".join(parts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="results/<model> directories")
    ap.add_argument("--data", default="data")
    ap.add_argument("--train", default=None, help="train data for the fitted refs (default <data>/train)")
    ap.add_argument("--ref", action="append", default=[], help="extra reference prediction directory (repeatable)")
    ap.add_argument("--no-refs", action="store_true", help="omit the built-in reference rows")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--skill-gate", type=float, default=SKILL_GATE)
    ap.add_argument("--sens-gate", type=float, default=SENS_GATE)
    a = ap.parse_args(argv)
    reps = load_reports(a.results, a.data, a.recompute, a.train, a.skill_gate, a.sens_gate)
    if not reps:
        print("no reports found", file=sys.stderr)
        return 1
    notes = []
    if not a.no_refs or a.ref:
        splits = sorted({s for r in reps for s in r.get("by_split", {})})
        refs, notes = reference_reports(splits, a.data, a.train, a.ref, builtin=not a.no_refs,
                                        skill_gate=a.skill_gate, sens_gate=a.sens_gate)
        reps = refs + reps
    print(render(reps, notes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
