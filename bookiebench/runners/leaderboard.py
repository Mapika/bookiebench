"""Release leaderboard: results/release/LEADERBOARD.md (+ leaderboard.json) from results/release/<model>/.

    python -m bookiebench.runners.leaderboard [--models m1 m2 ...]

Scoring uses bookiebench.metrics (evaluate / aggregate, the same code as `bookiebench.metrics.compare`) on the shared subset
results/release/_data/sub, one table per group:
    in_family (test) · prior (test_prior) · surface_transfer (heldout) · new_mechanics (val + */dev)
    realcoh, split into sources NOT in decider's training data and sources that are (manifest in_decider_train; decider
    rows there are marked †) · stress: per transform, paired deltas (stressed - unstressed) on the same source ids.
Reference rows: ref:uniform_joint and ref:indep_joint (bookiebench.metrics.refs, indep fitted on data/release/train) and, on
simulator groups, ref:oracle (every answer from the exact joint).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bookiebench.metrics.core import aggregate, evaluate, markdown_table
from bookiebench.metrics.exact import exact_answer
from bookiebench.metrics.loading import load_instances, load_predictions
from bookiebench.metrics.refs import fit_indep, ref_predictions

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "results" / "release"
SUB = REL / "_data" / "sub"
GROUPS = [("in_family", ["test"]), ("prior", ["test_prior"]), ("surface_transfer", ["heldout"]),
          ("new_mechanics", ["val", "mechanics/dev", "programs/dev", "tables/dev"])]
COLS = ["n", "answered_frac", "skill", "kl", "sens", "logscore", "acc", "ece", "dutch", "dutch_per_bet",
        "dutch_frac_exploitable", "optperm", "evperm", "mart", "cover@0.1", "size@0.1"]
RC_COLS = ["n", "answered_frac", "logscore", "acc", "ece", "dutch", "dutch_per_bet", "dutch_frac_exploitable",
           "optperm", "para", "cover@0.1", "size@0.1"]
DELTA = ["skill", "kl", "acc", "logscore", "dutch", "dutch_per_bet", "optperm"]


def _insts(dirs):
    out = {}
    for d in dirs:
        out.update(load_instances(SUB / d))
    return out


def _oracle(instances):
    """Every query answered exactly at its own step (the ceiling: skill 1, dutch 0, sens 1)."""
    out = {}
    for i, inst in instances.items():
        ans = {}
        for q in inst["queries"]:
            a = exact_answer(inst, q)
            if np.all(np.isfinite(a)):
                ans[q["id"]] = a.tolist()
        out[i] = {"base": {"answers": ans}}
    return out


def _score(instances, preds):
    preds = {i: v for i, v in preds.items() if i in instances}
    splits = {inst.get("split") for inst in instances.values()}
    recs, _ = evaluate(instances, preds, splits=splits)
    return recs


def _refs(instances, table, oracle=True):
    rows = [("ref:uniform_joint", _score(instances, ref_predictions(instances, "uniform_joint")))]
    if table is not None:
        rows.append(("ref:indep_joint", _score(instances, ref_predictions(instances, "indep_joint", table))))
    if oracle:
        rows.append(("ref:oracle", _score(instances, _oracle(instances))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--train", default=str(ROOT / "data" / "release" / "train"))
    ap.add_argument("--out", default=str(REL / "LEADERBOARD.md"))
    a = ap.parse_args(argv)
    models = a.models or sorted(p.name for p in REL.iterdir() if p.is_dir() and not p.name.startswith("_"))
    table = fit_indep(a.train)
    preds = {m: load_predictions(REL / m)[0] for m in models}
    runinfo = {}
    for m in models:
        for f in sorted((REL / m).glob("run_info__*.json")):
            runinfo.setdefault(m, []).append(json.load(open(f))["summary"])
    J = {"groups": {}, "stress": {}, "models": models}
    md = ["# BookieBench release leaderboard (runners)", "",
          "Shared subset `results/release/_data/sub` (`python -m bookiebench.runners.release_subset`): per family ≤100 "
          "instances (stress-paired ids first), realcoh 60 states per source, stress 30 per family and transform "
          "(10 for long_*). Every model is scored on the same instances; a missing answer is scored as uniform / "
          "adversarial by bookiebench.metrics. Options are shown in the stored (per-instance shuffled) order.", ""]

    def cover(instances, m):
        return sum(1 for i in instances if "base" in preds[m].get(i, {})) / max(len(instances), 1)

    partial = set()

    def table_rows(instances, pr, cols, oracle=True, mark=lambda m: m, allow_partial=False):
        rows = [(n, aggregate(r)) for n, r in _refs(instances, table, oracle)]
        for m in models:
            if not any(i in instances for i in pr[m]):
                continue
            if not allow_partial and cover(instances, m) < 0.5:        # a subset run (CoT): see the same-ids tables
                partial.add(m); continue
            rows.append((mark(m), aggregate(_score(instances, pr[m]))))
        return rows

    for g, dirs in GROUPS:
        inst = _insts(dirs)
        rows = table_rows(inst, preds, COLS)
        J["groups"][g] = {n: r for n, r in rows}
        md += [f"## {g} ({', '.join(dirs)}; {len(inst)} instances)", "", markdown_table(rows, COLS, "model"), ""]
        fams = defaultdict(dict)
        for i, x in inst.items():
            fams[x["family"]][i] = x
        if g == "new_mechanics":                               # skill per family (compact)
            fr = [(f, {m: aggregate(_score(fi, preds[m])).get("skill") for m in models}) for f, fi in sorted(fams.items())]
            md += ["skill per family:", "", "| family | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
            md += [f"| {f} | " + " | ".join("–" if v[m] is None else f"{v[m]:.3f}" for m in models) + " |" for f, v in fr]
            md.append("")
    # realcoh
    man = json.load(open(ROOT / "data" / "release" / "realcoh" / "manifest.json"))["sources"]
    rc = load_instances(SUB / "realcoh")
    for label, flag in (("realcoh, sources NOT in decider's training data", False),
                        ("realcoh, sources in decider's training data (decider rows marked †)", True)):
        inst = {i: x for i, x in rc.items() if bool(man[x["family"]]["in_decider_train"]) == flag}
        srcs = sorted({x["family"] for x in inst.values()})
        mark = (lambda m: m + " †" if flag and m.startswith("decider") else m)
        rows = table_rows(inst, preds, RC_COLS, oracle=False, mark=mark)
        J["groups"][label] = {n: r for n, r in rows}
        md += [f"## {label}", "", f"sources: {', '.join(srcs)} ({len(inst)} states)", "",
               markdown_table(rows, RC_COLS, "model"), ""]
    # partial (CoT) runs: every model on the same ids
    for pm in sorted(partial):
        md += [f"## same ids as {pm} (a subset run; every model scored on exactly these instances)", ""]
        for g, dirs in GROUPS + [("realcoh", ["realcoh"])]:
            inst = _insts(dirs)
            ids = {i for i in inst if "base" in preds[pm].get(i, {})}
            if not ids:
                continue
            sub = {i: inst[i] for i in ids}
            cols = RC_COLS if g == "realcoh" else COLS
            rows = table_rows(sub, preds, cols, oracle=(g != "realcoh"), allow_partial=True)
            J["groups"][f"{g} (ids of {pm})"] = {n: r for n, r in rows}
            md += [f"### {g} ({len(sub)} instances)", "", markdown_table(rows, cols, "model"), ""]
    # stress: paired deltas
    base_inst = _insts(["test", "heldout"])
    md += ["## stress: paired deltas (stressed − unstressed, same source ids)", "",
           "Each cell: Δskill / Δacc / Δdutch. Unstressed = the model's own test/heldout predictions on the source ids "
           "of the stressed instances it answered. long_* only for models whose context fits (decider: ≤ long_16k; "
           "Julia-1: long_4k at 8192 tokens; heads: none, they read ≤ 2048 context tokens).", ""]
    trs = sorted(p.name for p in (SUB / "stress").iterdir() if p.is_dir())
    md += ["| transform | " + " | ".join(models) + " |", "|---" * (len(models) + 1) + "|"]
    for t in trs:
        sinst = load_instances(SUB / "stress" / t)
        cells = []
        J["stress"][t] = {}
        for m in models:
            d = REL / m / "stress" / t
            sp = load_predictions(d)[0] if d.is_dir() else {}
            sp = {i: v for i, v in sp.items() if i in sinst}
            if not sp:
                cells.append("–"); continue
            si = {i: sinst[i] for i in sp}
            src = {i: (x.get("meta", {}).get("source_id") or i.split("~")[0]) for i, x in si.items()}
            bi = {s: base_inst[s] for s in set(src.values()) if s in base_inst}
            A = aggregate(_score(si, sp)); B = aggregate(_score(bi, preds[m]))
            dd = {k: (None if A.get(k) is None or B.get(k) is None else A[k] - B[k]) for k in DELTA}
            J["stress"][t][m] = {"stressed": A, "unstressed": B, "delta": dd, "n": len(si)}
            f = lambda x: "–" if x is None else f"{x:+.3f}"
            cells.append(f"{f(dd['skill'])} / {f(dd['acc'])} / {f(dd['dutch'])}")
        md.append(f"| {t} | " + " | ".join(cells) + " |")
    md += ["", "## runs", "", "| model | rows | seconds | rows/s | notes |", "|---|---|---|---|---|"]
    for m in models:
        for s in runinfo.get(m, []):
            note = {k: s[k] for k in ("temperature", "mode", "truncated_frac", "missing_letter_frac", "parse_failures")
                    if k in s}
            md.append(f"| {m} | {s.get('rows')} | {s.get('seconds')} | {s.get('rows_per_s')} | {note} |")
    Path(a.out).write_text("\n".join(md) + "\n")
    json.dump(J, open(Path(a.out).with_name("leaderboard.json"), "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
