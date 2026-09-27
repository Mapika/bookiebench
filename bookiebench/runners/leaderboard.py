"""Release leaderboard: results/release/LEADERBOARD.md (+ leaderboard.json) from results/release/<model>/.

    python -m bookiebench.runners.leaderboard [--models m1 m2 ...] [--jobs 8]

Scoring is bookiebench.metrics' own: `report.run` per model directory (writes its report.json) and
`compare.reference_reports` for the ref rows (ref:uniform_joint, ref:indep_joint, ref:prior), with groups from
sims_manifest.json and the coherence gate (dutch_inf only where skill >= 0.05 and sens >= 0.05). Data = the shared
subset results/release/_data/sub (sims_manifest.json copied in); train = data/release/train for the fitted refs.

Tables, in order: new_mechanics (headline), surface_transfer, in_family_v2, in_family, prior, realcoh (all sources, then
split by the realcoh manifest's in_decider_train; decider rows on in-train sources are marked †), stress (paired deltas,
stressed - unstressed on the same source ids). Each table lists the tempered models (-Tfit, headline: one temperature
per model, bookiebench.runners.finalize_release) with the ref rows, then the raw models; API models (deepseek-*) go in a
separate "LLM (API)" block; a model run on a subset (CoT) gets same-ids tables.
"""
import argparse
import json
import shutil
from multiprocessing import Pool
from pathlib import Path

from bookiebench.metrics.compare import reference_reports
from bookiebench.metrics.core import aggregate, evaluate, markdown_table
from bookiebench.metrics.loading import load_instances, load_predictions
from bookiebench.metrics.report import run

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "results" / "release"
SUB = REL / "_data" / "sub"
TRAIN = ROOT / "data" / "release" / "train"
ORDER = ["new_mechanics", "surface_transfer", "in_family_v2", "in_family", "prior"]
COLS = ["n", "skill", "skill_prior", "kl_marg", "kl_bin", "sens", "acc", "ece", "dutch_inf", "dutch@0.01", "answered_frac"]
RC_COLS = ["n", "acc", "ece", "logscore", "dutch", "dutch@0.01", "dutch_frac_exploitable", "optperm", "para",
           "answered_frac"]
DELTA = ["skill", "kl", "acc", "dutch", "dutch@0.01"]
API = ("deepseek",)


def _report(d):
    return run(d, SUB, write=True, train=TRAIN)


def _is_api(m):
    return m.startswith(API)


def _blocks(rows, names):
    """-> [(title, [(name, row)])]: tempered one-pass, raw one-pass, LLM (API)."""
    base = lambda n: n.replace(" †", "")
    one = [n for n in names if not _is_api(base(n))]
    return [("tempered (-Tfit, headline)", [(n, rows[n]) for n in one if base(n).endswith("-Tfit")]),
            ("raw (untempered)", [(n, rows[n]) for n in one if not base(n).endswith("-Tfit")]),
            ("LLM (API)", [(n, rows[n]) for n in names if _is_api(base(n))])]


def _preds(cache, m):
    if m not in cache:
        cache[m] = load_predictions(REL / m)[0]
    return cache[m]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=str(REL / "LEADERBOARD.md"))
    a = ap.parse_args(argv)
    shutil.copy(ROOT / "data" / "release" / "sims_manifest.json", SUB / "sims_manifest.json")
    models = a.models or sorted(p.name for p in REL.iterdir() if p.is_dir() and not p.name.startswith("_")
                                and any(p.glob("*.jsonl")))
    with Pool(max(1, min(a.jobs, len(models)))) as pool:
        reps = dict(zip(models, pool.map(_report, [str(REL / m) for m in models])))
    splits = sorted({s for r in reps.values() for s in r.get("by_split", {})})
    refs, notes = reference_reports(splits, SUB, TRAIN)
    refs = [r for r in refs if r["model"] in ("ref:uniform_joint", "ref:indep_joint", "ref:prior")]
    info = {m: r["info"] for m, r in reps.items()}
    partial = {m for m, i in info.items() if i.get("n_missing_inst", 0) > 0.5 * max(i.get("n_evaluated", 1), 1)}
    full = [m for m in models if m not in partial]
    J = {"models": models, "groups": {}, "stress": {}, "partial": sorted(partial), "notes": notes}
    md = ["# BookieBench release leaderboard (runners)", "",
          "Data: the shared subset `results/release/_data/sub` (`python -m bookiebench.runners.release_subset`): ≤100 "
          "instances per sims family (stress-paired ids first), 60 states per realcoh source, 30 stressed instances per "
          "family and transform (10 for long_*); every model is scored on the same instances. Scoring: bookiebench.metrics "
          "(report.run, compare.reference_reports; groups from sims_manifest.json; dutch_inf only where skill ≥ 0.05 "
          "and sens ≥ 0.05). Missing answers are scored as uniform (KL) and adversarial (dutch). Options are shown in "
          "the stored, per-instance shuffled order.", "",
          "**Tempering protocol.** Every model gets one temperature factor t, fitted with `bookiebench.runners.temper` "
          "(minimum mean KL) on its own predictions for 300 train instances per train family "
          "(`results/release/_data/calib`, never eval data; new_mechanics families have no train data, so their t comes "
          "from the other families). `-Tfit` rows are the headline; raw rows follow.", ""]
    temp = REL / "tempering.json"
    if temp.exists():
        md += ["| model | fitted factor t | absolute T | calibration KL raw → fitted |", "|---|---|---|---|"]
        for m, v in json.load(open(temp)).items():
            T = v.get("temperature")
            md.append(f"| {m} | {v['t_rel']:.3f} | {T:.3f} |" if isinstance(T, float) else f"| {m} | {v['t_rel']:.3f} | t × model default |")
            md[-1] += f" {v['kl_cal_run']:.4f} → {v['kl_cal_fitted']:.4f} |"
        md.append("")
    for g in ORDER:
        ref_rows = [(r["model"], r["by_group"][g]) for r in refs if g in r.get("by_group", {})]
        rows = {m: reps[m]["by_group"][g] for m in full if g in reps[m].get("by_group", {})}
        if not rows:
            continue
        n = next(iter(rows.values())).get("n")
        md += [f"## {g}{' (headline)' if g == 'new_mechanics' else ''} — {n} instances", ""]
        J["groups"][g] = {**dict(ref_rows), **rows}
        for title, rr in _blocks(rows, list(rows)):
            if rr:
                md += [f"**{title}**", "", markdown_table((ref_rows if title.startswith("tempered") else []) + rr, COLS,
                                                          "model"), ""]
    cache = {}
    rcman = json.load(open(ROOT / "data" / "release" / "realcoh" / "manifest.json"))["sources"]
    rc = load_instances(SUB / "realcoh")
    for label, flag in (("realcoh — all sources", None), ("realcoh — sources NOT in decider's training data", False),
                        ("realcoh — sources in decider's training data (decider rows marked †)", True)):
        inst = {i: x for i, x in rc.items() if flag is None or bool(rcman[x["family"]]["in_decider_train"]) == flag}
        srcs = sorted({x["family"] for x in inst.values()})
        rows = {}
        for m in full:
            p = {i: v for i, v in _preds(cache, m).items() if i in inst}
            if p:
                rows[m + (" †" if flag and m.startswith("decider") else "")] = aggregate(evaluate(inst, p, splits={"realcoh2"})[0])
        J["groups"][label] = rows
        md += [f"## {label}", "", f"{len(srcs)} sources, {len(inst)} states: {', '.join(srcs)}", ""]
        for title, rr in _blocks(rows, list(rows)):
            if rr:
                md += [f"**{title}**", "", markdown_table(rr, RC_COLS, "model"), ""]
    base_inst = {**load_instances(SUB / "test"), **load_instances(SUB / "heldout")}
    md += ["## stress — paired deltas (stressed − unstressed, same source ids)", "",
           "Cell: Δskill / Δacc / Δdutch@0.01 for the tempered models (raw deltas in leaderboard.json). Unstressed = "
           "the model's own test/heldout predictions on the source ids of the stressed instances it answered. long_* "
           "only where the context fits: decider ≤ long_16k, Julia-1 long_4k (8192 tokens), heads none (they read ≤ 2048 "
           "context tokens).", ""]
    sm = [m for m in full if m.endswith("-Tfit") and not _is_api(m)]
    md += ["| transform | " + " | ".join(sm) + " |", "|---" * (len(sm) + 1) + "|"]
    for t in sorted(p.name for p in (SUB / "stress").iterdir() if p.is_dir()):
        sinst = load_instances(SUB / "stress" / t)
        J["stress"][t] = {}
        cells = {}
        for m in full:
            d = REL / m / "stress" / t
            sp = {i: v for i, v in (load_predictions(d)[0] if d.is_dir() else {}).items() if i in sinst}
            if not sp:
                continue
            si = {i: sinst[i] for i in sp}
            src = {(x.get("meta", {}).get("source_id") or i.split("~")[0]) for i, x in si.items()}
            bi = {s: base_inst[s] for s in src if s in base_inst}
            P = _preds(cache, m)
            A = aggregate(evaluate(si, sp, splits={"stress"})[0])
            B = aggregate(evaluate(bi, {i: P[i] for i in bi if i in P}, splits={"test", "heldout"})[0])
            dd = {k: (None if A.get(k) is None or B.get(k) is None else A[k] - B[k]) for k in DELTA}
            J["stress"][t][m] = {"n": len(si), "delta": dd, "stressed": A, "unstressed": B}
            f = lambda x: "–" if x is None else f"{x:+.3f}"
            cells[m] = f"{f(dd['skill'])} / {f(dd['acc'])} / {f(dd['dutch@0.01'])}"
        md.append(f"| {t} | " + " | ".join(cells.get(m, "–") for m in sm) + " |")
    md.append("")
    if partial:
        allinst = load_instances(SUB, set(splits) - {"realcoh2"})
        for pm in sorted(partial):
            ids = set(_preds(cache, pm))
            md += [f"## same ids as {pm} (a subset run; every model scored on exactly these instances)", ""]
            for g in ORDER:
                inst = {i: x for i, x in allinst.items() if i in ids and
                        reps[pm].get("by_group", {}).get(g) is not None and x.get("split") in {"test", "test_prior", "heldout", "val", "dev"}}
                if not inst:
                    continue
                rows = [(m, aggregate(evaluate(inst, {i: v for i, v in _preds(cache, m).items() if i in inst})[0])) for m in models]
                md += [f"### {g} ({len(inst)} instances)", "", markdown_table(rows, COLS, "model"), ""]
    md += ["## runs", "", "| model | part | rows | seconds | rows/s | notes |", "|---|---|---|---|---|---|"]
    for m in models:
        for f in sorted((REL / m).glob("run_info__*.json")) + sorted((REL / m / "stress").glob("run_info__*.json")):
            s = json.load(open(f))["summary"]
            note = {k: s[k] for k in ("temperature", "mode", "truncated_frac", "missing_letter_frac", "parse_failures") if k in s}
            md.append(f"| {m} | {f.parent.name if f.parent.name == 'stress' else f.stem[10:]} | {s.get('rows')} | "
                      f"{s.get('seconds')} | {s.get('rows_per_s')} | {note} |")
    md += [""] + [f"_note: {n}_" for n in notes]
    Path(a.out).write_text("\n".join(md) + "\n")
    json.dump(J, open(Path(a.out).with_name("leaderboard.json"), "w"), indent=1, default=float)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
