"""Uniform tempering protocol for the release leaderboard (after run_release.sh and run_calib.sh).

    python -m bookiebench.runners.finalize_release [--dry-run]

For every model: fit ONE temperature factor t with bookiebench.runners.temper on its base-variant predictions for the train
calibration sample (results/release/_calib/<run>, 300 per train family, bookiebench.runners.calib_subset), then write
    results/release/<model>/        raw (untempered; the model's own default temperature)
    results/release/<model>-Tfit/   every answer (all groups, all variants, stress/*) rescaled by t
The logit baselines were run at an earlier fitted temperature T0 (directories named <base>-Tfit by run_release.sh).
Those runs move to results/release/_runs/<base>@T<T0>/; their raw copy is the exact rescale by 1/T0 (softmax(l/T0) ** T0
∝ softmax(l)), and their -Tfit copy is the rescale by the t fitted on calibration runs made at the same T0 (so the
absolute temperature is T0 * t). The fitted values go to results/release/tempering.json.
"""
import argparse
import json
import shutil
from pathlib import Path

from bookiebench.runners import temper

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "results" / "release"
CAL = REL / "_calib"
CAL_DATA = REL / "_data" / "calib" / "train"
SUB = REL / "_data" / "sub"   # instances for the joint rule of temper.apply (joint-emitting records)
T0 = {"Qwen3.5-4B-Base": 1.0949506966709828, "Qwen3.5-9B-Base": 1.3079565733946328,
      "gemma-4-12B": 1.0525045024081028, "Qwen3.8-27B": 1.0574877202165875}
NATIVE = ["decider-0.8b", "decider-2b", "decider-4b", "decider-35b-a3b", "Julia-1", "heads-p3-joint-r50-kd",
          "heads-p3-indep-r50-kd", "heads-joint-2b-lr3e6", "heads-indep-2b-lr3e6", "heads-hybrid-2b",
          "heads-decider-product",
          "deepseek-v41-flash-logit", "deepseek-v41-flash-cot"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", help="refit only these models (merged into the existing tempering.json)")
    a = ap.parse_args(argv)
    out = json.load(open(REL / "tempering.json")) if a.only and (REL / "tempering.json").exists() else {}
    runs = REL / "_runs"
    for base, t0 in T0.items():
        if a.only and base not in a.only:
            continue
        src = REL / f"{base}-Tfit"
        moved = runs / f"{base}@T{t0:.4f}"
        cal = CAL / f"{base}-Tfit"
        if not (src.exists() or moved.exists()) or not cal.exists():
            print(f"skip {base}: missing run or calibration"); continue
        fit = temper.fit(str(cal), str(CAL_DATA))
        out[base] = dict(run_temperature=t0, t_rel=fit["t"], temperature=t0 * fit["t"], kl_cal_run=fit["kl_t1"],
                         kl_cal_fitted=fit["kl_fitted"], n=fit["n"])
        print(base, out[base])
        if a.dry_run:
            continue
        if src.exists() and not moved.exists():
            runs.mkdir(exist_ok=True); shutil.move(str(src), str(moved))
        for d in (REL / base, REL / f"{base}-Tfit"):
            shutil.rmtree(d, ignore_errors=True)
        temper.apply(str(moved), 1.0 / t0, out_dir=str(REL / base), name=base)
        temper.apply(str(moved), fit["t"], out_dir=str(REL / f"{base}-Tfit"), name=f"{base}-Tfit")
    for m in NATIVE:
        if a.only and m not in a.only:
            continue
        src, cal = REL / m, CAL / m
        if not src.exists() or not cal.exists():
            print(f"skip {m}: missing run or calibration"); continue
        fit = temper.fit(str(cal), str(CAL_DATA))
        out[m] = dict(run_temperature="model default", t_rel=fit["t"], kl_cal_run=fit["kl_t1"], kl_cal_fitted=fit["kl_fitted"],
                      n=fit["n"], joint_records=fit["joint_records"],
                      rule="joint-tempered" if fit["joint_records"] else "per-answer")
        print(m, out[m])
        if not a.dry_run:
            shutil.rmtree(REL / f"{m}-Tfit", ignore_errors=True)
            temper.apply(str(src), fit["t"], out_dir=str(REL / f"{m}-Tfit"), name=f"{m}-Tfit", data=[SUB])
    if not a.dry_run:
        json.dump(out, open(REL / "tempering.json", "w"), indent=1)


if __name__ == "__main__":
    main()
