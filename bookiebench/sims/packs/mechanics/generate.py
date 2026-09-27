"""Generate the BookieBench v2 `mechanics` pack.

    python -m bookiebench.sims.packs.mechanics.generate --out data/v2/mechanics --n 300                 # dev only
    python -m bookiebench.sims.packs.mechanics.generate --out data/v2/mechanics --n 300 --n-train 20000 # dev + train

writes <out>/dev/<family>.jsonl (10 decimals) and <out>/train/<family>.jsonl (8 decimals).
Deterministic in --seed (independent of --workers). --scale > 1 enlarges configurations (joints may exceed 256 cells).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import FAMILIES, generate, validate
from bookiebench.sims.core import dumps, joint_array, marginal, var_index


def _chunk(args):
    fam, split, lo, hi, seed, scale, nuis = args
    out = []
    for i in range(lo, hi):
        inst = generate(fam, split, i, seed, scale=scale, nuisance=nuis if split == "train" else 0.0)
        validate(inst)
        out.append(dumps(inst))
    return out


def sanity(path):
    """Exact-posterior accuracy vs mean confidence on the final marginals (a calibrated oracle has acc ~= conf)."""
    acc, conf, n, lv = [], [], 0, {}
    with open(path) as f:
        for line in f:
            inst = json.loads(line)
            n += 1
            lv[inst["level"]] = lv.get(inst["level"], 0) + 1
            J = joint_array(inst)
            for v in inst["variables"]:
                p = marginal(J, var_index(inst, v["name"]))
                acc.append(float(np.argmax(p) == inst["gold"][v["name"]]))
                conf.append(float(p.max()))
    return {"n": n, "acc": float(np.mean(acc)), "conf": float(np.mean(conf)), "levels": dict(sorted(lv.items()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2/mechanics")
    ap.add_argument("--n", type=int, default=300, help="dev instances per family")
    ap.add_argument("--n-train", type=int, default=0, help="train instances per family (0 = skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--nuisance", type=float, default=0.0, help="fraction of train instances with nuisance text")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    fams = [f for f in FAMILIES if not args.families or f in args.families]
    jobs = [("dev", f, args.n) for f in fams if args.n > 0] + [("train", f, args.n_train) for f in fams if args.n_train > 0]
    report = {}
    with Pool(args.workers) as pool:
        for split, fam, n in jobs:
            t0 = time.time()
            step = 100
            chunks = [(fam, split, lo, min(n, lo + step), args.seed, args.scale, args.nuisance) for lo in range(0, n, step)]
            path = Path(args.out) / split / f"{fam}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for lines in pool.imap(_chunk, chunks):
                    f.write("\n".join(lines) + "\n")
            msg = f"{split:6s} {fam:12s} {n:7d} -> {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)"
            if split == "dev":
                s = sanity(path)
                report[fam] = s
                msg += f"  acc={s['acc']:.3f} conf={s['conf']:.3f} levels={s['levels']}"
            print(msg, flush=True)
    if report:
        with open(Path(args.out) / "dev_sanity.json", "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
