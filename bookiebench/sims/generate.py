"""Generate BookieBench simulator data.

    python -m bookiebench.sims.generate --out data --n-train 20000 --n-test 300                       # original splits
    python -m bookiebench.sims.generate --out data --splits train --families randbn randhmm --n-train 100000 \
        --nuisance 0.3                                                                         # broad prior
    python -m bookiebench.sims.generate --out data --splits val robust test_prior --n-test 300

writes data/<split>/<family>.jsonl. Output is deterministic in --seed (independent of --workers).
--nuisance only applies to split "train" (split "robust" always has it; other splits never).
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

from . import SPLITS, generate
from .core import dumps, validate


def _chunk(args):
    fam, split, lo, hi, seed, nuis, dec, tv = args
    out = []
    for i in range(lo, hi):
        inst = generate(fam, split, i, seed, nuisance=nuis if split == "train" else 0.0, decimals=dec, tv=tv)
        validate(inst)
        out.append(dumps(inst))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--splits", nargs="*", default=["train", "test", "heldout"])
    ap.add_argument("--nuisance", type=float, default=0.0, help="fraction of train instances with nuisance text")
    ap.add_argument("--decimals", type=int, default=None, help="override rounding (default: per family)")
    ap.add_argument("--out-split", default=None, help="write the (single) split under this directory name")
    ap.add_argument("--tv", type=int, default=None, help="template version: 1 legacy (internal data), 2 release (default)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args(argv)

    jobs = []
    for split in args.splits:
        n = args.n_train if split == "train" else args.n_test
        for fam in SPLITS[split]:
            if args.families and fam not in args.families:
                continue
            jobs.append((split, fam, n))
    with Pool(args.workers) as pool:
        for split, fam, n in jobs:
            t0 = time.time()
            step = 250
            chunks = [(fam, split, lo, min(n, lo + step), args.seed, args.nuisance, args.decimals, args.tv)
                      for lo in range(0, n, step)]
            path = Path(args.out) / (args.out_split or split) / f"{fam}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for lines in pool.imap(_chunk, chunks):
                    f.write("\n".join(lines) + "\n")
            print(f"{split:10s} {fam:10s} {n:7d} -> {path} ({path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.1f}s)",
                  flush=True)


if __name__ == "__main__":
    main()
