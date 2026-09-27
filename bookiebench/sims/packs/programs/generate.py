"""Generate the `programs` pack.

    python -m bookiebench.sims.packs.programs.generate --out data/v2/programs --n-dev 300 --n-train 20000

writes <out>/<split>/<family>.jsonl for split in {dev, train} (and `test` with --splits test --seed <secret>).
Deterministic in (--seed, family, split, idx), independent of --workers. Every instance is validated.
"""
from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

from bookiebench.sims.core import dumps, validate


def _chunk(args):
    pack, fam, split, lo, hi, seed = args
    mod = __import__(f"bookiebench.sims.packs.{pack}", fromlist=["generate"])
    out = []
    for i in range(lo, hi):
        inst = mod.generate(fam, split, i, seed)
        validate(inst)
        out.append(dumps(inst))
    return out


def main(argv=None, pack="programs"):
    mod = __import__(f"bookiebench.sims.packs.{pack}", fromlist=["FAMILIES"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"data/v2/{pack}")
    ap.add_argument("--n-dev", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="*", default=["dev", "train"])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    fams = args.families or list(mod.FAMILIES)
    with Pool(args.workers) as pool:
        for split in args.splits:
            n = {"dev": args.n_dev, "train": args.n_train, "test": args.n_test}[split]
            for fam in fams:
                t0 = time.time()
                step = 50
                chunks = [(pack, fam, split, lo, min(n, lo + step), args.seed) for lo in range(0, n, step)]
                path = Path(args.out) / split / f"{fam}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    for lines in pool.imap(_chunk, chunks):
                        f.write("\n".join(lines) + "\n")
                print(f"{pack:9s} {split:6s} {fam:14s} {n:7d} -> {path} ({path.stat().st_size / 1e6:.1f} MB, "
                      f"{time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
