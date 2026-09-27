"""Calibration sample for the uniform tempering protocol: 300 instances per TRAIN family from data/release/train.

    python -m bookiebench.runners.calib_subset     # -> results/release/_data/calib/train/<family>.jsonl

Only files whose sims_manifest group is "train" are used (never eval files; the 9 held-out v2 families have no train
file, so their temperature comes from the other families, like every new_mechanics family). Seeded per family.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "data" / "release"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "release" / "_data" / "calib"))
    a = ap.parse_args(argv)
    man = json.load(open(REL / "sims_manifest.json"))["files"]
    out = Path(a.out) / "train"; out.mkdir(parents=True, exist_ok=True)
    for rel, info in sorted(man.items()):
        if info["group"] != "train" or not (REL / rel).exists():
            continue
        with open(REL / rel) as f:
            n = sum(1 for _ in f)
        keep = set(random.Random(f"{a.seed}:{rel}").sample(range(n), min(a.n, n)))
        with open(REL / rel) as f, open(out / Path(rel).name, "w") as g:
            for i, line in enumerate(f):
                if i in keep:
                    g.write(line)
        print(rel, len(keep))


if __name__ == "__main__":
    main()
