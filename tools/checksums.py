#!/usr/bin/env python
"""Write or verify data/CHECKSUMS.sha256 (sha256sum format, paths relative to data/).

    python tools/checksums.py --write     # after changing any data file
    python tools/checksums.py             # verify (exit 1 on any mismatch, missing or unlisted file)
    cd data && sha256sum -c CHECKSUMS.sha256   # the same check without Python
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "CHECKSUMS.sha256"


REBUILT = ("release/stress/long_", "release/stress/scaling/")  # recipe outputs (git-ignored), verified by the recipe


def files():
    return sorted(p for p in DATA.rglob("*") if p.is_file() and p != OUT and not p.name.startswith(".")
                  and not str(p.relative_to(DATA)).startswith(REBUILT))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    cur = {str(p.relative_to(DATA)): sha(p) for p in files()}
    if a.write:
        OUT.write_text("".join(f"{h}  {n}\n" for n, h in cur.items()))
        print(f"{OUT}: {len(cur)} files")
        return 0
    want = dict(reversed(l.split("  ", 1)) for l in OUT.read_text().splitlines() if l.strip())
    want = {k.strip(): v for k, v in want.items()}
    bad = [n for n in want if cur.get(n) != want[n]] + [f"unlisted: {n}" for n in cur if n not in want]
    for b in bad:
        print("MISMATCH", b)
    print(f"{len(want)} listed, {len(bad)} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
