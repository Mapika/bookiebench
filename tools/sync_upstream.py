#!/usr/bin/env python
"""Maintainer tool: refresh the vendored code (and optionally the release metadata) from the upstream research repo.

    python tools/sync_upstream.py --src /path/to/upstream            # code + tests only
    python tools/sync_upstream.py --src /path/to/upstream --data     # also sims_manifest.json + eval/dev/realcoh files

The upstream package is called `honest`; here it is `bookiebench`. The sync copies the public parts only
(sims incl. packs, metrics, realcoh, the runner core and the model runners; never `honest/heads` or `review/`),
renames the imports, and replaces machine-specific defaults with environment variables. Files that exist only here
(bookiebench/cli.py, bookiebench/runners/baselines.py) are left alone. Run the test suite afterwards and review
`git diff` before committing.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIRS = ["sims", "metrics", "realcoh"]
RUNNERS = ["__init__.py", "core.py", "logit_runner.py", "decider_runner.py", "julia_runner.py", "api_runner.py",
           "temper.py", "leaderboard.py", "release_subset.py"]
LOCAL_ONLY = {"bookiebench/cli.py", "bookiebench/runners/baselines.py"}
SKIP_TESTS = {"test_heads.py", "conftest.py"}          # trained-heads tests; our local conftest.py
DATA_DIRS = ["test", "test_prior", "heldout", "val", "mechanics/dev", "programs/dev", "tables/dev"]

HF_DEFAULT = 'os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))'
LITERAL_FIXES = [  # (regex, replacement): machine paths -> env vars / generic commands
    (r'os\.environ\.get\("HF_HOME", "[^"]+"\)', HF_DEFAULT),
    (r'^PUBLIC_SRC = "[^"]+"', 'PUBLIC_SRC = os.environ.get("DECIDER_SRC", "")  # optional checkout of the public decider repo'),
    (r'else /\S+/decider-public\. --decider-src', 'else $DECIDER_SRC (a checkout of the public decider repo). --decider-src'),
    (r'^DECIDER_PUBLIC = "[^"]+"', 'DECIDER_PUBLIC = os.environ.get("BOOKIEBENCH_DECIDER_PUBLIC", "")  # optional audit'),
    (r'^DECIDER_PRIVATE = "[^"]+"', 'DECIDER_PRIVATE = os.environ.get("BOOKIEBENCH_DECIDER_PRIVATE", "")  # maintainers only'),
    (r'^HF = "[^"]+/datasets"', f'HF = os.path.join({HF_DEFAULT}, "datasets")'),
    (r'/\S+/\.venv/bin/python\b', 'python'),
    (r'\.venv/bin/python\b', 'python'),
]
RENAMES = [
    (r'\bhonest\.(sims|metrics|realcoh|runners)\b', r'bookiebench.\1'),
    (r'\bhonest/(sims|metrics|realcoh|runners)\b', r'bookiebench/\1'),
    (r'\bfrom honest import\b', 'from bookiebench import'),
    (r'HONEST_METRICS_OWN_EXACT', 'BOOKIEBENCH_METRICS_OWN_EXACT'),
    (r'\bHonestBench\b', 'BookieBench'),                  # the project's public name
]


def fix_text(s: str) -> str:
    for pat, rep in RENAMES:
        s = re.sub(pat, rep, s)
    for pat, rep in LITERAL_FIXES:
        s = re.sub(pat, rep, s, flags=re.M)
    if "os.environ" in s and not re.search(r"^import os\b", s, flags=re.M):
        s = re.sub(r"^(import re\b)", r"import os\n\1", s, count=1, flags=re.M)
    return s


def copy_tree(src: Path, dst: Path, skip=frozenset()):
    for p in sorted(src.rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc" or p.name in skip or p.is_dir():
            continue
        if p.suffix == ".sh":
            continue                                    # upstream lane scripts reference internal paths
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".py":
            out.write_text(fix_text(p.read_text()))
        else:
            shutil.copy2(p, out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="upstream repo root (contains honest/ and tests/)")
    ap.add_argument("--data", action="store_true", help="also copy data/release metadata and eval/dev/realcoh files")
    ap.add_argument("--no-realcoh", action="store_true", help="with --data: leave data/release/realcoh untouched")
    a = ap.parse_args(argv)
    src = Path(a.src)
    if not (src / "honest" / "sims").is_dir():
        sys.exit(f"{src}: no honest/sims")
    pkg = ROOT / "bookiebench"
    for d in PKG_DIRS:
        shutil.rmtree(pkg / d, ignore_errors=True)
        copy_tree(src / "honest" / d, pkg / d)
    for f in RUNNERS:
        (pkg / "runners" / f).write_text(fix_text((src / "honest" / "runners" / f).read_text()))
    copy_tree(src / "tests", ROOT / "tests", skip=SKIP_TESTS)
    if a.data:
        rel = src / "data" / "release"
        dst = ROOT / "data" / "release"
        for d in DATA_DIRS + ([] if a.no_realcoh else ["realcoh"]):
            (dst / d).mkdir(parents=True, exist_ok=True)
            for f in sorted((rel / d).glob("*")):
                if f.is_file():
                    shutil.copy2(f, dst / d / f.name)
        shutil.copy2(rel / "sims_manifest.json", dst / "sims_manifest.json")
        print("stress is not copied here: run tools/make_stress_recipe.py --src <upstream>/data/release/stress")
    left = [str(p.relative_to(ROOT)) for p in pkg.rglob("*.py") if re.search(r"/mnt/|/home/", p.read_text())]
    if left:
        print("WARNING: machine-specific paths remain in:", *left, sep="\n  ")
    print("synced; now run: python -m pytest -q && python tools/checksums.py --write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
