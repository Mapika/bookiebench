"""Generate the `tables` pack.

    python -m bookiebench.sims.packs.tables.generate --out data/v2/tables --n-dev 300 --n-train 20000

writes <out>/<split>/<family>.jsonl (families tab_pick, tab_stream, tab_big). Deterministic in
(--seed, family, split, idx), independent of --workers. Every instance is validated.
"""
from __future__ import annotations

from bookiebench.sims.packs.programs.generate import main as _main


def main(argv=None):
    _main(argv, pack="tables")


if __name__ == "__main__":
    main()
