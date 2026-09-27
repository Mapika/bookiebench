"""`bookiebench` command line.

    bookiebench generate [release-builder args]        build the release layout (bookiebench.sims.release)
    bookiebench rebuild-realcoh --release DIR --out DIR  fill the text of ids_only realcoh sources from the HF cache
    bookiebench rebuild-stress [--long] [--transforms scaling] [--sample K --check]   recipe-shipped stress transforms
    bookiebench run <runner> INPUTS... [runner args]   runner: uniform | oracle | logit | decider | julia | api
    bookiebench score results/<model> --data DIR        metrics report for one model (bookiebench.metrics.report)
    bookiebench compare results/<m1> results/<m2> ...   cross-model tables with reference rows (bookiebench.metrics.compare)

Every subcommand forwards its remaining arguments to the underlying module; `bookiebench <cmd> --help` shows them.
"""
from __future__ import annotations

import sys

RUNNERS = {
    "uniform": ("bookiebench.runners.baselines", "uniform"),
    "oracle": ("bookiebench.runners.baselines", "oracle"),
    "logit": ("bookiebench.runners.logit_runner", None),
    "decider": ("bookiebench.runners.decider_runner", None),
    "julia": ("bookiebench.runners.julia_runner", None),
    "api": ("bookiebench.runners.api_runner", None),
}


def _usage() -> int:
    print(__doc__.strip())
    return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        return _usage()
    cmd, rest = argv[0], argv[1:]
    if cmd == "generate":
        # the release builder wants single-threaded BLAS in every worker; set the limits before numpy loads by
        # running it as its own process (same as `python -m bookiebench.sims.release`)
        import os
        import subprocess
        env = dict(os.environ, **{v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                    "NUMEXPR_NUM_THREADS")})
        return subprocess.call([sys.executable, "-m", "bookiebench.sims.release", *rest], env=env)
    if cmd == "rebuild-realcoh":
        from bookiebench.realcoh import build_v2
        build_v2.main(["rebuild", *rest])
        return 0
    if cmd == "rebuild-stress":
        from bookiebench import stress_rebuild
        return stress_rebuild.main(rest)
    if cmd == "run":
        if not rest or rest[0] not in RUNNERS:
            print(f"usage: bookiebench run {{{'|'.join(RUNNERS)}}} INPUTS... [args]  (got {rest[:1]})")
            return 2
        mod_name, which = RUNNERS[rest[0]]
        import importlib
        mod = importlib.import_module(mod_name)
        res = mod.main(rest[1:], which=which) if which else mod.main(rest[1:])
        return 0 if res is None or res else 1
    if cmd == "score":
        from bookiebench.metrics import report
        return report.main(rest)
    if cmd == "compare":
        from bookiebench.metrics import compare
        return compare.main(rest) or 0
    print(f"unknown command {cmd!r}\n")
    return _usage()


if __name__ == "__main__":
    sys.exit(main())
