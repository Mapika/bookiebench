"""Build the deterministic stress transforms.

    PYTHONPATH=. HF_HUB_OFFLINE=1 python -m bookiebench.sims.packs.stress.build --n 300

reads <dir>/*.jsonl for every --sources dir (default data/test data/heldout; any directory of SPEC instances works,
e.g. --sources data/release/<split> --out data/release/stress), first --n instances per file, and writes
data/v2/stress/<transform>/<family>.jsonl, and a summary to logs/stress_build_stats.json (counts, lengths, sizes,
Dutch-book LP timings on the scaling instances).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bookiebench.sims.core import dumps, read_jsonl, validate

from . import SCALING, TRANSFORMS
from .scaling import validate_composite


def _strict_ok(inst) -> bool:
    """Outputs are validated as strictly as their source passes (sources from other packs may omit e.g. perms)."""
    try:
        return validate(inst, strict=True)
    except Exception:  # noqa: BLE001
        validate(inst, strict=False)
        return False


def _job(args):
    name, fam, path, n, seed, out_dir = args
    src = read_jsonl(path)[:n]
    fn = TRANSFORMS[name]
    t0 = time.time()
    lines, info = [], []
    for inst in src:
        o = fn(inst, seed=seed)
        o["meta"]["source_file"] = path
        validate(o, strict=_strict_ok(inst))
        lines.append(dumps(o))
        st = o["meta"]["stress"]
        info.append({k: st[k] for k in ("n_tokens",) if k in st} | {"n_added": len(st.get("added_qids", []))})
    p = Path(out_dir) / name / f"{fam}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return name, fam, len(lines), info, p.stat().st_size, time.time() - t0


def _scaling_job(args):
    fam, path, pool_paths, n, seed, out_dir = args
    from bookiebench.metrics.dutch import dutch_book
    from bookiebench.sims.core import exact_answer
    src = read_jsonl(path)[:n]
    pool = [i for pp in pool_paths for i in read_jsonl(pp)[:n]]
    lines, info = [], []
    for j, inst in enumerate(src):
        o = SCALING(inst, pool, seed=seed, tier_idx=j % 3)
        o["meta"]["source_file"] = path
        validate_composite(o)
        lines.append(dumps(o))
        # Dutch-book LP timing on noisy (incoherent) exact prices
        rng = np.random.default_rng(j)
        ans = {}
        for q in o["queries"]:
            a = np.asarray(exact_answer(o, q))
            a = np.clip(a + rng.normal(0, 0.05, a.shape), 0.001, 0.999)
            ans[q["id"]] = (a / a.sum()).tolist() if q["kind"] == "marginal" else a.tolist()
        t0 = time.time()
        r = dutch_book(o, ans)
        dt = time.time() - t0
        st = o["meta"]["stress"]
        info.append({"n_cells": st["n_cells"], "n_vars": st["n_vars"], "n_parts": len(st["source_ids"]),
                     "n_queries": len(o["queries"]), "dutch_s": dt, "dutch": r["dutch"], "n_bets": r["n_bets"]})
    p = Path(out_dir) / "scaling" / f"{fam}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "scaling", fam, len(lines), info, p.stat().st_size, 0.0


def _summ(x):
    x = np.asarray(x, dtype=float)
    if not x.size:
        return {}
    return {"min": float(x.min()), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
            "p90": float(np.percentile(x, 90)), "max": float(x.max()), "mean": float(x.mean())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="*", default=["data/test", "data/heldout"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--transforms", nargs="*", default=None)
    ap.add_argument("--out", default="data/v2/stress")
    ap.add_argument("--stats", default="logs/stress_build_stats.json")
    ap.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    names = args.transforms or list(TRANSFORMS) + ["scaling"]
    files = [(Path(d) / f"{p.stem}.jsonl", p.stem, d) for d in args.sources for p in sorted(Path(d).glob("*.jsonl"))]
    jobs = [(n, fam, str(path), args.n, args.seed, args.out) for n in names if n != "scaling" for path, fam, _ in files]
    sjobs = []
    if "scaling" in names:
        for path, fam, d in files:
            pool_paths = [str(p) for p in sorted(Path(d).glob("*.jsonl"))]
            sjobs.append((fam, str(path), pool_paths, args.n, args.seed, args.out))
    stats = defaultdict(lambda: {"counts": {}, "bytes": 0, "info": []})
    t0 = time.time()
    with Pool(args.workers) as pool:
        its = list(pool.imap_unordered(_job, jobs)) + list(pool.imap_unordered(_scaling_job, sjobs))
    for name, fam, cnt, info, size, dt in its:
        s = stats[name]
        s["counts"][fam] = cnt
        s["bytes"] += size
        s["info"] += info
    summary = {}
    for name, s in stats.items():
        info = s["info"]
        row = {"counts": dict(sorted(s["counts"].items())), "total": sum(s["counts"].values()),
               "MB": round(s["bytes"] / 1e6, 1), "added_queries": _summ([i["n_added"] for i in info if "n_added" in i])}
        if info and "n_tokens" in info[0]:
            row["n_tokens"] = _summ([i["n_tokens"] for i in info])
        if name == "scaling":
            for k in ("n_cells", "n_vars", "n_parts", "n_queries", "dutch_s", "n_bets"):
                row[k] = _summ([i[k] for i in info])
            row["cells_hist"] = {f"{lo}-{hi}": int(sum(lo <= i["n_cells"] <= hi for i in info))
                                 for lo, hi in [(1, 256), (257, 1024), (1025, 2048), (2049, 4096)]}
        summary[name] = row
        print(f"{name:16s} {row['total']:5d} inst {row['MB']:8.1f} MB", flush=True)
    Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(Path(args.stats).read_text()) if Path(args.stats).exists() else {}
    old.update(summary)
    Path(args.stats).write_text(json.dumps(old, indent=1))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
