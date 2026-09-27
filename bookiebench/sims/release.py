"""Build the BookieBench release (template version 2) into data/release/.

    python -m bookiebench.sims.release --out data/release                  # everything (train + eval)
    python -m bookiebench.sims.release --out /tmp/r --n-scale 0.01 --packs v1  # quick smoke build

Every instance goes through the shared release pipeline:
  1. generated with tv=2 by its pack's own `generate` (attempt a uses seed = base_seed + a);
  2. `core.finalize_release`: seeded per-instance option shuffle (remaps joints / events / gold);
  3. acceptance: the mart_var posterior must move after step 0 (`core.mart_moves`), and the fraction of degenerate
     final queries (near-certain or uniform, `core.degenerate_fraction`) must be <= DEGEN_CAP; otherwise the next
     attempt is drawn (up to MAX_ATTEMPTS, then the least degenerate moving attempt is kept);
  4. eval splits only: reject (redraw) instances whose world-parameter key or digit-masked text key matches an
     instance of the same family's train split, or an earlier instance of the same split.
Output layout (sims_manifest.json describes every file, its group and its audit rates):
  train/<family>.jsonl                           all trainable families (v1 train + prior + mechanics/programs/tables)
  test/<fam>, test_prior/<fam>                   in-family eval of v1 train / prior families
  heldout/<fam>                                  group "surface_transfer" (re-skinned v1 mechanics)
  val/<fam>, {mechanics,programs,tables}/dev/<fam>   group "new_mechanics"
"""
from __future__ import annotations

import os

# One BLAS/OpenMP thread per process: the builder parallelises over instances (--workers); nested BLAS pools on a
# shared machine oversubscribe it (48 workers x 6-8 threads pushed load to ~1950 on 256 cores). Set before numpy loads;
# main() re-execs itself if numpy was imported first (e.g. via the bookiebench.sims package), and workers also clamp
# their pools with threadpoolctl.
import sys as _sys

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# numpy already loaded without the limits -> its BLAS pool is already sized; main() then re-execs
_NEEDS_REEXEC = "numpy" in _sys.modules and any(os.environ.get(v) != "1" for v in THREAD_VARS)
for _v in THREAD_VARS:
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import template_version
from .core import degenerate_fraction, degenerate_query, dumps, finalize_release, mart_moves, validate

DEGEN_CAP = 0.10
MAX_ATTEMPTS = 6
TV = 2


# ----------------------------------------------------------------------------------------------------------------------
# pack adapters
# ----------------------------------------------------------------------------------------------------------------------

def _gen(pack, fam, split, idx, seed, nuisance):
    with template_version(TV):
        if pack == "v1":
            from . import generate
            inst = generate(fam, split, idx, seed=seed, nuisance=nuisance, tv=TV)
            key_inst = inst if not inst.get("meta", {}).get("nuisance") else generate(fam, split, idx, seed=seed, tv=TV)
            return inst, key_inst
        if pack == "mechanics":
            from .packs import mechanics as m
            inst = m.generate(fam, split, idx, seed=seed, nuisance=nuisance)
            key_inst = inst if not inst.get("meta", {}).get("nuisance") else m.generate(fam, split, idx, seed=seed)
            return inst, key_inst
        mod = __import__(f"bookiebench.sims.packs.{pack}", fromlist=["generate"])
        inst = mod.generate(fam, split, idx, seed=seed)
        return inst, inst


def _validate(pack, inst):
    if pack == "mechanics":
        from .packs import mechanics as m
        return m.validate(inst)
    return validate(inst)


_NUM = re.compile(r"\d+(?:\.\d+)?")


def keys(inst):
    """(world-parameter key, digit-masked text key) of an un-augmented, un-shuffled instance."""
    pre = inst["prelude"]
    ev = "\n".join(s["evidence"] for s in inst["steps"])
    shape = ",".join(str(len(v["options"])) for v in inst["variables"])
    kp = "|".join([inst["family"], shape, " ".join(_NUM.findall(pre)), " ".join(_NUM.findall(ev)),
                   _NUM.sub("#", ev)])
    kt = "|".join([inst["family"], _NUM.sub("#", pre), ev])
    h = lambda s: hashlib.blake2b(s.encode(), digest_size=12).hexdigest()  # noqa: E731
    return h(kp), h(kt)


def _one(pack, fam, split, idx, base_seed, nuisance, banned, start=0):
    """Accepted release instance for idx: (line, keys, attempt, degenerate_fraction, moves, dedupe_rejections)."""
    best, n_dup = None, 0
    for a in range(start, start + MAX_ATTEMPTS):
        inst, kinst = _gen(pack, fam, split, idx, base_seed + a, nuisance)
        k = keys(kinst)
        if banned is not None and (k[0] in banned or k[1] in banned):
            n_dup += 1
            continue
        fin = finalize_release(inst, seed=base_seed)
        fin["meta"]["release_attempt"] = a
        _validate(pack, fin)
        df, mv = degenerate_fraction(fin), mart_moves(fin)
        cand = (dumps(fin), k, a, df, mv)
        if mv and df <= DEGEN_CAP:
            return (*cand, n_dup)
        if best is None or (mv, -df) > (best[4], -best[3]):
            best = cand
    if best is None:  # every attempt was a duplicate: keep the last draw anyway, flagged
        fin = finalize_release(inst, seed=base_seed)
        fin["meta"]["release_attempt"] = a
        fin["meta"]["release_duplicate"] = True
        best = (dumps(fin), k, a, degenerate_fraction(fin), mart_moves(fin))
    return (*best, n_dup)


def _init_worker():
    for v in THREAD_VARS:
        os.environ[v] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass


def _ensure_single_threaded(argv):
    """numpy fixes its BLAS pool size at import: if it was imported before THREAD_VARS were set, re-exec once."""
    if not _NEEDS_REEXEC or os.environ.get("_SIMS_RELEASE_REEXEC") or argv is not None:
        return  # argv given = called as a library (tests): the worker initializer's threadpoolctl clamp suffices
    env = dict(os.environ, _SIMS_RELEASE_REEXEC="1", **{v: "1" for v in THREAD_VARS})
    os.execve(_sys.executable, [_sys.executable, "-m", "bookiebench.sims.release", *_sys.argv[1:]], env)


def _chunk(args):
    pack, fam, split, lo, hi, base_seed, nuisance, banned = args
    return [_one(pack, fam, split, i, base_seed, nuisance, banned) for i in range(lo, hi)]


# ----------------------------------------------------------------------------------------------------------------------
# plan
# ----------------------------------------------------------------------------------------------------------------------

# v2 pack families held out of train entirely (final shortcuts review: dev families with a train file are not "new").
# Their train split is still generated IN MEMORY during a build (group "keys_only": never written, no keys file, no
# manifest entry), only so that their dev files keep being deduped against it and stay byte-identical.
HOLDOUT_TRAIN = {"epidemic", "forensic", "raters", "recapture", "montyhall", "search", "queue",  # mechanics
                 "prog_domain",  # programs
                 "tab_stream"}  # tables
GROUP_DESC = {
    "train": "training data",
    "in_family": "v1 train families, unseen worlds (test/)",
    "prior": "procedural prior families randbn/randhmm, unseen worlds (test_prior/)",
    "surface_transfer": "v1 heldout: re-skinned versions of the v1 train mechanics (heldout/)",
    "in_family_v2": "v2 pack dev families whose train file is in train/ (unseen worlds of trained mechanics)",
    "new_mechanics": "families with NO train data: val (genetics, tracking) + the HOLDOUT_TRAIN v2 dev families",
}


def plan(n_scale=1.0, packs=("v1", "mechanics", "programs", "tables")):
    from . import HELDOUT_FAMILIES, PRIOR_FAMILIES, TRAIN_FAMILIES, VAL_FAMILIES
    S = lambda n: max(1, int(round(n * n_scale)))  # noqa: E731
    jobs = []  # (pack, fam, split, n, nuisance, out_rel, group, train_rel or None)
    if "v1" in packs:
        for f in TRAIN_FAMILIES:
            jobs.append(("v1", f, "train", S(20000), 0.0, f"train/{f}.jsonl", "train", None))
            jobs.append(("v1", f, "test", S(300), 0.0, f"test/{f}.jsonl", "in_family", f"train/{f}.jsonl"))
        for f in PRIOR_FAMILIES:
            jobs.append(("v1", f, "train", S(100000), 0.3, f"train/{f}.jsonl", "train", None))
            jobs.append(("v1", f, "test_prior", S(300), 0.0, f"test_prior/{f}.jsonl", "prior", f"train/{f}.jsonl"))
        for f in VAL_FAMILIES:
            jobs.append(("v1", f, "val", S(300), 0.0, f"val/{f}.jsonl", "new_mechanics", None))
        for f in HELDOUT_FAMILIES:
            jobs.append(("v1", f, "heldout", S(300), 0.0, f"heldout/{f}.jsonl", "surface_transfer", None))
    for pack in ("mechanics", "programs", "tables"):
        if pack not in packs:
            continue
        mod = __import__(f"bookiebench.sims.packs.{pack}", fromlist=["FAMILIES"])
        for f in mod.FAMILIES:
            held = f in HOLDOUT_TRAIN
            jobs.append((pack, f, "train", S(20000), 0.0, f"train/{f}.jsonl", "keys_only" if held else "train", None))
            jobs.append((pack, f, "dev", S(300), 0.0, f"{pack}/dev/{f}.jsonl",
                         "new_mechanics" if held else "in_family_v2", f"train/{f}.jsonl"))
    return jobs


def audit_file(path):
    """Per-file audit rates on the written release file."""
    from .core import iter_jsonl
    n = nq = det = unif = still = 0
    first = last = 0.0
    nm = 0
    for inst in iter_jsonl(path):
        n += 1
        still += not mart_moves(inst)
        for q in inst["queries"]:
            if "step" in q:
                continue
            nq += 1
            d = degenerate_query(inst, q)
            det += d
            from .core import exact_answer
            a = np.asarray(exact_answer(inst, q))
            if q["kind"] == "marginal":
                unif += (a.max() - a.min()) < 0.01
                k = len(a)
                first += a[0] - 1 / k
                last += a[-1] - 1 / k
                nm += 1
            else:
                unif += abs(a[0] - 0.5) < 0.01
    return {"instances": n, "queries": nq, "degenerate": round(det / max(1, nq), 4),
            "uniform": round(unif / max(1, nq), 4), "mart_still": round(still / max(1, n), 4),
            "excess_mass_first": round(first / max(1, nm), 4), "excess_mass_last": round(last / max(1, nm), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/release")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-scale", type=float, default=1.0)
    ap.add_argument("--packs", nargs="*", default=["v1", "mechanics", "programs", "tables"])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--eval-only", action="store_true", help="skip train files (no train dedupe; for audits)")
    ap.add_argument("--train-only", action="store_true", help="only the train files (pre-generation)")
    ap.add_argument("--reuse-train", action="store_true",
                    help="keep an existing train file (same seed/n/nuisance/tv in sims_manifest.json) and load its "
                         "dedupe keys from .train_keys/ instead of regenerating it")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    _ensure_single_threaded(argv)
    _init_worker()
    try:
        from threadpoolctl import threadpool_info
        pools = [(i["internal_api"], i["num_threads"]) for i in threadpool_info()]
    except ImportError:
        pools = "threadpoolctl missing"
    print(f"threads: {', '.join(f'{v}={os.environ.get(v)}' for v in THREAD_VARS)}; BLAS pools {pools}; "
          f"re-exec={bool(os.environ.get('_SIMS_RELEASE_REEXEC'))}; workers={args.workers}", flush=True)
    out = Path(args.out)
    jobs = [j for j in plan(args.n_scale, args.packs) if not args.families or j[1] in args.families]
    if args.eval_only:
        jobs = [j for j in jobs if j[2] != "train"]
    if args.train_only:
        jobs = [j for j in jobs if j[2] == "train"]
    jobs.sort(key=lambda j: j[2] != "train")  # train first: eval splits dedupe against it
    train_keys: dict[str, set] = {}
    manifest = {"template_version": TV, "seed": args.seed, "degenerate_cap": DEGEN_CAP, "max_attempts": MAX_ATTEMPTS,
                "groups": dict(GROUP_DESC), "holdout_train": sorted(HOLDOUT_TRAIN), "files": {}}
    old = out / "sims_manifest.json"
    if old.exists():  # merge: files built by an earlier (partial) run stay described
        prev = json.loads(old.read_text())
        if prev.get("template_version") == TV and prev.get("seed") == args.seed:
            manifest["files"].update(prev.get("files", {}))
    for _p, _f, _s, _n, _nu, _rel, _g, _t in plan(args.n_scale, args.packs):  # current grouping wins over old entries
        if _g == "keys_only":
            manifest["files"].pop(_rel, None)
        elif _rel in manifest["files"]:
            manifest["files"][_rel]["group"] = _g
    keydir = out / ".train_keys"
    with Pool(args.workers, initializer=_init_worker) as pool:
        for pack, fam, split, n, nuis, rel, group, train_rel in jobs:
            t0 = time.time()
            banned = train_keys.get(train_rel) if train_rel else None
            kfile = keydir / (Path(rel).stem + ".txt")
            prev = manifest["files"].get(rel, {})
            keys_only = group == "keys_only"
            if (split == "train" and not keys_only and args.reuse_train and (out / rel).exists() and kfile.exists()
                    and prev.get("n") == n and prev.get("nuisance") == nuis and prev.get("pack") == pack
                    and sum(1 for _ in open(out / rel, "rb")) == n):
                train_keys[rel] = set(kfile.read_text().split())
                print(f"{rel:38s} reused ({len(train_keys[rel])} keys)", flush=True)
                continue
            step = 50 if split != "train" else 250
            chunks = [(pack, fam, split, lo, min(n, lo + step), args.seed, nuis, banned) for lo in range(0, n, step)]
            path = out / rel
            if keys_only:
                ks = set()
                for res in pool.imap(_chunk, chunks):
                    for line, k, *_ in res:
                        ks.update(k)
                train_keys[rel] = ks
                print(f"{rel:38s} keys only (held out of train; {len(ks)} keys, {time.time() - t0:.1f}s)", flush=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".partial")  # renamed only when complete: no half file is ever reused
            ks, seen, stats = set(), set(), {"attempts": 0, "dedupe_train": 0, "dedupe_split": 0, "fallback": 0}
            with open(tmp, "w", encoding="utf-8") as f:
                idx = 0
                for res in pool.imap(_chunk, chunks):
                    for line, k, a, df, mv, ndup in res:
                        if split != "train" and (k[0] in seen or k[1] in seen):  # within-split duplicate: redraw
                            start = a + 1
                            while k[0] in seen or k[1] in seen:
                                line, k, a, df, mv, nd2 = _one(pack, fam, split, idx, args.seed, nuis, banned, start)
                                ndup += nd2
                                stats["dedupe_split"] += 1
                                start = a + 1
                                if start > 60:
                                    break
                        seen.update(k)
                        ks.update(k)
                        stats["attempts"] += a
                        stats["dedupe_train"] += ndup
                        stats["fallback"] += (not mv) or df > DEGEN_CAP
                        f.write(line + "\n")
                        idx += 1
            os.replace(tmp, path)
            if split == "train":
                train_keys[rel] = ks
                keydir.mkdir(parents=True, exist_ok=True)
                kfile.write_text("\n".join(sorted(ks)) + "\n")
            rates = audit_file(path)
            manifest["files"][rel] = {"pack": pack, "family": fam, "split": split, "group": group, "n": n,
                                      "nuisance": nuis, **rates,
                                      "mean_attempt": round(stats["attempts"] / max(1, n), 3),
                                      "dedupe_rejections_train": stats["dedupe_train"],
                                      "dedupe_rejections_split": stats["dedupe_split"],
                                      "fallback_instances": stats["fallback"]}
            print(f"{rel:38s} {n:7d} {time.time() - t0:7.1f}s {json.dumps(manifest['files'][rel])}", flush=True)
            (out / "sims_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
