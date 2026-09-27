#!/usr/bin/env python
"""Generate a hidden leaderboard test edition of the BookieBench simulator splits from a secret seed.

    export BOOKIEBENCH_SECRET_SEED='<long random string, kept offline>'      # e.g. `openssl rand -hex 32`
    python scripts/make_hidden_test.py --edition 2026-10 --out /secure/bookiebench-hidden/2026-10

Protocol (see README "Hidden test set"):
  * one fresh secret per leaderboard edition; never reuse a secret, never commit it, never pass it on the command line
    (it would land in shell history / process lists) -- it is read from the BOOKIEBENCH_SECRET_SEED env var only;
  * the edition uses exactly the public release plan (`bookiebench.sims.release.plan`): same families, groups, sizes,
    template version, degeneracy cap, attempt budget, option shuffle and acceptance rules -- only the seed differs;
  * every instance is deduplicated (world-parameter key and digit-masked text key, `release.keys`) against the public
    train split of its family (`--train-keys`, written by `bookiebench generate --train-only`), against the public
    eval/dev file of its family, and within the edition. Families held out of train permanently (`HOLDOUT_TRAIN`)
    have no train file and no keys file; their public train split is regenerated in memory (public seed, full size)
    for its keys only, exactly as the public build's `keys_only` group does;
  * the same built-in audits run on every written file (validate, degenerate / uniform / still-posterior rates,
    option-position excess mass, dedupe and fallback counts) plus the exact-posterior calibration check; a file that
    breaks a gate is reported in the manifest and makes the script exit non-zero;
  * the manifest records a commitment sha256("bookiebench-hidden:" + edition + ":" + secret), never the secret or
    the derived seed. Publish the commitment when the edition opens and the secret when it is retired, so anyone can
    regenerate the retired edition and check it.

Instance ids are prefixed with "hidden<edition>-" so predictions made on public files can never be scored against a
hidden edition by accident. Output layout = the public release layout (test/, test_prior/, heldout/, val/,
<pack>/dev/) plus hidden_manifest.json.

realcoh and stress are not regenerated here: realcoh needs new real-world states (a separate curation step), and the
stress transforms are applied to a hidden edition with `bookiebench.sims.packs.stress.build` afterwards if wanted.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from bookiebench.sims import release  # noqa: E402
from bookiebench.sims.core import exact_answer, iter_jsonl  # noqa: E402

ENV = "BOOKIEBENCH_SECRET_SEED"
PUBLIC_SEEDS = range(0, 1000)          # seeds (and attempt offsets) used by public builds; a derived seed never lands here
MIN_SECRET_LEN = 16
ROOT = Path(__file__).resolve().parents[1]

# audit gates (the public release satisfies all of them, see data/release/sims_manifest.json)
GATE_DEGENERATE = release.DEGEN_CAP    # fraction of degenerate final queries per file
GATE_MART_STILL = 0.02                 # instances whose martingale posterior never moves
GATE_POSITION = 0.05                   # |excess mass| on the first / last option
GATE_CALIB_SE = 3.0                    # |exact top-1 acc - mean confidence| in standard errors
GATE_MIN_N = 100                       # rate gates (position, still, degenerate) only on files this large (noise)


def secret_from_env() -> str:
    s = os.environ.get(ENV)
    if not s:
        sys.exit(f"{ENV} is not set. Generate a fresh secret per edition (e.g. `openssl rand -hex 32`), keep it offline.")
    if len(s) < MIN_SECRET_LEN:
        sys.exit(f"{ENV} must be at least {MIN_SECRET_LEN} characters")
    return s


def derive_seed(secret: str, edition: str) -> int:
    """63-bit seed from (secret, edition); disjoint from the public seed range."""
    h = hashlib.blake2b(f"{edition}:{secret}".encode(), digest_size=8, person=b"hb-hidden").digest()
    seed = int.from_bytes(h, "big") >> 1
    return seed if seed not in PUBLIC_SEEDS else seed + 1_000_003


def commitment(secret: str, edition: str) -> str:
    return hashlib.sha256(f"bookiebench-hidden:{edition}:{secret}".encode()).hexdigest()


# v2 pack families held out of train permanently. Upstream `release.HOLDOUT_TRAIN` is authoritative once synced; the
# fallback copy only covers the vendored release.py that predates it.
HOLDOUT_TRAIN = set(getattr(release, "HOLDOUT_TRAIN", {"epidemic", "forensic", "raters", "recapture", "montyhall",
                                                       "search", "queue", "prog_domain", "tab_stream"}))


def keys_only_train(pool, pack: str, family: str, n: int, public_seed: int) -> set:
    """Dedupe keys of a held-out family's public train split, generated in memory (never written), exactly as the
    public build does for its `keys_only` group: same seed, same n, same acceptance loop."""
    chunks = [(pack, family, "train", lo, min(n, lo + 250), public_seed, 0.0, None) for lo in range(0, n, 250)]
    ks: set = set()
    for res in pool.imap(release._chunk, chunks):
        for _line, k, *_ in res:
            ks.update(k)
    return ks


def load_banned(train_keys: Path | None, public_eval: Path, rel: str, train_rel: str | None, family: str,
                in_memory=None) -> set:
    """in_memory: callable returning the keys of a HOLDOUT_TRAIN family's train split (no keys file exists for it)."""
    banned: set = set()
    if train_keys is not None and train_rel is not None:
        kfile = train_keys / f"{Path(train_rel).stem}.txt"
        if family in HOLDOUT_TRAIN:
            banned |= in_memory()
        elif kfile.exists():
            banned |= set(kfile.read_text().split())
        else:
            raise FileNotFoundError(f"{kfile}: no train keys for {family} (run `bookiebench generate --train-only`)")
    pub = public_eval / rel
    if pub.exists():                                  # the public eval/dev file of the same family
        for inst in iter_jsonl(pub):
            banned.update(release.keys(inst))
    return banned


def calibration_check(path: Path) -> dict:
    """Exact-posterior top-1 accuracy vs mean confidence over final-step marginals with gold (SPEC calibration check).
    Under the stated model the gold is a draw from the exact posterior, so |gap| should be within a few SE."""
    hits, confs = [], []
    for inst in iter_jsonl(path):
        gold = inst.get("gold") or {}
        for q in inst["queries"]:
            if q["kind"] != "marginal" or "step" in q or gold.get(q["var"]) is None:
                continue
            p = np.asarray(exact_answer(inst, q), dtype=float)
            top = np.flatnonzero(p >= p.max() - 1e-12)
            hits.append((1.0 / top.size) if int(gold[q["var"]]) in top else 0.0)
            confs.append(float(p.max()))
    if not hits:
        return {"calib_n": 0}
    acc, conf = float(np.mean(hits)), float(np.mean(confs))
    se = math.sqrt(max(acc * (1 - acc), 1e-12) / len(hits))
    return {"calib_n": len(hits), "calib_acc": round(acc, 4), "calib_conf": round(conf, 4),
            "calib_gap": round(acc - conf, 4), "calib_se": round(se, 4)}


def gate(rates: dict) -> list[str]:
    bad = []
    if rates["instances"] < GATE_MIN_N:            # smoke-sized files: only the exactness / dedupe gates apply
        return [f"{rates['duplicates_kept']} instances could not be deduplicated"] if rates.get("duplicates_kept") else []
    if rates["degenerate"] > GATE_DEGENERATE:
        bad.append(f"degenerate {rates['degenerate']} > {GATE_DEGENERATE}")
    if rates["mart_still"] > GATE_MART_STILL:
        bad.append(f"mart_still {rates['mart_still']} > {GATE_MART_STILL}")
    for k in ("excess_mass_first", "excess_mass_last"):
        if abs(rates[k]) > GATE_POSITION:
            bad.append(f"{k} {rates[k]} beyond +-{GATE_POSITION}")
    if rates.get("calib_n") and abs(rates["calib_gap"]) > GATE_CALIB_SE * rates["calib_se"]:
        bad.append(f"calibration gap {rates['calib_gap']} > {GATE_CALIB_SE} SE")
    if rates.get("duplicates_kept"):
        bad.append(f"{rates['duplicates_kept']} instances could not be deduplicated")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edition", required=True, help="edition label, e.g. 2026-10 (part of the seed and of the ids)")
    ap.add_argument("--out", required=True, help="output directory (keep it off any public / synced location)")
    ap.add_argument("--train-keys", default=str(ROOT / "data" / "release" / ".train_keys"),
                    help="dedupe keys of the public train split (written by `bookiebench generate --train-only`)")
    ap.add_argument("--public-eval", default=str(ROOT / "data" / "release"),
                    help="public release root; its eval/dev files are also excluded")
    ap.add_argument("--n-scale", type=float, default=1.0, help="fraction of the public sizes (smoke tests only)")
    ap.add_argument("--packs", nargs="*", default=["v1", "mechanics", "programs", "tables"])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--no-train-dedupe", action="store_true",
                    help="skip train dedupe (smoke tests only; the manifest marks the edition as not publishable)")
    ap.add_argument("--workers", type=int, default=min(48, os.cpu_count() or 1))
    a = ap.parse_args(argv)

    secret = secret_from_env()
    seed = derive_seed(secret, a.edition)
    out = Path(a.out)
    if out.resolve().is_relative_to(ROOT.resolve()) and not os.environ.get("BOOKIEBENCH_ALLOW_IN_REPO"):
        sys.exit(f"refusing to write a hidden edition inside the repository ({out}); pick a path outside it")
    train_keys = None if a.no_train_dedupe else Path(a.train_keys)
    if train_keys is not None and not train_keys.is_dir():
        sys.exit(f"{train_keys} missing: build the public train split first "
                 "(`bookiebench generate --out data/release --train-only`), or pass --train-keys")
    prefix = f"hidden{a.edition}-"

    full_plan = release.plan(a.n_scale, a.packs)
    train_n = {j[5]: j[3] for j in full_plan if j[2] == "train"}  # public train size per file (keys_only included)
    jobs = [j for j in full_plan if j[2] != "train" and (not a.families or j[1] in a.families)]
    man_path = Path(a.public_eval) / "sims_manifest.json"
    public_seed = json.loads(man_path.read_text()).get("seed", 0) if man_path.exists() else 0
    manifest = {"edition": a.edition, "seed_commitment": commitment(secret, a.edition),
                "template_version": release.TV, "degenerate_cap": release.DEGEN_CAP,
                "max_attempts": release.MAX_ATTEMPTS, "n_scale": a.n_scale, "id_prefix": prefix,
                "train_dedupe": train_keys is not None,
                "publishable": train_keys is not None and a.n_scale == 1.0,
                "gates": {"min_n": GATE_MIN_N, "degenerate": GATE_DEGENERATE, "mart_still": GATE_MART_STILL, "position": GATE_POSITION,
                          "calibration_se": GATE_CALIB_SE},
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": {}}
    failures = 0
    with Pool(a.workers, initializer=release._init_worker) as pool:
        for pack, fam, split, n, nuis, rel, group, train_rel in jobs:
            t0 = time.time()
            n_train = train_n.get(train_rel, 0)
            banned = load_banned(train_keys, Path(a.public_eval), rel, train_rel, fam,
                                 in_memory=lambda: keys_only_train(pool, pack, fam, n_train, public_seed))
            step = 50
            chunks = [(pack, fam, split, lo, min(n, lo + step), seed, nuis, banned) for lo in range(0, n, step)]
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            seen, stats = set(), {"attempts": 0, "dedupe": 0, "dedupe_split": 0, "fallback": 0, "dup_kept": 0}
            with open(path, "w", encoding="utf-8") as f:
                idx = 0
                for res in pool.imap(release._chunk, chunks):
                    for line, k, att, df, mv, ndup in res:
                        start = att + 1
                        while k[0] in seen or k[1] in seen:       # within-edition duplicate: redraw
                            line, k, att, df, mv, nd2 = release._one(pack, fam, split, idx, seed, nuis, banned, start)
                            ndup += nd2
                            stats["dedupe_split"] += 1
                            start = att + 1
                            if start > 60:
                                break
                        seen.update(k)
                        inst = json.loads(line)
                        stats["dup_kept"] += bool(inst["meta"].get("release_duplicate")) or k[0] in banned
                        inst["id"] = prefix + inst["id"]
                        inst["meta"]["hidden_edition"] = a.edition
                        f.write(json.dumps(inst, ensure_ascii=False) + "\n")
                        stats["attempts"] += att
                        stats["dedupe"] += ndup
                        stats["fallback"] += (not mv) or df > release.DEGEN_CAP
                        idx += 1
            rates = release.audit_file(path)
            rates.update(calibration_check(path))
            rates["duplicates_kept"] = stats["dup_kept"]
            bad = gate(rates)
            failures += bool(bad)
            manifest["files"][rel] = {"pack": pack, "family": fam, "split": split, "group": group, "n": n, **rates,
                                      "mean_attempt": round(stats["attempts"] / max(1, n), 3),
                                      "dedupe_rejections": stats["dedupe"],
                                      "dedupe_rejections_split": stats["dedupe_split"],
                                      "fallback_instances": stats["fallback"],
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "gate_failures": bad}
            print(f"{rel:38s} {n:6d} {time.time() - t0:6.1f}s {'FAIL ' + '; '.join(bad) if bad else 'ok'}", flush=True)
            (out / "hidden_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"edition {a.edition}: {len(jobs)} files, {failures} failing a gate; commitment {manifest['seed_commitment']}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
