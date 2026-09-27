"""Rebuild the stress transforms that ship as recipes instead of text, and verify them hash by hash.

    bookiebench rebuild-stress --long                      # long_4k/8k/16k/32k -> data/release/stress/long_*/
    bookiebench rebuild-stress --transforms scaling        # the many-world composites (no external data needed)
    bookiebench rebuild-stress --long --sample 5 --check   # verify 5 instances per family and file, write nothing

The recipe (`data/release/stress/recipe.json`) pins everything a rebuild depends on. Every rebuilt instance is checked
against its sha256 in the recipe, and every full file against its file hash.

- Sources: the shipped `data/release/test/` and `data/release/heldout/` files, the first `n_per_family` instances
  (all 300).
- The transform code in this package, and the seed.
- For `long_*` only:
  - the filler datasets, as Hugging Face `datasets` cache files at pinned revisions: CNN/DailyMail validation,
    microsoft/wiki_qa train, SetFit/20_newsgroups test;
  - the Qwen3.5 tokenizer at a pinned revision, which measures lengths. The token counts decide which documents are
    added, so a different tokenizer or `tokenizers` version can change the output. The hash check catches that.

Why recipes. The long-context filler is third-party text (CNN/DailyMail is ids-only in the licence review), so it is
not redistributed. `scaling` is about 260 MB of text that is fully determined by the shipped source files, so it is
cheaper to rebuild than to ship.

Getting the filler (once, with network): `pip install datasets`, then run the `fetch` commands listed in the recipe.
They populate `$HF_HOME/datasets/...` at the pinned revisions. Then run with `HF_HUB_OFFLINE=1`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "data" / "release" / "stress" / "recipe.json"
LONG = ["long_4k", "long_8k", "long_16k", "long_32k"]


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))


def check_long_inputs(recipe: dict) -> list[str]:
    """Problems with the filler / tokenizer inputs (empty list = ready)."""
    probs = []
    for name, f in recipe["filler"].items():
        p = _hf_home() / "datasets" / f["cache_path"]
        if not p.exists():
            probs.append(f"filler {name}: {p} missing. Fetch with: {f['fetch']}")
    tok = recipe["tokenizer"]
    snap = _hf_home() / "hub" / f"models--{tok['repo'].replace('/', '--')}" / "snapshots" / tok["revision"]
    if not (snap / "tokenizer.json").exists():
        probs.append(f"tokenizer {tok['repo']}@{tok['revision']} missing under {snap}. Fetch with: {tok['fetch']}")
    else:
        ref = snap.parents[1] / "refs" / "main"
        if ref.exists() and ref.read_text().strip() != tok["revision"]:
            probs.append(f"tokenizer refs/main is {ref.read_text().strip()}, recipe pins {tok['revision']}: "
                         f"point refs/main at the pinned snapshot")
    return probs


def rebuild(transforms, out: Path | None, families=None, sample: int | None = None, root: Path = ROOT,
            recipe_path: Path = RECIPE, log=print) -> dict:
    """Returns {transform: {family: {"checked", "ok", "file_ok"}}}; raises on the first hash mismatch."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from bookiebench.sims.core import dumps, read_jsonl, validate
    from bookiebench.sims.packs.stress import SCALING, TRANSFORMS
    from bookiebench.sims.packs.stress.scaling import validate_composite

    recipe = json.loads(Path(recipe_path).read_text())
    seed, n = recipe["seed"], recipe["n_per_family"]
    if any(t in LONG for t in transforms):
        probs = check_long_inputs(recipe)
        if probs:
            raise SystemExit("long-context inputs not ready:\n  " + "\n  ".join(probs))
    srcs = recipe["sources"]                                  # family -> source file (relative to root)
    cache: dict = {}

    def load(rel):
        if rel not in cache:
            cache[rel] = read_jsonl(root / rel)[:n]
        return cache[rel]

    report: dict = {}
    for t in transforms:
        spec = recipe["transforms"][t]
        report[t] = {}
        for fam, fs in sorted(spec.items()):
            if families and fam not in families:
                continue
            rel = srcs[fam]
            src = load(rel)
            idx = range(len(src))
            if sample:
                step = max(1, len(src) // sample)
                idx = list(range(0, len(src), step))[:sample]
            pool = None
            if t == "scaling":
                pool_dir = (root / rel).parent
                pool = [i for pp in sorted(pool_dir.glob("*.jsonl")) for i in load(str(Path(rel).parent / pp.name))]
            lines = []
            for j in idx:
                inst = src[j]
                o = SCALING(inst, pool, seed=seed, tier_idx=j % 3) if t == "scaling" else TRANSFORMS[t](inst, seed=seed)
                o["meta"]["source_file"] = rel
                validate_composite(o) if t == "scaling" else validate(o, strict=False)
                line = dumps(o)
                if sha(line) != fs["line_sha256"][j]:
                    raise RuntimeError(f"{t}/{fam} instance {j} ({o['id']}): hash mismatch -- inputs differ from the "
                                       f"recipe (tokenizer/filler/library versions?)")
                lines.append(line)
            text = "\n".join(lines) + "\n"
            file_ok = None
            if not sample:
                file_ok = sha(text) == fs["file_sha256"]
                if not file_ok:
                    raise RuntimeError(f"{t}/{fam}: file hash mismatch")
            if out is not None:
                p = Path(out) / t / (f"{fam}.jsonl" if not sample else f"{fam}.sample.jsonl")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(text, encoding="utf-8")
            report[t][fam] = {"checked": len(lines), "ok": True, "file_ok": file_ok}
            log(f"{t:10s} {fam:10s} {len(lines):4d} verified" + (" (file hash ok)" if file_ok else ""))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", action="store_true", help="the long-context transforms (long_4k ... long_32k)")
    ap.add_argument("--transforms", nargs="*", default=None, help="explicit transforms (any in the recipe)")
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "data" / "release" / "stress"))
    ap.add_argument("--sample", type=int, default=None, help="verify K evenly spaced instances per family only")
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    ap.add_argument("--recipe", default=str(RECIPE))
    ap.add_argument("--root", default=str(ROOT), help="repo root the recipe's source paths are relative to")
    a = ap.parse_args(argv)
    recipe = json.loads(Path(a.recipe).read_text())
    ts = list(a.transforms or [])
    if a.long:
        ts += [t for t in LONG if t not in ts]
    if not ts:
        ts = list(recipe["transforms"])
    unknown = [t for t in ts if t not in recipe["transforms"]]
    if unknown:
        print(f"not in the recipe: {unknown}; available: {sorted(recipe['transforms'])}")
        return 2
    rebuild(ts, None if a.check else Path(a.out), a.families, a.sample, Path(a.root), Path(a.recipe))
    return 0


if __name__ == "__main__":
    sys.exit(main())
