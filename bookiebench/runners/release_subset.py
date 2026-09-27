"""The release evaluation subset used for the runner leaderboard (results/release/_data/sub).

    python -m bookiebench.runners.release_subset [--per-family 100] [--per-source 60] [--stress 30] [--long 10]

Why a subset: the full release is ~670k question rows per model outside stress and ~3M with stress; the slower
baselines (27B, 12B, 35B) would take days. Every model is scored on the SAME subset, so rows are comparable.

- in_family (test), prior (test_prior), surface_transfer (heldout), new_mechanics (val + {mechanics,programs,tables}/dev):
  the first --per-family instances of each file, except test/heldout families, where stress-paired ids come first:
  ids kept by every stress transform (manifest kept_by_all), then ids kept by the most transforms, then file order.
- realcoh: the first --per-source states of each rebuilt source (results/release/_data/realcoh).
- stress/<transform>: for each family, the first --stress stressed instances whose source id is in the family subset
  and in that transform's kept_source_ids (paired by construction); --long for long_*.
Files keep their split/family names, so a subset directory is a drop-in --data for runners and metrics.
Writes <out>/subset.json with the chosen ids and counts.
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "data" / "release"
GROUPS = {
    "in_family": ["test"], "prior": ["test_prior"], "surface_transfer": ["heldout"],
    "new_mechanics": ["val", "mechanics/dev", "programs/dev", "tables/dev"],
}


def _read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "results" / "release" / "_data" / "sub"))
    ap.add_argument("--realcoh", default=str(ROOT / "results" / "release" / "_data" / "realcoh"))
    ap.add_argument("--per-family", type=int, default=100)
    ap.add_argument("--per-source", type=int, default=60)
    ap.add_argument("--stress", type=int, default=30)
    ap.add_argument("--long", type=int, default=10)
    a = ap.parse_args(argv)
    out = Path(a.out)
    man = json.load(open(REL / "stress" / "manifest.json"))
    kept_count = Counter(i for t in man["transforms"].values() for f in t.values() for i in f["kept_source_ids"])
    chosen, counts = {}, {}
    for group, dirs in GROUPS.items():
        for d in dirs:
            for f in sorted((REL / d).glob("*.jsonl")):
                rows = _read(f)
                fam = f.stem
                if fam in man["families"]:
                    allk = set(man["families"][fam]["kept_by_all"])
                    rows.sort(key=lambda r: (r["id"] not in allk, -kept_count.get(r["id"], 0)))
                    rows = sorted(rows[:a.per_family], key=lambda r: r["id"])
                else:
                    rows = rows[:a.per_family]
                _write(out / d / f.name, rows)
                chosen[f"{d}/{fam}"] = [r["id"] for r in rows]
                counts[f"{d}/{fam}"] = len(rows)
    for f in sorted(Path(a.realcoh).glob("*.jsonl")):
        rows = _read(f)[:a.per_source]
        _write(out / "realcoh" / f.name, rows)
        counts[f"realcoh/{f.stem}"] = len(rows)
    for tr, fams in man["transforms"].items():
        n = a.long if tr.startswith("long_") else a.stress
        for fam, info in fams.items():
            src_dir = "test" if info["source_file"].split("/")[-2] == "test" else "heldout"
            ok = set(info["kept_source_ids"]) & set(chosen[f"{src_dir}/{fam}"])
            rows = []
            with open(REL / "stress" / tr / f"{fam}.jsonl") as fh:
                for line in fh:
                    r = json.loads(line)
                    sid = r.get("meta", {}).get("source_id") or r["id"].split("~")[0]
                    if sid in ok:
                        rows.append(r)
                        if len(rows) == n:
                            break
            _write(out / "stress" / tr / f"{fam}.jsonl", rows)
            counts[f"stress/{tr}/{fam}"] = len(rows)
    json.dump({"args": vars(a), "counts": counts, "ids": chosen}, open(out / "subset.json", "w"), indent=1)
    tot = Counter()
    for k, v in counts.items():
        tot[k.split("/")[0] if not k.startswith(("mechanics", "programs", "tables")) else "dev"] += v
    print(dict(tot))


if __name__ == "__main__":
    main()
