#!/usr/bin/env python
"""Maintainer tool: build the Hugging Face dataset layout (local only; uploading is a separate, manual step).

    python tools/make_hf_dataset.py --out ../bookiebench-hf

Why Parquet with a JSON-string column. Instances have nested fields of varying depth: a joint's nesting depth is the
number of variables, which differs between instances of the same file, and `meta` differs between sources. Neither
Arrow schema inference nor the HF viewer can type them. So every row keeps the instance **verbatim** as a JSON string
(`instance`), next to flat index columns for filtering and the viewer. The script checks the round trip. Writing the
`instance` values of one `source_file`, in `row` order and one per line, reproduces the shipped JSONL file byte for
byte, and the file sha256 matches data/CHECKSUMS.sha256.

Not included: the train split (regenerable), ids_only realcoh texts (not redistributable; rebuild with
`bookiebench rebuild-realcoh`), and the stress recipe transforms long_* and scaling (`bookiebench rebuild-stress`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data" / "release"
SIM_DIRS = ["test", "test_prior", "heldout", "val", "mechanics/dev", "programs/dev", "tables/dev"]


def checksums() -> dict:
    out = {}
    for line in (ROOT / "data" / "CHECKSUMS.sha256").read_text().splitlines():
        h, n = line.split("  ", 1)
        out[n.strip()] = h
    return out


def rows_of(path: Path, rel: str, group: str, extra=None):
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    lines = text.split("\n")
    assert lines[-1] == "", f"{rel}: no trailing newline"
    lines = lines[:-1]
    for i, line in enumerate(lines):
        inst = json.loads(line)
        meta = inst.get("meta") or {}
        steps = inst.get("steps") or []
        yield {
            "id": inst["id"], "family": inst.get("family"), "split": inst.get("split"), "group": group,
            "level": inst.get("level"), "pack": inst.get("pack"), "transform": inst.get("transform"),
            "source_file": rel, "row": i, "n_variables": len(inst.get("variables") or []), "n_steps": len(steps),
            "n_queries": len(inst.get("queries") or []),
            "has_exact_joint": bool(steps) and all(s.get("joint") is not None for s in steps),
            "evidence_withheld": any(s.get("evidence") is None for s in steps),
            "source_id": meta.get("source_id"), "release_mode": None, "beta": None, **(extra or {}),
            "instance": line,
        }


def write(rows, out: Path):
    cols = list(rows[0])
    types = {"row": pa.int64(), "n_variables": pa.int64(), "n_steps": pa.int64(), "n_queries": pa.int64(),
             "has_exact_joint": pa.bool_(), "evidence_withheld": pa.bool_(), "beta": pa.bool_()}
    schema = pa.schema([(c, types.get(c, pa.string())) for c in cols])  # same schema in every config
    table = pa.table({c: [r.get(c) for r in rows] for c in cols}, schema=schema)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd", row_group_size=2000)
    return table.num_rows


def verify(pq_path: Path, sums: dict):
    t = pq.read_table(pq_path, columns=["source_file", "row", "instance"]).to_pylist()
    by = {}
    for r in t:
        by.setdefault(r["source_file"], []).append((r["row"], r["instance"]))
    for rel, items in by.items():
        items.sort()
        assert [i for i, _ in items] == list(range(len(items))), rel
        text = "\n".join(s for _, s in items) + "\n"
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert h == sums["release/" + rel], f"{rel}: round trip does not reproduce the shipped file"
    return len(by)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    sums = checksums()
    man = json.loads((REL / "sims_manifest.json").read_text())["files"]
    groups: dict = {}
    for d in SIM_DIRS:
        for f in sorted((REL / d).glob("*.jsonl")):
            rel = str(f.relative_to(REL))
            groups.setdefault(man[rel]["group"], []).extend(rows_of(f, rel, man[rel]["group"]))
    report = {}
    for g, rows in groups.items():
        p = out / "data" / g / "dev.parquet"
        report[g] = (write(rows, p), verify(p, sums))
    rcm = json.loads((REL / "realcoh" / "manifest.json").read_text())["sources"]
    rows = []
    for f in sorted((REL / "realcoh").glob("*.jsonl")):
        rows += rows_of(f, f"realcoh/{f.name}", "realcoh", {"release_mode": rcm[f.stem]["release_mode"]})
    p = out / "data" / "realcoh" / "dev.parquet"
    report["realcoh"] = (write(rows, p), verify(p, sums))
    sman = json.loads((REL / "stress" / "manifest.json").read_text())
    for t in sman["release_format"]["data"]:
        rows = []
        for f in sorted((REL / "stress" / t).glob("*.jsonl")):
            rows += rows_of(f, f"stress/{t}/{f.name}", "stress", {"beta": t in sman.get("beta", {})})
        p = out / "data" / f"stress_{t}" / "dev.parquet"
        report[f"stress_{t}"] = (write(rows, p), verify(p, sums))
    meta = out / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for src, dst in [(REL / "sims_manifest.json", "sims_manifest.json"),
                     (REL / "realcoh" / "manifest.json", "realcoh_manifest.json"),
                     (REL / "realcoh" / "NOTICE.md", "realcoh_NOTICE.md"),
                     (REL / "stress" / "manifest.json", "stress_manifest.json"),
                     (REL / "stress" / "recipe.json", "stress_recipe.json"),
                     (ROOT / "NOTICE.md", "NOTICE.md"), (ROOT / "data" / "CHECKSUMS.sha256", "CHECKSUMS.sha256")]:
        shutil.copy2(src, meta / dst)
    shutil.copy2(ROOT / "LICENSE-DATA", out / "LICENSE")
    (out / "build_report.json").write_text(json.dumps({k: {"rows": v[0], "files_roundtrip_ok": v[1]}
                                                       for k, v in report.items()}, indent=1))
    for k, (n, nf) in report.items():
        print(f"{k:24s} {n:6d} rows  {nf:3d} files round-trip ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
