"""Loading instances (data/<split>/<family>.jsonl) and predictions (results/<model>/<split>__<family>.jsonl)."""
from __future__ import annotations

import json
from pathlib import Path


def _jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _first_split(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    return json.loads(line).get("split")
                except json.JSONDecodeError:
                    return None
    return None


def load_instances(data: str | Path, splits=None) -> dict[str, dict]:
    """Load every instance under `data` (a directory searched recursively for *.jsonl, or a single file).

    `splits`: optional set of split names to keep. Files in data/<split>/ are read directly; if some split has no
    such directory, all files are scanned and filtered on the instance's `split` field."""
    data = Path(data)
    if data.is_file():
        files = [data]
    elif splits is not None:
        files = []
        for s in sorted(splits):
            dirs = [data / s] if (data / s).is_dir() else [d for d in data.rglob(s) if d.is_dir()]
            if dirs:
                files += sorted(f for d in dirs for f in d.glob("*.jsonl"))
            else:  # no directory named after the split: peek at each file's first instance
                files += [f for f in sorted(data.rglob("*.jsonl")) if _first_split(f) == s]
        files = sorted(set(files))
    else:
        files = sorted(data.rglob("*.jsonl"))
    out: dict[str, dict] = {}
    for f in files:
        for inst in _jsonl(f):
            if "id" in inst and "queries" in inst:
                if "split" not in inst and not data.is_file():
                    inst["split"] = f.parent.name
                if splits is None or inst.get("split") in splits:
                    out[inst["id"]] = inst
    return out


def prediction_splits(results: str | Path) -> set[str] | None:
    """Split names from results/<model>/<split>__<family>.jsonl file names (None if not inferable)."""
    results = Path(results)
    if not results.is_dir():
        return None
    names = [f.name for f in results.glob("*.jsonl")]
    if not names or not all("__" in n for n in names):
        return None
    return {n.split("__", 1)[0] for n in names}


def load_predictions(results: str | Path) -> tuple[dict[str, dict[str, dict]], str]:
    """Return ({id: {variant: prediction}}, model name). Later lines override earlier ones."""
    results = Path(results)
    files = [results] if results.is_file() else sorted(results.glob("*.jsonl"))
    preds: dict[str, dict[str, dict]] = {}
    model = None
    for f in files:
        for p in _jsonl(f):
            model = model or p.get("model")
            preds.setdefault(p["id"], {})[p.get("variant", "base")] = p
    return preds, model or (results.name if results.is_dir() else results.stem)
