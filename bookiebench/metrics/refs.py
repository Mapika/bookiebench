"""Evidence-blind reference predictors, computed internally (no prediction files needed).

- uniform_joint: the uniform distribution over the joint cells; every query is answered from it. Needs nothing.
- indep_joint:   the product over variables of a train-mean marginal. The marginal for (family, variable) is the mean
                 exact final-step marginal over the first `n_per_file` train instances of that family; unseen families
                 (e.g. held-out) fall back to the mean over all train marginals with the same number of options, and
                 then to uniform.

Both are coherent by construction (dutch = 0) and ignore the evidence (sens = 0, mart = 0 when every step gets the
same joint), so they are the floor that skill scores and informative-coherence columns are measured against.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .exact import exact_answer, shape_of


def with_joint(inst: dict, joint: np.ndarray) -> dict:
    """Copy of `inst` whose every step has `joint` (so step-k queries are answered from it too)."""
    j = np.asarray(joint, dtype=float).tolist()
    return dict(inst, steps=[dict(s, joint=j) for s in inst["steps"]])


def uniform_joint(inst: dict) -> np.ndarray:
    shp = shape_of(inst)
    return np.full(shp, 1.0 / int(np.prod(shp)))


def answers_from_joint(inst: dict, joint: np.ndarray) -> dict:
    fake = with_joint(inst, joint)
    out = {}
    for q in inst["queries"]:
        a = exact_answer(fake, q)
        if q["kind"] == "cond" and not np.all(np.isfinite(a)):
            a = np.array([0.5])  # P(given) = 0 under this joint: every price is coherent, the bet is never on
        if np.all(np.isfinite(a)):
            out[q["id"]] = a.tolist()
    return out


class IndepTable:
    def __init__(self):
        self.fam: dict[tuple[str, str, int], np.ndarray] = {}
        self.by_k: dict[int, np.ndarray] = {}
        self.source = None

    def marginal(self, family: str, var: dict) -> np.ndarray:
        k = len(var["options"])
        m = self.fam.get((family, var["name"], k))
        if m is None:
            m = self.by_k.get(k)
        return np.ones(k) / k if m is None else m

    def joint(self, inst: dict) -> np.ndarray:
        J = None
        for v in inst["variables"]:
            m = self.marginal(inst.get("family", "?"), v)
            J = m if J is None else np.multiply.outer(J, m)
        return J


def fit_indep(train: str | Path, n_per_file: int = 1500) -> IndepTable | None:
    """Fit the indep_joint table from train files (None if `train` does not exist)."""
    train = Path(train)
    files = [train] if train.is_file() else sorted(train.rglob("*.jsonl")) if train.is_dir() else []
    if not files:
        return None
    fam, byk = defaultdict(list), defaultdict(list)
    for f in files:
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i >= n_per_file:
                    break
                if not line.strip():
                    continue
                inst = json.loads(line)
                if not inst.get("steps") or inst["steps"][-1].get("joint") is None:
                    continue
                j = np.asarray(inst["steps"][-1]["joint"], dtype=float).reshape(shape_of(inst))
                j = j / j.sum()
                for ax, v in enumerate(inst["variables"]):
                    m = j.sum(axis=tuple(a for a in range(j.ndim) if a != ax))
                    fam[(inst.get("family", "?"), v["name"], m.size)].append(m)
                    byk[m.size].append(m)
    t = IndepTable()
    t.fam = {k: np.mean(v, 0) for k, v in fam.items()}
    t.by_k = {k: np.mean(v, 0) for k, v in byk.items()}
    t.source = str(train)
    return t


def ref_predictions(instances: dict[str, dict], name: str, table: IndepTable | None = None) -> dict:
    """{id: {"base": prediction}} for reference `name` in {"uniform_joint", "indep_joint"} (base variant only)."""
    out = {}
    for iid, inst in instances.items():
        try:
            J = uniform_joint(inst) if name == "uniform_joint" else table.joint(inst)
            ans = answers_from_joint(inst, J)
        except Exception:  # noqa: BLE001 - malformed instance: reference abstains
            ans = {}
        out[iid] = {"base": {"id": iid, "model": name, "variant": "base", "answers": ans}}
    return out
