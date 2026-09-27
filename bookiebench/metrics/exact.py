"""Exact answers and event masks computed from an instance's joint (SPEC §1).

`exact_answer` delegates to `bookiebench.sims.core.exact_answer` when that module is importable and falls back to the
private implementation here otherwise. Set BOOKIEBENCH_METRICS_OWN_EXACT=1 to force the private implementation.
Whatever the source, results are returned as a 1-D float numpy array: the probability vector for a marginal query, or
`[p_yes]` for a noul/cond query.
"""
from __future__ import annotations

import os

import numpy as np


def shape_of(inst: dict) -> tuple[int, ...]:
    return tuple(len(v["options"]) for v in inst["variables"])


def var_index(inst: dict) -> dict[str, int]:
    return {v["name"]: i for i, v in enumerate(inst["variables"])}


def final_step(inst: dict) -> int:
    return len(inst["steps"]) - 1


def query_step(inst: dict, query: dict) -> int:
    """Step index a query is evaluated at (non-negative). Defaults to the final step."""
    s = query.get("step")
    n = len(inst["steps"])
    if s is None:
        return n - 1
    s = int(s)
    return s + n if s < 0 else s


def is_final(inst: dict, query: dict) -> bool:
    return query_step(inst, query) == final_step(inst)


def joint_at(inst: dict, step: int | None = None) -> np.ndarray:
    """Exact joint at `step` (default final), reshaped to the variable shape."""
    if step is None:
        step = final_step(inst)
    j = np.asarray(inst["steps"][step]["joint"], dtype=float).reshape(shape_of(inst))
    tot = j.sum()
    if tot > 0 and abs(tot - 1.0) > 1e-12:
        j = j / tot
    return j


def event_mask(inst: dict, event: dict | None, neg: bool = False) -> np.ndarray:
    """Boolean mask over joint cells (variable shape) of the conjunction `event` (var -> allowed option indices)."""
    shp = shape_of(inst)
    vi = var_index(inst)
    m = np.ones(shp, dtype=bool)
    for name, allowed in (event or {}).items():
        ax = vi[name]
        if isinstance(allowed, (int, np.integer)):
            allowed = [allowed]
        sel = np.zeros(shp[ax], dtype=bool)
        sel[list(allowed)] = True
        bshape = [1] * len(shp)
        bshape[ax] = shp[ax]
        m &= sel.reshape(bshape)
    return ~m if neg else m


def marginal_mask(inst: dict, var: str, option: int) -> np.ndarray:
    return event_mask(inst, {var: [option]})


def own_exact_answer(inst: dict, query: dict) -> list[float]:
    j = joint_at(inst, query_step(inst, query))
    kind = query["kind"]
    if kind == "marginal":
        ax = var_index(inst)[query["var"]]
        other = tuple(a for a in range(j.ndim) if a != ax)
        return [float(x) for x in j.sum(axis=other)]
    if kind == "noul":
        return [float(j[event_mask(inst, query["event"], bool(query.get("neg")))].sum())]
    if kind == "cond":
        g = event_mask(inst, query.get("given"))
        e = event_mask(inst, query["event"], bool(query.get("neg")))
        pg = float(j[g].sum())
        if pg <= 0:
            return [float("nan")]
        return [float(j[e & g].sum()) / pg]
    raise ValueError(f"unknown query kind {kind!r}")


_SIMS_EXACT = None
if not os.environ.get("BOOKIEBENCH_METRICS_OWN_EXACT"):
    try:  # later-compatible: prefer the shared implementation when sims is ready
        from bookiebench.sims.core import exact_answer as _SIMS_EXACT  # type: ignore
    except Exception:  # noqa: BLE001 - sims missing or broken -> fallback
        _SIMS_EXACT = None


def exact_source() -> str:
    return "bookiebench.sims.core" if _SIMS_EXACT is not None else "bookiebench.metrics.exact"


def exact_answer(inst: dict, query: dict) -> np.ndarray:
    res = _SIMS_EXACT(inst, query) if _SIMS_EXACT is not None else own_exact_answer(inst, query)
    arr = np.atleast_1d(np.asarray(res, dtype=float)).ravel()
    if query["kind"] in ("noul", "cond") and arr.size > 1:
        arr = arr[:1]  # tolerate a [p_yes, p_no] return format
    return arr
