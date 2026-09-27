"""Instance helpers shared by every BookieBench workstream.

The instance format is defined in SPEC.md section 1. The main entry point for other workstreams is
`exact_answer(instance, query, step=None)`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

MAX_CELLS = 256
SUM_TOL = 1e-5


# ----------------------------------------------------------------------------------------------------------------------
# indexing utilities
# ----------------------------------------------------------------------------------------------------------------------

def var_names(instance) -> list[str]:
    return [v["name"] for v in instance["variables"]]


def var_index(instance, name: str) -> int:
    for i, v in enumerate(instance["variables"]):
        if v["name"] == name:
            return i
    raise KeyError(f"unknown variable {name!r}")


def joint_shape(instance) -> tuple[int, ...]:
    return tuple(len(v["options"]) for v in instance["variables"])


def n_steps(instance) -> int:
    return len(instance["steps"])


def resolve_step(instance, step=None) -> int:
    T = n_steps(instance)
    if step is None:
        step = T - 1
    step = int(step)
    if step < 0:
        step += T
    if not 0 <= step < T:
        raise IndexError(f"step {step} out of range for {T} steps")
    return step


def joint_array(instance, step=None) -> np.ndarray:
    """Exact joint after the evidence prefix `step` (default: final step) as an ndarray in variable order
    (renormalised to sum to 1 exactly; stored values are rounded to 8-10 decimals)."""
    k = resolve_step(instance, step)
    J = np.asarray(instance["steps"][k]["joint"], dtype=float)
    s = J.sum()
    if s > 0:
        J = J / s  # absorb storage rounding, so marginals sum to 1 and neg is an exact complement
    return J.reshape(joint_shape(instance))


def cell_index(instance, assignment) -> int:
    """Row-major flat index of a full assignment ({var: option_idx} or a sequence of option indices)."""
    shape = joint_shape(instance)
    if isinstance(assignment, dict):
        assignment = [assignment[n] for n in var_names(instance)]
    return int(np.ravel_multi_index(tuple(int(a) for a in assignment), shape))


def cell_assignment(instance, flat: int) -> dict:
    shape = joint_shape(instance)
    idx = np.unravel_index(int(flat), shape)
    return {n: int(i) for n, i in zip(var_names(instance), idx)}


def iter_cells(instance) -> Iterator[dict]:
    for flat in range(int(np.prod(joint_shape(instance)))):
        yield cell_assignment(instance, flat)


def event_mask(instance, event: dict) -> np.ndarray:
    """Boolean array (joint shape) that is True on the cells where the conjunction `event` holds."""
    shape = joint_shape(instance)
    mask = np.ones(shape, dtype=bool)
    for name, allowed in event.items():
        ax = var_index(instance, name)
        m1 = np.zeros(shape[ax], dtype=bool)
        m1[list(allowed)] = True
        bshape = [1] * len(shape)
        bshape[ax] = shape[ax]
        mask &= m1.reshape(bshape)
    return mask


def marginal(J: np.ndarray, axis: int) -> np.ndarray:
    other = tuple(a for a in range(J.ndim) if a != axis)
    return J.sum(axis=other)


def query_mask(instance, query) -> tuple[np.ndarray, np.ndarray | None]:
    """(event-true mask with `neg` applied, given-mask or None) for a noul/cond query."""
    m = event_mask(instance, query["event"])
    if query.get("neg"):
        m = ~m
    g = event_mask(instance, query["given"]) if query["kind"] == "cond" else None
    return m, g


# ----------------------------------------------------------------------------------------------------------------------
# exact answers
# ----------------------------------------------------------------------------------------------------------------------

def exact_answer(instance, query, step=None) -> list[float]:
    """Exact answer of `query` on `instance`.

    Step resolution: explicit `step` argument > query["step"] > final step.
    marginal -> probability vector over the variable's options; noul/cond -> [P(yes)].
    `neg: true` negates the whole event (for cond: P(not event | given)).
    """
    if step is None:
        step = query.get("step")
    J = joint_array(instance, step)
    kind = query["kind"]
    if kind == "marginal":
        return marginal(J, var_index(instance, query["var"])).tolist()
    m, g = query_mask(instance, query)
    if kind == "noul":
        return [float(J[m].sum())]
    if kind == "cond":
        pg = float(J[g].sum())
        if pg <= 0:
            raise ZeroDivisionError(f"conditioning event of {query.get('id')} has probability 0")
        return [float(J[m & g].sum()) / pg]
    raise ValueError(f"unknown query kind {kind!r}")


def exact_answers(instance, step=None) -> dict[str, list[float]]:
    return {q["id"]: exact_answer(instance, q, step) for q in instance["queries"]}


def mart_queries(instance) -> dict[int, dict]:
    """step -> the marginal query of `mart_var` at that step. The unstepped final marginal serves as step T-1
    when no explicit step-(T-1) query exists (template version 2 drops that duplicate)."""
    out = {}
    mv = instance.get("mart_var")
    T = n_steps(instance)
    for q in instance["queries"]:
        if q["kind"] == "marginal" and q["var"] == mv:
            k = int(q["step"]) if "step" in q else T - 1
            if k not in out or "step" in q:
                out[k] = q
    return out


def state_text(instance, k=None, order=None) -> str:
    """The text shown to a model at step k: prelude + evidence_0..k (optionally in a permuted evidence order)."""
    T = n_steps(instance)
    k = resolve_step(instance, k)
    order = list(range(T)) if order is None else list(order)
    ev = [instance["steps"][i]["evidence"] for i in order[: k + 1]]
    return "\n".join([instance["prelude"], *ev])


# ----------------------------------------------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------------------------------------------

class InvalidInstance(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise InvalidInstance(msg)


def _check_event(instance, event, where):
    names = set(var_names(instance))
    shape = dict(zip(var_names(instance), joint_shape(instance)))
    _check(isinstance(event, dict) and len(event) >= 1, f"{where}: event must be a non-empty dict")
    for n, allowed in event.items():
        _check(n in names, f"{where}: unknown variable {n!r}")
        _check(isinstance(allowed, list) and len(allowed) >= 1, f"{where}: allowed options must be a non-empty list")
        _check(len(set(allowed)) == len(allowed), f"{where}: duplicate options")
        _check(all(isinstance(a, int) and 0 <= a < shape[n] for a in allowed), f"{where}: bad option index")


def validate(instance, strict: bool = True) -> bool:
    """Check the SPEC contract. Raises InvalidInstance on the first violation, returns True otherwise.

    strict=True additionally requires the full SPEC query set (all marginals, >=2 conjunctions, >=1 negation,
    >=1 conditional, step marginals of mart_var at every step).
    """
    for key in ("id", "family", "split", "variables", "steps", "prelude", "gold", "queries"):
        _check(key in instance, f"missing key {key!r}")
    vs = instance["variables"]
    _check(2 <= len(vs) <= 4, f"need 2-4 variables, got {len(vs)}")
    names = var_names(instance)
    _check(len(set(names)) == len(names), "duplicate variable names")
    for v in vs:
        _check(len(v["options"]) >= 2, f"variable {v['name']} needs >= 2 options")
        _check(len(set(v["options"])) == len(v["options"]), f"variable {v['name']} has duplicate options")
    shape = joint_shape(instance)
    ncell = int(np.prod(shape))
    _check(ncell <= MAX_CELLS, f"joint has {ncell} > {MAX_CELLS} cells")
    T = n_steps(instance)
    _check(T >= 1, "need at least one step")
    _check(isinstance(instance["prelude"], str) and instance["prelude"], "empty prelude")
    for k, st in enumerate(instance["steps"]):
        _check(isinstance(st.get("evidence"), str) and st["evidence"], f"step {k}: empty evidence")
        J = np.asarray(st["joint"], dtype=float)
        _check(J.shape == shape, f"step {k}: joint shape {J.shape} != {shape}")
        _check(np.all(np.isfinite(J)) and np.all(J >= 0), f"step {k}: joint has negative/non-finite cells")
        _check(abs(J.sum() - 1) < SUM_TOL, f"step {k}: joint sums to {J.sum()}")
        ne = st.get("next_evidence")
        if ne is not None:
            _check(k < T - 1, f"step {k}: next_evidence on the last step")
            _check(len(ne) >= 1, f"step {k}: empty next_evidence")
            ps = [a["prob"] for a in ne]
            _check(all(p >= 0 for p in ps) and abs(sum(ps) - 1) < SUM_TOL, f"step {k}: next_evidence probs sum {sum(ps)}")
            _check(all(isinstance(a["evidence"], str) and a["evidence"] for a in ne), f"step {k}: bad alt text")
            texts = [a["evidence"] for a in ne]
            _check(len(set(texts)) == len(texts), f"step {k}: duplicate next_evidence texts")
            _check(instance["steps"][k + 1]["evidence"] in texts, f"step {k}: realised next evidence not among alternatives")
    gold = instance["gold"]
    _check(set(gold) == set(names), "gold must assign every variable")
    for n, s in zip(names, shape):
        _check(isinstance(gold[n], int) and 0 <= gold[n] < s, f"gold[{n}] out of range")
    mv = instance.get("mart_var")
    if mv is not None:
        _check(mv in names, f"mart_var {mv!r} unknown")
    if "perms" in instance:
        for p in instance["perms"]:
            _check(sorted(p) == list(range(T)), f"bad perm {p}")

    qids = set()
    have_marg, n_conj, n_neg, n_cond, mart_steps = set(), 0, 0, 0, set()
    for q in instance["queries"]:
        qid = q.get("id")
        _check(isinstance(qid, str) and qid not in qids, f"bad/duplicate query id {qid!r}")
        qids.add(qid)
        _check(isinstance(q.get("text"), str) and q["text"].strip().endswith("?"), f"{qid}: text must be a question")
        if "step" in q:
            _check(isinstance(q["step"], int) and 0 <= q["step"] < T, f"{qid}: bad step")
        kind = q.get("kind")
        _check(kind in ("marginal", "noul", "cond"), f"{qid}: bad kind {kind!r}")
        if kind == "marginal":
            _check(q.get("var") in names, f"{qid}: unknown var")
            _check(q.get("options") == vs[names.index(q["var"])]["options"], f"{qid}: options differ from variable")
            if "step" not in q or q["step"] == T - 1:
                have_marg.add(q["var"])
            if q["var"] == mv:
                mart_steps.add(q["step"] if "step" in q else T - 1)
        else:
            _check_event(instance, q.get("event"), qid)
            if kind == "cond":
                _check_event(instance, q.get("given"), qid + ".given")
                _check(not set(q["event"]) & set(q["given"]), f"{qid}: event and given share a variable")
                pg = float(joint_array(instance, q.get("step"))[event_mask(instance, q["given"])].sum())
                _check(pg > 0, f"{qid}: given has probability 0")
                n_cond += 1
            elif q.get("neg"):
                n_neg += 1
            elif len(q["event"]) >= 2:
                n_conj += 1
        a = exact_answer(instance, q)
        if kind == "marginal":
            _check(abs(sum(a) - 1) < SUM_TOL, f"{qid}: marginal does not sum to 1")
        else:
            _check(-SUM_TOL <= a[0] <= 1 + SUM_TOL, f"{qid}: probability out of range")
    if strict:
        _check(have_marg == set(names), f"missing final marginals for {set(names) - have_marg}")
        _check(n_conj >= 2, "need >= 2 conjunction queries")
        _check(n_neg >= 1, "need >= 1 negation query")
        _check(n_cond >= 1, "need >= 1 conditional query")
        _check(mv is not None, "missing mart_var")
        _check(mart_steps == set(range(T)), f"mart_var step marginals missing for steps {set(range(T)) - mart_steps}")
    return True


# ----------------------------------------------------------------------------------------------------------------------
# (de)serialization
# ----------------------------------------------------------------------------------------------------------------------

def dumps(instance) -> str:
    return json.dumps(instance, ensure_ascii=False, separators=(",", ":"))


def loads(line: str):
    return json.loads(line)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def iter_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for l in f:
            if l.strip():
                yield json.loads(l)


def write_jsonl(path, instances: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(dumps(inst) + "\n")
            n += 1
    return n


def load_split(data_dir, split: str) -> dict[str, list[dict]]:
    """{family: instances} for data/<split>/*.jsonl."""
    d = Path(data_dir) / split
    return {p.stem: read_jsonl(p) for p in sorted(d.glob("*.jsonl"))}


# ----------------------------------------------------------------------------------------------------------------------
# release post-processing (template version 2): shared by every pack, applied at instance-finalisation time
# ----------------------------------------------------------------------------------------------------------------------

def _stable_seed(*parts) -> list[int]:
    import zlib
    return [zlib.crc32(str(p).encode()) for p in parts]


def shuffle_options(instance, rng) -> dict:
    """Permute every variable's option order (a fresh permutation per variable), remapping joints, next_evidence
    mart_post, gold, marginal-query options and event/given indices consistently. Exact answers are unchanged up to
    the relabelling. Returns a new instance; `meta.option_perm[var]` = original index of each new position."""
    import copy
    inst = copy.deepcopy(instance)
    names = var_names(inst)
    perms = {}
    for v in inst["variables"]:
        n = len(v["options"])
        p = [int(i) for i in rng.permutation(n)]  # new position j holds old option p[j]
        perms[v["name"]] = p
        v["options"] = [v["options"][i] for i in p]
    inv = {n: {old: new for new, old in enumerate(p)} for n, p in perms.items()}
    shape = joint_shape(inst)
    idx = np.ix_(*[perms[n] for n in names])
    for st in inst["steps"]:
        J = np.asarray(st["joint"], dtype=float).reshape(shape)
        st["joint"] = J[idx].tolist()
        for a in st.get("next_evidence", []) or []:
            if "mart_post" in a and inst.get("mart_var") in perms:
                mp = a["mart_post"]
                a["mart_post"] = [mp[i] for i in perms[inst["mart_var"]]]
    inst["gold"] = {n: inv[n][g] for n, g in inst["gold"].items()}
    opts = {v["name"]: v["options"] for v in inst["variables"]}
    for q in inst["queries"]:
        if q["kind"] == "marginal":
            q["options"] = list(opts[q["var"]])
        for key in ("event", "given"):
            if key in q:
                q[key] = {n: sorted(inv[n][i] for i in al) for n, al in q[key].items()}
    inst.setdefault("meta", {})["option_perm"] = perms
    return inst


def degenerate_query(instance, q, eps: float = 0.01) -> bool:
    """Final-step (or the query's step) answer near-certain (max > 1-eps / p outside [eps, 1-eps]) or uniform."""
    a = np.asarray(exact_answer(instance, q))
    if q["kind"] == "marginal":
        return bool(a.max() > 1 - eps or (a.max() - a.min()) < eps)
    p = a[0]
    return bool(p < eps or p > 1 - eps or abs(p - 0.5) < eps)


def degenerate_fraction(instance, eps: float = 0.01) -> float:
    qs = [q for q in instance["queries"] if "step" not in q]
    return sum(degenerate_query(instance, q, eps) for q in qs) / max(1, len(qs))


def mart_moves(instance, tol: float = 0.01) -> bool:
    """True when the mart_var posterior changes (TV > tol) at some step after step 0 (single-step: True)."""
    T = n_steps(instance)
    if T < 2 or instance.get("mart_var") is None:
        return True
    ax = var_index(instance, instance["mart_var"])
    p0 = marginal(joint_array(instance, 0), ax)
    return any(0.5 * np.abs(marginal(joint_array(instance, k), ax) - p0).sum() > tol for k in range(1, T))


def finalize_release(instance, seed: int = 0) -> dict:
    """Release (tv=2) finalisation shared by every pack: seeded per-instance option shuffle (shortcuts #5).
    The rng is derived from (seed, instance id), so the result is deterministic and independent of generation order.
    Degenerate/mart-still/dedupe *rejection* is done by the release builder (bookiebench.sims.release), which can redraw
    whole instances; `degenerate_fraction` and `mart_moves` are the shared criteria."""
    rng = np.random.default_rng(_stable_seed(seed, instance["id"], "optshuffle"))
    out = shuffle_options(instance, rng)
    out["meta"]["release"] = True
    return out


def unshuffle_options(instance) -> dict:
    """Inverse of `shuffle_options`: restore canonical option order using `meta.option_perm` (no-op without it).
    Lets order-dependent audit tools (review/exactness) run unchanged on release files."""
    perms = (instance.get("meta") or {}).get("option_perm")
    if not perms:
        return instance
    import copy
    inst = copy.deepcopy(instance)
    names = var_names(inst)
    inv = {n: [int(i) for i in np.argsort(p)] for n, p in perms.items()}  # old option i sits at new position inv[i]
    for v in inst["variables"]:
        p = perms[v["name"]]
        opts = [None] * len(p)
        for new, old in enumerate(p):
            opts[old] = v["options"][new]
        v["options"] = opts
    shape = joint_shape(inst)
    idx = np.ix_(*[inv[n] for n in names])
    for st in inst["steps"]:
        J = np.asarray(st["joint"], dtype=float).reshape(shape)
        st["joint"] = J[idx].tolist()
        for a in st.get("next_evidence", []) or []:
            if "mart_post" in a and inst.get("mart_var") in inv:
                mp = a["mart_post"]
                a["mart_post"] = [mp[i] for i in inv[inst["mart_var"]]]
    inst["gold"] = {n: perms[n][g] for n, g in inst["gold"].items()}
    opts = {v["name"]: v["options"] for v in inst["variables"]}
    for q in inst["queries"]:
        if q["kind"] == "marginal":
            q["options"] = list(opts[q["var"]])
        for key in ("event", "given"):
            if key in q:
                q[key] = {n: sorted(perms[n][i] for i in al) for n, al in q[key].items()}
    del inst["meta"]["option_perm"]
    return inst
