"""Many-variable scaling: compose independent sub-worlds into one instance with up to 4096 joint cells.

The composite joint at step k is the outer product of the sub-worlds' joints at step min(k, T_j - 1) (sub-world j has
seen its first min(k, T_j - 1) + 1 observations). Sub-world queries are kept (renamed, prefixed with the situation
label, same `step`); cross-situation conjunctions, disjunctions and conditionals are added, all exact from the product
joint. Composite instances have more than 4 variables and more than 256 cells, so they are checked with
`validate_composite` instead of `core.validate`. They carry no `next_evidence` (mart is skipped) and no `perms`.
"""
from __future__ import annotations

import numpy as np

from bookiebench.sims.common import pick
from bookiebench.sims.core import InvalidInstance, event_mask, exact_answer, joint_array

from .common import LEVEL, PACK, Clauses, add_query, transform_rng

LABELS = "ABCDEFGH"
TIERS = [(64, 256), (257, 1024), (1025, 4096)]
MAX_CELLS = 4096
DECIMALS = 12

CROSS_AND = ["Is it the case that, in situation {L1}, {a}, and, in situation {L2}, {b}?",
             "How likely is it that {a} (situation {L1}) and that {b} (situation {L2})?"]
CROSS_OR = ["Is it the case that, in situation {L1}, {a}, or, in situation {L2}, {b} (or both)?"]
CROSS_COND = ["If, in situation {L1}, {g}, how likely is it that, in situation {L2}, {a}?"]
CROSS_NEG = ["Is it false that, in situation {L1}, {a}, and, in situation {L2}, {b}?"]


def n_cells(inst) -> int:
    return int(np.prod([len(v["options"]) for v in inst["variables"]]))


def choose_partners(src, pool, rng, lo, hi, max_parts=6):
    """Pick partners (dicts from pool, not src) so that the product of cell counts lands in [lo, hi] if possible."""
    parts, cells = [src], n_cells(src)
    order = [int(i) for i in rng.permutation(len(pool))]
    for i in order:
        if cells >= lo or len(parts) >= max_parts:
            break
        p = pool[i]
        if p["id"] == src["id"] or any(p["id"] == q["id"] for q in parts):
            continue
        c = n_cells(p)
        if cells * c <= hi:
            parts.append(p)
            cells *= c
    return parts


def _rename_event(ev, L):
    return {f"{L}.{k}": list(v) for k, v in ev.items()}


def compose(parts, seed: int = 0, n_cross: int = 6, tier=None):
    src = parts[0]
    rng = transform_rng(src, "scaling", seed, extra=len(parts))
    labels = LABELS[: len(parts)]
    T = max(len(p["steps"]) for p in parts)
    variables, gold = [], {}
    for L, p in zip(labels, parts):
        for v in p["variables"]:
            variables.append({"name": f"{L}.{v['name']}", "options": list(v["options"])})
            gold[f"{L}.{v['name']}"] = int(p["gold"][v["name"]])
    steps = []
    for k in range(T):
        J = None
        ev = []
        for L, p in zip(labels, parts):
            kj = min(k, len(p["steps"]) - 1)
            Jj = joint_array(p, kj)
            J = Jj if J is None else np.multiply.outer(J, Jj)
            if k < len(p["steps"]):
                ev.append(f"Situation {L}: {p['steps'][k]['evidence']}")
        steps.append({"evidence": "\n".join(ev), "joint": np.round(J, DECIMALS).tolist()})
    fams = [p["family"] for p in parts]
    lab = f"{', '.join(labels[:-1])} and {labels[-1]}" if len(labels) > 1 else labels[0]
    intro = (f"The following notes describe {len(parts)} separate situation{'s' if len(parts) > 1 else ''}, labelled "
             f"{lab}. They have nothing to do with each other: everything in one "
             f"situation is independent of everything in the others. New observations for each situation are "
             f"reported with its label.")
    prelude = intro + "\n\n" + "\n\n".join(f"Situation {L}:\n{p['prelude']}" for L, p in zip(labels, parts))
    out = {
        "id": f"{src['id']}~scaling",
        "family": src["family"],
        "split": "stress",
        "transform": "scaling",
        "pack": PACK,
        "level": LEVEL,
        "variables": variables,
        "prelude": prelude,
        "steps": steps,
        "gold": gold,
        "mart_var": f"A.{src['mart_var']}" if src.get("mart_var") else None,
        "queries": [],
        "meta": {"source_id": src.get("meta", {}).get("source_id", src["id"]), "source_split": src.get("split"),
                 "stress": {"transform": "scaling", "source_ids": [p["id"] for p in parts], "families": fams,
                            "n_cells": int(np.prod([len(v["options"]) for v in variables])),
                            "n_vars": len(variables), "tier": tier}},
    }
    for L, p in zip(labels, parts):
        for q in p["queries"]:
            nq = dict(q)
            nq["id"] = f"{L}.{q['id']}"
            nq["text"] = f"In situation {L}: {q['text']}"
            if q["kind"] == "marginal":
                nq["var"] = f"{L}.{q['var']}"
            else:
                nq["event"] = _rename_event(q["event"], L)
                if "given" in q:
                    nq["given"] = _rename_event(q["given"], L)
            out["queries"].append(nq)
    # martingale step marginals of A's mart_var for composite steps beyond A's own horizon
    if out["mart_var"]:
        mv0 = src["mart_var"]
        have = {q["step"] for q in src["queries"] if q["kind"] == "marginal" and q["var"] == mv0 and "step" in q}
        tmpl = next(q for q in src["queries"] if q["kind"] == "marginal" and q["var"] == mv0)
        for k in range(T):
            if k not in have:
                out["queries"].append({"kind": "marginal", "var": out["mart_var"], "step": k, "id": f"A.m{k}",
                                       "text": f"In situation A: {tmpl['text']}", "options": list(tmpl["options"])})
    # cross-situation queries
    cls = [Clauses(p) for p in parts]
    nopt = {v["name"]: len(v["options"]) for v in variables}
    made, seen = 0, set()
    for _ in range(60):
        if made >= n_cross or len(parts) < 2:
            break
        i1, i2 = (int(x) for x in rng.choice(len(parts), size=2, replace=False))
        L1, L2 = labels[i1], labels[i2]
        v1 = pick(rng, parts[i1]["variables"])["name"]
        v2 = pick(rng, parts[i2]["variables"])["name"]
        a1 = int(rng.integers(len(parts[i1]["variables"][[v["name"] for v in parts[i1]["variables"]].index(v1)]["options"])))
        a2 = int(rng.integers(len(parts[i2]["variables"][[v["name"] for v in parts[i2]["variables"]].index(v2)]["options"])))
        c1, c2 = cls[i1].clause(v1, [a1]), cls[i2].clause(v2, [a2])
        n1, n2 = f"{L1}.{v1}", f"{L2}.{v2}"
        kind = ["and", "or", "cond", "neg"][made % 4]
        comp = lambda n, a: [i for i in range(nopt[n]) if i != a]  # noqa: E731
        if kind == "and":
            q = {"kind": "noul", "event": {n1: [a1], n2: [a2]}, "text": pick(rng, CROSS_AND).format(L1=L1, L2=L2, a=c1, b=c2)}
        elif kind == "or":
            q = {"kind": "noul", "event": {n1: comp(n1, a1), n2: comp(n2, a2)}, "neg": True,
                 "text": pick(rng, CROSS_OR).format(L1=L1, L2=L2, a=c1, b=c2)}
        elif kind == "neg":
            q = {"kind": "noul", "event": {n1: [a1], n2: [a2]}, "neg": True,
                 "text": pick(rng, CROSS_NEG).format(L1=L1, L2=L2, a=c1, b=c2)}
        else:
            q = {"kind": "cond", "event": {n2: [a2]}, "given": {n1: [a1]},
                 "text": pick(rng, CROSS_COND).format(L1=L1, L2=L2, g=c1, a=c2)}
        key = repr(sorted(q.items(), key=lambda kv: kv[0]))
        if key in seen:
            continue
        seen.add(key)
        try:
            add_query(out, q)
        except ValueError:
            continue
        made += 1
    return out


def scaling(inst, pool, seed: int = 0, tier_idx=None):
    rng = transform_rng(inst, "scaling", seed)
    if tier_idx is None:
        tier_idx = int(rng.integers(len(TIERS)))
    lo, hi = TIERS[tier_idx]
    parts = choose_partners(inst, pool, rng, lo, hi)
    return compose(parts, seed=seed, tier=f"{lo}-{hi}")


def validate_composite(inst) -> bool:
    """SPEC checks relaxed for >4 variables / >256 cells: shapes, sums, gold, answerable queries, mart steps."""
    def chk(c, m):
        if not c:
            raise InvalidInstance(m)
    shape = tuple(len(v["options"]) for v in inst["variables"])
    chk(int(np.prod(shape)) <= MAX_CELLS, "too many cells")
    names = [v["name"] for v in inst["variables"]]
    chk(len(set(names)) == len(names), "duplicate variable names")
    chk(set(inst["gold"]) == set(names), "gold must assign every variable")
    T = len(inst["steps"])
    for k, st in enumerate(inst["steps"]):
        J = np.asarray(st["joint"], dtype=float)
        chk(J.shape == shape, f"step {k} shape")
        chk(abs(J.sum() - 1) < 1e-5 and np.all(J >= 0), f"step {k} sum")
    ids = set()
    mart = set()
    for q in inst["queries"]:
        chk(q["id"] not in ids, f"dup qid {q['id']}")
        ids.add(q["id"])
        chk(q["text"].strip().endswith("?"), f"{q['id']} not a question")
        if q["kind"] == "cond":
            pg = float(joint_array(inst, q.get("step"))[event_mask(inst, q["given"])].sum())
            chk(pg > 0, f"{q['id']} given prob 0")
        a = exact_answer(inst, q)
        if q["kind"] == "marginal":
            chk(abs(sum(a) - 1) < 1e-5, "marginal sum")
            if q["var"] == inst.get("mart_var") and "step" in q:
                mart.add(q["step"])
    if inst.get("mart_var"):
        chk(mart == set(range(T)), "mart steps")
    return True
