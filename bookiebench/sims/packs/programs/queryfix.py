"""Post-hoc redraw of trivially certain queries (pack-local; the shared make_queries is not touched).

A noul/cond query whose exact final answer is degenerate (tv=1: 0 or 1; tv=2: the shared release criterion,
outside [0.01, 0.99] or within 0.01 of 1/2), or a cond whose given has P < 1e-3, is replaced by a fresh random
query of the same shape (same kind, same number of variables, same `neg`), keeping its id. tv=2 redraws also follow
the tv=2 wording rules (either-clauses last, no names absent from the state). If none is found the original stays.
"""
from __future__ import annotations

import numpy as np

from bookiebench.sims.common import pick
from bookiebench.sims.core import exact_answer, event_mask, joint_array

EPS = 1e-3


def is_trivial(inst, q) -> bool:
    a = exact_answer(inst, q)
    if q["kind"] == "marginal":
        return max(a) > 1 - 1e-9
    return a[0] < 1e-9 or a[0] > 1 - 1e-9


def _bad_answer(p, tv) -> bool:
    if tv >= 2:  # the shared release criterion (bookiebench.sims.world tv=2): near-certain or ~uniform
        from bookiebench.sims import world as W
        return p < W.DEGEN_EPS or p > 1 - W.DEGEN_EPS or abs(p - 0.5) < W.UNIFORM_EPS
    return not (EPS <= p <= 1 - EPS)


def _key(q):
    return (q["kind"], q.get("neg", False), tuple(sorted((k, tuple(v)) for k, v in q.get("event", {}).items())),
            tuple(sorted((k, tuple(v)) for k, v in q.get("given", {}).items())))


def redraw_trivial(inst, variables, rng, tpl: dict, fmt_kw=None, tries=80) -> dict:
    """variables: bookiebench.sims.world.Var list in instance order. tpl: {conj2, conj3, neg1, neg2, cond} templates
    with {a},{b},{c},{g} (and fmt_kw extra placeholders). Returns counts {before, after} of trivial noul/cond."""
    from bookiebench.sims import world as W
    from bookiebench.sims.common import get_tv
    from bookiebench.sims.core import state_text
    tv = get_tv()
    ctx = state_text(inst)
    fmt_kw = fmt_kw or {}
    V = variables

    def order(vids, e):  # tv=2: multi-option (either-) clauses last, so the either-scope is unambiguous
        pairs = [(vi, e[vi]) for vi in vids]
        return W.either_last(pairs) if tv >= 2 else pairs
    nv = len(V)
    J = joint_array(inst)
    seen = {_key(q) for q in inst["queries"] if q["kind"] != "marginal"}
    before = after = 0

    def ev(vi, multi=True):
        n = len(V[vi].options)
        if multi and n >= 3 and rng.random() < 0.25:
            return sorted(int(i) for i in rng.choice(n, size=2, replace=False))
        return [int(rng.integers(n))]

    def p_given(given):
        return float(J[event_mask(inst, given)].sum())

    for qi, q in enumerate(inst["queries"]):
        if q["kind"] == "marginal":
            continue
        bad = _bad_answer(exact_answer(inst, q)[0], tv) or (q["kind"] == "cond" and p_given(q["given"]) < EPS)
        if not bad:
            continue
        before += 1
        new = None
        for _ in range(tries):
            if q["kind"] == "cond":
                a, g = (int(i) for i in rng.choice(nv, size=2, replace=False))
                e = ev(a, False)
                gi = [int(rng.integers(len(V[g].options)))]
                cand = {"kind": "cond", "event": {V[a].name: e}, "given": {V[g].name: gi},
                        "text": pick(rng, tpl["cond"]).format(g=V[g].says(gi), a=V[a].says(e), **fmt_kw)}
                if p_given(cand["given"]) < EPS:
                    continue
            else:
                k = len(q["event"])
                k = min(k, nv)
                vids = sorted(int(i) for i in rng.choice(nv, size=k, replace=False))
                if q.get("neg"):
                    e = {vi: ev(vi, k == 1) for vi in vids}
                    if tv >= 2 and any(W._NEGWORD.search(V[vi].says(e[vi])) for vi in vids):
                        continue  # no double negation
                    cl = [V[vi].says(e[vi]) for vi in vids]
                    text = (pick(rng, tpl["neg1"]).format(a=cl[0], **fmt_kw) if k == 1
                            else pick(rng, tpl["neg2"]).format(a=cl[0], b=cl[1], **fmt_kw))
                else:
                    e = {vi: ev(vi) for vi in vids}
                    cl = [V[vi].says(idx) for vi, idx in order(vids, e)]
                    text = (pick(rng, tpl["conj2"]).format(a=cl[0], b=cl[1], **fmt_kw) if k == 2
                            else pick(rng, tpl["conj3"]).format(a=cl[0], b=cl[1], c=cl[2], **fmt_kw))
                cand = {"kind": "noul", "event": {V[vi].name: e[vi] for vi in vids}, "text": text}
                if q.get("neg"):
                    cand["neg"] = True
            if _key(cand) in seen or (tv >= 2 and W.names_absent(cand["text"], ctx)):
                continue
            p = exact_answer(inst, cand)[0]
            if not _bad_answer(p, tv):
                new = cand
                break
        if new is None:
            after += 1
            continue
        seen.add(_key(new))
        new["id"] = q["id"]
        inst["queries"][qi] = new
    return {"before": before, "after": after}


def pinned_vars(inst) -> list[str]:
    """Variables whose final marginal is a point mass."""
    J = joint_array(inst)
    out = []
    for i, v in enumerate(inst["variables"]):
        m = J.sum(axis=tuple(a for a in range(J.ndim) if a != i))
        if m.max() > 1 - 1e-9:
            out.append(v["name"])
    return out


def world_templates():
    """The shared templates of the active template version."""
    from bookiebench.sims import world as W
    from bookiebench.sims.common import get_tv
    if get_tv() >= 2:
        return {"conj2": W.CONJ2_V2, "conj3": W.CONJ3_V2, "neg1": W.NEG1_V2, "neg2": W.NEG2_V2, "cond": W.COND_V2}
    return {"conj2": W.CONJ2, "conj3": W.CONJ3, "neg1": W.NEG1, "neg2": W.NEG2, "cond": W.COND}
