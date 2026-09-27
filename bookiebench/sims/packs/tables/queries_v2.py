"""Release (tv=2) query set for the tables pack, following the shared tv=2 rules of bookiebench.sims.world:

  * every yes/no query is a polar question (no "what fraction / how likely" wording on yes/no queries);
  * conjunctions are fenced ("both that ... and that ...", "all of the following: ...; ...") and a multi-option
    "either" clause always comes last (`world.either_last`), so the either-scope is unambiguous;
  * negated conjunctions always say "both";
  * no query mentions a capitalised name that is absent from the state text (`world.names_absent`), e.g. a
    carrier that never occurs in the table;
  * events are redrawn while the final answer is degenerate (outside [0.01, 0.99] or within 0.01 of 1/2), with the
    SPEC minimum still guaranteed; conditionals need P(given) > 1e-3;
  * martingale marginals at steps 0..T-2 only (the unstepped final marginal is step T-1).
"""
from __future__ import annotations

import numpy as np

from bookiebench.sims import world as W
from bookiebench.sims.common import pick

PICK_V2 = dict(conj2=W.CONJ2_V2, conj3=W.CONJ3_V2, neg1=W.NEG1_V2, neg2=W.NEG2_V2, cond=W.COND_V2)

_R = "If we pick one of the {ents} so far uniformly at random, "
_D = "For a {ent} drawn uniformly at random from the {ents} so far, "
STREAM_V2 = dict(
    conj2=[_R + "is it true both that {a} and that {b}?", _D + "do both of the following hold: {a}; {b}?"],
    conj3=[_R + "is it true that {a}, that {b}, and that {c}?", _D + "do all of the following hold: {a}; {b}; {c}?"],
    neg1=[_R + "is it false that {a}?", _D + "is it not the case that {a}?"],
    neg2=[_R + "is it false that both {a} and {b}?", _D + "is it not the case that both {a} and {b}?"],
    cond=["Pick one of the {ents} so far uniformly at random among those for which {g}. Is it true that {a}?",
          "Restrict to the {ents} so far for which {g} and pick one of them uniformly at random. "
          "Does it hold that {a}?"],
)


def degenerate(p: float) -> bool:
    return p < W.DEGEN_EPS or p > 1 - W.DEGEN_EPS or abs(p - 0.5) < W.UNIFORM_EPS


def make_queries_v2(V, J, rng, T, mart_var, tpl, fmt_kw, context) -> list[dict]:
    nv = len(V)
    J = np.asarray(J, dtype=float)
    J = J / J.sum()
    shape = J.shape
    qs, seen = [], set()

    def key(q):
        return (q["kind"], q.get("neg", False), tuple(sorted((k, tuple(v)) for k, v in q.get("event", {}).items())),
                tuple(sorted((k, tuple(v)) for k, v in q.get("given", {}).items())))

    def add(q):
        if q["kind"] != "marginal":
            kk = key(q)
            if kk in seen:
                return False
            seen.add(kk)
        q["id"] = f"q{len(qs)}"
        qs.append(q)
        return True

    def ev(vi, multi=True):
        n = len(V[vi].options)
        if multi and n >= 3 and rng.random() < 0.25:
            return sorted(int(i) for i in rng.choice(n, size=2, replace=False))
        return [int(rng.integers(n))]

    def mask(e):
        return W._event_mask(shape, e)

    def F(s, **kw):
        return s.format(**fmt_kw, **kw)

    def fine(q, p, strict):
        if q is None or W.names_absent(q["text"], context):
            return False
        return not (strict and degenerate(p))

    def fill(n_want, n_min, draw):
        made = 0
        for attempt in range(120):
            if made >= n_want or (attempt >= 80 and made >= n_min):
                break
            q, p = draw()
            if not fine(q, p, attempt < 80):
                continue
            made += add(q)
        assert made >= n_min

    def draw_conj():
        k = 3 if (nv >= 3 and rng.random() < 0.3) else 2
        vids = sorted(int(i) for i in rng.choice(nv, size=k, replace=False))
        e = {vi: ev(vi) for vi in vids}
        cl = [V[vi].says(idx) for vi, idx in W.either_last([(vi, e[vi]) for vi in vids])]
        text = F(pick(rng, tpl["conj2"]), a=cl[0], b=cl[1]) if k == 2 else F(pick(rng, tpl["conj3"]), a=cl[0], b=cl[1], c=cl[2])
        p = float(J[mask(e)].sum())
        return {"kind": "noul", "event": {V[vi].name: e[vi] for vi in vids}, "text": text}, p

    def draw_neg():
        if rng.random() < 0.6:
            vi = int(rng.integers(nv))
            e = {vi: ev(vi)}
            for _ in range(4):  # prefer a positively phrased clause (no "= no" under a negation)
                if not W._NEGWORD.search(V[vi].says(e[vi])):
                    break
                e = {vi: ev(vi)}
            text = F(pick(rng, tpl["neg1"]), a=V[vi].says(e[vi]))
        else:
            vids = sorted(int(i) for i in rng.choice(nv, size=2, replace=False))
            e = {vi: ev(vi, False) for vi in vids}
            for _ in range(4):
                if not any(W._NEGWORD.search(V[vi].says(e[vi])) for vi in vids):
                    break
                e = {vi: ev(vi, False) for vi in vids}
            text = F(pick(rng, tpl["neg2"]), a=V[vids[0]].says(e[vids[0]]), b=V[vids[1]].says(e[vids[1]]))
        p = 1 - float(J[mask(e)].sum())
        return {"kind": "noul", "event": {V[vi].name: e[vi] for vi in e}, "neg": True, "text": text}, p

    def draw_cond():
        a, g = (int(i) for i in rng.choice(nv, size=2, replace=False))
        e = ev(a, False)
        mg = J.sum(axis=tuple(i for i in range(nv) if i != g))
        ok = [i for i in range(len(V[g].options)) if mg[i] > W.COND_GUARD_V2]
        if not ok:
            return None, 0.5
        gi = [int(pick(rng, ok))]
        gm = mask({g: gi})
        p = float(J[gm & mask({a: e})].sum() / J[gm].sum())
        text = F(pick(rng, tpl["cond"]), g=V[g].says(gi), a=V[a].says(e))
        return {"kind": "cond", "event": {V[a].name: e}, "given": {V[g].name: gi}, "text": text}, p

    for v in V:
        add({"kind": "marginal", "var": v.name, "text": pick(rng, W.visible_questions(v, context)),
             "options": list(v.options)})
    fill(int(rng.integers(2, 4)), 2, draw_conj)
    fill(int(rng.integers(1, 3)), 1, draw_neg)
    fill(int(rng.integers(1, 3)), 1, draw_cond)
    mv = next(v for v in V if v.name == mart_var)
    for k in range(T - 1):
        add({"kind": "marginal", "var": mv.name, "step": k, "text": pick(rng, W.visible_questions(mv, context)),
             "options": list(mv.options)})
    return qs
