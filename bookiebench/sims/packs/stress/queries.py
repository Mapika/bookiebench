"""Query-stress transforms: negation-heavy query sets, disjunctions, nested conditionals.

Each keeps the original queries and appends new ones. Every new query is an ordinary SPEC query (a conjunction of
per-variable option sets, optionally negated as a whole; conditionals condition on a conjunction), so its exact answer
comes from the joint via `exact_answer`. The logical form behind each text is recorded in `meta.stress.forms`.
Conditionals never use `neg` (a negated consequent is expressed by the complement option set instead).
"""
from __future__ import annotations

import re

from bookiebench.sims.common import pick

from .common import Clauses, add_query, positive_prob_options, start, transform_rng


def _key(q):
    return (q["kind"], bool(q.get("neg")), tuple(sorted((k, tuple(v)) for k, v in q.get("event", {}).items())),
            tuple(sorted((k, tuple(v)) for k, v in (q.get("given") or {}).items())))


class _Builder:
    def __init__(self, inst, name, seed):
        self.src = inst
        self.rng = transform_rng(inst, name, seed)
        self.out = start(inst, name)
        self.cl = Clauses(inst)
        self.V = inst["variables"]
        self.n = {v["name"]: len(v["options"]) for v in self.V}
        self.seen = {_key(q) for q in inst["queries"] if q["kind"] != "marginal"}
        self.forms = {}

    def var(self, exclude=()):
        c = [v["name"] for v in self.V if v["name"] not in exclude]
        return pick(self.rng, c) if c else None

    def opt(self, var, allowed=None):
        allowed = allowed if allowed is not None else list(range(self.n[var]))
        return int(pick(self.rng, allowed))

    def comp(self, var, idxs):
        s = set(idxs)
        return [i for i in range(self.n[var]) if i not in s]

    def ordered(self, ev):
        names = [v["name"] for v in self.V]
        return {k: sorted(ev[k]) for k in sorted(ev, key=names.index)}

    def add(self, q, form):
        q = dict(q)
        q["event"] = self.ordered(q["event"])
        if "given" in q:
            q["given"] = self.ordered(q["given"])
        k = _key(q)
        if k in self.seen:
            return None
        try:
            q = add_query(self.out, q)
        except ValueError:
            return None
        self.seen.add(k)
        self.forms[q["id"]] = form
        return q

    def finish(self, **info):
        self.out["meta"]["stress"].update({"forms": self.forms, "native_clauses": self.cl.native, **info})
        return self.out


# ----------------------------------------------------------------------------------------------------------------------

DNEG = ["Is it false that it is not the case that {a}?", "Is it not true that it is false that {a}?",
        "Would it be wrong to say that it is not the case that {a}?", "Is it false that it is untrue that {a}?",
        "Is it untrue that it is not the case that {a}?"]
TNEG = ["Is it false that it is not untrue that {a}?",
        "Is it not the case that it is false that it is untrue that {a}?",
        "Would it be wrong to deny that it is not the case that {a}?"]
FALSE_COMP = ["Is it false that {na}?", "Is it not the case that {na}?", "Is it untrue that {na}?"]
NEITHER = ["Is it false that neither {a} nor {b}?", "Is it not the case that neither {a} nor {b}?"]
NEITHER_SAFE = ["Is it false that both of the following fail: {a}; {b}?",
                "Is it not the case that each of these is false: {a}; {b}?"]
NOR_SAFE = ["How likely is it that both of the following are false: {a}; {b}?"]


def _has_nor(*cl):
    return any(re.search(r"\b(neither|nor)\b", c) for c in cl)


NOT_FALSE_AND = ["Is it not the case that it is false that {a} and {b}?",
                 "Is it false that it is not true that both {a} and {b}?"]
FALSE_A_NOTB = ["Is it false that {a} and it is not the case that {b}?",
                "Is it not true that {a} while it is false that {b}?"]
NOT_NOT_COND = ["If it is not the case that {g}, is it false that {a}?",
                "Suppose it is untrue that {g}. How likely is it then that it is not the case that {a}?"]


def negation(inst, seed: int = 0, n_add: int = 8):
    b = _Builder(inst, "negation", seed)
    cl = b.cl
    multi = len(b.V) >= 2
    kinds = ["dneg", "tneg", "false_comp"] + (["neither", "not_false_and", "false_a_notb", "not_not_cond"] if multi
                                               else [])
    for it in range(80):
        if len(b.forms) >= n_add:
            break
        kind = kinds[it % len(kinds)] if it < len(kinds) else pick(b.rng, kinds)
        A = b.var()
        a = b.opt(A)
        if kind == "dneg":
            b.add({"kind": "noul", "event": {A: [a]}, "text": pick(b.rng, DNEG).format(a=cl.clause(A, [a]))},
                  f"not(not({A}={a})) = {A}={a}")
        elif kind == "tneg":
            b.add({"kind": "noul", "event": {A: [a]}, "neg": True,
                   "text": pick(b.rng, TNEG).format(a=cl.clause(A, [a]))}, f"not(not(not({A}={a}))) = {A}!={a}")
        elif kind == "false_comp":
            b.add({"kind": "noul", "event": {A: [a]},
                   "text": pick(b.rng, FALSE_COMP).format(na=cl.not_clause(A, [a]))},
                  f"not({A} in comp({a})) = {A}={a}")
        else:
            B = b.var(exclude=(A,))
            bb = b.opt(B)
            ca, cb = cl.clause(A, [a]), cl.clause(B, [bb])
            if kind == "neither":
                tpl = pick(b.rng, NEITHER if not _has_nor(ca, cb) else NEITHER_SAFE)
                b.add({"kind": "noul", "event": {A: b.comp(A, [a]), B: b.comp(B, [bb])}, "neg": True,
                       "text": tpl.format(a=ca, b=cb)},
                      f"not(not {A}={a} and not {B}={bb}) = {A}={a} or {B}={bb}")
            elif kind == "not_false_and":
                b.add({"kind": "noul", "event": {A: [a], B: [bb]}, "text": pick(b.rng, NOT_FALSE_AND).format(a=ca, b=cb)},
                      f"not(not({A}={a} and {B}={bb})) = {A}={a} and {B}={bb}")
            elif kind == "false_a_notb":
                b.add({"kind": "noul", "event": {A: [a], B: b.comp(B, [bb])}, "neg": True,
                       "text": pick(b.rng, FALSE_A_NOTB).format(a=ca, b=cb)},
                      f"not({A}={a} and not {B}={bb})")
            elif kind == "not_not_cond":
                G, A2 = A, B
                ok = [i for i in positive_prob_options(inst, G) if True]
                gcomp_ok = [g for g in range(b.n[G]) if set(b.comp(G, [g])) & set(ok)]
                if not gcomp_ok:
                    continue
                g = int(pick(b.rng, gcomp_ok))
                b.add({"kind": "cond", "event": {A2: b.comp(A2, [bb])}, "given": {G: b.comp(G, [g])},
                       "text": pick(b.rng, NOT_NOT_COND).format(g=cl.clause(G, [g]), a=cl.clause(A2, [bb]))},
                      f"P(not {A2}={bb} | not {G}={g})")
    return b.finish()


# ----------------------------------------------------------------------------------------------------------------------

OR2 = ["Is it the case that {a} or {b}?", "How likely is it that {a} or {b} (or both)?",
       "What is the probability that at least one of these is true: {a}; {b}?",
       "Is it true that {a}, or that {b}, or both?"]
OR3 = ["How likely is it that at least one of the following holds: {a}; {b}; {c}?",
       "What is the probability that {a}, or {b}, or {c} (at least one of them)?"]
NOR = ["Is it the case that neither {a} nor {b}?", "How likely is it that it is neither true that {a} nor that {b}?"]
OR_NOT = ["Is it the case that {a} or it is not the case that {b}?",
          "How likely is it that {a}, or else that it is false that {b}?"]


def disjunction(inst, seed: int = 0, n_add: int = 6):
    b = _Builder(inst, "disjunction", seed)
    cl = b.cl
    if len(b.V) < 2:
        return b.finish()
    kinds = ["or2", "nor", "or_not"] + (["or3"] if len(b.V) >= 3 else [])
    for it in range(80):
        if len(b.forms) >= n_add:
            break
        kind = kinds[it % len(kinds)] if it < len(kinds) else pick(b.rng, kinds)
        A = b.var()
        B = b.var(exclude=(A,))
        a, bb = b.opt(A), b.opt(B)
        ca, cb = cl.clause(A, [a]), cl.clause(B, [bb])
        if kind == "or2":
            b.add({"kind": "noul", "event": {A: b.comp(A, [a]), B: b.comp(B, [bb])}, "neg": True,
                   "text": pick(b.rng, OR2).format(a=ca, b=cb)}, f"{A}={a} or {B}={bb}")
        elif kind == "nor":
            tpl = pick(b.rng, NOR if not _has_nor(ca, cb) else NOR_SAFE)
            b.add({"kind": "noul", "event": {A: b.comp(A, [a]), B: b.comp(B, [bb])},
                   "text": tpl.format(a=ca, b=cb)}, f"not {A}={a} and not {B}={bb}")
        elif kind == "or_not":
            b.add({"kind": "noul", "event": {A: b.comp(A, [a]), B: [bb]}, "neg": True,
                   "text": pick(b.rng, OR_NOT).format(a=ca, b=cb)}, f"{A}={a} or not {B}={bb}")
        elif kind == "or3":
            C = b.var(exclude=(A, B))
            c = b.opt(C)
            b.add({"kind": "noul", "event": {A: b.comp(A, [a]), B: b.comp(B, [bb]), C: b.comp(C, [c])}, "neg": True,
                   "text": pick(b.rng, OR3).format(a=ca, b=cb, c=cl.clause(C, [c]))},
                  f"{A}={a} or {B}={bb} or {C}={c}")
    return b.finish()


# ----------------------------------------------------------------------------------------------------------------------

NEST2 = ["If {g1}, then, if in addition {g2}, how likely is it that {a}?",
         "Suppose {g1}. Suppose further that {g2}. What is the probability, under both suppositions, that {a}?",
         "Assuming {g1}, and assuming also that {g2}, is it the case that {a}?"]
COND_CONJ = ["If {g}, how likely is it that {a} and {b}?", "Assuming {g}, is it true both that {a} and that {b}?"]
COND_NOT = ["If it is not the case that {g}, how likely is it that {a}?",
            "Suppose it is false that {g}. What is the probability then that {a}?"]
COND_NEGC = ["If {g}, how likely is it that it is not the case that {a}?",
             "Assuming {g}, is it false that {a}?"]
COND_EITHER = ["If {g}, how likely is it that {a}?", "Supposing {g}, what is the probability that {a}?"]


def nested(inst, seed: int = 0, n_add: int = 6):
    b = _Builder(inst, "nested", seed)
    cl = b.cl
    if len(b.V) < 2:
        return b.finish()
    kinds = ["cond_not", "cond_negc"] + (["nest2", "cond_conj"] if len(b.V) >= 3 else []) + ["cond_either"]
    for it in range(120):
        if len(b.forms) >= n_add:
            break
        kind = kinds[it % len(kinds)] if it < len(kinds) else pick(b.rng, kinds)
        G = b.var()
        okG = positive_prob_options(inst, G)
        A = b.var(exclude=(G,))
        a = b.opt(A)
        if kind == "nest2":
            G2 = b.var(exclude=(G, A))
            g1, g2 = b.opt(G, okG), b.opt(G2, positive_prob_options(inst, G2))
            b.add({"kind": "cond", "event": {A: [a]}, "given": {G: [g1], G2: [g2]},
                   "text": pick(b.rng, NEST2).format(g1=cl.clause(G, [g1]), g2=cl.clause(G2, [g2]),
                                                     a=cl.clause(A, [a]))},
                  f"P({A}={a} | {G}={g1}, {G2}={g2})")
        elif kind == "cond_conj":
            B = b.var(exclude=(G, A))
            bb = b.opt(B)
            g = b.opt(G, okG)
            b.add({"kind": "cond", "event": {A: [a], B: [bb]}, "given": {G: [g]},
                   "text": pick(b.rng, COND_CONJ).format(g=cl.clause(G, [g]), a=cl.clause(A, [a]),
                                                         b=cl.clause(B, [bb]))},
                  f"P({A}={a} and {B}={bb} | {G}={g})")
        elif kind == "cond_not":
            g = b.opt(G)
            if not set(b.comp(G, [g])) & set(okG):
                continue
            b.add({"kind": "cond", "event": {A: [a]}, "given": {G: b.comp(G, [g])},
                   "text": pick(b.rng, COND_NOT).format(g=cl.clause(G, [g]), a=cl.clause(A, [a]))},
                  f"P({A}={a} | not {G}={g})")
        elif kind == "cond_negc":
            g = b.opt(G, okG)
            b.add({"kind": "cond", "event": {A: b.comp(A, [a])}, "given": {G: [g]},
                   "text": pick(b.rng, COND_NEGC).format(g=cl.clause(G, [g]), a=cl.clause(A, [a]))},
                  f"P(not {A}={a} | {G}={g})")
        elif kind == "cond_either":
            if b.n[G] < 3 or len(okG) < 1:
                continue
            gs = sorted(int(x) for x in b.rng.choice(b.n[G], size=2, replace=False))
            if not set(gs) & set(okG):
                continue
            b.add({"kind": "cond", "event": {A: [a]}, "given": {G: gs},
                   "text": pick(b.rng, COND_EITHER).format(g=cl.clause(G, gs), a=cl.clause(A, [a]))},
                  f"P({A}={a} | {G} in {gs})")
    return b.finish()
