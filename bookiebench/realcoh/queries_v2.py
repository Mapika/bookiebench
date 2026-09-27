"""Linked-question sets for realcoh v2 (BookieBench v2, pack `realcoh`).

Every query is a price on an event over ONE outcome space: the product of the state's variables' options (SPEC §1).
All relations are therefore visible to the Dutch-book LP (`bookiebench.metrics.dutch`) with the existing query semantics
(`marginal`, `noul` with a conjunctive `event` and optional `neg`, `cond` with `given`); nothing new is needed in metrics:

    rel            kind      encoding                                                    linked with
    marginal       marginal  var                                                         everything
    coarse         noul      {fine: [options mapping to c]}                              fine marginal (sum)
    cum            noul      {ordinal var: [0..k]}                                       cum k-1 <= cum k (monotone)
    conj           noul      {va: [a], vb: [b]}                                          marginals, cond, disj
    conj3          noul      three variables                                             conj
    neg            noul      {v: [a]}, neg                                               marginal (complement)
    neg_conj       noul      {va: [a], vb: [b]}, neg                                     conj (complement)
    disj           noul      A or B = not(not A and not B) = {va: comp(A), vb: comp(B)}, neg   marginals + conj
    either         noul      {v: [a, a']}  (two options of one variable)                 marginal (sum)
    cond           cond      event A, given B                                            conj / P(B)  (Bayes)
    cond_rev       cond      event B, given A  (same pair as `cond`)                     conj / P(A)  (Bayes)
    cond_neg       cond      event A, neg, given B                                       1 - cond
    fc_event       noul      {v: [yes]} for a binary variable (forecast family)          marginal of v

Query ids are q0, q1, ...; each query carries "rel" and, for pairs, "link" (queries with the same link share the pair
(A, B)). `paraphrases[qid]` has >= 2 alternative wordings for every marginal and >= 1 for every other query.

A variable spec (see sources_v2.V2) is {"name", "options", "clauses", "questions", "cum"?, "coarse"?}:
    clauses   one declarative clause per option ("the clause limits the company's liability")
    questions >= 3 phrasings of the marginal question (first = base query, the rest = paraphrases)
    cum       ordinal variables only: K-1 clauses, cum[k] <=> option index <= k ("the fix takes under an hour")
    coarse    {"name", "options", "clauses", "map"}: a coarse variable that is a deterministic function of this one;
              map[i] = coarse option of fine option i. It is NOT a separate variable of the outcome space.
"""
from __future__ import annotations

import random

T_DEFAULT = {
    "yes": ["Is it true that {a}?", "Would you say that {a}?", "Is it the case that {a}?", "Do you think that {a}?"],
    "neg": ["Is it false that {a}?", "Would it be wrong to say that {a}?", "Is it not the case that {a}?"],
    "conj": ["Is it true both that {a} and that {b}?", "Would you say that {a}, and also that {b}?",
             "Are both of these correct: {a}; {b}?", "Is it the case that {a} and {b}?"],
    "conj3": ["Are all three of these correct: {a}; {b}; {c}?", "Is it the case that {a}, that {b}, and that {c}?",
              "Would you say that {a}, that {b} and also that {c}?"],
    "neg_conj": ["Is it false that {a} and {b} at the same time?", "Is it not the case that both {a} and {b}?",
                 "Is at least one of these false: {a}; {b}?"],
    "disj": ["Is at least one of these true: {a}; {b}?", "Is it the case that {a}, or that {b}, or both?",
             "Would you say that {a} or that {b} (or both)?"],
    "either": ["Is it the case that either {a} or {b}?", "Is one of these two true: {a}; {b}?",
               "Would you say that {a}, or else that {b}?"],
    "cond": ["Given that {g}, is it the case that {e}?", "Suppose {g}. In that case, would you say {e}?",
             "Assuming {g}, is it true that {e}?", "If {g}, would you conclude that {e}?"],
    "cond_neg": ["Given that {g}, is it false that {e}?", "Suppose {g}. Would it then be wrong to say that {e}?",
                 "Assuming {g}, is it not the case that {e}?"],
}

T_FORECAST = {
    "yes": ["Will it turn out that {a}?", "Is it going to be the case that {a}?", "Will it be true that {a}?"],
    "neg": ["Will it turn out to be false that {a}?", "Is it going to be the case that it is NOT true that {a}?",
            "Will the prediction that {a} turn out wrong?"],
    "conj": ["Will both of these happen: {a}; {b}?", "Will it turn out that {a}, and also that {b}?",
             "Is it going to be the case both that {a} and that {b}?"],
    "conj3": ["Will all three of these happen: {a}; {b}; {c}?", "Will it turn out that {a}, that {b} and that {c}?"],
    "neg_conj": ["Will it turn out false that both {a} and {b}?", "Will at least one of these fail to happen: {a}; {b}?"],
    "disj": ["Will at least one of these happen: {a}; {b}?", "Will it turn out that {a}, or that {b}, or both?",
             "Is it going to be the case that {a} or that {b} (or both)?"],
    "either": ["Will it turn out that either {a} or {b}?", "Will one of these two happen: {a}; {b}?"],
    "cond": ["If it turns out that {g}, will it also turn out that {e}?", "Suppose {g}. Will it then be the case that {e}?",
             "Assuming {g}, will it also be true that {e}?"],
    "cond_neg": ["If it turns out that {g}, will it be false that {e}?", "Suppose {g}. Will it then NOT be the case that {e}?"],
}


def _two(rng, pool):
    """Two different templates: (base, paraphrase)."""
    a, b = rng.sample(pool, 2)
    return a, b


def atoms(variables):
    """All yes/no atoms: (kind, var index, allowed option list, clause). kind in {opt, coarse, cum}."""
    out = []
    for vi, v in enumerate(variables):
        for k, c in enumerate(v["clauses"]):
            out.append(("opt", vi, [k], c))
        if v.get("coarse"):
            cz = v["coarse"]
            for c, cl in enumerate(cz["clauses"]):
                out.append(("coarse", vi, [i for i, m in enumerate(cz["map"]) if m == c], cl))
        for k, cl in enumerate(v.get("cum") or []):
            out.append(("cum", vi, list(range(k + 1)), cl))
    return out


def _comp(v, allowed):
    return [i for i in range(len(v["options"])) if i not in allowed]


def make_queries_v2(rng: random.Random, variables: list[dict], templates: dict | None = None, forecast: bool = False):
    T = templates or T_DEFAULT
    qs, para = [], {}
    names = [v["name"] for v in variables]

    def add(q, alt=None):
        q = dict(id=f"q{len(qs)}", **q)
        qs.append(q)
        if alt:
            para[q["id"]] = list(alt) if isinstance(alt, (list, tuple)) else [alt]
        return q["id"]

    def noul(rel, event, pool, fmt, neg=False, link=None):
        t0, t1 = _two(rng, T[pool])
        q = {"kind": "noul", "rel": rel, "event": event, "text": t0.format(**fmt)}
        if neg:
            q["neg"] = True
        if link:
            q["link"] = link
        return add(q, t1.format(**fmt))

    # 1. marginals, >= 2 paraphrases each
    for v in variables:
        assert len(v["questions"]) >= 3, v["name"]
        add({"kind": "marginal", "rel": "marginal", "var": v["name"], "text": v["questions"][0],
             "options": list(v["options"])}, v["questions"][1:])

    A = atoms(variables)
    opt_atoms = [a for a in A if a[0] == "opt"]
    rich_atoms = [a for a in A if a[0] != "opt"]

    # 2. option granularity: one yes/no per coarse option (sum over the fine options that map to it)
    for vi, v in enumerate(variables):
        if v.get("coarse"):
            for kind, vj, allowed, cl in A:
                if kind == "coarse" and vj == vi:
                    noul("coarse", {v["name"]: allowed}, "yes", {"a": cl})
    # 3. temporal / monotone: every cumulative threshold of every ordinal variable
    for vi, v in enumerate(variables):
        for k, cl in enumerate(v.get("cum") or []):
            noul("cum", {v["name"]: list(range(k + 1))}, "yes", {"a": cl})
    # forecast family: every binary event also asked as a plain yes/no question (vs its marginal)
    if forecast:
        for v in variables:
            if len(v["options"]) == 2 and not v.get("cum"):
                noul("fc_event", {v["name"]: [0]}, "yes", {"a": v["clauses"][0]})

    seen = set()

    def pick_pair(pool_a, pool_b):
        for _ in range(200):
            a = rng.choice(pool_a); b = rng.choice(pool_b)
            if a[1] == b[1]:
                continue
            key = tuple(sorted([(a[1], tuple(a[2])), (b[1], tuple(b[2]))]))
            if key in seen:
                continue
            seen.add(key)
            return a, b
        raise RuntimeError("could not pick a fresh pair")

    # 4. Bayes-linked pairs: A and B, A&B, P(A|B), P(B|A), P(not A|B)
    pools = [(opt_atoms, opt_atoms), (rich_atoms or opt_atoms, opt_atoms)]
    for li, (pa, pb) in enumerate(pools):
        a, b = pick_pair(pa, pb)
        link = f"b{li}"
        ev = {names[a[1]]: a[2], names[b[1]]: b[2]}
        noul("conj", ev, "conj", {"a": a[3], "b": b[3]}, link=link)
        for rel, e, g, pool in (("cond", a, b, "cond"), ("cond_rev", b, a, "cond"), ("cond_neg", a, b, "cond_neg")):
            if rel == "cond_neg" and li > 0:
                continue
            t0, t1 = _two(rng, T[pool])
            fmt = {"g": g[3], "e": e[3]}
            q = {"kind": "cond", "rel": rel, "event": {names[e[1]]: e[2]}, "given": {names[g[1]]: g[2]},
                 "text": t0.format(**fmt), "link": link}
            if rel == "cond_neg":
                q["neg"] = True
            add(q, t1.format(**fmt))
    # 5. one more conjunction, and a triple conjunction when there are >= 3 variables
    a, b = pick_pair(A, opt_atoms)
    noul("conj", {names[a[1]]: a[2], names[b[1]]: b[2]}, "conj", {"a": a[3], "b": b[3]})
    if len(variables) >= 3:
        vs = rng.sample(range(len(variables)), 3)
        tri = [rng.choice([x for x in A if x[1] == vi and (x[0] == "opt" or rng.random() < 0.3)]) for vi in vs]
        noul("conj3", {names[x[1]]: x[2] for x in tri}, "conj3", {"a": tri[0][3], "b": tri[1][3], "c": tri[2][3]})
    # 6. negations: one option atom, one coarse/cum atom when available
    a = rng.choice(opt_atoms)
    noul("neg", {names[a[1]]: a[2]}, "neg", {"a": a[3]}, neg=True)
    if rich_atoms:
        a = rng.choice(rich_atoms)
        noul("neg", {names[a[1]]: a[2]}, "neg", {"a": a[3]}, neg=True)
    # 7. negated conjunction
    a, b = pick_pair(opt_atoms, opt_atoms)
    noul("neg_conj", {names[a[1]]: a[2], names[b[1]]: b[2]}, "neg_conj", {"a": a[3], "b": b[3]}, neg=True)
    # 8. disjunctions across variables: A or B = not(comp A and comp B)
    for pa in (opt_atoms, A):
        a, b = pick_pair(pa, opt_atoms)
        va, vb = variables[a[1]], variables[b[1]]
        noul("disj", {va["name"]: _comp(va, a[2]), vb["name"]: _comp(vb, b[2])}, "disj", {"a": a[3], "b": b[3]}, neg=True)
    # 9. within-variable disjunction of two options
    multi = [v for v in variables if len(v["options"]) >= 3]
    if multi:
        v = rng.choice(multi)
        i, j = sorted(rng.sample(range(len(v["options"])), 2))
        noul("either", {v["name"]: [i, j]}, "either", {"a": v["clauses"][i], "b": v["clauses"][j]})
    return qs, para
