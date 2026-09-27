"""Birthday / occupancy problems: n individuals independently fall into one of d categories (uniform or stated
weights). Evidence: individual categories revealed, pairwise same/different comparisons, and membership checks.
Latent = the full assignment (d^n rows, exact enumeration). Evidence is deterministic given the assignment, hence
exchangeable. Variables: whether any two coincide / how many distinct categories occur, one individual's category,
whether two named individuals coincide.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps,
                    people, per_step, pick, round_pcts)

NAME = "birthday"
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _theme(rng, d_hint):
    t = pick(rng, ["month", "weekday", "bucket", "rune", "locker"])
    # (key, categories, attribute, singular verb phrase, group noun, plural "same" predicate, its negation,
    #  question about two people "{x}", questions about one person "{p}")
    if t == "month":
        return dict(t=t, cats=MONTHS[:], attr="birth month", verb="was born in {c}", gw="people",
                    pos="were born in the same month", neg="were not born in the same month",
                    q2="Were {x} born in the same month?", q1=["In which month was {p} born?", "What is {p}'s birth month?"])
    if t == "weekday":
        return dict(t=t, cats=DAYS[:], attr="birth weekday", verb="was born on a {c}", gw="people",
                    pos="were born on the same day of the week", neg="were not born on the same day of the week",
                    q2="Were {x} born on the same day of the week?",
                    q1=["On which day of the week was {p} born?", "What is {p}'s birth weekday?"])
    if t == "bucket":
        return dict(t=t, cats=[f"bucket {i}" for i in range(d_hint)], attr="hash bucket", verb="hashes to {c}", gw="keys",
                    pos="hash to the same bucket", neg="do not hash to the same bucket",
                    q2="Do {x} hash to the same bucket?", q1=["Which bucket does {p}'s key hash to?", "What is {p}'s hash bucket?"])
    if t == "rune":
        return dict(t=t, cats=[f"the {n} rune" for n in distinct_names(rng, d_hint)], attr="rune", verb="drew {c}",
                    gw="players", pos="drew the same rune", neg="did not draw the same rune",
                    q2="Did {x} draw the same rune?", q1=["Which rune did {p} draw?", "What rune did {p} get?"])
    return dict(t=t, cats=[f"locker {chr(65 + i)}" for i in range(d_hint)], attr="locker", verb="was assigned {c}",
                gw="members", pos="were assigned the same locker", neg="were not assigned the same locker",
                q2="Were {x} assigned the same locker?", q1=["Which locker was {p} assigned?", "What is {p}'s locker?"])


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    if level == 0:
        n, dh = 3, int(rng.integers(4, 7))
    elif level == 1:
        n, dh = 4, int(rng.integers(5, 8))
    else:
        n, dh = int(rng.integers(4, 6)), int(rng.integers(5, 9))
    n += scale - 1
    TH = _theme(rng, dh)
    theme, cats, attr, verb, grp_word = TH["t"], TH["cats"], TH["attr"], TH["verb"], TH["gw"]
    pos, neg = TH["pos"], TH["neg"]
    d = len(cats)
    while d ** n > 40000 * scale ** 2:
        if theme in ("month", "weekday"):
            n -= 1
        else:
            cats = cats[:-1]
            d -= 1
    weighted = rng.random() < 0.85  # stated, varied category weights keep answer vectors from repeating
    w = round_pcts(rng.dirichlet(np.full(d, 2.0)), 100, 1) if weighted else [1] * d
    pw = np.array(w, float) / sum(w)
    names = people(rng, n)
    T = n_steps(rng, level, 3, 5)
    X, Y = 0, 1  # target individual and its partner (after shuffling names these are arbitrary people)
    kinds = []
    for _ in range(T):
        kinds.append(pick(rng, ["reveal", "reveal", "pair", "member"] if level >= 1 else ["reveal", "pair"]))
    for _try in range(50):  # deterministic facts are never repeated
        ev = []
        free = [int(i) for i in rng.permutation(np.arange(2, n))]
        for kd in kinds:
            if kd == "reveal" and not free:
                kd = "pair"
            if kd == "reveal":
                ev.append(("reveal", free.pop(), None))
            elif kd == "pair":
                i, j = (int(a) for a in rng.choice(n, size=2, replace=False))
                if {i, j} == {X, Y}:
                    j = (max(i, j) + 1) % n if n > 2 else j
                    i, j = min(i, j), max(i, j)
                ev.append(("pair", min(i, j), max(i, j)))
            else:
                S = sorted(int(c) for c in rng.choice(d, size=int(rng.integers(2, max(3, d // 2 + 1))), replace=False))
                ev.append(("member", int(rng.integers(n)), tuple(S)))
        if len(set(ev)) == len(ev):
            break

    rows = np.array(list(itertools.product(range(d), repeat=n)))
    prior = np.prod(pw[rows], 1)
    ndist = np.array([len(set(r)) for r in rows])
    vs = {0: ["shared", "cat_x"], 1: ["shared", "cat_x", "same_xy"], 2: ["ndist", "cat_x", "same_xy"]}[level]
    lo_d = 1
    cols = []
    for v in vs:
        if v == "shared":
            cols.append(np.where(ndist < n, 0, 1))
        elif v == "ndist":
            cols.append(ndist - lo_d)
        elif v == "cat_x":
            cols.append(rows[:, X])
        else:
            cols.append(np.where(rows[:, X] == rows[:, Y], 0, 1))
    proj = np.stack(cols, 1)
    gw = grp_word
    nx, ny = names[X], names[Y]
    V = {
        "shared": Var("any_shared", ["yes", "no"],
                      [f"Do at least two of the {n} {gw} share a {attr}?", TH["q2"].format(x="any two of them"),
                       f"Is there a coincidence among the {n} {gw}, with two or more sharing a {attr}?"],
                      [f"at least two of the {gw} {pos}", f"no two of the {gw} {pos}"]),
        "ndist": Var("n_distinct", [f"{k} distinct" for k in range(1, n + 1)],
                     [f"How many different {attr}s occur among the {n} {gw}?",
                      f"How many distinct {attr}s appear across the group?"],
                     [f"exactly {k} different {attr}{'s occur' if k > 1 else ' occurs'} among the {gw}" for k in range(1, n + 1)]),
        "cat_x": Var(f"{attr.replace(' ', '_')}_of_{nx}", list(cats),
                     [q.format(p=nx) for q in TH["q1"]] + [f"Which {attr} does {nx} turn out to have?"],
                     [f"{nx} {verb.format(c=c)}" for c in cats]),
        "same_xy": Var("same_xy", ["yes", "no"],
                       [TH["q2"].format(x=f"{nx} and {ny}"), f"Do {nx} and {ny} share a {attr}?"],
                       [f"{nx} and {ny} {pos}", f"{nx} and {ny} {neg}"]),
    }
    variables = [V[v] for v in vs]

    js = is_json(rng)
    dist = (join_list([f"{c} with probability {fmt.p(x)}" for c, x in zip(cats, w)]) if weighted
            else f"chosen uniformly at random among the {d} possibilities ({join_list(cats)})")
    if js:
        prelude = json_block(rng, {"individuals": names, "attribute": attr, "categories": cats,
                                   "distribution": ({c: fmt.p(x) for c, x in zip(cats, w)} if weighted else "uniform"),
                                   "independence": "each individual independently of the others"})
        tr = [lambda i, c: json_line({"who": names[i], attr: cats[c]}),
              lambda i, c: json_line({"revealed": {names[i]: cats[c]}})]
        tp = [lambda i, j, s: json_line({"compare": [names[i], names[j]], "same": bool(s)}),
              lambda i, j, s: json_line({"pair": f"{names[i]}/{names[j]}", "match": "yes" if s else "no"})]
        tm = [lambda i, S, s: json_line({"who": names[i], "in": [cats[c] for c in S], "answer": "yes" if s else "no"}),
              lambda i, S, s: json_line({"check": names[i], "one_of": [cats[c] for c in S], "result": bool(s)})]
    else:
        game = cats[0].split()[1] if theme == "rune" else ""
        story = {
            "month": f"{join_list(names)} meet at a party.",
            "weekday": f"{join_list(names)} are in the same seminar group.",
            "bucket": f"A hash table receives {n} keys, owned by clients {join_list(names)}.",
            "rune": f"In the game of {game}-and-stones, {join_list(names)} each draw a rune from a bag that is refilled after every draw.",
            "locker": f"A gym assigns lockers to new members {join_list(names)}.",
        }[theme]
        prelude = pick(rng, [
            f"{story} For each of them, the {attr} is {dist}, independently of the others.",
            f"{story} Model: every one of the {n} {gw} independently gets a {attr}, {dist}.",
            f"{cap(gw)}: {join_list(names)}. {story.split('.')[0]}. Each {attr} is drawn independently, {dist}.",
        ])
        tr = [lambda i, c: f"{names[i]} {verb.format(c=c_(c))}.", lambda i, c: f"It turns out {names[i]} {verb.format(c=c_(c))}.",
              lambda i, c: f"We learn that {names[i]} {verb.format(c=c_(c))}."]
        tp = [lambda i, j, s: f"{names[i]} and {names[j]} compare: they {pos if s else neg}.",
              lambda i, j, s: f"{names[i]} and {names[j]} {pos if s else neg}.",
              lambda i, j, s: f"Asked together, {names[i]} and {names[j]} say they {pos if s else neg}."]
        tm = [lambda i, S, s: f"{names[i]} says the {attr} is {'' if s else 'not '}one of {join_list([cats[c] for c in S], 'or')}.",
              lambda i, S, s: f"Is {names[i]}'s {attr} one of {join_list([cats[c] for c in S], 'or')}? {'Yes' if s else 'No'}.",
              lambda i, S, s: f"A hint about {names[i]}: {'' if s else 'not '}in {{{', '.join(cats[c] for c in S)}}}."]

    def c_(c):
        return cats[c]

    ptr, ptp, ptm = per_step(rng, T, tr), per_step(rng, T, tp), per_step(rng, T, tm)

    def alternatives(k, past):
        kd = ev[k][0]
        return [(k, c) for c in range(d)] if kd == "reveal" else [(k, 1), (k, 0)]

    def lik(o, past):
        k, v = o
        kd, i, j = ev[k]
        if kd == "reveal":
            return (rows[:, i] == v).astype(float)
        if kd == "pair":
            return ((rows[:, i] == rows[:, j]) == bool(v)).astype(float)
        return (np.isin(rows[:, i], j) == bool(v)).astype(float)

    def render(k, o):
        _, v = o
        kd, i, j = ev[k]
        if kd == "reveal":
            return ptr[k](i, v)
        if kd == "pair":
            return ptp[k](i, j, v)
        return ptm[k](i, j, v)

    keep = prior > 0
    rows, prior, proj = rows[keep], prior[keep], proj[keep]
    return mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, variables[0].name, True,
                    "json" if js else "prose", theme,
                    {"n": n, "pw": pw, "ev": ev, "vs": vs, "X": X, "Y": Y})


def simulate(world, rng):
    p = world.params
    n, pw = p["n"], p["pw"]
    r = rng.choice(len(pw), size=n, p=pw)
    a = []
    for v in p["vs"]:
        if v == "shared":
            a.append(0 if len(set(r)) < n else 1)
        elif v == "ndist":
            a.append(len(set(r)) - 1)
        elif v == "cat_x":
            a.append(int(r[p["X"]]))
        else:
            a.append(0 if r[p["X"]] == r[p["Y"]] else 1)
    obs = []
    for k, (kd, i, j) in enumerate(p["ev"]):
        if kd == "reveal":
            obs.append((k, int(r[i])))
        elif kd == "pair":
            obs.append((k, int(r[i] == r[j])))
        else:
            obs.append((k, int(r[i] in j)))
    return tuple(a), obs
