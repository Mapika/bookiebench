"""Bayesian search: an object lies in one of N cells (stated prior); searching a cell detects it with a cell-specific
probability that drops under a latent poor-visibility condition; (L2) searches of empty cells can raise false contacts.

Latent = (location, visibility, outcome of the next planned search). Search reports are conditionally independent
given the latent, so exchangeable. Variables: location, visibility, next planned search outcome.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, repeat_tags, tag_text, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps,
                    per_step, pick, round_pcts)

NAME = "search"
THEMES = [
    dict(key="hiker", obj="the missing hiker", cell="sector", team="The mountain rescue team", vis=("clear", "foggy"),
         find="spots signs of the hiker", nothing="finds nothing"),
    dict(key="wreck", obj="the sunken probe", cell="grid square", team="The sonar boat", vis=("calm", "rough"),
         find="gets a sonar contact", nothing="gets no contact"),
    dict(key="keys", obj="the lost keys", pl=True, cell="room", team="Grandpa", vis=("tidy", "cluttered"),
         find="finds something that looks like the keys", nothing="finds nothing"),
    dict(key="drone", obj="the crashed drone", cell="field", team="The survey crew", vis=("dry", "waterlogged"),
         find="picks up the drone's beacon", nothing="hears nothing"),
]


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    be, it = ("are", "them") if th.get("pl") else ("is", "it")
    N = {0: int(rng.integers(2, 4)), 1: int(rng.integers(3, 5)), 2: int(rng.integers(4, 7))}[level] + (scale - 1)
    names = [f"{th['cell']} {x}" for x in distinct_names(rng, N)] if rng.random() < 0.5 else \
        [f"{th['cell']} {i + 1}" for i in range(N)]
    w = round_pcts(rng.dirichlet(np.full(N, 1.5)), 100, 5)
    two_vis = level >= 1
    pvis = int(rng.integers(5, 71)) if two_vis else 0
    d_good = [int(rng.integers(35, 100)) for _ in range(N)]
    d_bad = [int(rng.integers(5, max(6, g - 9))) for g in d_good]
    fa = int(rng.integers(1, 21)) if level == 2 else int(rng.integers(1, 13))  # false contacts keep "found" from being decisive
    T = n_steps(rng, level, 3, 6)
    plan = [int(rng.integers(N)) for _ in range(T)]
    nxt = int(rng.integers(N))
    vg, vb = th["vis"]
    rows = list(itertools.product(range(N), range(2 if two_vis else 1), (0, 1)))
    L = np.array([r[0] for r in rows])
    Vz = np.array([r[1] for r in rows])
    O = np.array([r[2] for r in rows])  # next search outcome: 0 = contact, 1 = nothing
    D = np.where(Vz[:, None] == 1, np.array(d_bad)[None, :], np.array(d_good)[None, :]) / 100  # rows x cells

    def p_contact(cell):
        return np.where(L == cell, D[:, cell], fa / 100)

    pc_next = p_contact(nxt)
    prior = (np.array(w, float)[L] / 100) * np.where(Vz == 1, pvis / 100, 1 - pvis / 100) * \
        np.where(O == 0, pc_next, 1 - pc_next)
    keep = prior > 0
    vs = ["loc", "vis", "next"] if two_vis else ["loc", "next"]
    if level == 1 and rng.random() < 0.3:
        vs = ["loc", "vis"]
    cols = {"loc": L, "vis": Vz, "next": O}
    proj = np.stack([cols[v] for v in vs], 1)
    V = {"loc": Var("location", list(names), [f"Where {be} {th['obj']}?", f"In which {th['cell']} {be} {th['obj']}?",
                                               f"Which {th['cell']} {'hold' if th.get('pl') else 'holds'} {th['obj']}?"],
                    [f"{th['obj']} {be} in {nm}" for nm in names]),
         "vis": Var("conditions", [vg, vb], ["What are the search conditions?", f"Are conditions {vg} or {vb}?"],
                    [f"conditions are {vg}", f"conditions are {vb}"]),
         "next": Var("next_search", ["contact", "nothing"],
                     [f"What will the next planned search, of {names[nxt]}, report?",
                      f"Will the upcoming search of {names[nxt]} report a contact?"],
                     [f"the next search of {names[nxt]} reports a contact", f"the next search of {names[nxt]} reports nothing"])}
    variables = [V[v] for v in vs]

    ptxt = "Prior to searching: " + "; ".join(f"{nm}: {fmt.p(x)}" for nm, x in zip(names, w)) + "."
    if two_vis:
        dtxt = (f"Conditions are {vb} with probability {fmt.p(pvis)} (otherwise {vg}) and stay that way all day. "
                f"If {th['obj']} {be} in the searched {th['cell']}, a search detects {it} with probability " +
                join_list([f"{fmt.p(g)} ({vg}) or {fmt.p(b)} ({vb}) in {nm}" for nm, g, b in zip(names, d_good, d_bad)]) + ".")
    else:
        dtxt = (f"If {th['obj']} {be} in the searched {th['cell']}, a search detects {it} with probability " +
                join_list([f"{fmt.p(g)} in {nm}" for nm, g in zip(names, d_good)]) + ".")
    ftxt = (f" A search of a {th['cell']} that does not hold {it} still raises a false contact with probability "
            f"{fmt.p(fa)}." if fa else f" A search never reports a contact in the wrong {th['cell']}.")
    itxt = (" Searches are independent given the location and conditions (repeat searches of the same "
            f"{th['cell']} are labelled sweep A, sweep B, ...), and nothing moves.")
    ntxt = f" After the searches below, the plan is to search {names[nxt]}." if "next" in vs else ""
    js = is_json(rng)
    if js:
        st = {"object": th["obj"], "prior": {nm: fmt.p(x) for nm, x in zip(names, w)},
              "detection_if_present": ({nm: {vg: fmt.p(g), vb: fmt.p(b)} for nm, g, b in zip(names, d_good, d_bad)}
                                       if two_vis else {nm: fmt.p(g) for nm, g in zip(names, d_good)}),
              "false_contact_if_absent": fmt.p(fa), "independence": "searches independent given location and conditions"}
        if two_vis:
            st[f"P({vb})"] = fmt.p(pvis)
            st["conditions"] = f"either {vg} or {vb}, the same for every search"
        st["object_moves"] = "no"
        if "next" in vs:
            st["next_planned_search"] = names[nxt]
        prelude = json_block(rng, st)
        tpl = [lambda c, r: json_line({"searched": names[c], "result": "contact" if r == 0 else "nothing"}),
               lambda c, r: json_line({"search": {"where": names[c], "contact": r == 0}})]
    else:
        prelude = pick(rng, [f"{th['team']} is looking for {th['obj']}. {ptxt} {dtxt}{ftxt}{itxt}{ntxt}",
                             f"{ptxt} {cap(th['obj'])} {be} being searched for. {dtxt}{ftxt}{itxt}{ntxt}",
                             f"Search log for {th['obj']}. {dtxt}{ftxt} {ptxt}{itxt}{ntxt}"])
        tpl = [lambda c, r: f"{th['team']} searches {names[c]} and {th['find'] if r == 0 else th['nothing']}.",
               lambda c, r: f"Search of {names[c]}: {'contact' if r == 0 else 'nothing'}.",
               lambda c, r: f"{cap(names[c])} is searched; result: {'a contact' if r == 0 else 'nothing found'}."]
    ptpl = per_step(rng, T, tpl)
    stag = [f"sweep {t}" if t else "" for t in repeat_tags(plan)]

    def lik(o, past):
        k, r = o
        pc = p_contact(plan[k])[keep]
        return pc if r == 0 else 1 - pc

    return mk_world(variables, prior[keep], proj[keep], T, lambda k, past: [(k, 0), (k, 1)], lik,
                    lambda k, o: tag_text(ptpl[k](plan[k], o[1]), stag[k]), prelude, "location", True, "json" if js else "prose",
                    th["key"], {"w": w, "pvis": pvis, "dg": d_good, "db": d_bad, "fa": fa, "plan": plan, "nxt": nxt,
                                "two": two_vis, "vs": vs})


def simulate(world, rng):
    p = world.params
    w = np.array(p["w"], float)
    loc = int(rng.choice(len(w), p=w / w.sum()))
    vis = int(p["two"] and rng.random() < p["pvis"] / 100)
    d = (p["db"] if vis else p["dg"])

    def srch(c):
        pc = d[c] / 100 if c == loc else p["fa"] / 100
        return 0 if rng.random() < pc else 1

    obs = [(k, srch(c)) for k, c in enumerate(p["plan"])]
    vals = {"loc": loc, "vis": vis, "next": srch(p["nxt"])}
    return tuple(vals[v] for v in p["vs"]), obs
