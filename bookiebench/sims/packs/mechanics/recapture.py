"""Capture-recapture: m animals were marked and released; the population size N is one of a stated grid of values.
In a second session animals are caught one at a time without replacement and checked for a mark (hypergeometric).
(L2) Each mark is lost before the second session with a stated probability, so the number of still-visible marks is
latent too.

Latent = (N, visible marks M', whether the next one / two catches are marked). With the future catches treated as
drawn first (exchangeability), lik(marked | N, M', future, past) = (M' - future marked - past marked)/(N - 2 - k).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._base import Fmt, Var, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps, per_step, pick, round_pcts

NAME = "recapture"
ANIMALS = [("newts", "newt"), ("voles", "vole"), ("carp", "carp"), ("lizards", "lizard"), ("moths", "moth"), ("crabs", "crab")]


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    animals, animal = pick(rng, ANIMALS)
    site = f"{distinct_names(rng, 1)[0]} {pick(rng, ['Pond', 'Meadow', 'Marsh', 'Copse', 'Cove'])}"
    G = {0: 3, 1: 5, 2: int(rng.integers(4, 7))}[level] + (scale - 1)
    m = int(rng.integers(3, 26))
    T = n_steps(rng, level, 3, 6)
    step = int(rng.integers(1, 13))
    lo = max(m + T + 3, int(rng.integers(m + T + 3, m + T + 40)))
    lo = int(math.ceil(lo / step) * step)
    grid = [lo + step * i for i in range(G)]
    uniform = rng.random() < 0.2
    pw = [1] * G if uniform else round_pcts(rng.dirichlet(np.full(G, 2.0)), 100, 5)
    tagloss = level == 2 and rng.random() < 0.6
    ell = int(rng.choice(np.arange(5, 35, 5))) if tagloss else 0
    two = level >= 1
    fut = 2 if two else 1
    rows = []
    for gi, N in enumerate(grid):
        for Mv in (range(m + 1) if tagloss else [m]):
            pM = math.comb(m, Mv) * (1 - ell / 100) ** Mv * (ell / 100) ** (m - Mv) if tagloss else 1.0
            for f in itertools.product((0, 1), repeat=fut):  # 1 = marked
                # sequential hypergeometric for the future catches (treated as drawn first)
                p, mk, nn = 1.0, Mv, N
                for x in f:
                    p *= (mk / nn) if x else ((nn - mk) / nn)
                    mk -= x
                    nn -= 1
                pr = pw[gi] / sum(pw) * pM * p
                if pr > 0:
                    rows.append((gi, Mv, f, pr))
    GZ = np.array([r[0] for r in rows])
    MZ = np.array([r[1] for r in rows])
    FZ = np.array([list(r[2]) for r in rows])
    NZ = np.array(grid)[GZ]
    prior = np.array([r[3] for r in rows])
    fsum = FZ.sum(1)
    mgroups = None
    vs = ["N", "next"] + (["second"] if two else [])
    if tagloss:
        vs = ["N", "next", "marks"]
        edges = [0, m // 3, (2 * m) // 3, m + 1] if m >= 6 else [0, m // 2, m + 1]
        edges = sorted(set(edges))
        mgroups = [(edges[i], edges[i + 1] - 1) for i in range(len(edges) - 1)]
    cols = {"N": GZ, "next": 1 - FZ[:, 0], "second": (1 - FZ[:, 1]) if two else None}
    if tagloss:
        cols["marks"] = np.array([next(i for i, (a, b) in enumerate(mgroups) if a <= x <= b) for x in MZ])
    proj = np.stack([cols[v] for v in vs], 1)
    V = {"N": Var("population", [f"{N} {animals}" for N in grid],
                  [f"How many {animals} live at {site}?", f"What is the population size at {site}?",
                   f"How large is the {animal} population?"],
                  [f"the population is {N} {animals}" for N in grid]),
         "next": Var("next_marked", ["marked", "unmarked"], [f"Will the next {animal} caught carry a mark?",
                                                             f"Is the next {animal} caught going to be marked?"],
                     [f"the next {animal} caught is marked", f"the next {animal} caught is unmarked"]),
         "second": Var("second_marked", ["marked", "unmarked"],
                       [f"Will the {animal} caught right after the next one carry a mark?",
                        f"Of the next two {animals} caught, will the second be marked?"],
                       [f"the second of the next two {animals} caught is marked",
                        f"the second of the next two {animals} caught is unmarked"])}
    if tagloss:
        V["marks"] = Var("visible_marks", [(f"{a} or {b} marks" if b == a + 1 else f"{a} to {b} marks") if a != b else f"{a} marks" for a, b in mgroups],
                         [f"How many of the {m} marks are still visible?", "How many marked animals still carry their mark?"],
                         [f"between {a} and {b} of the marks are still visible" if a != b else f"exactly {a} marks are still visible"
                          for a, b in mgroups])
    variables = [V[v] for v in vs]
    ptxt = (f"A priori the population size is equally likely to be any of {join_list([str(N) for N in grid], 'or')}" if uniform else
            "A priori the population is " + join_list([f"{N} with probability {fmt.p(x)}" for N, x in zip(grid, pw)]))
    ltxt = (f" Before the second session, each mark independently rubs off with probability {fmt.p(ell)}, and an animal "
            f"that lost its mark looks unmarked." if tagloss else " Marks are permanent.")
    js = is_json(rng)
    if js:
        st = {"site": site, "species": animals, "marked_and_released": m,
              "population_prior": ({str(N): "uniform" for N in grid} if uniform else {str(N): fmt.p(x) for N, x in zip(grid, pw)}),
              "second_session": "animals caught one at a time without replacement; each equally likely; mark checked",
              "mixing": "marked animals fully mixed back in; population closed"}
        if tagloss:
            st["mark_loss_probability"] = fmt.p(ell)
        prelude = json_block(rng, st)
        tpl = [lambda x: json_line({"catch": animal, "marked": bool(x)}),
               lambda x: json_line({"event": "capture", "mark": "yes" if x else "no"})]
    else:
        prelude = pick(rng, [
            f"Ecologists at {site} caught, marked and released {m} {animals}. The population is closed and fully mixed. "
            f"{ptxt}.{ltxt} In a second session they catch {animals} one at a time, never catching the same one twice, "
            f"each remaining {animal} equally likely to be caught next.",
            f"A mark-recapture survey: {m} {animals} at {site} were marked and let go. {ptxt}.{ltxt} Now {animals} are "
            f"caught one by one without replacement (every uncaught {animal} equally likely) and checked for a mark.",
            f"{ptxt} (for the {animal} population at {site}). Earlier, {m} of them were marked and released.{ltxt} "
            f"In the recapture session animals are caught one at a time, without replacement, uniformly at random.",
        ])
        tpl = [lambda x: f"A {animal} is caught: {'it carries a mark' if x else 'no mark'}.",
               lambda x: f"A catch comes up {'marked' if x else 'unmarked'}.",
               lambda x: f"The team nets {'a marked' if x else 'an unmarked'} {animal}.",
               lambda x: f"Caught {animal}: {'marked' if x else 'unmarked'}."]
    ptpl = per_step(rng, T, tpl)

    def lik(o, past):
        pm = sum(past)
        k = len(past)
        left = np.maximum(MZ - fsum - pm, 0)
        pmk = left / (NZ - fut - k)
        return pmk if o == 1 else np.clip(1 - pmk, 0, 1) * ((NZ - fut - k - left) > 0)

    return mk_world(variables, prior, proj, T, lambda k, past: [1, 0], lik, lambda k, o: ptpl[k](o), prelude,
                    "population", True, "json" if js else "prose", animals,
                    {"grid": grid, "pw": pw, "m": m, "ell": ell, "tagloss": tagloss, "fut": fut, "vs": vs,
                     "mgroups": mgroups})


def simulate(world, rng):
    p = world.params
    pw = np.array(p["pw"], float)
    gi = int(rng.choice(len(pw), p=pw / pw.sum()))
    N, m = p["grid"][gi], p["m"]
    Mv = int(rng.binomial(m, 1 - p["ell"] / 100)) if p["tagloss"] else m
    pop = np.zeros(N, int)
    pop[:Mv] = 1
    rng.shuffle(pop)
    T = world.T
    obs = [int(x) for x in pop[:T]]
    f = [int(x) for x in pop[T:T + p["fut"]]]
    vals = {"N": gi, "next": 1 - f[0], "second": 1 - f[1] if p["fut"] > 1 else 0}
    if p["tagloss"]:
        vals["marks"] = next(i for i, (a, b) in enumerate(p["mgroups"]) if a <= Mv <= b)
    return tuple(vals[v] for v in p["vs"]), obs
