"""Poisson arrivals under a latent traffic regime, with counts truncated at a stated cap ("c or more").

Latent = (regime, count in the next interval, count in the interval after). Counts are i.i.d. Poisson(lambda_regime)
given the regime; P(k) = e^-l l^k / k!, and the top category takes the tail mass. Evidence = observed interval counts,
exchangeable. Variables: regime, next interval's count, whether the interval after next overflows a stated capacity.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._base import Fmt, art, Var, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps, per_step, pick

NAME = "queue"
THEMES = [("customers", "customer", "the {X} bakery counter", "ten-minute window"),
          ("calls", "call", "the {X} help line", "five-minute slot"),
          ("trucks", "truck", "the {X} loading dock", "hour"),
          ("patients", "patient", "the {X} walk-in clinic", "half hour"),
          ("requests", "request", "the {X} API gateway", "ten-millisecond window")]


def pois(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


def trunc_pmf(lam, c):
    p = [pois(k, lam) for k in range(c)]
    return p + [max(0.0, 1 - sum(p))]


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    items, item, place, win = pick(rng, THEMES)
    place = place.replace("{X}", distinct_names(rng, 1)[0])
    R = {0: 2, 1: 2, 2: 3}[level]
    rnames = {2: ["quiet", "busy"], 3: ["quiet", "normal", "rush"]}[R]
    lams = sorted(round(float(x), 1) for x in rng.choice(np.arange(5, 91) / 10, size=R, replace=False))
    if level == 0:
        rp = [int(rng.integers(10, 91))]
        rp = [100 - rp[0], rp[0]]
    else:
        rp = [int(x) for x in rng.choice(np.arange(10, 90, 5), size=1)]
        rp = [100 - rp[0], rp[0]] if R == 2 else None
        if R == 3:
            a, b = int(rng.choice(np.arange(10, 50, 5))), int(rng.choice(np.arange(10, 40, 5)))
            rp = [a, 100 - a - b, b]
    c = int(min(9, max(3, math.ceil(lams[-1]) + 1))) + (scale - 1)
    capacity = int(rng.integers(2, c))
    T = n_steps(rng, level, 3, 6)
    pm = np.array([trunc_pmf(l, c) for l in lams])  # R x (c+1)
    after = level >= 1
    rows = list(itertools.product(range(R), range(c + 1), range(2) if after else [0]))
    RZ = np.array([r[0] for r in rows])
    NZ = np.array([r[1] for r in rows])
    AZ = np.array([r[2] for r in rows])
    p_over = pm[:, capacity + 1:].sum(1)
    prior = (np.array(rp, float)[RZ] / 100) * pm[RZ, NZ] * (np.where(AZ == 0, p_over[RZ], 1 - p_over[RZ]) if after else 1)
    keep = prior > 1e-300
    vs = ["regime", "next"] + (["over"] if after else [])
    if level == 2 and rng.random() < 0.3:
        vs = ["regime", "over", "next"]
    cols = {"regime": RZ, "next": NZ, "over": AZ}
    proj = np.stack([cols[v] for v in vs], 1)
    kl = [str(k) for k in range(c)] + [f"{c} or more"]
    V = {"regime": Var("regime", list(rnames), [f"What kind of day is it at {place}?", "Which traffic regime is in force today?",
                                                f"Is today {join_list(rnames, 'or')}?"],
                       [f"today is a {r} day" for r in rnames]),
         "next": Var("next_count", kl, [f"How many {items} will arrive in the next {win}?",
                                        f"What will the count be in the coming {win}?"],
                     [f"the next {win} brings {k} {items if k != '1' else item}" for k in kl]),
         "over": Var("overflow", ["yes", "no"], [f"Will the {win} after next bring more than {capacity} {items}?",
                                                 f"Is the {win} after the next one going to exceed capacity?"],
                     [f"the {win} after next brings more than {capacity} {items}",
                      f"the {win} after next brings at most {capacity} {items}"])}
    variables = [V[v] for v in vs]
    lt = join_list([f"{l:g} on a {r} day" for l, r in zip(lams, rnames)])
    rtxt = join_list([f"{r} with probability {fmt.p(p)}" for r, p in zip(rnames, rp)])
    base = (f"At {place}, the number of {items} arriving in each {win} is Poisson distributed, independently "
            f"across {win}s, with mean {lt} (so P(k) = e^-m m^k / k! for mean m).")
    dtxt = f"Today is {rtxt}, and the regime lasts all day."
    ttxt = f"The counter only records up to {c}: any {win} with {c} or more {items} is logged as '{c}+'."
    otxt = f" The capacity is {capacity} {items} per {win}." if after else ""
    js = is_json(rng)
    if js:
        st = {"place": place, "unit": win, "arrivals": "Poisson, independent across windows",
              "mean_per_window": {r: l for r, l in zip(rnames, lams)}, "regime_prior": {r: fmt.p(p) for r, p in zip(rnames, rp)},
              "logging": f"counts of {c} or more are recorded as '{c}+'", "regime": "fixed for the whole day",
              "pmf": "P(k) = e^-m m^k / k! for mean m"}
        if after:
            st["capacity"] = capacity
        prelude = json_block(rng, st)
        tpl = [lambda n: json_line({"window_count": f"{c}+" if n == c else n}),
               lambda n: json_line({"event": "count", items: f"{c}+" if n == c else n})]
    else:
        prelude = pick(rng, [f"{base} {dtxt} {ttxt}{otxt}", f"{dtxt} {base} {ttxt}{otxt}",
                             f"Arrival log for {place}. {base} {ttxt} {dtxt}{otxt}"])
        tpl = [lambda n: f"{cap(art(win))} logs {f'{c}+' if n == c else n} {items if n != 1 else item}.",
               lambda n: f"{cap(art(win))} is logged: {f'{c}+' if n == c else n}.",
               lambda n: f"The counter shows {f'{c}+' if n == c else n} for {art(win)}."]
    ptpl = per_step(rng, T, tpl)
    L = pm[RZ[keep]]

    return mk_world(variables, prior[keep], proj[keep], T, lambda k, past: list(range(c + 1)),
                    lambda o, past: L[:, o], lambda k, o: ptpl[k](o), prelude, "regime", True,
                    "json" if js else "prose", items, {"lams": lams, "rp": rp, "c": c, "capacity": capacity, "vs": vs})


def simulate(world, rng):
    p = world.params
    r = int(rng.choice(len(p["rp"]), p=np.array(p["rp"]) / 100))
    lam, c = p["lams"][r], p["c"]
    counts = [min(int(x), c) for x in rng.poisson(lam, size=world.T + 2)]
    vals = {"regime": r, "next": counts[world.T], "over": 0 if counts[world.T + 1] > p["capacity"] else 1}
    return tuple(vals[v] for v in p["vs"]), counts[:world.T]
