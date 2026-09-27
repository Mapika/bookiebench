"""Inventory with censored demand: a shop restocks to a stated level every morning; daily demand follows a stated
table that depends on a latent demand level; only sales are seen, and a sell-out hides how much demand there was.

Latent = (demand level, tomorrow's demand); on L2 the stock level varies by day. Daily demands are i.i.d. given the latent, so
the evidence (sales per day, with the day's stock stated) is exchangeable. Variables: demand level, tomorrow's
demand, and (L1+) whether tomorrow sells out at tomorrow's stated stock.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, cap, distinct_names, ints_row, is_json, join_list, json_block, json_line, mk_world,
                    n_steps, per_step, pick)

NAME = "inventory"
GOODS = [("sourdough loaves", "loaf"), ("umbrellas", "umbrella"), ("bouquets", "bouquet"), ("phone chargers", "charger"),
         ("jars of honey", "jar"), ("ice-cream tubs", "tub")]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    goods, unit = pick(rng, GOODS)
    shop = f"the {distinct_names(rng, 1)[0]} {pick(rng, ['kiosk', 'corner shop', 'market stall', 'station shop'])}"
    Dmax = 5 + (scale - 1)
    L = 2 if level < 2 else pick(rng, [2, 3])
    lnames = {2: ["slow", "strong"], 3: ["slow", "normal", "strong"]}[L]
    while True:
        tabs = [ints_row(rng, Dmax + 1, 100, 0, conc=1.2) for _ in range(L)]
        means = [sum(k * p for k, p in enumerate(t)) for t in tabs]
        if len(set(means)) == L:
            break
    order = np.argsort(means)
    tabs = [tabs[i] for i in order]
    lp = [50, 50] if level == 0 else None
    if lp is None:
        if L == 2:
            a = int(rng.choice(np.arange(15, 90, 5)))
            lp = [100 - a, a]
        else:
            a, b = int(rng.choice(np.arange(15, 45, 5))), int(rng.choice(np.arange(15, 45, 5)))
            lp = [a, 100 - a - b, b]
    T = n_steps(rng, level, 3, 6)
    vary = level == 2
    stock = [int(rng.integers(2, Dmax)) for _ in range(T + 1)] if vary else [int(rng.integers(2, Dmax))] * (T + 1)
    d0 = int(rng.integers(0, 7 - T))  # all observed days and tomorrow fall in the same Monday-Sunday week
    dnames = [DAYS[(d0 + i) % 7] for i in range(T + 1)]
    pm = np.array(tabs, float) / 100
    rows = list(itertools.product(range(L), range(Dmax + 1)))
    LZ = np.array([r[0] for r in rows])
    DZ = np.array([r[1] for r in rows])
    prior = (np.array(lp, float)[LZ] / 100) * pm[LZ, DZ]
    keep = prior > 0
    s_tom = stock[T]
    vs = ["level", "demand"] + (["sellout"] if level >= 1 else [])
    cols = {"level": LZ, "demand": DZ, "sellout": np.where(DZ >= s_tom, 0, 1)}
    proj = np.stack([cols[v] for v in vs], 1)
    tom = dnames[T]
    V = {"level": Var("demand_level", list(lnames), [f"Is demand at {shop} {join_list(lnames, 'or')} this week?",
                                                     "What is this week's demand level?",
                                                     f"Which demand level is {shop} facing?"],
                      [f"demand is {l} this week" for l in lnames]),
         "demand": Var("tomorrow_demand", [str(k) for k in range(Dmax + 1)],
                       [f"How many {goods} will customers want on {tom}?", f"What will {tom}'s demand be?"],
                       [f"{k} {goods if k != 1 else unit} {'are' if k != 1 else 'is'} wanted on {tom}" for k in range(Dmax + 1)]),
         "sellout": Var("tomorrow_sellout", ["yes", "no"], [f"Will {shop} sell out on {tom}?",
                                                           f"With {s_tom} in stock, will {tom} end in a sell-out?"],
                        [f"{shop} sells out on {tom}", f"{shop} does not sell out on {tom}"])}
    variables = [V[v] for v in vs]
    ttxt = "; ".join(f"in a {l} week: " + ", ".join(f"{k} with {fmt.p(p)}" for k, p in enumerate(t))
                     for l, t in zip(lnames, tabs))
    ltxt = join_list([f"{l} with probability {fmt.p(p)}" for l, p in zip(lnames, lp)])
    stxt = (f"Each morning the shelf is restocked to {stock[0]} {goods}." if not vary else
            "Each morning the shelf is restocked to that day's stock level (given below).")
    js = is_json(rng)
    if js:
        st = {"shop": shop, "goods": goods, "demand_level_prior": {l: fmt.p(p) for l, p in zip(lnames, lp)},
              "daily_demand_pmf": {l: {str(k): fmt.p(p) for k, p in enumerate(t)} for l, t in zip(lnames, tabs)},
              "restock": (stock[0] if not vary else "per day, as stated"),
              "sales": "min(demand, stock); unmet demand is lost and not observed",
              "independence": "daily demands independent given the week's level", "tomorrow": tom}
        if vary:
            st["tomorrow_stock"] = s_tom
        prelude = json_block(rng, st)
        tpl = [lambda d, s, n: json_line({"day": d, "stock": s, "sold": n, "sold_out": n == s}),
               lambda d, s, n: json_line({"date": d, "sales": n, "of": s})]
    else:
        prelude = pick(rng, [
            f"{cap(shop)} sells {goods}. The week's demand level is {ltxt}. Daily demand (number of {goods} customers "
            f"want) is independent from day to day given the level, with probabilities {ttxt}. {stxt} Sales are the "
            f"smaller of demand and stock; when the shelf empties, extra customers leave and are not counted.",
            f"{stxt} At {shop}, demand for {goods} depends on the week's level ({ltxt}). Given the level, each day's "
            f"demand is drawn independently: {ttxt}. Only sales are recorded, so a sell-out hides any extra demand.",
            f"Stock notes for {shop} ({goods}). Demand level: {ltxt}. Demand per day: {ttxt}. {stxt} If demand "
            f"exceeds the stock, the shop simply sells out.",
        ])
        if vary or level >= 1:
            prelude += f" Tomorrow is {tom}, when the stock will be {s_tom}."
        tpl = [lambda d, s, n: f"{d}: stocked {s}, sold {n}{' (sold out)' if n == s else ''}.",
               lambda d, s, n: (f"On {d} the shop sold out all {s}." if n == s else f"On {d}, {n} of the {s} {goods} sold."),
               lambda d, s, n: f"Sales on {d}: {n} out of {s}{', shelf empty by closing' if n == s else ''}."]
    ptpl = per_step(rng, T, tpl)
    PM = pm[LZ[keep]]

    def alternatives(k, past):
        return [(k, n) for n in range(stock[k] + 1)]

    def lik(o, past):
        k, n = o
        return PM[:, n] if n < stock[k] else PM[:, stock[k]:].sum(1)

    return mk_world(variables, prior[keep], proj[keep], T, alternatives, lik,
                    lambda k, o: ptpl[k](dnames[k], stock[k], o[1]), prelude, "demand_level", True,
                    "json" if js else "prose", goods, {"tabs": tabs, "lp": lp, "stock": stock, "vs": vs})


def simulate(world, rng):
    p = world.params
    l = int(rng.choice(len(p["lp"]), p=np.array(p["lp"]) / 100))
    t = np.array(p["tabs"][l], float) / 100
    d = rng.choice(len(t), size=world.T + 1, p=t)
    obs = [(k, int(min(d[k], p["stock"][k]))) for k in range(world.T)]
    vals = {"level": l, "demand": int(d[world.T]), "sellout": 0 if d[world.T] >= p["stock"][world.T] else 1}
    return tuple(vals[v] for v in p["vs"]), obs
