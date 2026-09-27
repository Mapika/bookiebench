"""Gaussian measurement with a discretised latent: a quantity takes one of a stated grid of integer values; each
reading equals the true value plus an integer error whose distribution is a stated discretised normal table, plus a
fixed offset if the (old) instrument is miscalibrated. (L2) A second, independent instrument with its own error table.

Latent = (true value, calibration state, next reading of the old instrument). Readings are conditionally independent
given the latent, so exchangeable.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from ._base import (Fmt, Var, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps,
                    per_step, pick, round_pcts)

NAME = "gauge"
THEMES = [("the mass of the {X} meteorite fragment", "grams", "g", "balance"),
          ("the length of the {X} bone", "millimetres", "mm", "calliper"),
          ("the depth of {X} Well", "metres", "m", "sounding line"),
          ("the temperature of the {X} vat", "degrees", "deg", "probe"),
          ("the voltage of the {X} cell", "decivolts", "dV", "meter")]


def dnorm_table(sd, K):
    w = [math.exp(-0.5 * (e / sd) ** 2) for e in range(-K, K + 1)]
    return round_pcts(w, 100, 1)


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    qty, unit, u, inst = pick(rng, THEMES)
    qty = qty.replace("{X}", distinct_names(rng, 1)[0])
    G = {0: 3, 1: 5, 2: 7}[level] + 2 * (scale - 1)
    base = int(rng.integers(10, 90))
    grid = list(range(base, base + G))
    if rng.random() < 0.5:
        pw = dnorm_table(float(rng.uniform(0.8, 2.5)), G // 2)
    else:
        pw = round_pcts(rng.dirichlet(np.full(G, 2.0)), 100, 2)
    pw = pw[:G]
    sdA = round(float(rng.uniform(0.5, 1.8)), 1)
    KA = 1 if sdA < 0.8 else 2
    eA = dnorm_table(sdA, KA)
    biased = level >= 1
    off = int(pick(rng, [1, -1, 2])) if biased else 0
    pbias = int(rng.integers(5, 61)) if biased else 0
    twoinst = level == 2 and rng.random() < 0.6
    sdB = round(float(rng.uniform(1.2, 2.4)), 1)
    KB = 3 if twoinst else 0
    eB = dnorm_table(sdB, KB) if twoinst else [100]
    T = n_steps(rng, level, 3, 5)
    who = [("A" if (not twoinst or rng.random() < 0.5) else "B") for _ in range(T)]
    offs = [0, off] if biased else [0]
    lo = grid[0] + min(offs) - KA
    hi = grid[-1] + max(offs) + KA
    vals = list(range(lo, hi + 1))
    rows = list(itertools.product(range(G), range(len(offs)), range(len(vals))))
    TH = np.array([r[0] for r in rows])
    BZ = np.array([r[1] for r in rows])
    NXv = np.array([vals[r[2]] for r in rows])
    NXi = np.array([r[2] for r in rows])
    thv = np.array(grid)[TH]
    offz = np.array(offs)[BZ]

    def pA(reading, th, of):
        e = reading - th - of
        return np.where(np.abs(e) <= KA, np.array(eA)[np.clip(e + KA, 0, 2 * KA)], 0) / 100

    def pB(reading, th):
        e = reading - th
        return np.where(np.abs(e) <= KB, np.array(eB)[np.clip(e + KB, 0, 2 * KB)], 0) / 100

    prior = (np.array(pw, float)[TH] / 100) * np.where(BZ == 1, pbias / 100, 1 - pbias / 100) * pA(NXv, thv, offz)
    keep = prior > 0
    vs = ["theta", "bias", "next"] if biased else ["theta", "next"]
    if level == 1 and rng.random() < 0.4:
        vs = ["theta", "bias"]
    used = sorted(set(NXi[keep]))
    nx_map = {v: i for i, v in enumerate(used)}
    cols = {"theta": TH, "bias": BZ, "next": np.array([nx_map.get(v, 0) for v in NXi])}
    proj = np.stack([cols[v] for v in vs], 1)
    iname = f"the old {inst}" if twoinst else f"the {inst}"
    V = {"theta": Var("true_value", [f"{g} {u}" for g in grid],
                      [f"What is {qty}?", f"What is the true value of {qty}?", f"How large is {qty} really?"],
                      [f"{qty} is {g} {u}" for g in grid]),
         "bias": Var("calibration", ["calibrated", "miscalibrated"],
                     [f"Is {iname} calibrated?", f"Is {iname} miscalibrated?"],
                     [f"{iname} is calibrated", f"{iname} is miscalibrated"]),
         "next": Var("next_reading", [f"{vals[i]} {u}" for i in used],
                     [f"What will the next reading on {iname} be?", f"If {iname} is read once more, what will it show?"],
                     [f"the next reading on {iname} is {vals[i]} {u}" for i in used])}
    variables = [V[v] for v in vs]

    def tab(e, K):
        return join_list([f"{'+' if k > 0 else ''}{k} with probability {fmt.p(p)}" for k, p in zip(range(-K, K + 1), e)])

    ptxt = f"Before measuring, {qty} is one of " + join_list([f"{g} {u} ({fmt.p(p)})" for g, p in zip(grid, pw)]) + "."
    atxt = (f"Each reading on {iname} equals the true value plus an independent error, in {unit}, of {tab(eA, KA)} "
            f"(a discretised normal distribution with standard deviation about {sdA:g})")
    btxt = (f" {cap(iname)} is miscalibrated with probability {fmt.p(pbias)}; if so, every one of its readings is "
            f"shifted by {'+' if off > 0 else ''}{off} {u} on top of the error." if biased else "")
    ctxt = (f" A second, new {inst} is always calibrated; its error, in {unit}, is independently {tab(eB, KB)}."
            if twoinst else "")
    js = is_json(rng)
    if js:
        st = {"quantity": qty, "unit": unit, "prior": {f"{g}": fmt.p(p) for g, p in zip(grid, pw)},
              (f"old_{inst}_error" if twoinst else f"{inst}_error"): {f"{k:+d}" if k else "0": fmt.p(p) for k, p in zip(range(-KA, KA + 1), eA)}}
        if biased:
            st["miscalibration"] = {"instrument": iname, "probability": fmt.p(pbias),
                                    "shift": f"{off:+d} {u} on every reading of {iname}"}
        if twoinst:
            st[f"new_{inst}_error"] = {f"{k:+d}" if k else "0": fmt.p(p) for k, p in zip(range(-KB, KB + 1), eB)}
        st["errors"] = "independent across readings" + (f"; the new {inst} is always calibrated" if twoinst else "")
        prelude = json_block(rng, st)
        tpl = [lambda w, r: json_line({"instrument": "old" if (w == "A" and twoinst) else ("new" if w == "B" else inst), "reading": r}),
               lambda w, r: json_line({"reading": f"{r} {u}", "on": ("new " if w == "B" else ("old " if twoinst else "")) + inst})]
    else:
        prelude = pick(rng, [f"{ptxt} {atxt}.{btxt}{ctxt}", f"A lab measures {qty}. {atxt}.{btxt}{ctxt} {ptxt}",
                             f"{ptxt}{btxt} {atxt}.{ctxt}"])
        nm = lambda w: ("the new " + inst) if w == "B" else iname  # noqa: E731
        tpl = [lambda w, r: f"{cap(nm(w))} reads {r} {u}.", lambda w, r: f"Reading from {nm(w)}: {r} {u}.",
               lambda w, r: f"A measurement with {nm(w)} gives {r} {u}."]
    ptpl = per_step(rng, T, tpl)
    loB, hiB = grid[0] - KB, grid[-1] + KB

    def alternatives(k, past):
        return [(k, r) for r in (vals if who[k] == "A" else range(loB, hiB + 1))]

    def lik(o, past):
        k, r = o
        return (pA(r, thv, offz) if who[k] == "A" else pB(r, thv))[keep]

    return mk_world(variables, prior[keep], proj[keep], T, alternatives, lik, lambda k, o: ptpl[k](who[k], o[1]),
                    prelude, "true_value", True, "json" if js else "prose", unit,
                    {"grid": grid, "pw": pw, "eA": eA, "KA": KA, "eB": eB, "KB": KB, "off": off, "pbias": pbias,
                     "biased": biased, "who": who, "vs": vs, "nx_map": nx_map, "lo": lo})


def simulate(world, rng):
    p = world.params
    pw = np.array(p["pw"], float)
    th = int(rng.choice(len(pw), p=pw / pw.sum()))
    b = int(p["biased"] and rng.random() < p["pbias"] / 100)
    tv = p["grid"][th]

    def read(w):
        if w == "A":
            e = int(rng.choice(np.arange(-p["KA"], p["KA"] + 1), p=np.array(p["eA"]) / 100))
            return tv + e + (p["off"] if b else 0)
        e = int(rng.choice(np.arange(-p["KB"], p["KB"] + 1), p=np.array(p["eB"]) / 100))
        return tv + e

    obs = [(k, read(w)) for k, w in enumerate(p["who"])]
    nx = read("A")
    vals = {"theta": th, "bias": b, "next": p["nx_map"][nx - p["lo"]]}
    return tuple(vals[v] for v in p["vs"]), obs
