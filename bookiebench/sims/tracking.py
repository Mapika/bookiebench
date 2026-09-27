"""Validation family: multi-sensor localisation of a static target on a line of positions.

Latent: target position (4-7 cells) and target size class (affects detectability). Sensors at stated positions give
detect / no-detect pings whose probability depends on the distance to the target, and range-finders report near /
far with an error rate. Pings are i.i.d. given (position, size) -> exchangeable.
Variables: position, size class, the next ping of a designated station.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import Fmt, cap, join_list, json_block, json_line, pick, rand_pct, sample
from .lexicon import pseudo
from .world import Var, World

DECIMALS = 8
THEMES = [
    dict(key="hiker", target="the lost hiker", line="the ridge trail", pos="marker", big="with a bright jacket",
         small="in dark clothing", det="spots", ping="search drone"),
    dict(key="sub", target="the survey submarine", line="the canyon floor", pos="grid line", big="running its lights",
         small="running silent", det="picks up", ping="sonar buoy"),
    dict(key="rover", target="the stalled rover", line="the crater rim track", pos="waypoint", big="with its beacon on",
         small="with its beacon off", det="registers", ping="relay mast"),
    dict(key="boat", target="the drifting boat", line="the estuary channel", pos="buoy", big="with its sail up",
         small="with its sail down", det="sights", ping="shore camera"),
]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    N = int(rng.integers(4, 8))
    posn = [f"{th['pos']} {i + 1}" for i in range(N)]
    prior_pos = np.array(rng.dirichlet(np.full(N, 2.0)))
    pp = [int(x) for x in np.round(prior_pos * 20)]
    while sum(pp) != 20:
        i = int(rng.integers(N))
        if sum(pp) > 20 and pp[i] > 0:
            pp[i] -= 1
        elif sum(pp) < 20:
            pp[i] += 1
    pp = [x * 5 for x in pp]
    p_big = rand_pct(rng, 30, 80, 5)
    det = sorted([rand_pct(rng, 40, 95, 5) for _ in range(3)], reverse=True)  # distance 0, 1, 2
    far = rand_pct(rng, 1, 10, 1)  # distance >= 3
    halve = pick(rng, [50, 60, 70])  # small target: detection multiplied by halve%
    n_st = int(rng.integers(2, 4))
    st_pos = sorted(int(x) for x in rng.choice(N, size=n_st, replace=False))
    st_names = [f"{pseudo(rng, 2)} {th['ping']}" for _ in range(n_st)]
    has_rf = rng.random() < 0.5
    rf_pos = int(rng.integers(N))
    rf_err = rand_pct(rng, 5, 25, 5)
    rf_name = f"the {pseudo(rng, 2)} range-finder"
    T = int(rng.integers(3, 7))
    srcs = [("st", int(rng.integers(n_st))) for _ in range(T)]
    if has_rf:
        srcs[int(rng.integers(T))] = ("rf", 0)
    nxt = int(rng.integers(n_st))

    Z = np.array(list(itertools.product(range(N), range(2), range(2))))  # pos, size(0=big), next ping(0=detect)
    pos, size, nping = Z[:, 0], Z[:, 1], Z[:, 2]

    def p_det(s):
        d = np.abs(pos - st_pos[s])
        base = np.where(d == 0, det[0], np.where(d == 1, det[1], np.where(d == 2, det[2], far))) / 100
        return np.where(size == 0, base, base * halve / 100)

    pn = p_det(nxt)
    prior = np.array(pp, float)[pos] / 100 * np.where(size == 0, p_big, 100 - p_big) / 100 * np.where(nping == 0, pn, 1 - pn)
    near = np.abs(pos - rf_pos) <= 1
    p_near_report = np.where(near, 1 - rf_err / 100, rf_err / 100)

    def lik(o, past):
        k, v = o
        kind, s = srcs[k]
        p = p_det(s) if kind == "st" else p_near_report
        return p if v == 0 else 1 - p

    big, small = th["big"], th["small"]
    json_style = rng.random() < 0.3
    if json_style:
        st = {"target": th["target"], "positions": posn, "P(position)": {p: fmt.p(x) for p, x in zip(posn, pp)},
              "P(target is " + big + ")": fmt.p(p_big),
              "stations": {n: posn[p] for n, p in zip(st_names, st_pos)},
              "P(detect | distance, " + big + ")": {"0": fmt.p(det[0]), "1": fmt.p(det[1]), "2": fmt.p(det[2]),
                                                   "3+": fmt.p(far)},
              "if " + small: f"multiply detection probabilities by {halve}%",
              "pings": "independent given position and visibility"}
        if has_rf:
            st["range-finder"] = {"name": rf_name, "at": posn[rf_pos], "reports": "near if within 1 position",
                                  "error rate": fmt.p(rf_err)}
        prelude = json_block(rng, st)
    else:
        s_pos = (f"{cap(th['target'])} is stationary at one of {N} positions along {th['line']}, "
                 f"{posn[0]} to {posn[-1]}. The prior chances are " +
                 join_list([f"{fmt.p(x)} for {p}" for p, x in zip(posn, pp)]) + ".")
        s_size = f"There is a {fmt.p(p_big)} chance the target is {big}; otherwise it is {small}."
        s_st = ("Stations: " + join_list([f"{n} at {posn[p]}" for n, p in zip(st_names, st_pos)]) + ".")
        s_det = (f"On each ping, a station {th['det']} a target {big} with probability {fmt.p(det[0])} if it is at the "
                 f"station's own position, {fmt.p(det[1])} one position away, {fmt.p(det[2])} two positions away, and "
                 f"{fmt.p(far)} further away. For a target {small}, each of these probabilities is multiplied by "
                 f"{halve}%.")
        s_rf = (f" {cap(rf_name)} at {posn[rf_pos]} reports 'near' if the target is within one position of it and "
                f"'far' otherwise, but gives the wrong report {fmt.p(rf_err)} of the time." if has_rf else "")
        s_ind = " All pings and reports are independent given the target's position and visibility."
        prelude = pick(rng, [f"{s_pos} {s_size} {s_st} {s_det}{s_rf}{s_ind}",
                             f"{s_st} {s_det} {s_pos} {s_size}{s_rf}{s_ind}",
                             f"A search is under way. {s_pos} {s_st} {s_size} {s_det}{s_rf}{s_ind}"])

    def render(k, o):
        _, v = o
        kind, s = srcs[k]
        if kind == "rf":
            return (json_line({"range_finder": "near" if v == 0 else "far"}) if json_style else
                    f"{cap(rf_name)} reports '{'near' if v == 0 else 'far'}'.")
        if json_style:
            return json_line({"station": st_names[s], "ping": "detect" if v == 0 else "no detect"})
        return (f"A ping from {st_names[s]} {'detects' if v == 0 else 'does not detect'} the target." if v in (0, 1)
                else "")

    variables = [
        Var("position", posn, [f"Where is {th['target']}?", f"At which {th['pos']} is {th['target']}?"],
            f"{th['target']} is at {{opt}}"),
        Var("visibility", [big, small], [f"Is {th['target']} {big} or {small}?"],
            [f"{th['target']} is {big}", f"{th['target']} is {small}"]),
        Var("next_ping", ["detect", "no detect"], [f"Will the next ping from {st_names[nxt]} detect the target?"],
            [f"the next ping from {st_names[nxt]} will detect the target",
             f"the next ping from {st_names[nxt]} will not detect the target"]),
    ]
    return World(variables, prior, Z, T, alternatives=lambda k, past: [(k, 0), (k, 1)], lik=lik, render=render,
                 prelude=prelude, mart_var="position", exchangeable=True,
                 meta={"style": "json" if json_style else "prose", "theme": th["key"]})
