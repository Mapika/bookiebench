"""Weather Markov chain with a hidden regime (e.g. a wet vs dry pattern), forecasting.

Days are observed exactly; the regime selects the transition matrix. Latent space = (regime, full day sequence
including the two forecast days). Evidence is ordered in time (NOT exchangeable).
Variables: regime, the weather on the day after the last observation, and the day after that.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, TOWNS, Fmt, cap, join_list, json_block, json_line, pick, pct_row, rand_pct
from .world import Var, World

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
THEMES = [
    dict(key="weather", states=["sunny", "cloudy", "rainy"], what="the weather in {town}",
         regimes=[["a dry spell", "a wet spell"], ["a high-pressure pattern", "a low-pressure pattern"],
                  ["the Aster pattern", "the Boreal pattern"]],
         obs="On {day} it was {s} in {town}.", obs2="{day}: {s}.", q_day="What will the weather be in {town} on {day}?",
         clause="it will be {opt} in {town} on {day}"),
    dict(key="traffic", states=["light", "moderate", "heavy"], what="the morning traffic on the {town} ring road",
         regimes=[["a normal period", "a roadworks period"], ["the school-term pattern", "the holiday pattern"]],
         obs="On {day} the morning traffic was {s}.", obs2="{day} traffic: {s}.",
         q_day="How heavy will the morning traffic be on {day}?", clause="the morning traffic on {day} will be {opt}"),
]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    town = pick(rng, TOWNS)
    S = th["states"]
    regimes = pick(rng, th["regimes"])
    if get_tv() >= 2:  # F10: neutral regime labels (real words like "a dry spell" can contradict the stated numbers)
        regimes = ["pattern A", "pattern B"] if th["key"] == "weather" else ["period type A", "period type B"]
    T = int(rng.integers(3, 5))
    d0 = int(rng.integers(0, 7))
    days = [DAYS[(d0 + i) % 7] for i in range(T + 2)]
    pr = rand_pct(rng, 20, 80, 5)  # P(regime 0)
    init = [pct_row(rng, 3, lo=5, step=5, conc=2.0) for _ in range(2)]
    trans = []
    for r in range(2):
        rows = []
        for i in range(3):
            stay = rand_pct(rng, 30, 80, 5)
            rest = pct_row(rng, 2, lo=5, step=5, total=100 - stay, conc=1.0)
            row = list(rest)
            row.insert(i, stay)
            rows.append(row)
        trans.append(rows)
    A = np.array(trans, float) / 100
    I0 = np.array(init, float) / 100

    Z = np.array([(r,) + seq for r in range(2) for seq in itertools.product(range(3), repeat=T + 2)])
    rr = Z[:, 0]
    prior = np.where(rr == 0, pr, 100 - pr) / 100 * I0[rr, Z[:, 1]]
    for t in range(T + 1):
        prior = prior * A[rr, Z[:, 1 + t], Z[:, 2 + t]]
    proj = Z[:, [0, T + 1, T + 2]]
    what = th["what"].format(town=town)

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {
            "subject": what, "states": S, "hidden_regime": {regimes[0]: fmt.p(pr), regimes[1]: fmt.p(100 - pr)},
            f"P({days[0]} | regime)": {regimes[r]: {s: fmt.p(x) for s, x in zip(S, init[r])} for r in range(2)},
            "P(tomorrow | today, regime)": {regimes[r]: {s: {s2: fmt.p(x) for s2, x in zip(S, row)}
                                                         for s, row in zip(S, trans[r])} for r in range(2)},
            "observed_days": days[:T],
        })
        render = lambda k, o: json_line({"day": days[o[0]], "observed": S[o[1]]})  # noqa: E731
    else:
        s_reg = (f"{'This week' if get_tv() < 2 else 'Throughout these days'} {what} follows one of two hidden patterns: {regimes[0]} (probability {fmt.p(pr)}) or "
                 f"{regimes[1]} (probability {fmt.p(100 - pr)}).")
        parts = []
        for r in range(2):
            p_init = join_list([f"{s} with probability {fmt.p(x)}" for s, x in zip(S, init[r])], "or")
            p_tr = " ".join(
                f"After a {s} day, the next day is " + join_list(
                    [f"{s2} with probability {fmt.p(x)}" for s2, x in zip(S, row) if x > 0], "or") + "."
                for s, row in zip(S, trans[r]))
            parts.append(f"Under {regimes[r]}, {days[0]} is {p_init}. {p_tr}")
        s_tail = "Each day depends only on the previous day and the pattern."
        prelude = pick(rng, [
            f"{s_reg} {' '.join(parts)} {s_tail}",
            f"A forecaster models {what} as a Markov chain whose transition probabilities depend on a hidden pattern. "
            f"{s_reg} {' '.join(parts)}",
            f"{' '.join(parts)} {s_reg} {s_tail} Observations so far are listed below.",
        ])
        ev = [th["obs"], th["obs2"]]
        pos = [pick(rng, ev) for _ in range(T)]
        render = lambda k, o: pos[o[0]].format(day=days[o[0]], s=S[o[1]], town=town)  # noqa: E731

    variables = [
        Var("pattern", list(regimes), ["Which pattern is in effect this week?" if get_tv() < 2 else "Which pattern is in effect over these days?",
             "Which hidden pattern is it?"],
            "{opt} is in effect"),
        Var("day_next", list(S), [th["q_day"].format(town=town, day=days[T])],
            th["clause"].format(town=town, day=days[T], opt="{opt}")),
        Var("day_after", list(S), [th["q_day"].format(town=town, day=days[T + 1])],
            th["clause"].format(town=town, day=days[T + 1], opt="{opt}")),
    ]
    cols = [Z[:, 1 + k] for k in range(T)]
    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: [(k, s) for s in range(3)],
                 lik=lambda o, past: (cols[o[0]] == o[1]).astype(float),
                 render=render, prelude=prelude, mart_var="pattern", exchangeable=False,
                 meta={"style": "json" if json_style else "prose", "theme": th["key"]},
                 params={"p_regime0": pr / 100, "init": I0, "trans": A})
