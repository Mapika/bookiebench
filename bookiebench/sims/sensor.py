"""Noisy-sensor hidden Markov model: a hidden state evolves hourly, a noisy sensor reads it each hour.

Latent space = the full state sequence (enumerated). Variables: state at the first reading, at the last reading
and one hour after the last reading. Evidence is ordered in time, so it is NOT exchangeable (no perms).
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, Fmt, cap, join_list, json_block, json_line, person, pick, pct_row, rand_pct
from .world import Var, World

THEMES = [
    dict(key="machine", subject="the {name} press", states=["running normally", "worn", "faulty"],
         reads=["low", "medium", "high"], sensor="a vibration sensor", read_word="vibration",
         names=["Kessler", "Atlas", "Borealis", "Mark IV", "Tandem", "Hydra"]),
    dict(key="robot", subject="the delivery robot {name}", states=["in the kitchen", "in the hallway", "in the lab"],
         reads=["kitchen", "hallway", "lab"], sensor="a Wi-Fi beacon that reports the robot's nearest room",
         read_word="beacon", names=["Pip", "Rover-7", "Bolt", "Scout", "Tink", "Orbit"]),
    dict(key="patient", subject="{name}'s heart rhythm", states=["normal", "elevated", "irregular"],
         reads=["green", "amber", "red"], sensor="a wrist monitor that shows a coloured light",
         read_word="monitor light", names=None),
]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    name = person(rng) if th["names"] is None else pick(rng, th["names"])
    subj = th["subject"].format(name=name)
    K = int(rng.integers(2, 4))
    states = th["states"][:K]
    R = K if th["key"] == "robot" else int(rng.integers(2, 4))
    reads = th["reads"][:R] if R == 3 or th["key"] == "robot" else [th["reads"][0], th["reads"][-1]]
    T = int(rng.integers(3, 6))
    h0 = int(rng.integers(7, 15))
    times = [f"{h0 + t}:00" for t in range(T + 1)]

    init = pct_row(rng, K, lo=5, step=5, conc=2.0)
    trans = []
    for i in range(K):
        stay = rand_pct(rng, 50, 90, 5)
        rest = pct_row(rng, K - 1, lo=0, step=5, total=100 - stay, conc=1.0) if K > 1 else []
        row = list(rest)
        row.insert(i, stay)
        trans.append(row)
    emis = []
    for i in range(K):
        if R == K:
            hit = rand_pct(rng, 55, 90, 5)
            rest = pct_row(rng, R - 1, lo=0, step=5, total=100 - hit)
            row = list(rest)
            row.insert(i, hit)
        else:
            row = pct_row(rng, R, lo=5, step=5, conc=1.5)
            # make higher states produce higher readings on average
        emis.append(row)
    if R != K:
        emis = sorted(emis, key=lambda r: sum(j * x for j, x in enumerate(r)))
    A = np.array(trans, float) / 100
    E = np.array(emis, float) / 100
    p0 = np.array(init, float) / 100

    Z = np.array(list(itertools.product(range(K), repeat=T + 1)))
    prior = p0[Z[:, 0]].copy()
    for t in range(T):
        prior *= A[Z[:, t], Z[:, t + 1]]
    proj = Z[:, [0, T - 1, T]]

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {
            "subject": subj, "states": states, "sensor": th["sensor"], "readings": reads,
            f"P(state at {times[0]})": {s: fmt.p(x) for s, x in zip(states, init)},
            "hourly transition P(next | current)": {s: {s2: fmt.p(x) for s2, x in zip(states, row)}
                                                    for s, row in zip(states, trans)},
            "P(reading | state)": {s: {r: fmt.p(x) for r, x in zip(reads, row)} for s, row in zip(states, emis)},
            "readings_at": times[:T],
            **({"model": "Markov chain over hours; each reading depends only on the current state"}
               if get_tv() >= 2 else {}),
        })
        render = lambda k, o: json_line({"time": times[o[0]], th["read_word"]: reads[o[1]]})  # noqa: E731
    else:
        s_init = f"At {times[0]}, {subj} is " + join_list(
            [f"{s} with probability {fmt.p(x)}" for s, x in zip(states, init)], "or") + "."
        s_tr = " ".join(
            f"If it is {s} in one hour, then in the next hour it is " + join_list(
                [f"{s2} with probability {fmt.p(x)}" for s2, x in zip(states, row) if x > 0], "or") + "."
            for s, row in zip(states, trans))
        s_em = " ".join(
            f"When it is {s}, the {th['read_word']} reading is " + join_list(
                [f"{r} with probability {fmt.p(x)}" for r, x in zip(reads, row) if x > 0], "or") + "."
            for s, row in zip(states, emis))
        prelude = pick(rng, [
            f"{cap(subj)} is tracked with {th['sensor']}, read once per hour. {s_init} {s_tr} {s_em} Readings depend "
            f"only on the current state.",
            f"We monitor {subj} using {th['sensor']}. The hidden state changes from hour to hour as a Markov chain. "
            f"{s_init} {s_tr} {s_em}",
            f"{s_init} {s_tr} Each hour, {th['sensor']} gives a noisy reading. {s_em} The readings follow.",
        ])
        if get_tv() >= 2:  # state every model assumption in every template
            prelude += (" The state in each hour depends only on the state in the hour before (a Markov chain), and "
                        "each reading depends only on the state at the time of that reading.")
        ev = [lambda t, r: f"At {t}, the {th['read_word']} reading is {r}.",
              lambda t, r: f"{t} reading: {r}.",
              lambda t, r: f"The sensor shows {r} at {t}."]
        pos = [pick(rng, ev) for _ in range(T)]
        render = lambda k, o: pos[o[0]](times[o[0]], reads[o[1]])  # noqa: E731

    variables = [
        Var("state_last", states, [f"What is the state of {subj} at {times[T - 1]}?",
                                   f"At {times[T - 1]}, which state is {subj} in?"],
            f"at {times[T - 1]} {subj} is {{opt}}"),
        Var("state_first", states, [f"What was the state of {subj} at {times[0]}?",
                                    f"At {times[0]}, which state was {subj} in?"],
            f"at {times[0]} {subj} was {{opt}}"),
        Var("state_next", states, [f"What will the state of {subj} be at {times[T]}?",
                                   f"At {times[T]}, one hour after the last reading, which state will {subj} be in?"],
            f"at {times[T]} {subj} will be {{opt}}"),
    ]
    if get_tv() >= 2:  # step-k martingale questions: the last reading time is still in the future before step T-1
        tl = times[T - 1]
        variables[0].questions_at = lambda k: (
            [f"What will the state of {subj} be at {tl}?", f"At {tl}, which state will {subj} be in?"]
            if k < T - 1 else variables[0].questions)
    proj = proj[:, [1, 0, 2]]
    L = [E[Z[:, t]] for t in range(T)]
    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: [(k, r) for r in range(R)],
                 lik=lambda o, past: L[o[0]][:, o[1]],
                 render=render, prelude=prelude, mart_var="state_last", exchangeable=False,
                 meta={"style": "json" if json_style else "prose", "theme": th["key"]})
