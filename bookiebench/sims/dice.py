"""Biased coin / loaded die identification.

A device is picked at random from a stated collection (prior from counts); tosses/rolls are i.i.d. given the
device (exchangeable). Variables: device type, next outcome, (coins only) the outcome after that.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, Fmt, a_an, cap, join_list, json_block, json_line, person, pick, sample
from .world import Var, World

FACES = ["one", "two", "three", "four", "five", "six"]
CONTAINERS = ["drawer", "pouch", "jar", "box", "cup"]


def _coin_world(rng):
    fmt = Fmt(rng)
    who = person(rng)
    K = int(rng.integers(2, 5))
    heads = sorted(sample(rng, [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100], K))
    if 50 not in heads and rng.random() < 0.6:
        heads[int(rng.integers(K))] = 50
        heads = sorted(set(heads))
        K = len(heads)
    if K < 2:
        heads = [50, 90]
        K = 2

    def lab(h):
        if h == 50:
            return "fair coin"
        if h == 100:
            return "double-headed coin"
        return f"{h}%-heads coin"

    labels = [lab(h) for h in heads]
    counts = [int(rng.integers(1, 6)) for _ in range(K)]
    cont = pick(rng, CONTAINERS)
    prior_d = np.array(counts, float) / sum(counts)
    th = np.array([h / 100 for h in heads])
    T = int(rng.integers(3, 7))
    Z = np.array(list(itertools.product(range(K), range(2), range(2))))
    ph = np.stack([th, 1 - th], 1)  # outcome 0 = heads
    prior = prior_d[Z[:, 0]] * ph[Z[:, 0], Z[:, 1]] * ph[Z[:, 0], Z[:, 2]]
    outs = ["heads", "tails"]

    def kind_desc(h, c):
        noun = "coin" if c == 1 else "coins"
        if h == 50:
            return f"{c} fair {noun}"
        if h == 100:
            return f"{c} double-headed {noun}"
        verb = "lands" if (c == 1 and get_tv() >= 2) else "land"
        return f"{c} {noun} that {verb} heads {fmt.p(h)} of the time"

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {"container": cont, "coins": [{"type": labels[i], "count": counts[i],
                                                               "p_heads": fmt.p(heads[i])} for i in range(K)],
                                   "picked": "one coin uniformly at random, type hidden", "tosser": who})
        ev = [lambda o: json_line({"toss": outs[o]}), lambda o: json_line({"result": outs[o]})]
    else:
        inv = join_list([kind_desc(h, c) for h, c in zip(heads, counts)])
        prelude = pick(rng, [
            f"A {cont} contains {inv}. {who} takes out one coin at random without looking at it and starts tossing it.",
            f"{who} has a {cont} with {inv}. One coin is selected uniformly at random from the {cont} and tossed "
            f"repeatedly.",
            f"In a probability class, a {cont} holds {inv}. A student picks one coin blindly, and {who} records its "
            f"tosses.",
        ])
        ev = [lambda o: f"The coin lands {outs[o]}.", lambda o: f"{who} tosses it: {outs[o]}.",
              lambda o: f"Toss result: {outs[o]}.", lambda o: f"It comes up {outs[o]}."]
    pos = [pick(rng, ev) for _ in range(T)]
    variables = [
        Var("coin", labels, ["Which kind of coin was picked?", f"Which type of coin is {who} tossing?",
                             "What kind of coin is being tossed?"],
            [f"the coin being tossed is {a_an(l)}" for l in labels]),
        Var("next_toss", outs, ["What will the next toss show?", "If the coin is tossed once more, how will it land?"],
            "the next toss will come up {opt}"),
        Var("toss_after_next", outs, ["If the coin is tossed twice more, how will the second of those tosses land?"],
            "the second of the next two tosses will come up {opt}"),
    ]
    L = ph[Z[:, 0]]
    return World(variables, prior, Z, T, lambda k, p: [0, 1], lambda o, p: L[:, o], lambda k, o: pos[k](o),
                 prelude, "coin", True, {"style": "json" if json_style else "prose", "device": "coin"})


def _die_world(rng):
    fmt = Fmt(rng)
    who = person(rng)
    K = int(rng.integers(2, 4))
    faces_loaded = sample(rng, list(range(6)), K)
    kinds = []  # (label, probs in pct over faces, description)
    has_fair = rng.random() < 0.7
    for i in range(K):
        if i == 0 and has_fair:
            kinds.append(("fair die", [None] * 6, "fair"))
            continue
        f = faces_loaded[i]
        p = pick(rng, [25, 30, 35, 40, 45, 50, 55, 60])
        kinds.append((f"die loaded toward {FACES[f]}", (f, p), "loaded"))
    probs = []
    for lab, spec, kind in kinds:
        if kind == "fair":
            probs.append(np.full(6, 1 / 6))
        else:
            f, p = spec
            v = np.full(6, (1 - p / 100) / 5)
            v[f] = p / 100
            probs.append(v)
    probs = np.array(probs)
    labels = [k[0] for k in kinds]
    counts = [int(rng.integers(1, 5)) for _ in range(K)]
    cont = pick(rng, CONTAINERS)
    prior_d = np.array(counts, float) / sum(counts)
    T = int(rng.integers(3, 7))
    Z = np.array(list(itertools.product(range(K), range(6))))
    prior = prior_d[Z[:, 0]] * probs[Z[:, 0], Z[:, 1]]

    def desc(i):
        lab, spec, kind = kinds[i]
        c = counts[i]
        if kind == "fair":
            return f"{c} fair {'die' if c == 1 else 'dice'}"
        f, p = spec
        return (f"{c} {'die' if c == 1 else 'dice'} loaded so that {FACES[f]} comes up {fmt.p(p)} of the time "
                f"(the other faces equally likely)")

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {"container": cont, "dice": [
            {"type": labels[i], "count": counts[i],
             "face_probs": ("uniform" if kinds[i][2] == "fair" else
                            {FACES[kinds[i][1][0]]: fmt.p(kinds[i][1][1]), "each other face": "equal share of the rest"})}
            for i in range(K)], "picked": "one die at random, type hidden", "roller": who})
        ev = [lambda o: json_line({"roll": o + 1}), lambda o: json_line({"face": FACES[o]})]
    else:
        inv = join_list([desc(i) for i in range(K)])
        prelude = pick(rng, [
            f"A {cont} contains {inv}. {who} draws one die at random and rolls it several times.",
            f"{who}'s {cont} holds {inv}. One die is taken out uniformly at random, and the rolls are reported.",
            f"At a game night there is a {cont} with {inv}. Someone secretly picks one die at random and {who} rolls "
            f"it.",
        ])
        ev = [lambda o: f"The die shows {FACES[o]}.", lambda o: f"{who} rolls a {o + 1}.",
              lambda o: f"Roll result: {o + 1}.", lambda o: f"It lands on {FACES[o]}."]
    pos = [pick(rng, ev) for _ in range(T)]
    face_opts = [str(i + 1) for i in range(6)]
    variables = [
        Var("die", labels, ["Which kind of die was picked?", f"Which die is {who} rolling?",
                            "What type of die is being rolled?"],
            [f"the die being rolled is {a_an(l)}" for l in labels]),
        Var("next_roll", face_opts, ["What number will the next roll show?", "If the die is rolled again, which face "
                                     "will come up?"],
            "the next roll will show a {opt}"),
    ]
    L = probs[Z[:, 0]]
    return World(variables, prior, Z, T, lambda k, p: list(range(6)), lambda o, p: L[:, o], lambda k, o: pos[k](o),
                 prelude, "die", True, {"style": "json" if json_style else "prose", "device": "die"})


def make_world(rng) -> World:
    return _coin_world(rng) if rng.random() < 0.5 else _die_world(rng)
