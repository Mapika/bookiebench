"""Urn selection + draws with replacement.

Latent: which container was chosen (prior from a die / stated weights / uniform). Draws are i.i.d. given the
container, so the evidence is exchangeable. Variables: container, the next draw, (optionally) the draw after that.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import (FIRST_NAMES, a_an, LETTERS, COLORS, Fmt, cap, composition, join_list, json_block, json_line, person, pick,
                     sample)
from .world import Var, World

CONTAINERS = ["urn", "jar", "bag", "box", "bucket", "tin"]
ITEMS = [("ball", "balls"), ("marble", "marbles"), ("bead", "beads"), ("token", "tokens"), ("chip", "chips")]
ADJ = ["wooden", "metal", "glass", "striped", "tall", "small", "clay", "canvas"]


def make_world(rng) -> World:
    K = int(rng.integers(2, 4))
    C = int(rng.integers(2, 4))
    cont = pick(rng, CONTAINERS)
    item, items = pick(rng, ITEMS)
    colors = sample(rng, COLORS, C)
    who = person(rng)
    fmt = Fmt(rng)
    label_style = pick(rng, ["letter", "adj", "owner"])
    if label_style == "letter":
        labels = [f"{cap(cont)} {LETTERS[i]}" for i in range(K)]
    elif label_style == "adj":
        labels = [f"the {a} {cont}" for a in sample(rng, ADJ, K)]
    else:
        labels = [f"{n}'s {cont}" for n in sample(rng, [n for n in FIRST_NAMES if n != who], K)]

    # compositions: distinct across containers
    while True:
        comps = [composition(rng, C, int(rng.integers(4, 13))) for _ in range(K)]
        fr = [tuple(np.array(c) / sum(c)) for c in comps]
        if len(set(fr)) == K:
            break

    sel = pick(rng, ["uniform", "die", "pct"])
    if sel == "uniform":
        w = [1] * K
        sel_txt = f"one {cont} is picked uniformly at random"
    elif sel == "die":
        w = composition(rng, K, 6, min_each=1)
        ranges, lo = [], 1
        for i in range(K):
            hi = lo + w[i] - 1
            ranges.append((f"a {lo}" if lo == hi else f"{lo}-{hi}", labels[i]))
            lo = hi + 1
        sel_txt = f"{who} rolls a fair six-sided die to choose: " + join_list([f"{r} means {l}" for r, l in ranges])
    else:
        w = composition(rng, K, 20, min_each=1)
        w = [x * 5 for x in w]
        sel_txt = "the " + cont + " is chosen with probabilities " + join_list(
            [f"{fmt.p(x)} for {l}" for x, l in zip(w, labels)])
    prior_c = np.array(w, float) / sum(w)
    theta = np.array([np.array(c, float) / sum(c) for c in comps])  # K x C

    three = rng.random() < 0.5
    T = int(rng.integers(3, 6))
    Z = list(itertools.product(range(K), range(C), range(C))) if three else list(itertools.product(range(K), range(C)))
    Za = np.array(Z)
    prior = prior_c[Za[:, 0]] * theta[Za[:, 0], Za[:, 1]]
    if three:
        prior = prior * theta[Za[:, 0], Za[:, 2]]
    proj = Za

    def desc(i):
        parts = [f"{comps[i][j]} {colors[j]}" for j in range(C) if comps[i][j] > 0]
        return f"{labels[i]} holds " + join_list(parts) + f" {items}"

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {
            "containers": {labels[i]: {colors[j]: comps[i][j] for j in range(C)} for i in range(K)},
            "selection": {labels[i]: (f"{w[i]}/{sum(w)}" if sel != "pct" else fmt.p(w[i])) for i in range(K)},
            "draws": f"one {item} at a time, with replacement, colour recorded",
            "which_container": "hidden",
            "drawn_by": who,
        })
    else:
        tpl = pick(rng, [0, 1, 2])
        descs = ". ".join(cap(desc(i)) for i in range(K)) + "."
        if tpl == 0:
            prelude = (f"{who} has {K} {cont}s. {descs} To start the game, {sel_txt}. Without saying which {cont} "
                       f"was chosen, {who} draws {items} from it one at a time, announcing the colour and putting "
                       f"each {item} back before the next draw.")
        elif tpl == 1:
            prelude = (f"There are {K} {cont}s on a table. {descs} Behind a screen, {sel_txt}. {cap(items)} are then "
                       f"drawn from that {cont} with replacement, and only their colours are reported.")
        else:
            prelude = (f"A guessing game uses {K} {cont}s: {join_list([desc(i) for i in range(K)])}. {cap(sel_txt)}. "
                       f"The chosen {cont} stays hidden; {item}s are drawn from it with replacement and each colour is "
                       f"called out.")

    ev_tpls = ([lambda c: json_line({"draw": {"colour": c}}), lambda c: json_line({"event": "draw", "color": c})]
               if json_style else [
        lambda c: f"{cap(item)} drawn: it is {c}.",
        lambda c: f"{who} pulls out {a_an(c + ' ' + item)} and puts it back.",
        lambda c: f"The drawn {item} is {c}; it is returned to the {cont}.",
        lambda c: f"Draw result: {c}.",
    ])
    pos_tpl = [pick(rng, ev_tpls) for _ in range(T)]

    variables = [
        Var("container", labels, [f"Which {cont} are the {items} being drawn from?", f"Which {cont} was chosen?",
                                  f"From which {cont} is {who} drawing?"],
            f"the {items} are being drawn from {{opt}}"),
        Var("next_draw", colors, [f"What colour will the next {item} drawn be?",
                                  f"If one more {item} is drawn, what colour will it be?"],
            f"the next {item} drawn will be {{opt}}"),
    ]
    if three:
        variables.append(Var("draw_after_next", colors,
                             [f"If two more {items} are drawn, what colour will the second of them be?",
                              f"What colour will the {item} drawn two draws from now be?"],
                             f"the second of the next two {items} drawn will be {{opt}}"))

    lik_tab = theta[Za[:, 0]]  # n x C

    return World(
        variables=variables, prior=prior, proj=proj, T=T,
        alternatives=lambda k, past: list(range(C)),
        lik=lambda o, past: lik_tab[:, o],
        render=lambda k, o: pos_tpl[k](colors[o]),
        prelude=prelude, mart_var="container", exchangeable=True,
        meta={"style": "json" if json_style else "prose"},
        params={"prior": prior_c, "theta": theta, "three": three},
    )
