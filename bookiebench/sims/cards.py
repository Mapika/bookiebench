"""Drawing without replacement from one of several small decks of unknown identity.

Draws without replacement are exchangeable, so the latent z = (deck, next card, card after next) can put the two
future cards "first": prior(z) = P(deck) P(c1|deck) P(c2|deck,c1) and each observed draw has likelihood
(remaining count of its type) / (remaining cards), with the two future cards and the past draws removed.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, Fmt, cap, composition, join_list, json_block, json_line, person, pick, sample
from .world import Var, World

THEMES = [
    dict(key="cards", unit="card", units="cards", pile="deck", types=["hearts", "spades", "clubs", "diamonds"],
         type_of="a {t} card", deck_labels=[["Deck A", "Deck B", "Deck C"],
                                          ["the red-backed deck", "the blue-backed deck", "the green-backed deck"]]),
    dict(key="tiles", unit="tile", units="tiles", pile="bag", types=["vowel", "consonant", "blank"],
         type_of="a {t} tile", deck_labels=[["Bag 1", "Bag 2", "Bag 3"], ["the felt bag", "the leather bag",
                                                                         "the paper bag"]]),
    dict(key="tickets", unit="ticket", units="tickets", pile="drum", types=["gold", "silver", "plain", "blue"],
         type_of="a {t} ticket", deck_labels=[["Drum A", "Drum B", "Drum C"], ["the north drum", "the south drum",
                                                                              "the east drum"]]),
]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    who = person(rng)
    K = int(rng.integers(2, 4))
    C = int(rng.integers(2, 4))
    types = sample(rng, th["types"], C)
    labels = pick(rng, th["deck_labels"])[:K]
    T = int(rng.integers(3, 6))
    while True:
        comps = [composition(rng, C, int(rng.integers(max(T + 3, 6), 15))) for _ in range(K)]
        fr = [tuple(np.array(c) / sum(c)) for c in comps]
        if len(set(fr)) == K:
            break
    cnt = np.array(comps, float)  # K x C
    N = cnt.sum(1)  # K
    sel = pick(rng, ["uniform", "pct"])
    if sel == "uniform":
        w = [1] * K
        sel_txt = "one of them is picked uniformly at random"
    else:
        w = [x * 5 for x in composition(rng, K, 20, min_each=1)]
        sel_txt = "one of them is picked, " + join_list([f"{l} with probability {fmt.p(x)}" for l, x in zip(labels, w)])
    pdk = np.array(w, float) / sum(w)

    Z = np.array(list(itertools.product(range(K), range(C), range(C))))
    d, c1, c2 = Z[:, 0], Z[:, 1], Z[:, 2]
    p1 = cnt[d, c1] / N[d]
    p2 = (cnt[d, c2] - (c1 == c2)) / (N[d] - 1)
    prior = pdk[d] * p1 * np.clip(p2, 0, None)

    def lik(o, past):
        used = sum(1 for x in past if x == o)
        rem = cnt[d, o] - used - (c1 == o) - (c2 == o)
        return np.clip(rem, 0, None) / (N[d] - len(past) - 2)

    def desc(i):
        return f"{labels[i]} has " + join_list([f"{comps[i][j]} {types[j]}" for j in range(C) if comps[i][j] > 0])

    u, us, pile = th["unit"], th["units"], th["pile"]
    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {f"{pile}s": {labels[i]: {types[j]: comps[i][j] for j in range(C)} for i in range(K)},
                                   "selection": {l: (fmt.p(x) if sel == "pct" else f"1/{K}") for l, x in zip(labels, w)},
                                   "drawing": f"{us} drawn one by one WITHOUT replacement from the chosen {pile}",
                                   "dealer": who})
        render = lambda k, o: json_line({"drawn": types[o]})  # noqa: E731
    else:
        descs = "; ".join(desc(i) for i in range(K))
        prelude = pick(rng, [
            f"{who} has {K} {pile}s of {us}: {descs}. {cap(sel_txt)}, and {who} draws {us} from it one at a time "
            f"without putting any back.",
            f"There are {K} {pile}s. {cap(descs)}. Out of sight, {sel_txt}. {us.capitalize()} are then drawn from "
            f"that {pile} without replacement and shown one by one.",
            f"A parlour game uses these {pile}s: {descs}. {cap(sel_txt)}. {who} deals {us} from the chosen {pile}, "
            f"never returning a drawn {u}.",
        ])
        ev = [lambda t: f"A {u} is drawn: {t}.", lambda t: f"{who} draws {th['type_of'].format(t=t)}.",
              lambda t: f"The drawn {u} is {t}; it is set aside.", lambda t: f"Out comes {th['type_of'].format(t=t)}."]
        pos = [pick(rng, ev) for _ in range(T)]
        render = lambda k, o: pos[k](types[o])  # noqa: E731

    variables = [
        Var(pile, labels, [f"Which {pile} are the {us} coming from?", f"Which {pile} was picked?",
                           f"From which {pile} is {who} drawing?"],
            f"the {us} are being drawn from {{opt}}"),
        Var("next", types, [f"What type will the next {u} drawn be?", f"If one more {u} is drawn, what will it be?"],
            f"the next {u} drawn will be {{opt}}" if get_tv() < 2 else
            f"the next {u} drawn will be {th['type_of'].replace('{t}', '{opt}')}"),
        Var("after_next", types, [f"If two more {us} are drawn, what will the second of them be?"],
            f"the second of the next two {us} drawn will be {{opt}}" if get_tv() < 2 else
            f"the second of the next two {us} drawn will be {th['type_of'].replace('{t}', '{opt}')}"),
    ]
    return World(variables, prior, Z, T,
                 alternatives=lambda k, past: list(range(C)), lik=lik, render=render, prelude=prelude,
                 mart_var=pile, exchangeable=True, meta={"style": "json" if json_style else "prose", "theme": th["key"]},
                 params={"counts": cnt, "prior": pdk})
