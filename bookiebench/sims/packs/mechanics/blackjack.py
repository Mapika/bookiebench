"""Blackjack with a stated shoe composition, a hidden dealer hole card and cards revealed to other players.

Latent: (dealer hole card value h, the player's next card value x). All unseen cards are a uniformly random
permutation of the stated shoe, so (h, x) is a uniform ordered pair of distinct cards and each face-up card dealt to
another player is a uniform draw from what is left: lik(r | h, x, past) = (c_r - [h=r] - [x=r] - #past r) / (N-2-k).
Evidence is exchangeable. Variables: the hole card (value or value group), the player's total after hitting, the
dealer's two-card total.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, people, per_step,
                    pick, sample)

NAME = "blackjack"
VNAME = {1: "ace", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
         10: "ten-valued card"}


def card(v):
    return {1: "an ace", 8: "an eight"}.get(v, f"a {VNAME[v]}")


def plural(v, n):
    if v == 10:
        return f"{n} ten-valued card" + ("s" if n != 1 else "")
    nm = VNAME[v]
    if n == 1:
        return f"1 {nm}"
    return f"{n} {nm}s" if nm != "six" else f"{n} sixes"


def total(cards):
    t = sum(cards)
    if 1 in cards and t + 10 <= 21:
        return t + 10, True
    return t, False


GROUPS = [("an ace", {1}), ("a ten-valued card", {10}), ("a 7, 8 or 9", {7, 8, 9}), ("a 2 to 6", {2, 3, 4, 5, 6})]


def player_cat(hand, x):
    t, _ = total(hand + [x])
    return 0 if t > 21 else (1 if t >= 17 else 2)


PLAYER_OPTS = ["bust (over 21)", "17 to 21", "16 or less"]
DEALER_OPTS = ["a natural blackjack", "17 to 21", "12 to 16", "11 or less"]


def dealer_cat(u, h):
    if {u, h} == {1, 10}:
        return 0
    t, _ = total([u, h])
    return 1 if t >= 17 else (2 if t >= 12 else 3)


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    T = n_steps(rng, level, 3, 6)
    while True:
        if level == 0:
            vals = sorted(int(v) for v in rng.choice(np.arange(1, 11), size=int(rng.integers(3, 5)), replace=False))
            counts = {v: int(rng.integers(1, 7)) for v in vals}
        elif level == 1:
            vals = sorted(int(v) for v in rng.choice(np.arange(1, 11), size=int(rng.integers(5, 8)), replace=False))
            counts = {v: int(rng.integers(1, 8)) for v in vals}
        else:
            counts = {v: int(rng.integers(0, 4 * scale + 1)) for v in range(1, 10)}
            counts[10] = int(rng.integers(2, 10 * scale + 1))
            counts = {v: c for v, c in counts.items() if c > 0}
            vals = sorted(counts)
        N = sum(counts.values())
        if N < T + 4 or len(vals) < 3:
            continue
        u = int(rng.integers(1, 11))
        hand = [int(rng.integers(1, 11)), int(rng.integers(1, 11))]
        if total(hand)[0] >= 21:
            continue
        pc = {player_cat(hand, x) for x in vals}
        dc = {dealer_cat(u, h) for h in vals}
        if len(pc) >= 2:
            break

    grouped = len(vals) > 5
    if grouped:
        gl = [(nm, s) for nm, s in GROUPS if s & set(vals)]
        hole_opts = [nm for nm, _ in gl]
        hole_of = {v: next(i for i, (_, s) in enumerate(gl) if v in s) for v in vals}
    else:
        hole_opts = [card(v) for v in vals]
        hole_of = {v: i for i, v in enumerate(vals)}
    p_opts_ids = sorted(pc)
    d_opts_ids = sorted(dc)
    use_dealer = len(d_opts_ids) >= 2 and (level == 2 or (level == 1 and rng.random() < 0.5))
    use_player = level != 1 or not use_dealer

    rows = [(h, x) for h, x in itertools.product(vals, vals) if counts[h] - (h == x) > 0 and counts[x] > 0]
    prior = np.array([counts[h] / N * (counts[x] - (h == x)) / (N - 1) for h, x in rows])
    H = np.array([r[0] for r in rows])
    X = np.array([r[1] for r in rows])

    def assign(h, x):
        a = [hole_of[h]]
        if use_player:
            a.append(p_opts_ids.index(player_cat(hand, x)))
        if use_dealer:
            a.append(d_opts_ids.index(dealer_cat(u, h)))
        return tuple(a)

    proj = np.array([assign(h, x) for h, x in rows])

    hand_txt = join_list([card(c) for c in hand])
    pt, soft = total(hand)
    variables = [Var("hole_card", hole_opts,
                     ["What is the dealer's face-down card?", "Which card is the dealer's hole card?",
                      "What is the dealer hiding face down?"],
                     "the dealer's face-down card is {opt}")]
    if use_player:
        variables.append(Var("after_hit", [PLAYER_OPTS[i] for i in p_opts_ids],
                             ["If you take one more card, what will your total be?",
                              "What will your hand total after hitting once?",
                              "If you hit, where does your total end up?"],
                             [f"your total after hitting is {PLAYER_OPTS[i].replace('bust (over 21)', 'over 21')}"
                              for i in p_opts_ids]))
    if use_dealer:
        variables.append(Var("dealer_total", [DEALER_OPTS[i] for i in d_opts_ids],
                             ["What is the dealer's two-card total?", "What do the dealer's two cards add up to?",
                              "Which range does the dealer's two-card hand fall in?"],
                             [("the dealer has " + DEALER_OPTS[i]) if i == 0 else f"the dealer's two cards total {DEALER_OPTS[i]}"
                              for i in d_opts_ids]))

    others = people(rng, 6)
    seat = [others[int(rng.integers(len(others)))] for _ in range(T)]
    js = is_json(rng)
    rule = "The dealer does not peek at the face-down card (nothing is revealed about it before the end of the hand). Aces count 11 unless that would take a hand over 21, in which case they count 1; ten-valued cards are 10, J, Q and K."
    comp = join_list([plural(v, counts[v]) for v in vals])
    if js:
        prelude = json_block(rng, {
            "shoe_before_hole_card": {("ten-valued (10/J/Q/K)" if v == 10 else VNAME[v]): counts[v] for v in vals},
            "dealer_upcard": VNAME[u], "your_cards": [VNAME[c] for c in hand],
            "dealing": "the dealer's face-down card, then face-up cards to other players, then your next card, "
                       "all from the shoe in uniformly shuffled order",
            "ace_rule": "11 unless over 21, then 1",
            "dealer_peek": "the dealer does not peek: nothing is revealed about the face-down card before the end of the hand"})
        tpls = [lambda nm, v: json_line({"player": nm, "face_up_card": VNAME[v]}),
                lambda nm, v: json_line({"dealt_to": nm, "card": VNAME[v]}),
                lambda nm, v: json_line({"event": "card shown", "seat": nm, "value": VNAME[v]})]
    else:
        prelude = pick(rng, [
            f"You are playing blackjack from a small, well-shuffled shoe. Just before the dealer's face-down card was "
            f"dealt, the shoe held exactly {comp}. The dealer shows {card(u)} face up, and you hold {hand_txt} "
            f"(total {pt}{', soft' if soft else ''}). Next, the dealer deals face-up cards to the other players, and "
            f"only after that would your own next card come from the shoe. {rule}",
            f"Blackjack table. Your hand: {hand_txt}. Dealer's up card: {card(u)}. The shoe (shuffled uniformly) "
            f"contained {comp} at the moment the dealer took the face-down card. Before you act, face-up cards go to "
            f"the other players; your hit card, if you take one, is the card after those. {rule}",
            f"A dealer shuffles a stripped shoe of {comp}. From it she deals her own face-down card; her face-up card "
            f"({card(u)}) and your two cards ({hand_txt}) came from elsewhere and are not part of that count. The other "
            f"players then receive face-up cards from the shoe, and your next card would follow them. {rule}",
        ])
        tpls = [lambda nm, v: f"{nm} is dealt {card(v)} face up.",
                lambda nm, v: f"A card goes face up to {nm}: {card(v)}.",
                lambda nm, v: f"{nm} receives {card(v)}.",
                lambda nm, v: f"The card shown to {nm} is {card(v)}."]
    ptpl = per_step(rng, T, tpls)

    def alternatives(k, past):
        return [v for v in vals if counts[v] - past.count(v) > 0]

    def lik(o, past):
        c = counts[o] - (H == o) - (X == o) - past.count(o)
        return np.maximum(c, 0) / (N - 2 - len(past))

    return mk_world(variables, prior, proj, T, alternatives, lik, lambda k, o: ptpl[k](seat[k], o), prelude,
                    "hole_card", True, "json" if js else "prose", "blackjack",
                    {"counts": counts, "assign": assign, "hand": hand, "up": u})


def simulate(world, rng):
    p = world.params
    deck = [v for v, c in p["counts"].items() for _ in range(c)]
    rng.shuffle(deck)
    T = world.T
    return p["assign"](deck[0], deck[T + 1]), list(deck[1:T + 1])
