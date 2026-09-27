"""Short-deck hold'em draw odds with a partially seen opponent hand.

Latent: the opponent's two hidden cards and the next community card, drawn without replacement from the unseen
cards (uniform). Evidence: accidentally flashed cards from the stub (uniform draws from what is left), one of the
opponent's cards exposed at random, and the opponent's betting action with stated raise rates per hand category.
All evidence is conditionally independent / exchangeable given the latent, so perms are provided.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import Fmt, Var, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, people, per_step, pick

NAME = "poker"
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]
SUITS = ["hearts", "spades", "diamonds", "clubs"]
SUIT1 = {"hearts": "heart", "spades": "spade", "diamonds": "diamond", "clubs": "club"}


def cname(c):
    r, s = divmod(c, 4)
    return f"the {RANKS[r]} of {SUITS[s]}"


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    nr = {0: int(rng.integers(4, 7)), 1: int(rng.integers(4, 7)), 2: int(rng.integers(5, 9))}[level]
    nr = min(13, nr + 2 * (scale - 1))
    ranks = sorted(int(r) for r in rng.choice(13, size=nr, replace=False))
    deck = [r * 4 + s for r in ranks for s in range(4)]
    ts = int(rng.integers(4))  # target suit
    suit = SUITS[ts]
    tsuit = [c for c in deck if c % 4 == ts]
    mine = [int(c) for c in rng.choice(tsuit, size=2, replace=False)]
    rest = [c for c in deck if c not in mine]
    for _ in range(100):  # keep at least two unseen target-suit cards, so suit questions stay uncertain
        board = [int(c) for c in rng.choice(rest, size=3, replace=False)]
        U = [c for c in rest if c not in board]
        if 2 <= sum(c % 4 == ts for c in U) <= len(U) - 2:
            break
    nU = len(U)
    T = n_steps(rng, level, 3, 5)
    T = min(T, nU - 4)

    # evidence schedule
    kinds = ["bet"]  # every hand has one betting tell, which carries the (varied) raise rates
    for k in range(len(kinds), T):
        opts = ["burn", "burn"] + (["expose"] if "expose" not in kinds else []) + \
               (["bet", "bet"] if "bet" not in kinds else [])
        kinds.append(pick(rng, opts))
    kinds = [kinds[int(i)] for i in rng.permutation(len(kinds))]
    n_burn = kinds.count("burn")
    raise_p = sorted([int(x) for x in rng.choice(np.arange(5, 96), size=3, replace=False)], reverse=True)

    rows = [(i, j, n) for i, j in itertools.combinations(U, 2) for n in U if n not in (i, j)]
    O1 = np.array([r[0] for r in rows])
    O2 = np.array([r[1] for r in rows])
    NX = np.array([r[2] for r in rows])
    prior = np.ones(len(rows))

    def cat(i, j):  # 0 = pair, 1 = at least one target-suit card, 2 = other
        return 0 if i // 4 == j // 4 else (1 if (i % 4 == ts or j % 4 == ts) else 2)

    CAT = np.array([cat(i, j) for i, j, _ in rows])
    vset = {0: ["opp_suit", "next_card"], 1: ["opp_suit", "opp_pair"] + (["next_card"] if rng.random() < 0.5 else []),
            2: ["opp_suit", "opp_pair", "next_card"]}[level]

    def assign(i, j, n):
        a = []
        for v in vset:
            if v == "opp_suit":
                a.append(int(i % 4 == ts) + int(j % 4 == ts))
            elif v == "opp_pair":
                a.append(0 if i // 4 == j // 4 else 1)
            else:
                a.append(0 if n % 4 == ts else 1)
        return tuple(a)

    proj = np.array([assign(*r) for r in rows])
    S1 = SUIT1[suit]
    V = {"opp_suit": Var("opp_suit", [f"no {suit}", f"one {S1}", f"two {suit}"],
                         [f"How many {suit} does the opponent hold?", f"How many of the opponent's two cards are {suit}?",
                          f"How many {suit} are in the opponent's hand?"],
                         [f"the opponent holds no {suit}", f"the opponent holds exactly one {S1}",
                          f"the opponent holds two {suit}"]),
         "opp_pair": Var("opp_pair", ["a pocket pair", "no pair"],
                         ["Does the opponent hold a pocket pair?", "Are the opponent's two cards a pair?"],
                         ["the opponent holds a pocket pair", "the opponent does not hold a pair"]),
         "next_card": Var("next_card", [f"a {S1}", f"not a {S1}"],
                          ["Will the next community card be a " + S1 + "?", f"Is the next card dealt to the board a {S1}?",
                           f"Will the next board card be a {S1}?"],
                          [f"the next community card is a {S1}", f"the next community card is not a {S1}"])}
    variables = [V[v] for v in vset]

    opp = pick(rng, people(rng, 3))
    js = is_json(rng)
    rtxt = (f"{opp} raises {fmt.p(raise_p[0])} of the time with a pocket pair, {fmt.p(raise_p[1])} of the time with "
            f"no pair but at least one {S1}, and {fmt.p(raise_p[2])} of the time otherwise (and checks otherwise)")
    ranks_txt = join_list([RANKS[r] + "s" for r in ranks])
    if js:
        st = {"deck": f"{len(deck)} cards: ranks {', '.join(RANKS[r] for r in ranks)} in all four suits",
              "your_hole_cards": [cname(c) for c in mine], "board": [cname(c) for c in board],
              "opponent": f"{opp}, holding two hidden cards dealt uniformly from the {nU} unseen cards",
              "next_community_card": "dealt after any flashed cards, from the remaining unseen cards",
              "flashed_cards": "a flashed card is a uniformly random card from the undealt stub"}
        if "bet" in kinds:
            st["betting"] = rtxt
        if "expose" in kinds:
            st["exposure"] = f"one of {opp}'s two cards, chosen at random, may get exposed"
        prelude = json_block(rng, st)
        tb = [lambda c: json_line({"flashed_from_stub": cname(c)}), lambda c: json_line({"event": "card flashed", "card": cname(c)})]
        te = [lambda c: json_line({"exposed_opponent_card": cname(c)}), lambda c: json_line({"event": "opponent card seen", "card": cname(c)})]
        tr = [lambda b: json_line({"opponent_action": "raise" if b == 0 else "check"}),
              lambda b: json_line({"event": "bet", "action": "raises" if b == 0 else "checks"})]
    else:
        base = (f"The game uses a short deck of {len(deck)} cards (only the {ranks_txt}, in all four suits), "
                f"shuffled uniformly. You hold {cname(mine[0])} and {cname(mine[1])}; the board shows "
                f"{join_list([cname(c) for c in board])}. {opp} holds two hidden cards dealt from the {nU} cards you "
                f"cannot see.")
        extra = []
        if "burn" in kinds:
            extra.append("Now and then the dealer clumsily flashes a card from the undealt stub; such a card is a "
                         "uniformly random card among those still undealt, and it is set aside.")
        if "expose" in kinds:
            extra.append(f"At one point one of {opp}'s two cards, chosen at random, may be accidentally exposed.")
        if "bet" in kinds:
            extra.append(cap(rtxt) + ".")
        nxt = pick(rng, ["The next community card will come from the stub after any flashed cards.",
                         "Once the flashing is over, the dealer turns the next community card from the stub.",
                         "The next card to the board is dealt from what remains of the stub."])
        prelude = pick(rng, [" ".join([base] + extra + [nxt]),
                             " ".join(["Poker hand in progress."] + [base, nxt] + extra),
                             " ".join([base.replace("The game uses", "You are playing hold'em with")] + extra + [nxt])])
        tb = [lambda c: f"The dealer flashes {cname(c)} from the stub.", lambda c: f"A card slips out of the stub: {cname(c)}.",
              lambda c: f"Flashed card: {cname(c)}.", lambda c: f"You glimpse {cname(c)} as the dealer handles the stub."]
        te = [lambda c: f"One of {opp}'s cards is exposed: {cname(c)}.", lambda c: f"{opp} fumbles and shows {cname(c)}.",
              lambda c: f"You catch sight of one of {opp}'s cards: {cname(c)}."]
        tr = [lambda b: f"{opp} {'raises' if b == 0 else 'checks'}.", lambda b: f"Action: {opp} {'puts in a raise' if b == 0 else 'checks'}.",
              lambda b: f"{opp} decides to {'raise' if b == 0 else 'check'}."]
    rb, re_, rr = per_step(rng, T, tb), per_step(rng, T, te), per_step(rng, T, tr)

    def seen(past):
        return {o[1] for o in past if o[0] in ("burn", "expose")}

    def alternatives(k, past):
        if kinds[k] == "bet":
            return [("bet", 0), ("bet", 1)]
        s = seen(past)
        return [(kinds[k], c) for c in U if c not in s]

    rp = np.array(raise_p, float)[CAT] / 100

    def lik(o, past):
        kind, c = o
        if kind == "bet":
            return rp if c == 0 else 1 - rp
        inhand = (O1 == c) | (O2 == c)
        if kind == "expose":
            return inhand * 0.5
        nb = sum(1 for p in past if p[0] == "burn")
        return (~inhand & (NX != c)) / (nU - 3 - nb)

    def render(k, o):
        kind, c = o
        return {"burn": rb, "expose": re_, "bet": rr}[kind][k](c)

    return mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, "opp_suit", True,
                    "json" if js else "prose", "holdem",
                    {"U": U, "ts": ts, "kinds": kinds, "raise_p": raise_p, "cat": cat, "assign": assign})


def simulate(world, rng):
    p = world.params
    U = list(p["U"])
    rng.shuffle(U)
    i, j = U[0], U[1]
    pos = 2
    obs = []
    for kind in p["kinds"]:
        if kind == "burn":
            obs.append(("burn", U[pos]))
            pos += 1
        elif kind == "expose":
            obs.append(("expose", i if rng.random() < 0.5 else j))
        else:
            obs.append(("bet", 0 if rng.random() < p["raise_p"][p["cat"](i, j)] / 100 else 1))
    a, b = min(i, j), max(i, j)
    return p["assign"](a, b, U[pos]), obs
