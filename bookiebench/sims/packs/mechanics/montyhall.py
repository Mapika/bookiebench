"""Generalised Monty Hall: N doors, stated car placement odds, host knowledge variants, optional second prize and
a peeking friend.

Host variants: `knows` (always opens a goat door, uniformly among eligible ones), `lazy` (knows; opens the
lowest-numbered eligible door with a stated probability, otherwise uniformly among the other eligible doors),
`ignorant` (opens a uniformly random unopened, unpicked door; whatever is behind it is shown).
The host's choices depend on what he has already opened, so the evidence is not exchangeable (no perms).
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import Fmt, Var, cap, is_json, join_list, json_block, json_line, mk_world, people, per_step, pick, round_pcts

NAME = "montyhall"
PRIZES = [("a car", "car"), ("a sports car", "sports car"), ("a holiday voucher", "voucher"), ("a gold bar", "gold bar")]
SECOND = [("a bicycle", "bicycle"), ("a toaster", "toaster"), ("a scooter", "scooter")]
GOAT = [("a goat", "goat", "goats"), ("a rubber chicken", "rubber chicken", "rubber chickens"),
        ("an empty box", "empty box", "empty boxes")]


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    Nd = {0: 3, 1: int(rng.integers(3, 5)), 2: int(rng.integers(4, 6))}[level] + (scale - 1)
    bike = level == 2 and rng.random() < 0.6
    friend = rng.random() < (0.7 if level >= 1 else 0.5)
    pending = level >= 1 and not bike and friend
    host = pick(rng, ["knows", "lazy", "lazy", "ignorant"] if level >= 1 else ["knows", "lazy"])
    prize, prize_s = pick(rng, PRIZES)
    second, second_s = pick(rng, SECOND)
    goat, goat_s, goats = pick(rng, GOAT)
    pick_door = int(rng.integers(Nd))
    weighted = True  # stated, varied placement odds (a uniform prior makes answers repeat)
    w = round_pcts(rng.dirichlet(np.full(Nd, 2.0)), 100, 3) if weighted else [1] * Nd
    wsum = sum(w)
    lazy_q = int(rng.integers(30, 100))
    fa = int(rng.integers(40, 96))
    max_open = Nd - 2 - (1 if bike else 0)
    if host == "ignorant":
        max_open = Nd - 2
    n_open = int(rng.integers(1, max_open + 1))
    kinds = ["host"] * n_open
    if friend:
        kinds.insert(int(rng.integers(len(kinds) + 1)), "friend")
    if level == 0:
        kinds = kinds[:2]
    T = len(kinds)

    def contents(door, car, bk):
        return 0 if door == car else (1 if bike and door == bk else 2)

    rows = []
    for car in range(Nd):
        for bk in (range(Nd) if bike else [None]):
            if bike and bk == car:
                continue
            for pend in (range(Nd) if pending else [None]):
                p = w[car] / wsum
                if bike:
                    p /= (Nd - 1)
                if pending:
                    p *= fa / 100 if pend == car else (1 - fa / 100) / (Nd - 1)
                rows.append((car, bk, pend, p))
    CAR = np.array([r[0] for r in rows])
    BK = np.array([(-1 if r[1] is None else r[1]) for r in rows])
    PEND = np.array([(-1 if r[2] is None else r[2]) for r in rows])
    prior = np.array([r[3] for r in rows])
    D = lambda i: f"door {i + 1}"  # noqa: E731

    def assign(car, bk, pend):
        a = [car, 0 if car == pick_door else 1]
        if bike:
            a.append(bk)
        if pending:
            a.append(pend)
        return tuple(a)

    proj = np.array([assign(r[0], r[1], r[2]) for r in rows])
    doors = [D(i) for i in range(Nd)]
    Dc = [f"Door {i + 1}" for i in range(Nd)]
    variables = [Var("car_door", Dc, [f"Which door hides {prize}?", f"Behind which door is {prize}?",
                                      f"Where is {prize}?"], [f"{prize} is behind {d}" for d in doors]),
                 Var("stay_wins", ["yes", "no"], [f"Does sticking with {D(pick_door)} win {prize}?",
                                                  f"Would staying with the original pick win {prize}?"],
                     [f"staying with {D(pick_door)} wins {prize}", f"staying with {D(pick_door)} does not win {prize}"])]
    if bike:
        variables.append(Var("second_prize", Dc, [f"Which door hides {second}?", f"Behind which door is {second}?"],
                             [f"{second} is behind {d}" for d in doors]))
    if pending:
        variables.append(Var("second_friend", Dc, ["Which door will the second friend point to?",
                                                   "What door will the other friend signal when asked?"],
                             [f"the second friend points to {d}" for d in doors]))

    host_nm, you, fr = people(rng, 3)
    place = (f"The producers put {prize} behind " + join_list([f"{D(i)} with probability {fmt.p(w[i])}" for i in range(Nd)])
             if weighted else f"{cap(prize)} was placed behind one of the {Nd} doors uniformly at random")
    htxt = {
        "knows": f"The host, {host_nm}, knows where everything is and, each time, opens a door that {you} did not pick, "
                 f"that is still closed and that hides {goat}; if several doors qualify, the host chooses uniformly among them",
        "lazy": f"The host, {host_nm}, knows where everything is and always opens an unpicked, closed door hiding {goat}; "
                f"when several qualify, the host opens the lowest-numbered one with probability {fmt.p(lazy_q)} and otherwise "
                f"picks uniformly among the remaining qualifying doors",
        "ignorant": f"The host, {host_nm}, has no idea where anything is: each time the host opens a uniformly random closed door "
                    f"that {you} did not pick, and whatever is behind it is shown (the game goes on regardless)",
    }[host]
    ftxt = (f"{fr}, who peeked backstage, signals a door: the right one (the {prize_s}) with probability {fmt.p(fa)}, "
            f"otherwise one of the other {Nd - 1} doors uniformly at random") if friend else ""
    ptxt = (f"A second friend who also peeked has not signalled yet; when asked, they will signal the right door with "
            f"probability {fmt.p(fa)} and otherwise one of the other {Nd - 1} doors uniformly at random, independently "
            f"of {fr}.") if pending else ""
    js = is_json(rng)
    if js:
        st = {"doors": Nd, "contestant": you, "contestant_pick": D(pick_door),
              "prize_placement": ({D(i): fmt.p(w[i]) for i in range(Nd)} if weighted else "uniform"),
              "other_doors": goats + (f", except one door hiding {second}, placed uniformly at random among the doors "
                                      f"without {prize}" if bike else ""), "host": htxt}
        if friend:
            st["friend"] = ftxt
        if pending:
            st["second_friend"] = ptxt
        prelude = json_block(rng, st)
        ht = [lambda d, c: json_line({"host_opens": D(d), "reveals": [prize_s, second_s, goat_s][c]}),
              lambda d, c: json_line({"event": "door opened", "door": d + 1, "behind": [prize_s, second_s, goat_s][c]})]
        ft = [lambda d: json_line({"friend_signals": D(d)}), lambda d: json_line({"event": "signal", "door": d + 1})]
    else:
        body = [f"On a game show there are {Nd} doors. {place}.",
                (f"{cap(second)} is behind one of the other doors, chosen uniformly at random, and the rest hide {goats}."
                 if bike else f"All the other doors hide {goats}."),
                f"{you} picks {D(pick_door)}.", cap(htxt) + "."]
        if friend:
            body.append(cap(ftxt) + ".")
        if ptxt:
            body.append(ptxt)
        order = pick(rng, [[0, 1, 2, 3, 4, 5], [2, 0, 1, 3, 4, 5], [0, 1, 3, 2, 4, 5]])
        prelude = " ".join(body[i] for i in order if i < len(body))
        ht = [lambda d, c: f"{host_nm} opens {D(d)}, revealing {[prize, second, goat][c]}.",
              lambda d, c: f"{cap(D(d))} is opened by the host: {[prize, second, goat][c]}.",
              lambda d, c: f"The host swings open {D(d)} and shows {[prize, second, goat][c]}."]
        ft = [lambda d: f"{fr} signals {D(d)}.", lambda d: f"From the audience, {fr} points at {D(d)}.",
              lambda d: f"{fr}'s signal: {D(d)}."]
    htp, ftp = per_step(rng, T, ht), per_step(rng, T, ft)

    def opened(past):
        return [o[1] for o in past if o[0] == "host"]

    def alternatives(k, past):
        if kinds[k] == "friend":
            return [("friend", d) for d in range(Nd)]
        op = opened(past)
        closed = [d for d in range(Nd) if d != pick_door and d not in op]
        cs = [2] if host != "ignorant" else [0, 1, 2] if bike else [0, 2]
        return [("host", d, c) for d in closed for c in cs]

    def lik(o, past):
        if o[0] == "friend":
            d = o[1]
            return np.where(CAR == d, fa / 100, (1 - fa / 100) / (Nd - 1))
        _, d, c = o
        op = opened(past)
        closed = [x for x in range(Nd) if x != pick_door and x not in op]
        cont = np.where(CAR == d, 0, np.where(BK == d, 1, 2))
        match = (cont == c).astype(float)
        if host == "ignorant":
            return match / len(closed)
        # eligible = closed doors hiding a goat
        elig = np.zeros(len(CAR))
        for x in closed:
            elig += (CAR != x) & (BK != x)
        is_elig = (CAR != d) & (BK != d)
        if host == "knows":
            return np.where(is_elig, 1.0 / np.maximum(elig, 1), 0.0)
        # lazy: lowest-numbered eligible door with prob q
        q = lazy_q / 100
        lowest = np.full(len(CAR), 10 ** 6)
        for x in sorted(closed, reverse=True):
            lowest = np.where((CAR != x) & (BK != x), x, lowest)
        p_lo = np.where(elig == 1, 1.0, q)
        p_other = np.where(elig > 1, (1 - q) / np.maximum(elig - 1, 1), 0.0)
        return np.where(is_elig, np.where(lowest == d, p_lo, p_other), 0.0)

    def render(k, o):
        return ftp[k](o[1]) if o[0] == "friend" else htp[k](o[1], o[2])

    rows_keep = prior > 0
    assert rows_keep.all()
    return mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, "car_door", False,
                    "json" if js else "prose", "gameshow",
                    {"Nd": Nd, "w": w, "bike": bike, "pending": pending, "fa": fa, "host": host, "lazy_q": lazy_q,
                     "kinds": kinds, "pick": pick_door, "assign": assign})


def simulate(world, rng):
    p = world.params
    Nd = p["Nd"]
    w = np.array(p["w"], float)
    car = int(rng.choice(Nd, p=w / w.sum()))
    bk = int(rng.choice([d for d in range(Nd) if d != car])) if p["bike"] else None

    def signal():
        if rng.random() < p["fa"] / 100:
            return car
        return int(rng.choice([d for d in range(Nd) if d != car]))

    pend = signal() if p["pending"] else None
    obs, op = [], []
    for kind in p["kinds"]:
        if kind == "friend":
            obs.append(("friend", signal()))
            continue
        closed = [d for d in range(Nd) if d != p["pick"] and d not in op]
        if p["host"] == "ignorant":
            d = int(rng.choice(closed))
        else:
            el = [d for d in closed if d != car and d != bk]
            if p["host"] == "knows" or len(el) == 1:
                d = int(rng.choice(el))
            elif rng.random() < p["lazy_q"] / 100:
                d = min(el)
            else:
                d = int(rng.choice([x for x in el if x != min(el)]))
        op.append(d)
        obs.append(("host", d, 0 if d == car else (1 if d == bk else 2)))
    return p["assign"](car, bk, pend), obs
