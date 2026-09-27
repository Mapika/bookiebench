"""Matching problems (hat-check / derangements): n items are handed back to n owners -- with a stated probability
everything goes back correctly, otherwise in a uniformly random order.

Latent = the permutation (n! rows). Evidence: people's reports (own item? whose item? do you hold X's?), each wrong
with a stated probability; conditionally independent given the permutation, so exchangeable.
Variables: the number of people who got their own item back, whether a named person did, and (L1+) who holds a
named person's item.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import Fmt, Var, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, people, per_step, pick

NAME = "matching"
THEMES = [
    dict(key="hats", item="hat", items="hats", setup="{names} check their hats at a party, and the attendant loses the "
         "tickets.", care="the attendant still remembers every face and returns each hat to its owner",
         got="{a} got {b}'s hat", own="{a} got their own hat back"),
    dict(key="coats", item="coat", items="coats", setup="After a concert, the cloakroom at the {x} Hall has to return "
         "the {n} coats of {names} after the numbered tags got mixed up.",
         care="the cloakroom staff sort it out and every coat goes back to its owner",
         got="{a} was handed {b}'s coat", own="{a} received their own coat"),
    dict(key="letters", item="letter", items="letters", setup="A clerk puts {n} letters, addressed to {names}, into "
         "the {n} addressed envelopes, and each person receives the letter in their envelope.",
         care="the clerk is careful and every letter goes into its own envelope",
         got="{a} received the letter meant for {b}", own="{a} received their own letter"),
    dict(key="santa", item="name slip", items="name slips", setup="For a gift swap, {names} each write their name "
         "on a slip, and the {n} slips are dealt out one per person (drawing your own name is allowed).",
         care="the organiser forgets to shuffle and everyone is dealt their own slip",
         got="{a} drew {b}'s name", own="{a} drew their own name"),
]


def make_world(rng, level, scale=1):
    th = pick(rng, THEMES)
    n = {0: 3, 1: 4, 2: int(rng.integers(4, 6))}[level] + (scale - 1)
    names = people(rng, n)
    X, Y = 0, 1
    T = n_steps(rng, level, 3, 5)
    for _try in range(50):  # deterministic facts are never repeated
        ev, done = [], set()
        for _ in range(T):
            kd = pick(rng, ["own", "own", "got", "check"] if level >= 1 else ["own", "check"])
            free = [i for i in range(1, n) if i not in done]
            if kd == "own":
                ev.append(("own", int(pick(rng, free)), None))
            elif kd == "got" and [i for i in free if i >= 2]:
                i = int(pick(rng, [i for i in free if i >= 2]))
                done.add(i)
                ev.append(("got", i, None))
            else:
                i = int(pick(rng, free))
                j = int(pick(rng, [x for x in range(n) if x != i]))
                ev.append(("check", i, j))
        owns = {e[1] for e in ev if e[0] == "own"}
        gots = {e[1] for e in ev if e[0] == "got"}
        # a "got" reveal of someone's own item reads like an "own" report, so never pair the two for one person
        if len(set(ev)) == len(ev) and not owns & gots:
            break
    rows = np.array(list(itertools.permutations(range(n))))  # rows[:, i] = whose item person i holds
    fixed = (rows == np.arange(n)).sum(1)
    # mixture prior: with probability q everything goes back to its owner, otherwise a uniform random matching
    q = int(rng.integers(3, 61))
    err = int(rng.integers(2, 26))  # own-item reports are wrong with this probability
    prior = (1 - q / 100) / len(rows) + (q / 100) * (fixed == n)
    possible = sorted(set(int(f) for f in fixed))
    if n >= 4:
        cats = [0, 1, 2, 3]
        fcol = np.minimum(fixed, 3)
        words = ["nobody", "exactly one person", "exactly two people", "three or more people"]
    else:
        cats = possible
        fcol = np.searchsorted(np.array(cats), fixed)
        words = {0: "nobody", 1: "exactly one person", 3: "all three people"}
        words = [words[c] for c in cats]
    vs = ["fixed", "own_x"] + (["holder_y"] if level >= 1 else [])
    cols = [fcol, (rows[:, X] != X).astype(int)]
    if level >= 1:
        cols.append(np.argmax(rows == Y, 1))
    proj = np.stack(cols, 1)
    it, its = th["item"], th["items"]
    variables = [
        Var("n_own", words, [f"How many people end up with their own {it}?",
                             f"For how many of the {n} people does the {it} match?",
                             f"How many of them got their own {it}?"],
            [f"{w} got their own {it}" for w in words]),
        Var(f"{names[X]}_own", ["yes", "no"], [f"Did {names[X]} get their own {it}?",
                                               f"Is the {it} {names[X]} got their own?"],
            [th["own"].format(a=names[X]), f"{names[X]} did not get their own {it}"]),
    ]
    if level >= 1:
        variables.append(Var(f"holder_of_{names[Y]}", list(names),
                             [f"Who ended up with {names[Y]}'s {it}?", f"Which person holds {names[Y]}'s {it}?"],
                             [th["got"].format(a=a, b=names[Y]) if a != names[Y] else th["own"].format(a=a) for a in names]))

    js = is_json(rng)
    setup = th["setup"].format(names=join_list(names), n=n, x=pick(rng, ["Orsolya", "Vantor", "Kestrel", "Lumen"]))
    fmt = Fmt(rng)
    mix = (f"With probability {fmt.p(q)} {th['care']}; otherwise the {its} are handed out in a completely random order, "
           f"every matching of {its} to people equally likely.")
    rtxt = (f"The {its} look alike, so every report people make is wrong with probability {fmt.p(err)}, independently: "
            f"a yes/no answer (is it your own {it}? do you hold a given person's {it}?) is flipped, and a person naming "
            f"whose {it} they hold names one of the other {n - 1} people uniformly at random instead.")
    setup = f"{setup} {mix} {rtxt}"
    if js:
        prelude = json_block(rng, {"people": names, "items": its,
                                   "assignment": (f"with probability {fmt.p(q)} everyone gets their own {it}; otherwise "
                                                  f"a uniformly random matching (all n! equally likely)"),
                                   "reports": (f"every report is wrong with probability {fmt.p(err)}, independently: yes/no "
                                               f"answers (own item? holds a given person's item?) are flipped; a wrong "
                                               f"'whose item' answer (asked_whose_item_they_hold_names, "
                                               f"names_owner_of_held_item) names one of the other {n - 1} people uniformly"),
                                   "story": setup})
        to = [lambda i, v: json_line({"report_by": names[i], "says_own_" + it.replace(' ', '_'): bool(v)}),
              lambda i, v: json_line({"own_item_report": names[i], "says": "own" if v else "not own"})]
        tg = [lambda i, v: json_line({"person": names[i], "asked_whose_item_they_hold_names": names[v]}),
              lambda i, v: json_line({"person": names[i], "names_owner_of_held_item": names[v]})]
        tc = [lambda i, j, v: json_line({"question": f"does {names[i]} hold {names[j]}'s {it}", "answer_by_" + names[i]: "yes" if v else "no"}),
              lambda i, j, v: json_line({"person": names[i], "says_has_" + names[j] + "_item": bool(v)})]
    else:
        prelude = pick(rng, [setup, f"{setup} Nobody swaps afterwards.",
                             f"Setup: {setup} Some of them then report on what they got."])
        to = [lambda i, v: f"{names[i]} says the {it} they got {'is' if v else 'is not'} their own.",
              lambda i, v: f"{names[i]} reports: {'my own' if v else 'not my own'} {it}.",
              lambda i, v: f"{names[i]} reports that the {it} {'is' if v else 'is not'} their own."]
        tg = [lambda i, v: f"{names[i]} names {names[v]} as the owner of the {it} they hold.",
              lambda i, v: f"{names[i]} looks at the {it} they hold and names {names[v]} as its owner.",
              lambda i, v: f"Asked whose {it} they hold, {names[i]} names {names[v]}."]
        tc = [lambda i, j, v: f"{names[i]} says they {'do' if v else 'do not'} hold {names[j]}'s {it}.",
              lambda i, j, v: f"Asked whether they have {names[j]}'s {it}, {names[i]} says {'yes' if v else 'no'}.",
              lambda i, j, v: f"{names[i]} reports that {names[j]}'s {it} {'is' if v else 'is not'} the one they have."]
    pto, ptg, ptc = per_step(rng, T, to), per_step(rng, T, tg), per_step(rng, T, tc)

    def alternatives(k, past):
        return [(k, v) for v in range(n)] if ev[k][0] == "got" else [(k, 1), (k, 0)]

    def lik(o, past):
        k, v = o
        kd, i, j = ev[k]
        if kd == "own":  # noisy report
            says_own = np.where(rows[:, i] == i, 1 - err / 100, err / 100)
            return says_own if v else 1 - says_own
        if kd == "got":  # noisy naming of whose item i holds
            return np.where(rows[:, i] == v, 1 - err / 100, err / 100 / (n - 1))
        says_yes = np.where(rows[:, i] == j, 1 - err / 100, err / 100)
        return says_yes if v else 1 - says_yes

    def render(k, o):
        _, v = o
        kd, i, j = ev[k]
        if kd == "got" and v == i:  # naming oneself: worded apart from the yes/no own-item reports
            return (json_line({"person": names[i], "names_owner_of_held_item": names[i]}) if js else
                    f"Asked whose {it} they hold, {names[i]} names themself.")
        return pto[k](i, v) if kd == "own" else (ptg[k](i, v) if kd == "got" else ptc[k](i, j, v))

    return mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, "n_own", True,
                    "json" if js else "prose", th["key"], {"n": n, "ev": ev, "level": level, "cats": cats, "q": q, "err": err})


def simulate(world, rng):
    p = world.params
    n = p["n"]
    s = np.arange(n) if rng.random() < p["q"] / 100 else rng.permutation(n)
    f = int((s == np.arange(n)).sum())
    a = [min(f, 3) if n >= 4 else p["cats"].index(f), int(s[0] != 0)]
    if p["level"] >= 1:
        a.append(int(np.argmax(s == 1)))
    obs = []
    for k, (kd, i, j) in enumerate(p["ev"]):
        if kd == "own":
            own = s[i] == i
            obs.append((k, int(own if rng.random() >= p["err"] / 100 else not own)))
        elif kd == "got":
            wrong = rng.random() < p["err"] / 100
            obs.append((k, int(s[i]) if not wrong else int(rng.choice([x for x in range(n) if x != s[i]]))))
        else:
            yes = s[i] == j
            obs.append((k, int(yes if rng.random() >= p["err"] / 100 else not yes)))
    return tuple(a), obs
