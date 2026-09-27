"""Knockout tournament under a Bradley-Terry (or Elo) model with latent team form.

Latent = (form of every team, semifinal winners, champion). Match win probabilities follow BT, P(i beats j) =
s_i / (s_i + s_j), or Elo, 1 / (1 + 10^((R_j - R_i)/400)), with the rating determined by the team's (hidden) form.
Evidence: warm-up friendlies (independent given forms) and semifinal results (deterministic given the latent);
conditionally independent given the latent, so exchangeable. Variables: champion, a semifinal winner, one team's form.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, repeat_tags, tag_text, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps,
                    per_step, pick)

NAME = "tournament"
SPORTS = [("curling", "rink"), ("water polo", "club"), ("chess", "team"), ("hurling", "county side"),
          ("robot combat", "crew"), ("quiz", "squad")]


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    sport, unit = pick(rng, SPORTS)
    nt = 4
    teams = [f"{n} {pick(rng, ['United', 'Rovers', 'Athletic', 'Wanderers', 'Comets', 'Owls'])}" if rng.random() < 0.4
             else n for n in distinct_names(rng, nt)]
    nf = {0: 1, 1: 2, 2: pick(rng, [2, 3])}[level]
    model = "elo" if (level == 2 and rng.random() < 0.4) else "bt"
    fnames = {1: ["steady"], 2: ["off form", "in form"], 3: ["poor form", "normal form", "peak form"]}[nf]
    fprior = [100] if nf == 1 else [int(x) for x in rng.choice(np.arange(20, 85, 5), size=1)] if nf == 2 else None
    if nf == 2:
        fprior = [100 - fprior[0], fprior[0]]
    elif nf == 3:
        a = int(rng.choice(np.arange(10, 40, 5)))
        b = int(rng.choice(np.arange(10, 40, 5)))
        fprior = [a, 100 - a - b, b]
    rating = []
    for t in range(nt):
        if model == "bt":
            base = int(rng.integers(1, 31))
            r = sorted(int(x) for x in rng.choice(np.arange(1, 31), size=nf, replace=False)) if nf > 1 else [base]
        else:
            r = sorted(int(x) for x in rng.choice(np.arange(1300, 1810, 10), size=nf, replace=False))
        rating.append(r)
    rating = np.array(rating)

    def pwin(ri, rj):
        if model == "bt":
            return ri / (ri + rj)
        return 1.0 / (1.0 + 10 ** ((rj - ri) / 400.0))

    rows = []
    for forms in itertools.product(range(nf), repeat=nt):
        pf = np.prod([fprior[f] / 100 for f in forms])
        R = [rating[t, forms[t]] for t in range(nt)]
        for w1 in (0, 1):
            for w2 in (2, 3):
                p1 = pwin(R[w1], R[1 - w1])
                p2 = pwin(R[w2], R[5 - w2])
                for ch in (w1, w2):
                    pc = pwin(R[ch], R[w1 + w2 - ch])
                    rows.append((forms, w1, w2, ch, pf * p1 * p2 * pc))
    F = np.array([r[0] for r in rows])
    W1 = np.array([r[1] for r in rows])
    W2 = np.array([r[2] for r in rows])
    CH = np.array([r[3] for r in rows])
    prior = np.array([r[4] for r in rows])
    Rz = rating[np.arange(nt)[None, :], F]  # rows x teams

    fteam = int(rng.integers(nt))
    vs = ["champion", "semi1"] + (["form"] if nf > 1 else [])
    cols = [CH, W1] + ([F[:, fteam]] if nf > 1 else [])
    proj = np.stack(cols, 1)
    variables = [Var("champion", list(teams), ["Who will win the tournament?", "Which side lifts the trophy?",
                                                "Who becomes champion?"],
                     [f"{t} win{'s' if not t.endswith('s') else ''} the tournament" for t in teams]),
                 Var("semifinal_1", [teams[0], teams[1]], [f"Who wins the semifinal between {teams[0]} and {teams[1]}?",
                                                           f"Which of {teams[0]} and {teams[1]} reaches the final?"],
                     [f"{teams[0]} beat {teams[1]} in their semifinal", f"{teams[1]} beat {teams[0]} in their semifinal"])]
    if nf > 1:
        variables.append(Var(f"form_{fteam}", list(fnames), [f"What form is {teams[fteam]} in?",
                                                              f"Which form state is {teams[fteam]} in this season?"],
                             [f"{teams[fteam]} are {fst(fn)}" for fn in fnames]))

    T = n_steps(rng, level, 3, 6)
    ev = []
    semis_left = [0, 1]
    for k in range(T):
        if level == 0 or (semis_left and rng.random() < 0.25 and k >= T - 2):
            if not semis_left:
                break
            ev.append(("semi", semis_left.pop(int(rng.integers(len(semis_left))))))
        else:
            i, j = (int(x) for x in rng.choice(nt, size=2, replace=False))
            ev.append(("friendly", min(i, j), max(i, j)))
    T = len(ev)

    def rtxt(t):
        if nf == 1:
            return f"{teams[t]} {fmt_r(rating[t, 0])}"
        return f"{teams[t]} " + ", ".join(f"{fmt_r(rating[t, f])} when {fst(fnames[f])}"
                                          for f in range(nf))

    def fmt_r(r):
        return f"rating {r}"

    rule = ("the chance that one side beats another equals its rating divided by the sum of the two ratings"
            if model == "bt" else "the chance that a side with rating A beats a side with rating B is "
            "1 / (1 + 10^((B - A) / 400)) (the Elo formula)")
    formtxt = ("" if nf == 1 else
               f"Each side is, independently and for the whole competition, " +
               join_list([f"{fst(fn)} with probability {fmt.p(p)}" for fn, p in zip(fnames, fprior)]) + ".")
    js = is_json(rng)
    if js:
        st = {"competition": f"{sport} knockout", "semifinals": [f"{teams[0]} v {teams[1]}", f"{teams[2]} v {teams[3]}"],
              "final": "the two semifinal winners", "ratings": {teams[t]: ({fnames[f]: int(rating[t, f]) for f in range(nf)}
                                                                          if nf > 1 else int(rating[t, 0])) for t in range(nt)},
              "win_model": rule, "matches": "independent given ratings, no draws"}
        if nf > 1:
            st["form_prior"] = {fn: fmt.p(p) for fn, p in zip(fnames, fprior)}
            st["friendlies"] = "warm-up games played beforehand with the same forms"
            st["forms"] = "drawn independently for each side, fixed for the whole competition"
        prelude = json_block(rng, st)
        tf = [lambda w, l: json_line({"friendly": {"winner": teams[w], "loser": teams[l]}}),
              lambda w, l: json_line({"warmup_result": f"{teams[w]} beat {teams[l]}"})]
        ts = [lambda w, l: json_line({"semifinal": {"winner": teams[w], "loser": teams[l]}}),
              lambda w, l: json_line({"result": "semifinal", "winner": teams[w], "beaten": teams[l]})]
    else:
        s_b = (f"Four {unit}s contest the {pseudo_cup(rng)} {sport} cup: {teams[0]} play {teams[1]} and {teams[2]} play "
               f"{teams[3]} in the semifinals, and the winners meet in the final. There are no draws.")
        s_r = "Ratings: " + "; ".join(rtxt(t) for t in range(nt)) + f". In any match, {rule}, independently of other matches."
        s_f = formtxt + (" Before the cup, some warm-up friendlies are played (repeat pairings are labelled match A, match B, ...); "
                         "forms do not change in between." if nf > 1 else "")
        prelude = pick(rng, [" ".join(x for x in o if x.strip()) for o in [(s_b, s_r, s_f), (s_r, s_b, s_f), (s_b, s_f, s_r)]])
        tf = [lambda w, l: f"In a warm-up friendly, {teams[w]} beat {teams[l]}.",
              lambda w, l: f"Friendly result: {teams[w]} defeated {teams[l]}.",
              lambda w, l: f"{teams[l]} lost a friendly to {teams[w]}."]
        ts = [lambda w, l: f"Semifinal: {teams[w]} knocked out {teams[l]}.",
              lambda w, l: f"{teams[w]} won their semifinal against {teams[l]}.",
              lambda w, l: f"{teams[l]} are out, beaten by {teams[w]} in the semifinal."]
    ptf, pts = per_step(rng, T, tf), per_step(rng, T, ts)
    ftag = [f"match {t}" if t else "" for t in repeat_tags(ev)]

    def alternatives(k, past):
        e = ev[k]
        if e[0] == "friendly":
            return [(k, e[1]), (k, e[2])]
        a = 0 if e[1] == 0 else 2
        return [(k, a), (k, a + 1)]

    def lik(o, past):
        k, w = o
        e = ev[k]
        if e[0] == "semi":
            return ((W1 if e[1] == 0 else W2) == w).astype(float)
        l = e[1] + e[2] - w
        return pwin(Rz[:, w].astype(float), Rz[:, l].astype(float))

    def render(k, o):
        k, w = o
        e = ev[k]
        l = (e[1] + e[2] - w) if e[0] == "friendly" else ((1 - w) if e[1] == 0 else (5 - w))
        return tag_text((ptf if e[0] == "friendly" else pts)[k](w, l), ftag[k])

    return mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, "champion", True,
                    "json" if js else "prose", sport, {"nf": nf, "fprior": fprior, "rating": rating, "model": model,
                                                       "ev": ev, "fteam": fteam}, model=model)


def fst(fn):
    return fn if fn in ("off form", "in form") else f"in {fn}"


def pseudo_cup(rng):
    return pick(rng, ["Harwick", "Solenne", "Varro", "Quillon", "Brask", "Ederly"])


def simulate(world, rng):
    p = world.params
    nf, rating = p["nf"], p["rating"]
    forms = [int(rng.choice(nf, p=np.array(p["fprior"]) / 100)) for _ in range(4)]
    R = [rating[t, forms[t]] for t in range(4)]

    def pw(i, j):
        return R[i] / (R[i] + R[j]) if p["model"] == "bt" else 1 / (1 + 10 ** ((R[j] - R[i]) / 400))

    def play(i, j):
        return i if rng.random() < pw(i, j) else j

    obs = []
    semis = {}
    for k, e in enumerate(p["ev"]):
        if e[0] == "friendly":
            obs.append((k, play(e[1], e[2])))
        else:
            obs.append((k, None))
    w1, w2 = play(0, 1), play(2, 3)
    ch = play(w1, w2)
    for k, e in enumerate(p["ev"]):
        if e[0] == "semi":
            obs[k] = (k, w1 if e[1] == 0 else w2)
    a = [ch, w1] + ([forms[p["fteam"]]] if nf > 1 else [])
    return tuple(a), obs
