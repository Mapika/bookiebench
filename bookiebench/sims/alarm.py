"""Burglary/earthquake -> alarm -> reports network (classic explaining-away BN), several surface themes.

Reports are conditionally independent given (cause1, cause2, alarm): exchangeable.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, Fmt, a_an, cap, join_list, json_block, json_line, people, person, pick, rand_pct
from .world import Var, World

THEMES = [
    dict(key="home", place="{owner}'s house",
         c1=("burglary", "a burglary happened", "no burglary happened", "burglary"),
         c2=("earthquake", "a small earthquake struck", "no earthquake struck", "earthquake"),
         al=("the burglar alarm went off", "the burglar alarm stayed silent", "burglar alarm"), ev="the burglar alarm goes off",
         rep_yes="{r} calls {owner} to say the alarm is ringing.", rep_no="{r} does not call {owner}.",
         rep_desc="neighbour {r} calls {owner}", rep_role="neighbour",
         news=("the local radio", "The local radio reports an earthquake in the area.",
               "The local radio reports no earthquake.")),
    dict(key="office", place="the {owner} server room",
         c1=("intrusion", "an intruder entered the server room", "no intruder entered", "intruder"),
         c2=("power_surge", "a power surge occurred", "no power surge occurred", "power surge"),
         al=("the security alarm was triggered", "the security alarm was not triggered", "security alarm"),
         ev="the security alarm is triggered",
         rep_yes="{r} messages the on-call engineer that an alarm is sounding.", rep_no="{r} sends no message.",
         rep_desc="{r} (a night-shift worker) messages the on-call engineer", rep_role="night-shift worker",
         news=("the utility company's status page", "The utility company's status page lists a power surge.",
               "The utility company's status page lists no power surge.")),
    dict(key="farm", place="{owner}'s farm",
         c1=("fox", "a fox got into the henhouse", "no fox got into the henhouse", "fox in the henhouse"),
         c2=("storm", "a thunderstorm passed over", "no thunderstorm passed over", "thunderstorm"),
         al=("the farm dogs started barking", "the farm dogs stayed quiet", "barking"), ev="the farm dogs start barking",
         rep_yes="{r} phones {owner} about the barking.", rep_no="{r} does not phone {owner}.",
         rep_desc="neighbour {r} phones {owner} about barking dogs", rep_role="neighbour",
         news=("the weather service", "The weather service confirms a thunderstorm passed over.",
               "The weather service reports no thunderstorm.")),
]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    owner = person(rng) if th["key"] != "office" else pick(rng, ["Halvard", "Zentek", "Norvik", "Octavo", "Luma"])
    place = th["place"].format(owner=owner)
    p1 = rand_pct(rng, 2, 30, 1)
    p2 = rand_pct(rng, 2, 30, 1)
    pa = {(1, 1): rand_pct(rng, 85, 99, 1), (1, 0): rand_pct(rng, 60, 95, 5), (0, 1): rand_pct(rng, 15, 60, 5),
          (0, 0): rand_pct(rng, 1, 10, 1)}
    n_rep = int(rng.integers(2, 5))
    has_news = rng.random() < 0.6
    if not has_news:
        n_rep = max(n_rep, 3)
    has_pend = rng.random() < 0.5
    reps = people(rng, n_rep + 2)
    reps = [r for r in reps if r != owner][:n_rep + int(has_pend)]
    rep_p = [(rand_pct(rng, 50, 95, 5), rand_pct(rng, 1, 25, 1)) for _ in range(len(reps))]
    news_p = (rand_pct(rng, 60, 95, 5), rand_pct(rng, 1, 10, 1))
    sources = [("rep", i) for i in range(n_rep)] + ([("news", 0)] if has_news else [])
    order = rng.permutation(len(sources))
    sources = [sources[int(i)] for i in order]
    T = len(sources)

    Z = np.array(list(itertools.product(*[range(2)] * (4 if has_pend else 3))))  # option 0 = yes
    c1y = Z[:, 0] == 0
    c2y = Z[:, 1] == 0
    ay = Z[:, 2] == 0
    pa_z = np.array([pa[(int(a), int(b))] for a, b in zip(c1y, c2y)]) / 100
    prior = (np.where(c1y, p1, 100 - p1) / 100) * (np.where(c2y, p2, 100 - p2) / 100) * np.where(ay, pa_z, 1 - pa_z)
    if has_pend:
        hp, fp = rep_p[n_rep]
        pc = np.where(ay, hp, fp) / 100
        prior = prior * np.where(Z[:, 3] == 0, pc, 1 - pc)

    def lik_src(src):
        kind, i = src
        if kind == "rep":
            hit, fa = rep_p[i]
            p = np.where(ay, hit, fa) / 100
        else:
            hit, fa = news_p
            p = np.where(c2y, hit, fa) / 100
        return np.stack([p, 1 - p], 1)

    L = [lik_src(s) for s in sources]

    c1n, c2n, aln = th["c1"][3], th["c2"][3], th["al"][2]
    Ac1n, Ac2n = a_an(c1n), a_an(c2n)
    json_style = rng.random() < 0.3
    if json_style:
        state = {"location": place, f"P({c1n})": fmt.p(p1), f"P({c2n})": fmt.p(p2),
                 f"P({aln} | cause)": {f"{c1n} and {c2n}": fmt.p(pa[(1, 1)]), f"{c1n} only": fmt.p(pa[(1, 0)]),
                                       f"{c2n} only": fmt.p(pa[(0, 1)]), "neither": fmt.p(pa[(0, 0)])},
                 "reporters": {r: {f"P(report | {aln})": fmt.p(h), f"P(report | no {aln})": fmt.p(f)}
                               for r, (h, f) in zip(reps, rep_p)}}
        if has_news:
            state[th["news"][0]] = {f"P(report | {c2n})": fmt.p(news_p[0]), f"P(report | no {c2n})": fmt.p(news_p[1])}
        state["independence"] = "reports are independent given the causes and the alarm"
        if get_tv() >= 2:
            state["cause_independence"] = f"{c1n} and {c2n} occur independently of each other"
            if has_news:
                state[th["news"][0]]["otherwise"] = f"reports that there was no {c2n}"
        if has_pend:
            state["not_yet_heard_from"] = reps[n_rep]
        prelude = json_block(rng, state)

        def render(k, o):
            kind, i = sources[o[0]]
            who = reps[i] if kind == "rep" else th["news"][0]
            return json_line({"source": who, "reported": bool(o[1] == 0)})
    else:
        s_pri = (f"On any given night, the chance of {Ac1n} at {place} is {fmt.p(p1)} and, independently, the chance of "
                 f"{Ac2n} is {fmt.p(p2)}.")
        s_al = (f"{cap(th['ev'])} with probability {fmt.p(pa[(1, 1)])} if there is both {Ac1n} and {Ac2n}, "
                f"{fmt.p(pa[(1, 0)])} with only {Ac1n}, {fmt.p(pa[(0, 1)])} with only {Ac2n}, and {fmt.p(pa[(0, 0)])} "
                f"with neither.")
        s_rep = " ".join(
            f"If {th['ev']}, {th['rep_desc'].format(r=r, owner=owner)} with probability {fmt.p(h)}; otherwise "
            f"{r} still does so with probability {fmt.p(f)}." for r, (h, f) in zip(reps, rep_p))
        s_news = (f" {cap(th['news'][0])} reports {Ac2n} with probability {fmt.p(news_p[0])} when there was one and "
                  f"{fmt.p(news_p[1])} when there was not." if has_news else "")
        if has_news and get_tv() >= 2:
            s_news += f" If it does not report {Ac2n}, it reports that there was no {c2n}."
        tail = " All reports are independent of one another given what actually happened."
        if has_pend:
            tail += f" {reps[n_rep]} has not been heard from yet."
        prelude = pick(rng, [
            f"Consider {place}. {s_pri} {s_al} {s_rep}{s_news}{tail}",
            f"{s_pri} {s_al} {s_rep}{s_news}{tail} Tonight, the following reports come in.",
            f"This is about one night at {place}. {s_pri} {s_al} {s_rep}{s_news}{tail}",
        ])

        def render(k, o):
            kind, i = sources[o[0]]
            if kind == "rep":
                return (th["rep_yes"] if o[1] == 0 else th["rep_no"]).format(r=reps[i], owner=owner)
            return th["news"][1] if o[1] == 0 else th["news"][2]

    c1, c2, al = th["c1"], th["c2"], th["al"]
    TV = get_tv()
    variables = [
        Var(c1[0], [f"yes, {c1n}", f"no {c1n}"], [f"Was there {Ac1n} tonight?", f"Did {Ac1n} happen tonight?" if TV < 2 else f"Tonight, was there {Ac1n}?"],
            [c1[1], c1[2]]),
        Var(c2[0], [f"yes, {c2n}", f"no {c2n}"], [f"Was there {Ac2n} tonight?", f"Did {Ac2n} occur tonight?" if TV < 2 else f"Tonight, was there {Ac2n}?"],
            [c2[1], c2[2]]),
        Var("alarm", ["occurred", "did not occur"], [f"Did the {aln} occur?", f"Was there {aln} tonight?"]
            if aln == "barking" else [f"Did the {aln} go off?", f"Was the {aln} activated tonight?"],
            [al[0], al[1]]),
    ]
    if has_pend:
        pr = reps[n_rep]
        yes = th["rep_yes"].format(r=pr, owner=owner).rstrip(".")
        no = th["rep_no"].format(r=pr, owner=owner).rstrip(".")
        variables.append(Var("pending_report", ["reports", "does not report"],
                             [f"Will {pr} report tonight?", f"Is {pr} going to get in touch tonight?"],
                             [yes, no]))
    return World(variables, prior, Z, T,
                 alternatives=lambda k, past: [(k, 0), (k, 1)],
                 lik=lambda o, past: L[o[0]][:, o[1]],
                 render=render, prelude=prelude, mart_var=c1[0], exchangeable=True,
                 meta={"style": "json" if json_style else "prose", "theme": th["key"]})
