"""Unreliable witnesses with stated reliabilities (taxi-cab-problem style), reliability depends on visibility.

Each witness names one option; correct with the stated accuracy for the (latent) visibility, otherwise names one of
the other options uniformly. Statements are conditionally independent given the truth and visibility, so the
evidence is exchangeable. A further witness who has not yet spoken is a predictive variable.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, Fmt, cap, composition, join_list, json_block, json_line, people, pick, rand_pct, sample
from .world import Var, World

THEMES = [
    dict(key="taxi", event="Late last night a taxi clipped a cyclist and drove off.",
         opts_pool=["Blue Cab", "Green Cab", "Yellow Cab", "Red Line Taxis"], base="{p} of the city's taxis belong to "
         "{o}", q="Which company's taxi was involved?", q2="Which taxi company was responsible?", clause="the taxi belonged to {opt}",
         says="{w} says the taxi was from {o}.", vis=("clear", "foggy", "in clear weather", "in fog"), wrole="witness", wroles="witnesses"),
    dict(key="bird", event="A rare bird was briefly seen at the {town} marsh this morning.",
         opts_pool=["a grey heron", "a great egret", "a white stork", "a common crane"], base="{p} of large wading "
         "birds seen at this marsh are {o_bare}s", q="Which species was the bird?", q2="What kind of bird was it?", clause="the bird was {opt}",
         says="{w} reports that the bird was {o}.", vis=("good light", "poor light", "in good light", "in poor light"), wrole="birdwatcher",
         wroles="birdwatchers"),
    dict(key="jacket", event="A bicycle was stolen outside the {town} library.",
         opts_pool=["a red jacket", "a black jacket", "a green jacket", "a yellow jacket"],
         base="{p} of people matching the suspect's description in CCTV archives wear {o}", q="What was the thief wearing?",
         q2="What kind of jacket did the thief have on?", clause="the thief wore {opt}", says="{w} says the thief wore {o}.", vis=("daylight", "dusk", "in daylight", "at dusk"),
         wrole="passer-by", wroles="passers-by"),
]
TOWNS = ["Millbrook", "Redfield", "Oakhaven", "Larkhill", "Westmere", "Pinecrest"]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    town = pick(rng, TOWNS)
    K = int(rng.integers(2, 4))
    opts = sample(rng, th["opts_pool"], K)
    base = [x * 5 for x in composition(rng, K, 20, min_each=1)]
    pv = rand_pct(rng, 10, 60, 5)  # P(poor visibility)
    vgood, vpoor, in_good, in_poor = th["vis"]
    n_w = int(rng.integers(2, 5))
    has_log = rng.random() < 0.5
    if not has_log:
        n_w = max(n_w, 3)
    names = people(rng, n_w + 1)
    wit, pend = names[:n_w], names[n_w]
    acc = []
    for _ in range(n_w + 1):
        g = rand_pct(rng, 60, 95, 5)
        p = rand_pct(rng, max(40, 100 // K + 5), g, 5) if g > max(40, 100 // K + 5) else g
        acc.append((g, p))
    log_acc = rand_pct(rng, 70, 95, 5)
    sources = [("w", i) for i in range(n_w)] + ([("log", 0)] if has_log else [])
    sources = [sources[int(i)] for i in rng.permutation(len(sources))]
    T = len(sources)

    Z = np.array(list(itertools.product(range(K), range(2), range(K))))
    t, v, s = Z[:, 0], Z[:, 1], Z[:, 2]

    def say_prob(i, said):
        a = np.array(acc[i], float)[v] / 100
        return np.where(said == t, a, (1 - a) / (K - 1))

    prior = (np.array(base, float)[t] / 100) * np.where(v == 1, pv, 100 - pv) / 100 * say_prob(n_w, s)
    Ls = []
    for kind, i in sources:
        if kind == "w":
            Ls.append(np.stack([say_prob(i, np.full_like(t, o)) for o in range(K)], 1))
        else:
            a = log_acc / 100
            Ls.append(np.stack([np.where(v == 0, a, 1 - a), np.where(v == 1, a, 1 - a)], 1))

    def bare(o):
        return o.split(" ", 1)[1] if o.startswith(("a ", "an ")) else o

    event = th["event"].format(town=town)
    json_style = rng.random() < 0.3
    if json_style:
        st = {"event": event, "conditions": [vgood, vpoor], "base_rates": {o: fmt.p(b) for o, b in zip(opts, base)},
              "P(" + vpoor + ")": fmt.p(pv),
              "witnesses": {w: {f"accuracy in {vgood}": fmt.p(acc[i][0]), f"accuracy in {vpoor}": fmt.p(acc[i][1])}
                            for i, w in enumerate(wit + [pend])},
              "error_model": "a mistaken witness names one of the other options uniformly at random",
              "not_yet_interviewed": pend}
        if has_log:
            st["conditions_log"] = f"records the conditions correctly with probability {fmt.p(log_acc)}"
        if get_tv() >= 2:
            st["independence"] = "all statements are independent given what happened and the conditions"
        prelude = json_block(rng, st)

        def render(k, o):
            kind, i = sources[o[0]]
            if kind == "w":
                return json_line({"witness": wit[i], "says": opts[o[1]]})
            return json_line({"conditions_log": [vgood, vpoor][o[1]]})
    else:
        s_base = cap(join_list([th["base"].format(p=fmt.p(b), o=o, o_bare=bare(o)) for o, b in zip(opts, base)])) + "."
        s_vis = f"At that time of day there is a {fmt.p(pv)} chance the conditions were {vpoor} rather than {vgood}."
        s_w = " ".join(
            f"{w} identifies correctly {fmt.p(g)} of the time {in_good} but only {fmt.p(p)} {in_poor}."
            if p != g else f"{w} identifies correctly {fmt.p(g)} of the time regardless of conditions."
            for w, (g, p) in zip(wit + [pend], acc))
        s_err = (f"When a {th['wrole']} is wrong, they name one of the other possibilities at random. "
                 f"{pend} has not been interviewed yet.")
        s_log = (f" A conditions log records whether it was {vgood} or {vpoor}; it is correct "
                 f"{fmt.p(log_acc)} of the time." if has_log else "")
        prelude = pick(rng, [
            f"{event} {s_base} {s_vis} {s_w} {s_err}{s_log} Statements are independent given what happened.",
            f"{event} The {th['wroles']} are {join_list(wit + [pend])}. {s_w} {s_err} {s_base} {s_vis}{s_log}",
            f"Case notes. {event} Background: {s_base} {s_vis} Reliability: {s_w} {s_err}{s_log}",
        ])
        if get_tv() >= 2 and "Statements are independent" not in prelude:
            prelude += " Statements are independent given what happened."

        def render(k, o):
            kind, i = sources[o[0]]
            if kind == "w":
                return th["says"].format(w=wit[i], o=opts[o[1]])
            return f"The conditions log says it was {[vgood, vpoor][o[1]]}."

    variables = [
        Var("truth", opts, [th["q"], th["q2"]], th["clause"]),
        Var("conditions", [vgood, vpoor], ["What were the conditions at the time?", f"Was it {vgood} or {vpoor}?"],
            "the conditions were {opt}"),
        Var("pending_witness", opts, [f"What will {pend} say when interviewed?",
                                      f"Which answer will {pend} give?"],
            [f"{pend} will say {th['clause'].format(opt=o)}" for o in opts]),
    ]
    return World(variables, prior, Z, T,
                 alternatives=lambda k, past: [(k, j) for j in range(K if sources[k][0] == "w" else 2)],
                 lik=lambda o, past: Ls[o[0]][:, o[1]],
                 render=render, prelude=prelude, mart_var="truth", exchangeable=True,
                 meta={"style": "json" if json_style else "prose", "theme": th["key"]})
