"""Legal / forensic evidence stated as likelihoods and likelihood ratios, with a possibly contaminated lab sample.

Latent = (guilt, lab contamination, result of a still-pending check). Each evidence item has stated P(E | guilty)
and P(E | innocent) (also given as a likelihood ratio); a contaminated sample reports a DNA match with a stated
probability regardless of guilt. Items are conditionally independent given the latent, so exchangeable.
Prior: the suspect is one of N people with access (equally likely), or stated prior odds.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import Fmt, Var, cap, distinct_names, is_json, join_list, json_block, json_line, mk_world, n_steps, people, per_step, pct_str, pick

NAME = "forensic"
LRS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 40, 50]
ITEMS = [
    dict(key="fiber", pos="the fibres on the victim's coat match the suspect's jumper",
         neg="the fibres do not match the suspect's jumper", name="fibre comparison", pg=(50, 95)),
    dict(key="witness", pos="the eyewitness picks the suspect out of the line-up", neg="the eyewitness does not pick the suspect",
         name="line-up identification", pg=(40, 85)),
    dict(key="phone", pos="the suspect's phone pinged the mast next to the scene", neg="the suspect's phone did not ping the mast near the scene",
         name="phone-mast record", pg=(60, 95)),
    dict(key="shoe", pos="the shoe print at the scene matches the suspect's trainers", neg="the shoe print does not match the suspect's trainers",
         name="shoe-print comparison", pg=(50, 95)),
    dict(key="glass", pos="glass fragments in the suspect's cuffs match the broken window", neg="no matching glass is found on the suspect",
         name="glass-fragment analysis", pg=(30, 80)),
    dict(key="cctv", pos="the CCTV analyst says the figure on camera has the suspect's gait", neg="the CCTV analyst says the gait does not match",
         name="gait analysis of CCTV", pg=(50, 90)),
]


def _item(rng, it):
    for _ in range(100):
        pg = int(rng.choice(np.arange(it["pg"][0], it["pg"][1] + 1, 5)))
        lr = int(pick(rng, LRS))
        if (pg * 100) % lr == 0:
            return pg, pg / lr, lr
    return pg, pg / 2, 2


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    suspect = people(rng, 1)[0]
    crime = pick(rng, ["the burglary at the {X} gallery", "the arson at {X} Mill", "the theft from the {X} museum",
                       "the break-in at {X} Pharmacy"]).replace("{X}", distinct_names(rng, 1)[0])
    if rng.random() < 0.5:
        Npool = int(rng.choice([2, 3, 4, 5, 8, 10, 20, 25, 50]))
        pG = 1.0 / Npool
        ptxt = (f"Before any forensic evidence, investigators consider {suspect} one of {Npool} people who could have "
                f"done it, all equally likely")
    else:
        a, b = int(rng.integers(1, 4)), int(rng.choice([1, 2, 3, 4, 5, 9, 19, 49]))
        if a == b:
            b += 1
        pG = a / (a + b)
        ptxt = f"The prior odds that {suspect} is the culprit are {a} to {b}"
    use_dna = level >= 1 and rng.random() < 0.8
    nitems = {0: int(rng.integers(1, 3)), 1: int(rng.integers(2, 4)), 2: int(rng.integers(3, 5))}[level]
    chosen = [ITEMS[int(i)] for i in rng.choice(len(ITEMS), size=nitems + 1, replace=False)]
    pend_item, obs_items = chosen[-1], chosen[:-1]
    specs = []
    for it in obs_items:
        pg, pi, lr = _item(rng, it)
        specs.append(dict(it, pgv=pg, piv=pi, lr=lr))
    ppg, ppi, plr = _item(rng, pend_item)
    rmp = float(pick(rng, [0.1, 0.2, 0.5, 1.0, 2.0]))  # random match probability in %
    dna_sens = int(rng.choice(np.arange(90, 100, 1)))
    pcont = int(rng.choice(np.arange(5, 35, 5)))
    pcm = int(rng.choice(np.arange(30, 90, 10)))
    ev = [("item", i) for i in range(len(specs))] + ([("dna", None)] if use_dna else [])
    ev = [ev[int(i)] for i in rng.permutation(len(ev))]
    T = len(ev)
    rows = list(itertools.product((0, 1), (0, 1) if use_dna else (0,), (0, 1)))
    G = np.array([r[0] for r in rows])  # 0 = guilty
    C = np.array([r[1] for r in rows])
    P = np.array([r[2] for r in rows])  # pending: 0 = positive
    p_pend = np.where(G == 0, ppg / 100, ppi / 100)
    prior = np.where(G == 0, pG, 1 - pG) * np.where(C == 1, pcont / 100, 1 - pcont / 100) * np.where(P == 0, p_pend, 1 - p_pend)
    vs = ["guilt", "pending"] + (["cont"] if use_dna else [])
    cols = {"guilt": G, "pending": P, "cont": 1 - C}
    proj = np.stack([cols[v] for v in vs], 1)
    V = {"guilt": Var("culprit", [f"{suspect} did it", f"{suspect} did not do it"],
                      [f"Is {suspect} the culprit?", f"Did {suspect} commit {crime}?", f"Is {suspect} guilty of {crime}?"],
                      [f"{suspect} is the culprit", f"{suspect} is not the culprit"]),
         "pending": Var("pending_check", ["positive", "negative"],
                        [f"What will the pending {pend_item['name']} show?",
                         f"Will the {pend_item['name']} come back positive?"],
                        [pend_item["pos"], pend_item["neg"]]),
         # latent C == 1 is a contaminated sample; the variable is coded 0 = contaminated, 1 = clean (cols = 1 - C)
         "cont": Var("contamination", ["contaminated", "clean"], ["Was the DNA sample contaminated in the lab?",
                                                                  "Is the lab sample contaminated?"],
                     ["the DNA sample was contaminated", "the DNA sample was clean"])}
    variables = [V[v] for v in vs]

    P_ = lambda x: pct_str(fmt, x)  # noqa: E731

    def itxt(s, lr, pg, pi):
        form = pick(rng, [0, 1, 2])
        means = f" (positive means {s['pos']}; negative means {s['neg']})"
        if form == 0:
            return (f"For the {s['name']}{means}: a positive result has probability {P_(pg)} if {suspect} is the "
                    f"culprit and {P_(pi)} if not.")
        if form == 1:
            return (f"The {s['name']}{means} is positive with probability {P_(pg)} for the true culprit and {P_(pi)} "
                    f"for an innocent person, a likelihood ratio of {lr}.")
        return (f"A positive {s['name']}{means} is {lr} times as likely if {suspect} did it as if not "
                f"({P_(pg)} versus {P_(pi)}).")

    parts = [itxt(s, s["lr"], s["pgv"], s["piv"]) for s in specs]
    ptxt2 = itxt(dict(pend_item), plr, ppg, ppi) + " Its result is still pending."
    dtxt = (f"The lab's DNA comparison reports a match with probability {P_(dna_sens)} if {suspect} is the culprit and "
            f"{P_(rmp)} (the random-match probability) if not, provided the sample is clean. With probability "
            f"{P_(pcont)} the sample was contaminated in the lab, in which case it reports a match with probability "
            f"{P_(pcm)} whatever the truth.") if use_dna else ""
    indep = "All results are independent of one another given the truth" + (" and the sample's condition." if use_dna else ".")
    js = is_json(rng)
    if js:
        st = {"case": crime, "suspect": suspect, "prior": ptxt + ".",
              "evidence_model": {s["name"]: {"P(positive | culprit)": P_(s["pgv"]), "P(positive | innocent)": P_(s["piv"]),
                                             "positive_means": s["pos"], "negative_means": s["neg"],
                                             "likelihood_ratio": s["lr"]} for s in specs},
              "pending": {pend_item["name"]: {"P(positive | culprit)": P_(ppg), "P(positive | innocent)": P_(ppi),
                                             "positive_means": pend_item["pos"], "negative_means": pend_item["neg"]}},
              "independence": indep}
        if use_dna:
            st["DNA comparison"] = {"P(match | culprit, clean)": P_(dna_sens), "P(match | innocent, clean)": P_(rmp),
                         "P(contaminated)": P_(pcont), "P(match | contaminated)": P_(pcm)}
        prelude = json_block(rng, st)
        ti = [lambda s, r: json_line({"result": s["name"], "outcome": "positive" if r == 0 else "negative"}),
              lambda s, r: json_line({"evidence": s["name"], "positive": r == 0})]
        td = [lambda r: json_line({"result": "DNA comparison", "outcome": "match" if r == 0 else "no match"}),
              lambda r: json_line({"evidence": "DNA comparison", "match": r == 0})]
    else:
        body = [f"Case file: {crime}. {ptxt}."] + parts + ([dtxt] if dtxt else []) + [ptxt2, indep]
        mid = body[1:-1]
        mid = [mid[int(i)] for i in rng.permutation(len(mid))]
        prelude = " ".join([body[0]] + mid + [body[-1]]) if rng.random() < 0.7 else " ".join([body[-1], body[0]] + mid)
        ti = [lambda s, r: f"Result: {s['pos'] if r == 0 else s['neg']}.",
              lambda s, r: f"The {s['name']} comes back {'positive' if r == 0 else 'negative'}: {s['pos'] if r == 0 else s['neg']}.",
              lambda s, r: f"It is reported that {s['pos'] if r == 0 else s['neg']}."]
        td = [lambda r: f"The DNA comparison reports {'a match' if r == 0 else 'no match'}.",
              lambda r: f"DNA result: {'match' if r == 0 else 'no match'}.",
              lambda r: f"The lab says the DNA {'matches' if r == 0 else 'does not match'} {suspect}."]
    pti, ptd = per_step(rng, T, ti), per_step(rng, T, td)

    def lik(o, past):
        k, r = o
        kind, i = ev[k]
        if kind == "dna":
            pm = np.where(C == 1, pcm / 100, np.where(G == 0, dna_sens / 100, rmp / 100))
        else:
            pm = np.where(G == 0, specs[i]["pgv"] / 100, specs[i]["piv"] / 100)
        return pm if r == 0 else 1 - pm

    def render(k, o):
        kind, i = ev[k]
        return ptd[k](o[1]) if kind == "dna" else pti[k](specs[i], o[1])

    return mk_world(variables, prior, proj, T, lambda k, past: [(k, 0), (k, 1)], lik, render, prelude, "culprit",
                    True, "json" if js else "prose", "forensic",
                    {"pG": pG, "use_dna": use_dna, "pcont": pcont, "pcm": pcm, "dna": dna_sens, "rmp": rmp,
                     "specs": specs, "pend": (ppg, ppi), "ev": ev, "vs": vs})


def simulate(world, rng):
    p = world.params
    g = 0 if rng.random() < p["pG"] else 1
    c = int(p["use_dna"] and rng.random() < p["pcont"] / 100)
    obs = []
    for k, (kind, i) in enumerate(p["ev"]):
        if kind == "dna":
            pm = p["pcm"] / 100 if c else (p["dna"] / 100 if g == 0 else p["rmp"] / 100)
        else:
            pm = p["specs"][i]["pgv"] / 100 if g == 0 else p["specs"][i]["piv"] / 100
        obs.append((k, 0 if rng.random() < pm else 1))
    pp = p["pend"][0] / 100 if g == 0 else p["pend"][1] / 100
    vals = {"guilt": g, "pending": 0 if rng.random() < pp else 1, "cont": 1 - c}
    return tuple(vals[v] for v in p["vs"]), obs
