"""Crowd raters with confusion matrices (Dawid-Skene with known confusions), plus an anonymous worker who is either
diligent or a spammer, whose type can be probed with control items of known label.

Latent = (true class of the target item, anonymous worker type, label a pending rater will give). Every label is
conditionally independent given the latent, so the evidence is exchangeable.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, art, Var, cap, distinct_names, ints_row, is_json, join_list, json_block, json_line, mk_world,
                    n_steps, people, per_step, pick, round_pcts)

NAME = "raters"
TASKS = [
    dict(key="species", item="photo {X}-17", what="the species shown in {I}",
         classes=lambda rng: [art(f"{n} beetle") for n in distinct_names(rng, 3)]),
    dict(key="sentiment", item="review #{X}", what="the sentiment of {I}", classes=lambda rng: ["positive", "neutral", "negative"]),
    dict(key="triage", item="ticket {X}", what="the priority of {I}", classes=lambda rng: ["low", "medium", "high"]),
    dict(key="galaxy", item="galaxy image {X}", what="the shape of the galaxy in {I}", classes=lambda rng: ["spiral", "elliptical", "irregular"]),
]


def _conf(rng, K, lo=55, hi=95):
    M = []
    for c in range(K):
        d = int(rng.choice(np.arange(lo, hi + 1, 5)))
        rest = ints_row(rng, K - 1, 100 - d, 0, conc=2.0) if K > 2 else [100 - d]
        row = rest[:c] + [d] + rest[c:]
        M.append(row)
    return M


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    task = pick(rng, TASKS)
    K = {0: 2, 1: int(rng.integers(2, 4)), 2: 3}[level]
    classes = task["classes"](rng)[:K]
    item = task["item"].replace("{X}", distinct_names(rng, 1)[0][:4].upper())
    W = task["what"].replace("{I}", item)
    prev = round_pcts(rng.dirichlet(np.full(K, 2.0)), 100, 5)
    nR = {0: int(rng.integers(1, 3)), 1: int(rng.integers(2, 4)), 2: int(rng.integers(2, 5))}[level]
    names = people(rng, nR + 1)
    raters, pend = names[:nR], names[nR]
    conf = [_conf(rng, K) for _ in range(nR + 1)]
    anon = level >= 1
    wid = f"worker #{int(rng.integers(1000, 9999))}"
    p_spam = int(rng.choice(np.arange(10, 60, 5))) if anon else 0
    dil = _conf(rng, K, 70, 95)
    spam = [round_pcts(rng.dirichlet(np.full(K, 1.0)), 100, 5)] * K  # same label distribution whatever the truth
    n_ctrl = int(rng.integers(1, 3)) if level == 2 else 0
    ctrl_truth = [int(rng.integers(K)) for _ in range(n_ctrl)]
    ev = [("rater", i) for i in range(nR)] + ([("anon", None)] if anon else []) + [("ctrl", j) for j in range(n_ctrl)]
    ev = [ev[int(i)] for i in rng.permutation(len(ev))]
    T = len(ev)
    rows = list(itertools.product(range(K), (0, 1) if anon else (0,), range(K)))
    Cz = np.array([r[0] for r in rows])
    Az = np.array([r[1] for r in rows])  # 1 = spammer
    Pz = np.array([r[2] for r in rows])
    CF = [np.array(c, float) / 100 for c in conf]
    DIL, SPM = np.array(dil, float) / 100, np.array(spam, float) / 100
    prior = (np.array(prev, float)[Cz] / 100) * np.where(Az == 1, p_spam / 100, 1 - p_spam / 100) * CF[nR][Cz, Pz]
    keep = prior > 0
    vs = ["truth", "pending"] + (["anon"] if anon else [])
    if level == 1 and rng.random() < 0.3:
        vs = ["truth", "anon"]
    cols = {"truth": Cz, "pending": Pz, "anon": Az}
    proj = np.stack([cols[v] for v in vs], 1)
    V = {"truth": Var("true_label", list(classes), [f"What is the true label of {item}?", f"What is {W}, really?",
                                                    f"Which class does {item} truly belong to?"],
                      [f"{item} truly is {c}" if task["key"] == "species" else f"the true label of {item} is {c}" for c in classes]),
         "pending": Var("pending_label", list(classes), [f"Which label will {pend} give {item}?",
                                                         f"When {pend} rates {item}, what will they choose?"],
                        [f"{pend} labels {item} as {c}" for c in classes]),
         "anon": Var("worker_type", ["diligent", "spammer"], [f"Is {wid} diligent or a spammer?", f"Is {wid} a spammer?"],
                     [f"{wid} is diligent", f"{wid} is a spammer"])}
    variables = [V[v] for v in vs]

    def mtxt(nm, M):
        return f"{nm}: " + "; ".join(
            f"when the truth is {classes[c]}, says " + ", ".join(f"{classes[l]} {fmt.p(M[c][l])}" for l in range(K) if M[c][l] > 0)
            for c in range(K))

    prv = "Base rates: " + join_list([f"{c} {fmt.p(p)}" for c, p in zip(classes, prev)]) + "."
    rtxt = " ".join(mtxt(nm, conf[i]) + "." for i, nm in enumerate(raters + [pend]))
    atxt = (f" An anonymous crowd {wid} is a spammer with probability {fmt.p(p_spam)}. If diligent, their labels "
            f"follow this table: {mtxt('the matrix', dil).split(': ', 1)[1]}; a spammer ignores the item and says " +
            join_list([f"{classes[l]} {fmt.p(spam[0][l])}" for l in range(K)]) + "." if anon else "")
    ctxt = (f" {cap(wid)} also labels distinct control items (C1, C2, ...) whose true label is known; those labels "
            f"follow the same model (diligent table or spammer rates), independently." if n_ctrl else "")
    js = is_json(rng)
    if js:
        st = {"item": item, "task": f"label {W}", "base_rates": {c: fmt.p(p) for c, p in zip(classes, prev)},
              "confusion": {nm: {classes[c]: {classes[l]: fmt.p(conf[i][c][l]) for l in range(K)} for c in range(K)}
                            for i, nm in enumerate(raters + [pend])},
              "pending_rater": pend, "independence": "labels independent given the truth (and worker type)"}
        if anon:
            st["anonymous_worker"] = {"id": wid, "P(spammer)": fmt.p(p_spam),
                                      "diligent_confusion": {classes[c]: {classes[l]: fmt.p(dil[c][l]) for l in range(K)} for c in range(K)},
                                      "spammer_labels": {classes[l]: fmt.p(spam[0][l]) for l in range(K)},
                                      "spammer_behaviour": "ignores the item"}
            if n_ctrl:
                st["anonymous_worker"]["control_items"] = ("distinct items C1, C2, ... with known true label, labelled "
                                                           "by the same model, independently")
        prelude = json_block(rng, st)
        tr = [lambda who, l: json_line({"rater": who, "item": item, "label": classes[l]}),
              lambda who, l: json_line({"label": {who: classes[l]}})]
        tc = [lambda j, t, l: json_line({"rater": wid, "control_item": f"C{j + 1}", "known_true_label": classes[t],
                                         "label": classes[l]})]
    else:
        head = pick(rng, [f"Raters are labelling {W}.", f"A labelling job asks for {W}.",
                          f"Quality control: several people judge {W}."])
        conf_intro = pick(rng, ["Each rater's answers follow a known confusion table.",
                                "Rater accuracies from past audits:", "Known rater behaviour:"])
        prelude = " ".join(x for x in [head, prv, conf_intro, rtxt + atxt + ctxt,
                                       f"{pend} has not rated it yet. Labels are independent given the truth" + (" and the anonymous worker's type." if anon else ".")] if x)
        tr = [lambda who, l: f"{who} labels {item} as {classes[l]}.", lambda who, l: f"{who}'s label: {classes[l]}.",
              lambda who, l: f"{who} says {classes[l]}."]
        tc = [lambda j, t, l: f"On control item C{j + 1}, whose true label is {classes[t]}, {wid} answers {classes[l]}.",
              lambda j, t, l: f"Control check C{j + 1}: truth {classes[t]}, {wid} said {classes[l]}.",
              lambda j, t, l: f"{cap(wid)} labels control item C{j + 1}, known to be {classes[t]}, as {classes[l]}."]
    ptr, ptc = per_step(rng, T, tr), per_step(rng, T, tc)

    def lik(o, past):
        k, l = o
        kind, i = ev[k]
        if kind == "rater":
            return CF[i][Cz, l][keep]
        if kind == "anon":
            return np.where(Az == 1, SPM[Cz, l], DIL[Cz, l])[keep]
        t = ctrl_truth[i]
        return np.where(Az == 1, SPM[t, l], DIL[t, l])[keep]

    def render(k, o):
        kind, i = ev[k]
        if kind == "rater":
            return ptr[k](raters[i], o[1])
        if kind == "anon":
            return cap(ptr[k](wid, o[1]))
        return ptc[k](i, ctrl_truth[i], o[1])

    return mk_world(variables, prior[keep], proj[keep], T, lambda k, past: [(k, l) for l in range(K)], lik, render,
                    prelude, "true_label", True, "json" if js else "prose", task["key"],
                    {"prev": prev, "conf": conf, "anon": anon, "p_spam": p_spam, "dil": dil, "spam": spam[0],
                     "ctrl": ctrl_truth, "ev": ev, "vs": vs, "K": K})


def simulate(world, rng):
    p = world.params
    K = p["K"]
    c = int(rng.choice(K, p=np.array(p["prev"]) / 100))
    a = int(p["anon"] and rng.random() < p["p_spam"] / 100)

    def lab(row):
        return int(rng.choice(K, p=np.array(row, float) / sum(row)))

    def anon_lab(t):
        return lab(p["spam"]) if a else lab(p["dil"][t])

    obs = []
    for k, (kind, i) in enumerate(p["ev"]):
        if kind == "rater":
            obs.append((k, lab(p["conf"][i][c])))
        elif kind == "anon":
            obs.append((k, anon_lab(c)))
        else:
            obs.append((k, anon_lab(p["ctrl"][i])))
    vals = {"truth": c, "pending": lab(p["conf"][-1][c]), "anon": a}
    return tuple(vals[v] for v in p["vs"]), obs
