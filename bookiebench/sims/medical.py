"""Medical diagnosis Bayes net: exposure -> disease -> findings (symptoms / test results), plus a pending test.

Findings are conditionally independent given the disease (exchangeable). Disease names are invented so that
real-world medical knowledge cannot substitute for the stated numbers.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, Fmt, SURNAMES, cap, join_list, json_block, json_line, person, pick, pct_row, rand_pct, sample
from .world import Var, World

D_PRE = ["Varn", "Toller", "Kessin", "Morrow", "Halden", "Quell", "Dray", "Fenwick", "Oster", "Brannock", "Ilsen",
         "Corran", "Maddox", "Ytter", "Pell"]
D_SUF = ["fever", "syndrome", "disease", "infection", "anaemia"]
EXPOSURES = ["contaminated well water", "bites from the marsh tick", "industrial solvents", "raw shellfish",
             "a mould found in old buildings", "bat colonies", "unpasteurised milk", "a cave system"]
# (key, present text, absent text, noun phrase for the table)
SYMPTOMS = [
    ("fever", "The patient has a fever.", "The patient does not have a fever.", "a fever"),
    ("rash", "A rash is found on the patient's arms.", "There is no rash.", "a rash"),
    ("cough", "The patient reports a persistent cough.", "The patient reports no cough.", "a persistent cough"),
    ("fatigue", "The patient complains of severe fatigue.", "The patient has no unusual fatigue.", "severe fatigue"),
    ("joint_pain", "The patient has joint pain.", "The patient has no joint pain.", "joint pain"),
    ("headache", "The patient reports headaches.", "The patient reports no headaches.", "headaches"),
    ("swelling", "The lymph nodes are swollen.", "The lymph nodes are not swollen.", "swollen lymph nodes"),
    ("nausea", "The patient has been nauseous.", "The patient has not been nauseous.", "nausea"),
]
TESTS = ["the Ferrin panel", "a throat swab", "the RX-4 blood test", "a urine screen", "the Delmar antigen test",
         "a chest X-ray", "the K-titre assay", "a skin-prick test"]


def make_world(rng) -> World:
    TV = get_tv()
    fmt = Fmt(rng)
    doc = f"Dr. {pick(rng, SURNAMES)}"
    pat = person(rng)
    names = sample(rng, D_PRE, 2)
    d1, d2 = f"{names[0]} {pick(rng, D_SUF)}", f"{names[1]} {pick(rng, D_SUF)}"
    exp = pick(rng, EXPOSURES)
    T = int(rng.integers(3, 6))
    n_sym = int(rng.integers(1, T + 1))
    syms = sample(rng, SYMPTOMS, min(n_sym, len(SYMPTOMS)))
    tests = sample(rng, TESTS, T - len(syms) + 1)
    pend = tests[-1]
    findings = []  # (key, pos text, neg text, table noun)
    for s in syms:
        findings.append(s)
    for t in tests[:-1]:
        findings.append((t, f"{cap(t)} comes back positive.", f"{cap(t)} comes back negative.",
                         f"a positive {t}" if TV < 2 else f"a positive result on {t}"))
    order = rng.permutation(len(findings))
    findings = [findings[int(i)] for i in order]

    pe = rand_pct(rng, 10, 70)
    rowE = pct_row(rng, 3, lo=5, step=5, conc=2.0)
    rowU = pct_row(rng, 3, lo=5, step=5, conc=2.0)
    rowU = sorted(rowU, reverse=True) if rng.random() < 0.7 else rowU  # unexposed mostly healthy
    tabs = []
    for _ in range(T + 1):  # per finding and the pending test: P(present | none, d1, d2)
        base = rand_pct(rng, 5, 40)
        tabs.append([base, rand_pct(rng, 10, 95), rand_pct(rng, 10, 95)])
    tab = np.array(tabs, float) / 100
    pd = np.array([rowE, rowU], float) / 100  # [E=exposed/not] x D

    Z = np.array(list(itertools.product(range(2), range(3), range(2))))
    pE = np.array([pe, 100 - pe]) / 100
    ptest = np.stack([tab[T], 1 - tab[T]], 1)  # D x {pos, neg}
    prior = pE[Z[:, 0]] * pd[Z[:, 0], Z[:, 1]] * ptest[Z[:, 1], Z[:, 2]]
    dnames = ["neither condition", d1, d2]

    def table_sentence(i, noun):
        a, b, c = (int(round(x * 100)) for x in tab[i])
        if TV < 2:
            first = (f"{cap(noun)} occurs in {fmt.p(b)} of patients with {d1}, {fmt.p(c)} of patients with {d2}, and "
                     f"{fmt.p(a)} of patients with neither.")
        else:  # F9: no "Headaches occurs"
            first = (f"{cap(fmt.p(b))} of patients with {d1}, {fmt.p(c)} of patients with {d2}, and {fmt.p(a)} of "
                     f"patients with neither condition have {noun}.")
        return pick(rng, [
            first,
            f"The chance of {noun} is {fmt.p(b)} with {d1}, {fmt.p(c)} with {d2}, and {fmt.p(a)} otherwise.",
        ])

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {
            "patient": pat, "clinician": doc,
            "P(exposed to " + exp + ")": fmt.p(pe),
            "P(condition | exposed)": {dn: fmt.p(r) for dn, r in zip(dnames, rowE)},
            "P(condition | not exposed)": {dn: fmt.p(r) for dn, r in zip(dnames, rowU)},
            "P(finding | condition)": {f[0]: {dn: fmt.p(int(round(x * 100))) for dn, x in zip(dnames, tab[i])}
                                       for i, f in enumerate(findings)},
            f"P({pend} positive | condition)": {dn: fmt.p(int(round(x * 100))) for dn, x in zip(dnames, tab[T])},
            "note": f"patients have at most one of {d1} and {d2}; findings are independent given the condition",
        })
        render = lambda k, o: json_line({"finding": findings[o[0]][0], "present": bool(o[1] == 0)})  # noqa: E731
    else:
        intro = pick(rng, [
            f"{doc} is examining {pat}. In this clinic's population, {fmt.p(pe)} of patients have been exposed to {exp}.",
            f"{pat} visits {doc}. Records show that {fmt.p(pe)} of the clinic's patients have had contact with {exp}.",
            f"At a rural clinic, {doc} sees a new patient, {pat}. About {fmt.p(pe)} of patients there were exposed "
            f"to {exp}.",
        ])
        dz = (f"Among exposed patients, {fmt.p(rowE[1])} have {d1}, {fmt.p(rowE[2])} have {d2}, and {fmt.p(rowE[0])} "
              f"have neither; among unexposed patients the figures are {fmt.p(rowU[1])}, {fmt.p(rowU[2])} and "
              f"{fmt.p(rowU[0])}. Nobody has both conditions.")
        tabs_txt = " ".join(table_sentence(i, f[3]) for i, f in enumerate(findings))
        pend_txt = (f"{cap(pend)} has also been ordered; it is positive for {fmt.p(int(round(tab[T][1] * 100)))} of "
                    f"patients with {d1}, {fmt.p(int(round(tab[T][2] * 100)))} with {d2}, and "
                    f"{fmt.p(int(round(tab[T][0] * 100)))} with neither. Its result is not back yet.")
        tail = "All findings are independent of each other once the condition is known."
        prelude = " ".join([intro, dz, tabs_txt, pend_txt, tail])
        render = lambda k, o: findings[o[0]][1] if o[1] == 0 else findings[o[0]][2]  # noqa: E731

    variables = [
        Var("condition", dnames, [f"What condition does {pat} have?", f"Which diagnosis is correct for {pat}?",
                                  "What is the patient's condition?"],
            [f"{pat} has neither {d1} nor {d2}" if TV < 2 else f"{pat} has no condition", f"{pat} has {d1}",
             f"{pat} has {d2}"]),
        Var("exposure", ["exposed", "not exposed"], [f"Was {pat} exposed to {exp}?"],
            [f"{pat} was exposed to {exp}", f"{pat} was not exposed to {exp}"]),
        Var("pending_test", ["positive", "negative"], [f"What will the result of {pend} be?",
                                                       f"Will {pend} come back positive or negative?"],
            f"{pend} will come back {{opt}}"),
    ]
    # reorder Z columns to variable order (condition, exposure, pending)
    proj = Z[:, [1, 0, 2]]
    L = [np.stack([tab[i][Z[:, 1]], 1 - tab[i][Z[:, 1]]], 1) for i in range(T)]
    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: [(k, 0), (k, 1)],
                 lik=lambda o, past: L[o[0]][:, o[1]],
                 render=render, prelude=prelude, mart_var="condition", exchangeable=True,
                 meta={"style": "json" if json_style else "prose"})
