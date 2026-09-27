"""Multi-cause assessment (hiring / credit): two independent latent causes, several noisy assessments that depend on one
or both causes (explaining away), and an outcome that depends on both.

Assessments are conditionally independent given the causes (exchangeable).
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, COMPANIES, Fmt, cap, join_list, json_block, json_line, person, pick, pct_row, rand_pct, sample
from .world import Var, World

THEMES = [
    dict(key="hiring", subj="the candidate {who}", org="{firm}",
         a=("skill", ["strong", "average", "weak"], "{who}'s technical skill is {opt}", "How strong is {who}'s "
            "technical skill?"),
         b=("motivation", ["highly motivated", "not very motivated"], ["{who} is highly motivated",
                                                                         "{who} is not very motivated"],
            "Is {who} highly motivated?"),
         out=("success", ["succeeds", "does not succeed"], ["{who} will succeed in the role",
                                                            "{who} will not succeed in the role"],
              "Will {who} succeed in the role if hired?", "succeeds in the role"),
         a_tests=[("coding exercise", "passes the coding exercise", "fails the coding exercise"),
                  ("technical quiz", "scores well on the technical quiz", "scores poorly on the technical quiz"),
                  ("portfolio review", "gets a positive portfolio review", "gets a negative portfolio review")],
         b_tests=[("reference call", "gets an enthusiastic reference", "gets a lukewarm reference"),
                  ("cover letter", "writes a compelling cover letter", "writes a generic cover letter")],
         ab_tests=[("panel interview", "impresses the interview panel", "does not impress the interview panel"),
                   ("take-home project", "submits an excellent take-home project",
                    "submits a mediocre take-home project")]),
    dict(key="credit", subj="the loan applicant {who}", org="{firm} Bank",
         a=("finances", ["strong", "moderate", "weak"], "{who}'s financial position is {opt}",
            "How strong is {who}'s financial position?"),
         b=("diligence", ["diligent", "careless"], ["{who} is diligent with money", "{who} is careless with money"],
            "Is {who} diligent with money?"),
         out=("repayment", ["repays", "defaults"], ["{who} will repay the loan", "{who} will default on the loan"],
              "Will {who} repay the loan?", "repays the loan"),
         a_tests=[("income check", "passes the income check", "fails the income check"),
                  ("savings review", "shows healthy savings", "shows little savings")],
         b_tests=[("payment-history check", "has a clean payment history", "has missed payments on record"),
                  ("landlord reference", "gets a good landlord reference", "gets a poor landlord reference")],
         ab_tests=[("credit score pull", "has a high credit score", "has a low credit score"),
                   ("underwriter interview", "satisfies the underwriter", "does not satisfy the underwriter")]),
]


def _two_factor(rng):
    base = sorted([rand_pct(rng, 20, 95, 5) for _ in range(3)], reverse=True)
    pen = rand_pct(rng, 5, min(base) - 5, 5)
    return base, pen  # P(yes | a, b=0) = base[a]; P(yes | a, b=1) = base[a] - pen


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    who = person(rng)
    firm = pick(rng, COMPANIES)
    org = th["org"].format(firm=firm)
    an, aopts, acl, aq = th["a"]
    bn, bopts, bcl, bq = th["b"]
    on, oopts, ocl, oq, odesc = th["out"]
    pa = pct_row(rng, 3, lo=10, step=5, conc=2.0)
    pb = rand_pct(rng, 20, 80, 5)
    T = int(rng.integers(3, 6))
    pool = ([("a",) + t for t in th["a_tests"]] + [("b",) + t for t in th["b_tests"]] +
            [("ab",) + t for t in th["ab_tests"]])
    tests = sample(rng, pool, min(T, len(pool)))
    T = len(tests)
    params = []
    for kind, *_ in tests:
        if kind == "a":
            params.append(sorted([rand_pct(rng, 10, 95, 5) for _ in range(3)], reverse=True))
        elif kind == "b":
            params.append(sorted([rand_pct(rng, 10, 95, 5) for _ in range(2)], reverse=True))
        else:
            params.append(_two_factor(rng))
    out_base, out_pen = _two_factor(rng)

    Z = np.array(list(itertools.product(range(3), range(2), range(2))))
    A, B, O = Z[:, 0], Z[:, 1], Z[:, 2]
    po = (np.array(out_base, float)[A] - np.where(B == 1, out_pen, 0)) / 100
    prior = (np.array(pa, float)[A] / 100 * np.where(B == 0, pb, 100 - pb) / 100 * np.where(O == 0, po, 1 - po))

    def p_yes(i):
        kind = tests[i][0]
        if kind == "a":
            return np.array(params[i], float)[A] / 100
        if kind == "b":
            return np.array(params[i], float)[B] / 100
        base, pen = params[i]
        return (np.array(base, float)[A] - np.where(B == 1, pen, 0)) / 100

    L = [np.stack([p_yes(i), 1 - p_yes(i)], 1) for i in range(T)]

    def desc(i):
        s = _desc(i)
        if get_tv() >= 2:  # define the negative outcome phrase
            s += f" Otherwise the applicant {tests[i][3]}."
        return s

    def _desc(i):
        kind, name, yes, no = tests[i]
        if kind == "a":
            return (f"The {name} depends only on {an}: someone with {aopts[0]} {an} {yes} with probability "
                    f"{fmt.p(params[i][0])}; with {aopts[1]} {an}, {fmt.p(params[i][1])}; with {aopts[2]} {an}, "
                    f"{fmt.p(params[i][2])}.")
        if kind == "b":
            return (f"The {name} depends only on {bn}: someone who is {bopts[0]} {yes} with probability "
                    f"{fmt.p(params[i][0])}, someone {bopts[1]} with probability {fmt.p(params[i][1])}.")
        base, pen = params[i]
        return (f"The {name} depends on both: someone {bopts[0]} {yes} with probability {fmt.p(base[0])}, "
                f"{fmt.p(base[1])} or {fmt.p(base[2])} for {aopts[0]}, {aopts[1]} or {aopts[2]} {an}, and the "
                f"probability is {pen} percentage points lower for someone {bopts[1]}.")

    subj = th["subj"].format(who=who)
    json_style = rng.random() < 0.3
    if json_style:
        st = {"subject": subj, "organisation": org,
              f"P({an})": {o: fmt.p(p) for o, p in zip(aopts, pa)},
              f"P({bn})": {bopts[0]: fmt.p(pb), bopts[1]: fmt.p(100 - pb)},
              "independence": f"{an} and {bn} are independent; assessments independent given both"}
        if get_tv() >= 2:
            st["assessment_outcomes"] = {t[1]: [t[2], t[3]] for t in tests}
            st["outcome_independence"] = f"given {an} and {bn}, whether the applicant {odesc} is independent of the assessments"
        for i, (kind, name, yes, no) in enumerate(tests):
            if kind == "a":
                st[f"P({yes} | {an})"] = {o: fmt.p(x) for o, x in zip(aopts, params[i])}
            elif kind == "b":
                st[f"P({yes} | {bn})"] = {o: fmt.p(x) for o, x in zip(bopts, params[i])}
            else:
                base, pen = params[i]
                st[f"P({yes} | {an}, {bn})"] = {f"{o}, {bo}": fmt.p(x - (pen if j else 0))
                                                for o, x in zip(aopts, base) for j, bo in enumerate(bopts)}
        st[f"P({odesc} | {an}, {bn})"] = {f"{o}, {bo}": fmt.p(x - (out_pen if j else 0))
                                          for o, x in zip(aopts, out_base) for j, bo in enumerate(bopts)}
        prelude = json_block(rng, st)
        render = lambda k, o: json_line({"assessment": tests[o[0]][1], "result": tests[o[0]][2 if o[1] == 0 else 3]})  # noqa: E731
    else:
        s_pri = (f"{org} is evaluating {subj}. Among such applicants, {an} is " +
                 join_list([f"{o} with probability {fmt.p(p)}" for o, p in zip(aopts, pa)], "or") +
                 f". Independently, {fmt.p(pb)} are {bopts[0]} and the rest are {bopts[1]}.")
        s_t = " ".join(desc(i) for i in range(T))
        s_o = (f"Someone {bopts[0]} {odesc} with probability {fmt.p(out_base[0])}, {fmt.p(out_base[1])} or "
               f"{fmt.p(out_base[2])} for {aopts[0]}, {aopts[1]} or {aopts[2]} {an}; for someone {bopts[1]} each "
               f"figure is {out_pen} percentage points lower.")
        s_ind = "All assessments are independent of each other given the applicant's true characteristics."
        if get_tv() >= 2:
            s_ind += (f" Given {an} and {bn}, whether the applicant {odesc} is independent of the assessment "
                      f"results.")
        prelude = pick(rng, [
            f"{s_pri} {s_t} {s_ind} {s_o}",
            f"{s_pri} {s_o} The following assessments are run. {s_t} {s_ind}",
            f"{s_t} {s_ind} {s_o} {s_pri}",
        ])
        render = lambda k, o: f"{who} {tests[o[0]][2 if o[1] == 0 else 3]}."  # noqa: E731

    variables = [
        Var(an, list(aopts), [aq.format(who=who)], acl.replace("{who}", who)),
        Var(bn, list(bopts), [bq.format(who=who)], [x.format(who=who) for x in bcl]),
        Var(on, list(oopts), [oq.format(who=who)], [x.format(who=who) for x in ocl]),
    ]
    return World(variables, prior, Z, T, alternatives=lambda k, past: [(k, 0), (k, 1)],
                 lik=lambda o, past: L[o[0]][:, o[1]], render=render, prelude=prelude, mart_var=an,
                 exchangeable=True, meta={"style": "json" if json_style else "prose", "theme": th["key"]})
