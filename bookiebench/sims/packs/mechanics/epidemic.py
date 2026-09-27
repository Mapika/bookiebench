"""A tiny SIR epidemic on a contact graph, run for a stated number of days from an unknown patient zero, followed by
imperfect tests (antibody: ever infected; swab: currently infectious).

Exact: the day-by-day chain over the 3^n joint S/I/R states is propagated exactly (each susceptible person is
infected with probability 1 - (1 - beta)^(#infectious contacts), each infectious person recovers with probability
gamma, independently). Latent = (patient zero, final joint state). Tests are conditionally independent given the
latent, so exchangeable. Variables: patient zero, one person's final status, the number of people ever infected.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import Fmt, Var, repeat_tags, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, people, per_step, pick

NAME = "epidemic"
S_, I_, R_ = 0, 1, 2
SETTINGS = ["a remote research station", "a small sailing crew", "a shared flat", "a rural school's staff room",
            "an expedition camp", "a start-up office"]
BUGS = ["Velt fever", "Orrin flu", "the Kessa virus", "Marn cough", "Tolly pox"]


def _graph(rng, n):
    while True:
        edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.45]
        adj = {i: set() for i in range(n)}
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)
        seen, st = {0}, [0]
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y)
                    st.append(y)
        if len(seen) == n and len(edges) <= n + 1:
            return edges, adj


def step_dist(state, adj, beta, gamma):
    """Distribution over next joint states from one joint state (dict state -> prob)."""
    per = []
    for i, s in enumerate(state):
        if s == S_:
            k = sum(state[j] == I_ for j in adj[i])
            q = 1 - (1 - beta) ** k
            per.append([(S_, 1 - q), (I_, q)] if q > 0 else [(S_, 1.0)])
        elif s == I_:
            per.append([(I_, 1 - gamma), (R_, gamma)] if gamma > 0 else [(I_, 1.0)])
        else:
            per.append([(R_, 1.0)])
    out = {}
    for combo in itertools.product(*per):
        st = tuple(c[0] for c in combo)
        p = float(np.prod([c[1] for c in combo]))
        if p > 0:
            out[st] = out.get(st, 0.0) + p
    return out


def run(n, adj, src, days, beta, gamma):
    dist = {tuple(I_ if i == src else S_ for i in range(n)): 1.0}
    for _ in range(days):
        nd = {}
        for st, p in dist.items():
            for s2, q in step_dist(st, adj, beta, gamma).items():
                nd[s2] = nd.get(s2, 0.0) + p * q
        dist = nd
    return dist


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    n = {0: 3, 1: 4, 2: 5}[level] + (scale - 1)
    names = people(rng, n)
    edges, adj = _graph(rng, n)
    days = int(rng.integers(1, 3)) if level == 0 else int(rng.integers(2, 4))
    beta = int(rng.choice(np.arange(20, 75, 5)))
    gamma = 0 if level == 0 else int(rng.choice(np.arange(10, 55, 5)))
    src_w = [1] * n
    rows, prior = [], []
    for src in range(n):
        for st, p in run(n, adj, src, days, beta / 100, gamma / 100).items():
            rows.append((src, st))
            prior.append(p * src_w[src] / n)
    SRC = np.array([r[0] for r in rows])
    ST = np.array([r[1] for r in rows])  # rows x n
    ever = (ST != S_).sum(1)
    target = int(rng.integers(n))
    sirw = ["susceptible (never infected)", "infectious", "recovered"] if gamma else ["never infected", "infected"]
    k_opts = list(range(1, n + 1))
    if n >= 4:
        cats = [1, 2, 3] + ([4] if n >= 5 else [])
        cat_of = np.minimum(ever, cats[-1]) - 1
        cw = [f"{c}" if c < cats[-1] else f"{c} or more" for c in cats]
    else:
        cats = k_opts
        cat_of = ever - 1
        cw = [str(c) for c in cats]
    vs = ["src", "status"] + (["ever"] if level >= 1 else [])
    cols = {"src": SRC, "status": ST[:, target], "ever": cat_of}
    proj = np.stack([cols[v] for v in vs], 1)
    bug = pick(rng, BUGS)
    V = {"src": Var("patient_zero", list(names), [f"Who brought {bug} into the group?", "Who was patient zero?",
                                                   f"Which person was infected first?"],
                    [f"{nm} was patient zero" for nm in names]),
         "status": Var(f"status_{names[target]}", sirw,
                       [f"What is {names[target]}'s status now?", f"Is {names[target]} {'susceptible, infectious or recovered' if gamma else 'infected'} at this point?",
                        f"Where does {names[target]} stand with {bug} today?"],
                       [f"{names[target]} is {w.split(' (')[0]}" if gamma else f"{names[target]} has {'never been' if i == 0 else 'been'} infected"
                        for i, w in enumerate(sirw)]),
         "ever": Var("n_infected", [f"{c} {'person' if c == '1' else 'people'}" for c in cw],
                     ["How many people have been infected so far, patient zero included?",
                      f"How many of the {n} have caught {bug} by now?"],
                     [f"{c} {'person has' if c == '1' else 'people have'} been infected so far" for c in cw])}
    variables = [V[v] for v in vs]

    T = n_steps(rng, level, 3, 5)
    tests = []
    for _ in range(T):
        kind = pick(rng, ["antibody", "swab"]) if gamma else "antibody"
        tests.append((kind, int(rng.integers(n))))
    ab_sens, ab_spec = int(rng.choice(np.arange(70, 100, 5))), int(rng.choice(np.arange(80, 100, 2)))
    sw_sens, sw_spec = int(rng.choice(np.arange(60, 100, 5))), int(rng.choice(np.arange(80, 100, 2)))
    setting = pick(rng, SETTINGS)
    contacts = join_list([f"{names[i]}-{names[j]}" for i, j in edges])
    dyn = (f"On each day, every infectious person passes {bug} to each susceptible contact independently with "
           f"probability {fmt.p(beta)}" + (f"; at the end of each day, each person who was infectious at the start of "
                                           f"that day recovers (becomes immune) with probability {fmt.p(gamma)}"
                                           if gamma else "; nobody recovers within this period") +
           ". Someone infected during a day becomes infectious from the next day on.")
    start = (f"On day 0 exactly one of them, equally likely to be any of the {n}, came back infectious; "
             f"{days} day{'s have' if days > 1 else ' has'} passed since.")
    ttxt = (f"An antibody test detects past or current infection with probability {fmt.p(ab_sens)} and gives a false "
            f"positive for someone never infected with probability {fmt.p(100 - ab_spec)}.")
    if gamma:
        ttxt += (f" A swab test detects current infectiousness with probability {fmt.p(sw_sens)} and is falsely "
                 f"positive with probability {fmt.p(100 - sw_spec)} for anyone not currently infectious.")
    tags = repeat_tags(tests)
    rep_txt = (" Repeated tests of the same person use separate samples (labelled A, B, ...), and every test result is "
               "independent of the others given everyone's true status." if any(tags) else "")
    ttxt += rep_txt or " Test results are independent given everyone's true status."
    lab = [(kind, t) for (kind, _), t in zip(tests, tags)]

    def L(kt):  # prose label of a test
        return f"{kt[0]} test" + (f" (sample {kt[1]})" if kt[1] else "")

    def J(kt, **kw):  # JSON fields of a test
        return dict(test=kt[0], **({"sample": kt[1]} if kt[1] else {}), **kw)
    js = is_json(rng)
    if js:
        st = {"setting": setting, "people": names, "contacts": [f"{names[i]}-{names[j]}" for i, j in edges],
              "disease": bug, "patient_zero": "uniformly one of the people, infectious on day 0", "days_elapsed": days, "rounds_of_transmission": days,
              "daily_transmission_per_contact": fmt.p(beta), "daily_recovery": fmt.p(gamma) if gamma else "none",
              "timing": ("each day, every infectious person infects each susceptible contact independently with the "
                         "daily transmission probability; " + ("at the end of the day, each person who was infectious "
                         "at the start of that day recovers (becomes immune) with the daily recovery probability; "
                         if gamma else "nobody recovers; ") + "people infected during a day are infectious from the next day"),
              "antibody_test": {"detects": "past or current infection", "sensitivity": fmt.p(ab_sens),
                                "false_positive": fmt.p(100 - ab_spec)},
              "test_independence": (rep_txt.strip() or "test results are independent given everyone's true status")}
        if gamma:
            st["swab_test"] = {"detects": "current infectiousness", "sensitivity": fmt.p(sw_sens),
                               "false_positive": fmt.p(100 - sw_spec)}
        prelude = json_block(rng, st)
        tt = [lambda kt, i, r: json_line(J(kt, person=names[i], result="positive" if r else "negative")),
              lambda kt, i, r: json_line({"person": names[i], **J(kt), "outcome": "+" if r else "-"})]
    else:
        g = f"The {n} people in {setting} ({join_list(names)}) have these close contacts: {contacts}."
        prelude = pick(rng, [f"{g} {start} {dyn} {ttxt}", f"Outbreak of {bug}. {g} {dyn} {start} {ttxt}",
                             f"{start.replace('them', 'the ' + str(n) + ' people in ' + setting)} Contacts: {contacts}. {dyn} {ttxt}"])
        tt = [lambda kt, i, r: f"{names[i]}'s {L(kt)} is {'positive' if r else 'negative'}.",
              lambda kt, i, r: f"{cap(L(kt))} on {names[i]}: {'positive' if r else 'negative'}.",
              lambda kt, i, r: f"{names[i]} tests {'positive' if r else 'negative'} on the {L(kt)}."]
    ptt = per_step(rng, T, tt)

    def lik(o, past):
        k, r = o
        kind, i = tests[k]
        if kind == "antibody":
            pos = np.where(ST[:, i] != S_, ab_sens / 100, 1 - ab_spec / 100)
        else:
            pos = np.where(ST[:, i] == I_, sw_sens / 100, 1 - sw_spec / 100)
        return pos if r else 1 - pos

    return mk_world(variables, np.array(prior), proj, T, lambda k, past: [(k, 1), (k, 0)], lik,
                    lambda k, o: ptt[k](lab[k], tests[k][1], o[1]), prelude, "patient_zero", True,
                    "json" if js else "prose", "sir",
                    {"n": n, "adj": adj, "days": days, "beta": beta, "gamma": gamma, "tests": tests, "target": target,
                     "ab": (ab_sens, ab_spec), "sw": (sw_sens, sw_spec), "cats": cats, "vs": vs})


def simulate(world, rng):
    p = world.params
    n, adj = p["n"], p["adj"]
    src = int(rng.integers(n))
    st = [S_] * n
    st[src] = I_
    for _ in range(p["days"]):
        new = list(st)
        for i in range(n):
            if st[i] == S_:
                for j in adj[i]:
                    if st[j] == I_ and rng.random() < p["beta"] / 100:
                        new[i] = I_
            elif st[i] == I_ and rng.random() < p["gamma"] / 100:
                new[i] = R_
        st = new
    ever = sum(s != S_ for s in st)
    cats = p["cats"]
    cat = min(ever, cats[-1]) - 1
    obs = []
    for k, (kind, i) in enumerate(p["tests"]):
        if kind == "antibody":
            pos = p["ab"][0] / 100 if st[i] != S_ else 1 - p["ab"][1] / 100
        else:
            pos = p["sw"][0] / 100 if st[i] == I_ else 1 - p["sw"][1] / 100
        obs.append((k, int(rng.random() < pos)))
    vals = {"src": src, "status": st[p["target"]], "ever": cat}
    return tuple(vals[v] for v in p["vs"]), obs
