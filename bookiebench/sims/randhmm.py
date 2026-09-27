"""Procedural hidden-Markov worlds: random state spaces, transition/emission matrices, lengths, gaps and sensors,
in random lexicon themes. Latent space = the full hidden trajectory (plus an optional future reading), enumerated.
Evidence is ordered in time (not exchangeable).
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, cap, join_list, json_block, json_line, pick
from .lexicon import DOMAINS, instantiate, pseudo
from .procedural import ALPHAS, DENOMS, NumStyle, Row, round_counts
from .world import Var, World

DECIMALS = 8
MAX_Z = 8192
INSTRUMENTS = ["gauge", "monitor", "indicator", "meter", "survey", "scanner", "probe", "panel", "tracker", "log"]
UNITS = [("day", "days"), ("week", "weeks"), ("hour", "hours"), ("round", "rounds"), ("month", "months"),
         ("shift", "shifts"), ("session", "sessions")]


def _row(rng, n, D, alpha, stick_i=None, stick=0.0):
    w = rng.dirichlet(np.full(n, alpha))
    if stick_i is not None:
        w = (1 - stick) * w
        w[stick_i] += stick
    return Row(round_counts(w, D), D)


def make_world(rng) -> World:
    for _ in range(200):
        w = _try(rng)
        if w is not None:
            return w
    raise RuntimeError("randhmm: could not build a world")


def _try(rng):
    dom = pick(rng, DOMAINS)
    setting = dom["setting"].replace("{X}", pseudo(rng))
    if rng.random() < 0.3:  # abstract regime with invented state names (allows K = 4)
        hid_np = f"the {pseudo(rng, 2)} regime of {setting}"
        K0 = int(rng.integers(2, 5))
        S = []
        while len(S) < K0:
            w = pseudo(rng, 2).lower() + " " + pick(rng, ["mode", "phase", "state", "level"])
            if w not in S:
                S.append(w)
    else:
        hid_np, S = instantiate(rng, pick(rng, dom["g"]))
    if hid_np.startswith("whether"):
        return None
    K = len(S)
    unit, units = pick(rng, UNITS)
    n_sens = 1 if rng.random() < 0.6 else 2
    sensors = []
    pool = [c for c in dom["p"]]
    for _ in range(n_sens):
        _, R = instantiate(rng, pick(rng, pool))
        name = f"the {pseudo(rng)} {pick(rng, INSTRUMENTS)}"
        D = int(pick(rng, DENOMS))
        alpha = float(pick(rng, ALPHAS))
        rows = [_row(rng, len(R), D, alpha) for _ in range(K)]
        sensors.append((name, R, rows))
    if n_sens == 2 and sensors[0][0] == sensors[1][0]:
        return None
    ns = NumStyle(rng)
    D = int(pick(rng, DENOMS))
    alpha = float(pick(rng, ALPHAS))
    stick = float(pick(rng, [0.0, 0.3, 0.5, 0.7, 0.85]))
    init = _row(rng, K, D, float(pick(rng, ALPHAS)))
    trans = [_row(rng, K, D, alpha, i, stick) for i in range(K)]

    T = int(rng.integers(3, 8))
    gaps = rng.random() < 0.4
    times, t = [], 1
    for _ in range(T):
        times.append(t)
        t += 1 + (int(rng.integers(0, 2)) if gaps else 0)
    last = times[-1]
    fut = int(rng.integers(1, 3))
    H = last + fut
    fut_read = rng.random() < 0.4
    nZ = K ** H * (len(sensors[0][1]) if fut_read else 1)
    if nZ > MAX_Z:
        return None
    both = n_sens == 2 and rng.random() < 0.5
    step_sens = [list(range(n_sens)) if both else [int(rng.integers(n_sens))] for _ in range(T)]

    Z = np.array(list(itertools.product(*([range(K)] * H + ([range(len(sensors[0][1]))] if fut_read else [])))))
    A = np.array([r.probs for r in trans])
    E = [np.array([r.probs for r in rows]) for _, _, rows in sensors]
    prior = init.probs[Z[:, 0]].copy()
    for h in range(H - 1):
        prior = prior * A[Z[:, h], Z[:, h + 1]]
    if fut_read:
        prior = prior * E[0][Z[:, H - 1], Z[:, H]]

    lab = pick(rng, ["num", "num", "ord"])
    tl = (lambda h: f"{unit} {h}") if lab == "num" else (lambda h: f"the {_ordinal(h)} {unit}")

    # variables: state at last observed time (mart), plus 1-3 others
    cand = [("state", h) for h in range(1, H + 1) if h != last]
    if fut_read:
        cand.append(("read", H))
    k = int(rng.integers(1, min(3, len(cand)) + 1))
    others = [cand[int(i)] for i in rng.choice(len(cand), size=k, replace=False)]
    spec = [("state", last)] + sorted(others, key=lambda x: (x[1], x[0]))
    shape = [K if s[0] == "state" else len(sensors[0][1]) for s in spec]
    if np.prod(shape) > 256:
        return None
    variables, cols = [], []
    for kind, h in spec:
        tense_fut = h > last
        verb = "will be" if tense_fut else ("was" if h < last else "is")
        if kind == "state":
            q = [f"What {'will ' + hid_np + ' be' if tense_fut else ('was ' + hid_np if h < last else 'is ' + hid_np)} "
                 f"in {tl(h)}?", f"In {tl(h)}, which state {'will' if tense_fut else 'did'} {hid_np} "
                 f"{'be in' if tense_fut else 'have'}?"]
            variables.append(Var(f"state_{h}", list(S), q, f"in {tl(h)} {hid_np} {verb} {{opt}}"))
            cols.append(h - 1)
        else:
            nm, R, _ = sensors[0]
            variables.append(Var(f"reading_{h}", list(R), [f"What will {nm} read in {tl(h)}?"],
                                 f"in {tl(h)} {nm} will read {{opt}}"))
            cols.append(H)
    proj = Z[:, cols]
    if get_tv() >= 2:  # tense-correct step-k martingale questions (the last reading lies in the future before T-1)
        v0 = variables[0]
        v0.questions_at = lambda k: ([f"What will {hid_np} be in {tl(last)}?",
                                      f"In {tl(last)}, which state will {hid_np} be in?"] if k < T - 1 else v0.questions)

    layout = pick(rng, ["prose", "prose", "bullets", "json"])
    if layout == "json":
        st = {"setting": setting, "hidden quantity": hid_np, "states": S, "time step": unit,
              f"P({hid_np} in {tl(1)})": ns.cells(init, S),
              f"P(next {unit} | this {unit})": {s: ns.cells(r, S) for s, r in zip(S, trans)}}
        for nm, R, rows in sensors:
            st[f"P({nm} reading | {hid_np})"] = {s: ns.cells(r, R) for s, r in zip(S, rows)}
        st["observations"] = f"sensor readings at {join_list([tl(h) for h in times])}"
        prelude = json_block(rng, st)
    else:
        u = []
        u.append(pick(rng, [
            f"At {setting}, {hid_np} changes from {unit} to {unit} as a Markov chain and is only seen through "
            f"noisy instruments.",
            f"We track {hid_np} at {setting}, {unit} by {unit}. It cannot be observed directly.",
            f"This concerns {hid_np} at {setting}, which evolves over {units}.",
        ]))
        if layout == "bullets":
            u.append(f"{cap(hid_np)} in {tl(1)}: " + ", ".join(f"{k} {v}" for k, v in ns.cells(init, S).items()) + ".")
            lines = [f"Transitions from one {unit} to the next:"]
            for s, r in zip(S, trans):
                lines.append(f"- from {s}: " + ", ".join(f"{k} {v}" for k, v in ns.cells(r, S).items()))
            u.append("\n".join(lines))
            for nm, R, rows in sensors:
                lines = [f"Readings of {nm}, depending on {hid_np}:"]
                for s, r in zip(S, rows):
                    lines.append(f"- when {s}: " + ", ".join(f"{k} {v}" for k, v in ns.cells(r, R).items()))
                u.append("\n".join(lines))
        else:
            u.append(f"In {tl(1)}, " + ns.row_text(init, S, hid_np) + ".")
            for s, r in zip(S, trans):
                u.append(f"If {hid_np} is {s} in one {unit}, then in the next {unit} " + ns.row_text(r, S, "it") + ".")
            for nm, R, rows in sensors:
                for s, r in zip(S, rows):
                    u.append(f"When {hid_np} is {s}, " + ns.row_text(r, R, f"the reading of {nm}") + ".")
        u.append(pick(rng, [
            f"Each reading depends only on {hid_np} in that {unit}.",
            f"Readings are independent of each other given the hidden states.",
        ]))
        prelude = ("\n" if layout == "bullets" else " ").join(u)

    def render(k, o):
        vals = o[1]
        h = times[k]
        items = [(sensors[s][0], sensors[s][1][v]) for s, v in zip(step_sens[k], vals)]
        if layout == "json":
            return json_line({unit: h if lab == "num" else tl(h), **{nm: v for nm, v in items}})
        body = join_list([f"{nm} reads {v}" for nm, v in items])
        return cap(pick_form[k].format(t=tl(h), b=body))

    forms = ["In {t}, {b}.", "{t}: {b}.", "Observation for {t}: {b}."]
    pick_form = [pick(rng, forms) for _ in range(T)]
    alts_by_step = [list(itertools.product(*[range(len(sensors[s][1])) for s in step_sens[k]])) for k in range(T)]
    Lcache = {}

    def lik(o, past):
        k, vals = o
        key = (k, vals)
        if key not in Lcache:
            l = np.ones(len(Z))
            for s, v in zip(step_sens[k], vals):
                l = l * E[s][Z[:, times[k] - 1], v]
            Lcache[key] = l
        return Lcache[key]

    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: [(k, a) for a in alts_by_step[k]],
                 lik=lik, render=render, prelude=prelude, mart_var=f"state_{last}", exchangeable=False,
                 meta={"style": "json" if layout == "json" else layout, "theme": dom["name"], "K": K, "H": H,
                       "sensors": n_sens})


def _ordinal(n):
    words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
             "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth"]
    return words[n - 1] if n <= len(words) else f"{n}th"
