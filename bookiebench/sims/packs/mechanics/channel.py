"""Noisy-channel decoding: a message from a stated prior is encoded (repetition, single parity, or a small block
code), sent over a binary symmetric channel whose flip rate depends on a latent line condition, and the received
bits arrive one per step (optionally with erasures whose rate also depends on the line).

Latent = (message, line condition, a future retransmission of one position). Received bits are conditionally
independent given the latent, so the evidence is exchangeable. Variables: the message, the line condition, the value
a retransmitted copy of a named position will be received as.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, per_step, pick,
                    pseudo, round_pcts)

NAME = "channel"


def _code(rng, level, scale):
    kind = pick(rng, ["rep"] if level == 0 else (["rep", "parity"] if level == 1 else ["parity", "block", "rep"]))
    if kind == "rep":
        n = int(rng.integers(3, 6))
        return kind, ["0", "1"], [[0] * n, [1] * n]
    if kind == "parity":
        k = 2 if level < 2 else pick(rng, [2, 3])
        msgs = list(itertools.product((0, 1), repeat=k))
        return kind, ["".join(map(str, m)) for m in msgs], [list(m) + [sum(m) % 2] for m in msgs]
    # a small block code: 4 codewords of length 5 at distance >= 3
    base = [[0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 1, 0, 1, 1]]
    perm = rng.permutation(5)
    cws = [[c[i] for i in perm] for c in base]
    return kind, ["A", "B", "C", "D"], cws


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    kind, labels, cws = _code(rng, level, scale)
    M, n = len(cws), len(cws[0])
    mprior = round_pcts(rng.dirichlet(np.full(M, 2.0)), 100, 3) if rng.random() < 0.85 else None
    pm = np.array(mprior if mprior else [1] * M, float)
    pm /= pm.sum()
    two_state = level >= 1
    flips = sorted(int(x) for x in rng.choice(np.arange(1, 46) if two_state else np.arange(2, 40),
                                              size=2 if two_state else 1, replace=False))
    p_bad = int(rng.integers(5, 71)) if two_state else 0
    erasures = level == 2 and rng.random() < 0.5
    er = sorted(int(x) for x in rng.choice(np.arange(2, 46), size=2, replace=False)) if erasures else [0, 0]
    T = min(n, n_steps(rng, level, 3, 5) + (1 if kind == "block" else 0))
    if kind == "rep" and level == 0:
        T = min(T, n)
    order = [int(i) for i in rng.permutation(n)][:T]
    retx = int(rng.integers(n))
    cw = np.array(cws)
    rows = list(itertools.product(range(M), range(2 if two_state else 1), (0, 1)))
    Mz = np.array([r[0] for r in rows])
    Lz = np.array([r[1] for r in rows])
    Rz = np.array([r[2] for r in rows])
    fz = np.array(flips, float)[Lz] / 100
    ez = np.array(er, float)[Lz] / 100 if erasures else np.zeros(len(rows))
    sent_retx = cw[Mz, retx]
    p_retx = np.where(Rz == sent_retx, 1 - fz, fz)  # the retransmitted copy is never erased
    prior = pm[Mz] * np.where(Lz == 1, p_bad / 100, 1 - p_bad / 100) * p_retx
    vs = ["msg", "line", "retx"] if two_state else ["msg", "retx"]
    if level == 1 and rng.random() < 0.4:
        vs = ["msg", "line"]
    cols = {"msg": Mz, "line": Lz, "retx": Rz}
    proj = np.stack([cols[v] for v in vs], 1)
    link = f"the {pseudo(rng)} {pick(rng, ['radio link', 'serial cable', 'telegraph line', 'deep-space relay', 'laser link'])}"
    lnames = ["quiet", "noisy"]
    mword = {"rep": "bit", "parity": "data word", "block": "codeword"}[kind]
    V = {"msg": Var("message", list(labels), [f"Which {mword} was sent?", f"What was the transmitted {mword}?",
                                              f"Which {mword} did the sender transmit?"],
                    [f"the sender transmitted {mword} {l}" for l in labels]),
         "line": Var("line", lnames, [f"Is {link} quiet or noisy right now?", f"What condition is {link} in?"],
                     [f"{link} is quiet", f"{link} is noisy"]),
         "retx": Var("retransmission", ["0", "1"], [f"When position {retx + 1} is retransmitted, what will be received?",
                                                    f"What will the retransmitted copy of position {retx + 1} read?"],
                     [f"the retransmitted position {retx + 1} arrives as 0", f"the retransmitted position {retx + 1} arrives as 1"])}
    variables = [V[v] for v in vs]
    if kind == "rep":
        enc = f"the sender repeats a single bit {n} times"
    elif kind == "parity":
        enc = (f"the sender transmits a {len(labels[0])}-bit data word followed by one parity bit (the sum of the data "
               f"bits modulo 2), {n} bits in all")
    else:
        enc = "the sender picks one of four codewords: " + join_list([f"{l} = {''.join(map(str, c))}" for l, c in zip(labels, cws)])
    ptxt = ("The message is " + join_list([f"{l} with probability {fmt.p(x)}" for l, x in zip(labels, mprior)])
            if mprior else f"Every possible {mword} is equally likely a priori")
    if two_state:
        ctxt = (f"{cap(link)} is noisy with probability {fmt.p(p_bad)} (and quiet otherwise) for the whole "
                f"transmission; each bit is flipped independently with probability {fmt.p(flips[0])} when quiet and "
                f"{fmt.p(flips[1])} when noisy")
    else:
        ctxt = f"Each bit sent over {link} is flipped independently with probability {fmt.p(flips[0])}"
    etxt = (f" A bit may also be lost (erased) in transit, independently, with probability {fmt.p(er[0])} when quiet "
            f"and {fmt.p(er[1])} when noisy." if erasures else "")
    same = "in the same (quiet or noisy) condition" if two_state else "with the same flip probability"
    rtxt = (f" Later, position {retx + 1} will be retransmitted once more over the same line, {same}; the retransmission "
            f"is acknowledged and resent until it arrives, so it is never erased, but it can still be flipped like any "
            f"other bit." if "retx" in vs else "")
    js = is_json(rng)
    if js:
        st = {"link": link, "encoding": enc, "prior": ({l: fmt.p(x) for l, x in zip(labels, mprior)} if mprior else "uniform"),
              "flip_probability": ({"quiet": fmt.p(flips[0]), "noisy": fmt.p(flips[1])} if two_state else fmt.p(flips[0]))}
        if two_state:
            st["P(noisy)"] = fmt.p(p_bad)
            st["line_condition"] = "fixed for the whole transmission (and the retransmission)"
        if erasures:
            st["erasure_probability"] = {"quiet": fmt.p(er[0]), "noisy": fmt.p(er[1])}
        if "retx" in vs:
            st["retransmission"] = (f"position {retx + 1}, sent once more over the same line {same}; acknowledged and "
                                    f"resent until it arrives, so never erased, but flipped like any other bit")
        prelude = json_block(rng, st)
        tpl = [lambda i, b: json_line({"position": i + 1, "received": "erased" if b == 2 else b}),
               lambda i, b: json_line({"rx": {f"bit{i + 1}": "?" if b == 2 else str(b)}})]
    else:
        prelude = pick(rng, [
            f"Over {link}, {enc}. {ptxt}. {ctxt}.{etxt}{rtxt}",
            f"A receiver listens to {link}. Encoding: {enc}. {ctxt}.{etxt} {ptxt}.{rtxt}",
            f"{ptxt}; {enc}, and the bits travel over {link}. {ctxt}.{etxt}{rtxt}",
        ])
        tpl = [lambda i, b: f"Position {i + 1} is received as {b}." if b != 2 else f"Position {i + 1} was erased in transit.",
               lambda i, b: f"Received bit at position {i + 1}: {'erased' if b == 2 else b}.",
               lambda i, b: f"The receiver reads position {i + 1}: {'nothing (erasure)' if b == 2 else b}."]
    ptpl = per_step(rng, T, tpl)

    def alternatives(k, past):
        return [(k, 0), (k, 1)] + ([(k, 2)] if erasures else [])

    def lik(o, past):
        k, b = o
        pos = order[k]
        if b == 2:
            return ez
        sent = cw[Mz, pos]
        return np.where(sent == b, 1 - fz, fz) * (1 - ez)

    return mk_world(variables, prior, proj, T, alternatives, lik, lambda k, o: ptpl[k](order[k], o[1]), prelude,
                    "message", True, "json" if js else "prose", kind,
                    {"cws": cw, "pm": pm, "flips": flips, "p_bad": p_bad, "two": two_state, "er": er,
                     "erasures": erasures, "order": order, "retx": retx, "vs": vs})


def simulate(world, rng):
    p = world.params
    m = int(rng.choice(len(p["pm"]), p=p["pm"]))
    line = int(p["two"] and rng.random() < p["p_bad"] / 100)
    f = p["flips"][line] / 100
    e = p["er"][line] / 100 if p["erasures"] else 0.0
    obs = []
    for k, pos in enumerate(p["order"]):
        if rng.random() < e:
            obs.append((k, 2))
            continue
        b = int(p["cws"][m, pos])
        obs.append((k, b ^ int(rng.random() < f)))
    r = int(p["cws"][m, p["retx"]]) ^ int(rng.random() < f)
    vals = {"msg": m, "line": line, "retx": r}
    return tuple(vals[v] for v in p["vs"]), obs
