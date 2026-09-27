"""System reliability: series / parallel / k-of-n structures with independent component failures, an optional
shared supplier batch (common cause), imperfect diagnostics and block status lights.

Latent = (batch quality, up/down state of every component): <= 2 * 2^7 rows. Evidence: diagnostic reports on
components (stated detection and false-alarm rates) and exact block indicators; conditionally independent given the
latent, so exchangeable. Variables: whether the whole system works, one component's state, batch quality, one
block's state.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import (Fmt, Var, repeat_tags, tag_text, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, per_step, pick,
                    pseudo)

NAME = "reliability"
PARTS = [("pump", "P"), ("valve", "V"), ("sensor", "S"), ("relay", "R"), ("battery", "B"), ("fan", "F"),
         ("controller", "C"), ("link", "L")]
SYSTEMS = ["cooling loop", "irrigation rig", "signal repeater", "ventilation unit", "backup generator", "lift drive"]


def _blocks(rng, level, scale):
    target = {0: int(rng.integers(2, 4)), 1: int(rng.integers(4, 6)), 2: int(rng.integers(5, 8))}[level] + (scale - 1)
    blocks, n = [], 0
    while n < target:
        room = target - n
        opts = ["single"] + (["parallel"] if room >= 2 else []) + (["2of3"] if room >= 3 else [])
        b = pick(rng, opts)
        size = {"single": 1, "parallel": 2, "2of3": 3}[b]
        blocks.append((b, list(range(n, n + size))))
        n += size
    if len(blocks) == 1 and blocks[0][0] == "single":
        blocks = [("parallel", [0, 1])] if n == 1 else blocks
        n = max(n, 2)
    return blocks, n


def make_world(rng, level, scale=1):
    fmt = Fmt(rng)
    blocks, n = _blocks(rng, level, scale)
    kinds = [PARTS[int(i)] for i in rng.choice(len(PARTS), size=len(blocks), replace=True)]
    names, ptype = [], []
    for bi, (b, comps) in enumerate(blocks):
        part, letter = kinds[bi]
        for c in comps:
            names.append(f"{letter}{c + 1}")
            ptype.append(part)
    sysname = f"the {pseudo(rng)} {pick(rng, SYSTEMS)}"
    fail = [int(rng.choice(np.arange(5, 45, 5))) for _ in range(n)]
    has_batch = level == 2 or (level == 1 and rng.random() < 0.5)
    batch_members = []
    if has_batch:
        k = max(2, n // 2)
        batch_members = sorted(int(i) for i in rng.choice(n, size=k, replace=False))
        pb = int(rng.choice(np.arange(10, 55, 5)))
        f_good = int(rng.choice(np.arange(5, 25, 5)))
        f_bad = int(rng.choice(np.arange(f_good + 15, 75, 5)))
        for c in batch_members:
            fail[c] = f_good
    else:
        pb, f_good, f_bad = 0, 0, 0
    supplier = pseudo(rng)

    rows = np.array(list(itertools.product(range(2 if has_batch else 1), *([range(2)] * n))))
    B, S = rows[:, 0], rows[:, 1:]  # S[:, c] = 1 if component c works
    fp = np.array(fail, float)[None, :].repeat(len(rows), 0) / 100
    if has_batch:
        fp[:, batch_members] = np.where(B[:, None] == 1, f_bad, f_good) / 100
    prior = np.prod(np.where(S == 1, 1 - fp, fp), 1) * (np.where(B == 1, pb / 100, 1 - pb / 100) if has_batch else 1)

    def block_up(Sx, blk):
        b, comps = blk
        up = Sx[..., comps].sum(-1)
        return up >= {"single": 1, "parallel": 1, "2of3": 2}[b]

    SYS = np.all(np.stack([block_up(S, b) for b in blocks], -1), -1)
    comp_var = int(rng.integers(n))
    multi = [i for i, (b, _) in enumerate(blocks) if b != "single"]
    blk_var = int(pick(rng, multi)) if multi and level == 2 and len(blocks) >= 2 else None
    T = n_steps(rng, level, 3, 5)
    sens = int(rng.choice(np.arange(70, 100, 5)))
    fa = int(rng.choice(np.arange(2, 21, 2)))
    ev = []
    for _ in range(T):
        unused = [b for b in multi if ("light", b) not in ev]
        if unused and rng.random() < 0.25:  # a light is exact, so it is never repeated
            ev.append(("light", int(pick(rng, unused))))
        else:
            ev.append(("test", int(rng.integers(n))))

    vs = ["system", "comp"] + (["batch"] if has_batch and level >= 1 else []) + (["block"] if blk_var is not None else [])
    vs = vs[:4]
    cols = []
    for v in vs:
        if v == "system":
            cols.append((~SYS).astype(int))
        elif v == "comp":
            cols.append(1 - S[:, comp_var])
        elif v == "batch":
            cols.append(B)
        else:
            cols.append((~block_up(S, blocks[blk_var])).astype(int))
    proj = np.stack(cols, 1)

    def bdesc(blk):
        b, comps = blk
        nm = [names[c] for c in comps]
        if b == "single":
            return f"{ptype[comps[0]]} {nm[0]}"
        if b == "parallel":
            return f"the redundant {ptype[comps[0]]} pair {nm[0]}/{nm[1]} (at least one must work)"
        return f"the {ptype[comps[0]]} bank {', '.join(nm)} (at least 2 of the 3 must work)"

    def bshort(blk):
        return "the " + "/".join(names[c] for c in blk[1]) + " block"

    V = {"system": Var("system", ["working", "failed"], [f"Is {sysname} working?", f"Does {sysname} function?",
                                                          f"What is the state of {sysname}?"],
                       [f"{sysname} is working", f"{sysname} has failed"]),
         "comp": Var(f"component_{names[comp_var]}", ["working", "failed"],
                     [f"Is component {names[comp_var]} working?", f"Has {ptype[comp_var]} {names[comp_var]} failed?",
                      f"What is the state of {names[comp_var]}?"],
                     [f"{names[comp_var]} is working", f"{names[comp_var]} has failed"]),
         "batch": Var("batch", ["sound", "defective"], [f"Is the {supplier} batch sound or defective?",
                                                        f"Was the {supplier} batch defective?"],
                      [f"the {supplier} batch is sound", f"the {supplier} batch is defective"]),
         }
    if blk_var is not None:
        V["block"] = Var("block", ["up", "down"], [f"Is {bshort(blocks[blk_var])} up?",
                                                   f"Is {bshort(blocks[blk_var])} functioning?"],
                         [f"{bshort(blocks[blk_var])} is up", f"{bshort(blocks[blk_var])} is down"])
    variables = [V[v] for v in vs]

    structure = join_list([bdesc(b) for b in blocks])
    ftxt = join_list([f"{names[c]} {fmt.p(fail[c])}" for c in range(n) if c not in batch_members])
    btxt = (f"Components {join_list([names[c] for c in batch_members])} {'both' if len(batch_members) == 2 else 'all'} come from one {supplier} batch, which is "
            f"defective with probability {fmt.p(pb)}; each of them fails with probability {fmt.p(f_bad)} if the batch "
            f"is defective and {fmt.p(f_good)} if it is sound.") if has_batch else ""
    rtag = [f"run {t}" if t else "" for t in repeat_tags(ev)]
    dtxt = (f"A diagnostic on a failed component reports FAULT with probability {fmt.p(sens)}; on a working one it "
            f"wrongly reports FAULT with probability {fmt.p(fa)}. Repeated diagnostics of the same component (labelled run A, "
            f"run B, ...) are independent given its state.")
    ltxt = "Each redundant block has a status light that shows exactly whether the block is up."
    js = is_json(rng)
    if js:
        st = {"system": sysname, "works_if_all_of": [bdesc(b) for b in blocks],
              "failure_probability": {names[c]: fmt.p(fail[c]) for c in range(n) if c not in batch_members},
              "independence": "components fail independently" + (" given the batch" if has_batch else ""),
              "diagnostic": {"P(FAULT | failed)": fmt.p(sens), "P(FAULT | working)": fmt.p(fa),
                             "repeats": "repeated diagnostics (run A, run B, ...) are independent given the state"}}
        if has_batch:
            st["shared_batch"] = {"name": f"the {supplier} batch", "members": [names[c] for c in batch_members], "P(defective)": fmt.p(pb),
                                  "fail_if_defective": fmt.p(f_bad), "fail_if_sound": fmt.p(f_good)}
        if multi:
            st["status_lights"] = "exact up/down light on every redundant block"
        prelude = json_block(rng, st)
        tt = [lambda c, v: json_line({"diagnostic": names[c], "result": "FAULT" if v else "OK"}),
              lambda c, v: json_line({"component": names[c], "test": "FAULT" if v else "OK"})]
        tl = [lambda b, v: json_line({"status_light": bshort(blocks[b]), "shows": "up" if v else "down"})]
    else:
        s1 = pick(rng, [f"{cap(sysname)} works only if all of these work: {structure}.",
                        f"For {sysname} to run, every stage must be working: {structure}.",
                        f"{cap(sysname)} is a chain of stages ({structure}); it fails if any stage fails."])
        s2 = (f"Failure probabilities for this run: {ftxt}; components fail independently"
              f"{' of one another' if not has_batch else ' given the batch'}." if ftxt else
              "All components fail independently given the batch.")
        parts = [s1, s2, btxt, dtxt] + ([ltxt] if multi else [])
        prelude = " ".join(p for p in pick(rng, [parts, [parts[0], parts[2], parts[1]] + parts[3:],
                                                  parts[:2] + parts[3:] + [parts[2]]]) if p)
        tt = [lambda c, v: f"A diagnostic on {names[c]} reports {'FAULT' if v else 'OK'}.",
              lambda c, v: f"The technician tests {names[c]}: {'FAULT' if v else 'OK'}.",
              lambda c, v: f"Test result for {ptype[c]} {names[c]}: {'FAULT' if v else 'OK'}."]
        tl = [lambda b, v: f"The status light of {bshort(blocks[b])} shows {'up' if v else 'down'}.",
              lambda b, v: f"{cap(bshort(blocks[b]))} light: {'up' if v else 'down'}.",
              lambda b, v: f"According to its status light, {bshort(blocks[b])} is {'up' if v else 'down'}."]
    ptt, ptl = per_step(rng, T, tt), per_step(rng, T, tl)
    BU = {b: block_up(S, blocks[b]) for b in multi}

    def lik(o, past):
        k, v = o
        kd, x = ev[k]
        if kd == "light":
            return (BU[x] == bool(v)).astype(float)
        pf = np.where(S[:, x] == 0, sens / 100, fa / 100)
        return pf if v else 1 - pf

    def render(k, o):
        _, v = o
        kd, x = ev[k]
        return tag_text(ptt[k](x, v), rtag[k]) if kd == "test" else ptl[k](x, v)

    keep = prior > 0
    return mk_world(variables, prior[keep], proj[keep], T, lambda k, past: [(k, 0), (k, 1)],
                    lambda o, past: lik(o, past)[keep], render, prelude, "system", True, "json" if js else "prose",
                    "reliability",
                    {"blocks": blocks, "n": n, "fail": fail, "has_batch": has_batch, "members": batch_members,
                     "pb": pb, "f_good": f_good, "f_bad": f_bad, "sens": sens, "fa": fa, "ev": ev, "vs": vs,
                     "comp": comp_var, "blk": blk_var})


def simulate(world, rng):
    p = world.params
    bad = bool(p["has_batch"] and rng.random() < p["pb"] / 100)
    f = np.array(p["fail"], float) / 100
    for c in p["members"]:
        f[c] = (p["f_bad"] if bad else p["f_good"]) / 100
    up = rng.random(p["n"]) >= f

    def bu(blk):
        b, comps = blk
        return up[comps].sum() >= (2 if b == "2of3" else 1)

    system = all(bu(b) for b in p["blocks"])
    a = []
    for v in p["vs"]:
        a.append({"system": int(not system), "comp": int(not up[p["comp"]]), "batch": int(bad),
                  "block": int(not bu(p["blocks"][p["blk"]])) if p["blk"] is not None else 0}[v])
    obs = []
    for k, (kd, x) in enumerate(p["ev"]):
        if kd == "light":
            obs.append((k, int(bu(p["blocks"][x]))))
        else:
            pf = p["sens"] / 100 if not up[x] else p["fa"] / 100
            obs.append((k, int(rng.random() < pf)))
    return tuple(a), obs
