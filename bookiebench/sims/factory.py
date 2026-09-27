"""Factory quality control: a batch comes from one of several machines with different defect rates, and an imperfect
inspector flags sampled units. Flags are i.i.d. given the machine (exchangeable).

Variables: source machine, whether the next unit is truly defective, whether the inspector will flag it.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, COMPANIES, Fmt, cap, composition, join_list, json_block, json_line, person, pick, rand_pct, sample
from .world import Var, World

PRODUCTS = [("bottle", "bottles"), ("gear", "gears"), ("circuit board", "circuit boards"), ("tile", "tiles"),
            ("valve", "valves"), ("lens", "lenses"), ("battery cell", "battery cells")]
MACHINE_NAMES = [["Line 1", "Line 2", "Line 3"], ["the old press", "the new press", "the backup press"],
                 ["Machine A", "Machine B", "Machine C"], ["the day shift", "the night shift", "the weekend shift"]]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    firm = pick(rng, COMPANIES)
    insp = person(rng)
    unit, units = pick(rng, PRODUCTS)
    K = int(rng.integers(2, 4))
    labels = pick(rng, MACHINE_NAMES)[:K]
    share = [x * 5 for x in composition(rng, K, 20, min_each=1)]
    rates = sorted(sample(rng, [1, 2, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40], K))
    rates = [rates[int(i)] for i in rng.permutation(K)]
    sens = rand_pct(rng, 70, 99, 1)
    fa = rand_pct(rng, 1, 15, 1)
    T = int(rng.integers(3, 7))

    Z = np.array(list(itertools.product(range(K), range(2), range(2))))  # machine, defective(0=yes), flagged(0=yes)
    m, dfc, fl = Z[:, 0], Z[:, 1], Z[:, 2]
    d = np.array(rates, float)[m] / 100
    pdf = np.where(dfc == 0, d, 1 - d)
    pfl_given = np.where(dfc == 0, sens / 100, fa / 100)
    pfl = np.where(fl == 0, pfl_given, 1 - pfl_given)
    prior = np.array(share, float)[m] / 100 * pdf * pfl
    q = d * sens / 100 + (1 - d) * fa / 100  # P(flag | machine)
    Lf = np.stack([q, 1 - q], 1)

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {
            "plant": f"{firm} {unit} plant", "batch_source": "hidden, one of " + ", ".join(labels),
            "P(source)": {l: fmt.p(s) for l, s in zip(labels, share)},
            "defect_rate": {l: fmt.p(r) for l, r in zip(labels, rates)},
            "inspector": {"name": insp, "P(flag | defective)": fmt.p(sens), "P(flag | good)": fmt.p(fa)},
            "sampling": "units drawn at random from a very large batch; units independent given the source",
        })
        render = lambda k, o: json_line({"inspected_unit": "flagged" if o == 0 else "passed"})  # noqa: E731
    else:
        s_src = (f"A large batch of {units} at the {firm} plant was produced entirely by one source, but the label is "
                 f"missing. " + cap(join_list([f"{l} makes {fmt.p(s)} of all batches" for l, s in zip(labels, share)]))
                 + ".")
        s_def = cap(join_list([f"{l} produces defective {units} at a rate of {fmt.p(r)}" for l, r in
                               zip(labels, rates)])) + "."
        s_in = (f"Inspector {insp} flags a defective {unit} {fmt.p(sens)} of the time and wrongly flags a good one "
                f"{fmt.p(fa)} of the time.")
        prelude = pick(rng, [
            f"{s_src} {s_def} {s_in} {cap(units)} are sampled at random from the batch and inspected one at a time.",
            f"Quality control at {firm}. {s_def} {s_src} {s_in} Each sampled {unit} is independent of the others "
            f"given the source.",
            f"{s_in} {insp} is checking a batch of {units}. {s_src} {s_def}",
        ])
        if get_tv() >= 2 and "sampled at random" not in prelude:
            prelude += (f" {cap(units)} are sampled at random from the batch; given the source, each sampled {unit} "
                        f"is independent of the others.")
        ev = [lambda f: f"A sampled {unit} is {'flagged as defective' if f == 0 else 'passed'} by {insp}.",
              lambda f: f"Inspection result: {'flagged' if f == 0 else 'passed'}.",
              lambda f: f"{insp} {'flags' if f == 0 else 'passes'} the {unit} just checked."]
        pos = [pick(rng, ev) for _ in range(T)]
        render = lambda k, o: pos[k](o)  # noqa: E731

    variables = [
        Var("source", labels, ["Which source produced this batch?", f"Where did this batch of {units} come from?"],
            "the batch was produced by {opt}"),
        Var("next_defective", ["defective", "not defective"],
            [f"Is the next {unit} sampled from the batch truly defective?",
             f"Will the next sampled {unit} actually be defective?"],
            [f"the next sampled {unit} is truly defective", f"the next sampled {unit} is not defective"]),
        Var("next_flagged", ["flagged", "passed"], [f"Will {insp} flag the next sampled {unit}?"],
            [f"{insp} will flag the next sampled {unit}", f"{insp} will pass the next sampled {unit}"]),
    ]
    return World(variables, prior, Z, T, alternatives=lambda k, past: [0, 1], lik=lambda o, past: Lf[:, o],
                 render=render, prelude=prelude, mart_var="source", exchangeable=True,
                 meta={"style": "json" if json_style else "prose"})
