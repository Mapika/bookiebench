"""BookieBench v2 `programs` pack: random probabilistic programs rendered as code, exact posteriors by enumeration.

Families:
  prog_abstract  grammar-generated programs with abstract variable names
  prog_domain    domain templates (retry logic, A/B routing, load balancers, loot drops, caches, alerting, ...)

`generate(family, split, idx, seed)` is deterministic in its arguments. Splits: dev, train, test.
"""
from __future__ import annotations

import zlib

import numpy as np

from bookiebench.sims.world import build_instance

from . import gen_abstract, gen_domain, queryfix

PACK = "programs"
DECIMALS = 8
FAMILIES = {"prog_abstract": gen_abstract, "prog_domain": gen_domain}
SPLIT_CODE = {"dev": 11, "train": 12, "test": 13}


def instance_rng(family: str, split: str, idx: int, seed: int = 0):
    return np.random.default_rng([int(seed), zlib.crc32(PACK.encode()), zlib.crc32(family.encode()),
                                  SPLIT_CODE[split], int(idx)])


def make_world(family: str, split: str, idx: int, seed: int = 0):
    rng = instance_rng(family, split, idx, seed)
    return FAMILIES[family].make_world(rng), rng


MAX_TRIES = 40


def generate(family: str, split: str, idx: int, seed: int = 0, return_trace: bool = False, return_world: bool = False):
    rng = instance_rng(family, split, idx, seed)
    # resample the world while the realised evidence pins a query variable (its marginal would be trivially certain)
    for attempt in range(MAX_TRIES):
        world = FAMILIES[family].make_world(rng)
        inst, trace = build_instance(world, rng, family=family, split=split, idx=idx, return_trace=True,
                                     decimals=DECIMALS)
        if not queryfix.pinned_vars(inst):
            break
    fix = queryfix.redraw_trivial(inst, world.variables, rng, queryfix.world_templates())
    inst["meta"]["redrawn_queries"] = fix["before"] - fix["after"]
    level = world.meta.get("level", "L3")
    inst = {"id": inst["id"], "pack": PACK, "family": family, "level": level, "transform": "none",
            **{k: v for k, v in inst.items() if k not in ("id", "family")}}
    if return_world:
        return inst, trace, world
    return (inst, trace) if return_trace else inst
