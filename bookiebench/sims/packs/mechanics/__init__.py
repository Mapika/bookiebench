"""BookieBench v2 `mechanics` pack: closed-form / enumerable worlds with diverse reasoning mechanics.

    from bookiebench.sims.packs.mechanics import FAMILIES, generate
    inst = generate("minesweeper", "dev", 0)

Instances follow SPEC section 1 and carry `pack="mechanics"`, `level` ("L0".."L2") and `transform="none"`.
Generation is deterministic in (seed, family, split, idx, scale).
"""
from __future__ import annotations

import numpy as np

from bookiebench.sims import core
from bookiebench.sims.world import build_instance

from . import (birthday, blackjack, channel, epidemic, forensic, gauge, inventory, matching, minesweeper, montyhall,
               poker, queue, raters, recapture, reliability, search, tournament)
from ._base import PACK, draw_level, instance_rng

_MODULES = [blackjack, poker, minesweeper, montyhall, birthday, matching, reliability, tournament, channel, search,
            epidemic, gauge, queue, inventory, forensic, raters, recapture]
FAMILIES = {m.NAME: m for m in _MODULES}
SPLITS = {"train": list(FAMILIES), "dev": list(FAMILIES)}


def make_world(family: str, split: str, idx: int, seed: int = 0, scale: int = 1, level: int | None = None):
    """(world, rng, level); the rng continues into build_instance."""
    rng = instance_rng(family, split, idx, seed, scale)
    lv = draw_level(rng)
    if level is not None:
        lv = int(level)
    return FAMILIES[family].make_world(rng, level=lv, scale=scale), rng, lv


def generate(family: str, split: str, idx: int, seed: int = 0, scale: int = 1, level: int | None = None,
             decimals: int | None = None, nuisance: float = 0.0, return_trace: bool = False):
    world, rng, lv = make_world(family, split, idx, seed, scale, level)
    dec = decimals if decimals is not None else (8 if split == "train" else 10)
    inst, trace = build_instance(world, rng, family=family, split=split, idx=idx, return_trace=True, decimals=dec)
    inst["pack"] = PACK
    inst["level"] = f"L{lv}"
    inst["transform"] = "none"
    inst["meta"]["level"] = lv
    inst["meta"]["scale"] = int(scale)
    if nuisance > 0:
        from bookiebench.sims.nuisance import augment
        nrng = np.random.default_rng([int(seed), idx, 7331, len(family)])
        if nrng.random() < nuisance:
            inst = augment(inst, nrng)
    return (inst, trace) if return_trace else inst


def validate(inst, strict: bool = True) -> bool:
    """core.validate, allowing > 256 cells for scale > 1 instances."""
    cells = int(np.prod([len(v["options"]) for v in inst["variables"]]))
    if cells <= core.MAX_CELLS:
        return core.validate(inst, strict)
    old = core.MAX_CELLS
    core.MAX_CELLS = cells
    try:
        return core.validate(inst, strict)
    finally:
        core.MAX_CELLS = old
