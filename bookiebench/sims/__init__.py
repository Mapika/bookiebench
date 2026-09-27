"""BookieBench simulators: text-rendered latent-variable worlds with exact posteriors.

Registry:
  TRAIN_FAMILIES   (7 hand-built)  -> splits "train", "test" (in-family test), "robust" (test worlds + nuisance text)
  PRIOR_FAMILIES   (procedural)    -> splits "train" and "test_prior" (broad prior: random BNs / random HMMs)
  VAL_FAMILIES     (2 hand-built)  -> split "val" only (checkpoint selection; never trained on)
  HELDOUT_FAMILIES (4 hand-built)  -> split "heldout" only (never trained on, never used for selection)
"""
from __future__ import annotations

import zlib

import numpy as np

from . import (alarm, cards, dice, factory, genetics, hiring, medical, randbn, randhmm, sensor, spam, tracking, urn,
               weather, witness)
from .common import DEFAULT_TV, get_tv, template_version  # noqa: F401
from .core import exact_answer, validate  # noqa: F401
from .world import ROUND, build_instance

FAMILIES = {
    "urn": urn, "dice": dice, "medical": medical, "alarm": alarm, "sensor": sensor, "cards": cards,
    "witness": witness, "factory": factory, "weather": weather, "spam": spam, "hiring": hiring,
    "randbn": randbn, "randhmm": randhmm, "genetics": genetics, "tracking": tracking,
}
TRAIN_FAMILIES = ["urn", "dice", "medical", "alarm", "sensor", "cards", "witness"]
PRIOR_FAMILIES = ["randbn", "randhmm"]
VAL_FAMILIES = ["genetics", "tracking"]
HELDOUT_FAMILIES = ["factory", "weather", "spam", "hiring"]
SPLITS = {
    "train": TRAIN_FAMILIES + PRIOR_FAMILIES,
    "test": TRAIN_FAMILIES,
    "test_prior": PRIOR_FAMILIES,
    "robust": TRAIN_FAMILIES,
    "val": VAL_FAMILIES,
    "heldout": HELDOUT_FAMILIES,
}
_SPLIT_CODE = {"train": 1, "test": 2, "heldout": 3, "test_prior": 4, "val": 5, "robust": 6}


def instance_rng(family: str, split: str, idx: int, seed: int = 0):
    return np.random.default_rng([int(seed), zlib.crc32(family.encode()), _SPLIT_CODE[split], int(idx)])


def nuisance_rng(family: str, split: str, idx: int, seed: int = 0):
    """Independent stream for nuisance decisions/text (does not perturb the world's own rng)."""
    return np.random.default_rng([int(seed), zlib.crc32(family.encode()), _SPLIT_CODE[split], int(idx), 991])


def make_world(family: str, split: str, idx: int, seed: int = 0):
    """(world, rng) for one instance; the rng continues into build_instance (sampling gold + queries)."""
    rng = instance_rng(family, split, idx, seed)
    return FAMILIES[family].make_world(rng), rng


def generate(family: str, split: str, idx: int, seed: int = 0, return_trace: bool = False, nuisance: float = 0.0,
             decimals: int | None = None, tv: int | None = None):
    """See _generate. tv: template version (1 = legacy/internal data, byte-identical; 2 = release default)."""
    with template_version(tv):
        return _generate(family, split, idx, seed, return_trace, nuisance, decimals)


def _generate(family: str, split: str, idx: int, seed: int = 0, return_trace: bool = False, nuisance: float = 0.0,
              decimals: int | None = None):
    """Deterministically generate instance `idx` of `family` in `split`.

    nuisance: probability in [0, 1] that the instance gets nuisance-augmented surface text (decided by an
    independent rng, so the world/posteriors are unchanged). Split "robust" = the "test" world idx with nuisance 1.
    decimals: rounding of joints/probabilities (default: the family's DECIMALS, 10 for the original families).
    """
    from .nuisance import augment

    base_split = "test" if split == "robust" else split
    if split == "robust":
        nuisance = 1.0
    world, rng = make_world(family, base_split, idx, seed)
    dec = decimals if decimals is not None else getattr(FAMILIES[family], "DECIMALS", ROUND)
    out = build_instance(world, rng, family=family, split=base_split, idx=idx, return_trace=return_trace,
                         decimals=dec)
    inst, trace = (out if return_trace else (out, None))
    if nuisance > 0:
        nrng = nuisance_rng(family, split, idx, seed)
        if nrng.random() < nuisance:
            inst = augment(inst, nrng)
    if split == "robust":
        inst["id"] = f"{family}-robust-{idx:06d}"
        inst["split"] = "robust"
        inst["meta"]["paired_with"] = f"{family}-test-{idx:06d}"
    return (inst, trace) if return_trace else inst
