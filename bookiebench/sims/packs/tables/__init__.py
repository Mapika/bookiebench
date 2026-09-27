"""BookieBench v2 `tables` pack: a table of records / logs in context; questions under the table's empirical
distribution, so exact answers are ratios of counts.

Families:
  tab_pick    10-200 rows; one row is selected uniformly at random, facts about it are revealed step by step
  tab_stream  20-200 rows arriving in 2-5 batches; questions about all rows received so far
  tab_big     scale knob: 2k-20k token tables (both modes), level L4

`generate(family, split, idx, seed)` is deterministic in its arguments. Splits: dev, train, test.
"""
from __future__ import annotations

import zlib

import numpy as np

from bookiebench.sims.common import get_tv
from bookiebench.sims.packs.programs import queryfix

from . import build, model
from .schema import DOMAINS

PACK = "tables"
DECIMALS = build.DECIMALS
FAMILIES = {"tab_pick": "pick", "tab_stream": "stream", "tab_big": "big"}
LEVELS = {"tab_pick": "L3", "tab_stream": "L3", "tab_big": "L4"}
SPLIT_CODE = {"dev": 11, "train": 12, "test": 13}
CHARS_PER_TOKEN = {"json": 2.27, "jsonl": 2.29, "log": 2.56, "csv": 1.58, "markdown": 1.78}  # Qwen3.5 tokenizer


def instance_rng(family: str, split: str, idx: int, seed: int = 0):
    return np.random.default_rng([int(seed), zlib.crc32(PACK.encode()), zlib.crc32(family.encode()),
                                  SPLIT_CODE[split], int(idx)])


def _loguniform_int(rng, lo, hi):
    return int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))


def generate(family: str, split: str, idx: int, seed: int = 0, return_trace: bool = False,
             return_world: bool = False, n_rows: int | None = None, target_tokens: int | None = None):
    """n_rows / target_tokens override the family's size distribution (scale knob)."""
    rng = instance_rng(family, split, idx, seed)
    kind = FAMILIES[family]
    world = None
    for attempt in range(100):
        if kind == "big":
            mode = "pick" if rng.random() < 0.6 else "stream"
            tt = target_tokens or _loguniform_int(rng, 2000, 18000)
            dom = DOMAINS[int(rng.integers(len(DOMAINS)))]
            n_cols = int(rng.integers(4, 8))
            # probe: tokens per row for this domain/format (formats differ a lot)
            probe_rng = np.random.default_rng(int(rng.integers(2**31)))
            cols = model.choose_columns(probe_rng, dom, n_cols)
            prow, _ = model.generate_table(probe_rng, dom, 20, cols)
            columns = [dom["id_col"], dom["time_col"]] + [c[1] for c in cols]
            specs = {c[1]: c[2] for c in cols if c[0] == "num"}
            fmt = model.FORMATS[int(rng.integers(len(model.FORMATS)))]
            per_row = len(model.render_rows(prow, columns, fmt, specs)) / 20
            n = n_rows or max(60, int(tt * CHARS_PER_TOKEN[fmt] / per_row))
            n_cols = cols
            extra = {"target_tokens": tt}
        else:
            mode = kind
            n = n_rows or (_loguniform_int(rng, 10, 200) if kind == "pick" else _loguniform_int(rng, 20, 200))
            dom, n_cols, fmt, extra = None, None, None, {}
        try:
            if mode == "pick":
                world = build.pick_world(rng, n, domain=dom, n_cols=n_cols, fmt=fmt)
                inst, trace = build.pick_instance(world, rng, family=family, split=split, idx=idx)
                variables, ent = world.variables, world.meta["entity"]
                tpl = build.PICK_V2 if get_tv() >= 2 else build.PICK_TPL
            else:
                inst, trace = build.stream_instance(rng, n, family=family, split=split, idx=idx, domain=dom,
                                                    n_cols=n_cols, fmt=fmt)
                variables, ent = trace["variables"], inst["meta"]["entity"]
                tpl = build.STREAM_V2 if get_tv() >= 2 else build.STREAM_TPL
        except build.Reject:
            continue
        # resample while the revealed facts pin a query column (its marginal would be trivially certain)
        if queryfix.pinned_vars(inst) and attempt < 60:
            continue
        break
    else:
        raise RuntimeError("could not generate a table instance")
    fix = queryfix.redraw_trivial(inst, variables, rng, tpl, fmt_kw={"ent": ent, "ents": ent + "s"})
    extra["redrawn_queries"] = fix["before"] - fix["after"]
    inst["meta"].update(extra)
    inst = {"id": inst["id"], "pack": PACK, "family": family, "level": LEVELS[family], "transform": "none",
            **{k: v for k, v in inst.items() if k not in ("id", "family")}}
    if return_world:
        return inst, trace, world
    return (inst, trace) if return_trace else inst
