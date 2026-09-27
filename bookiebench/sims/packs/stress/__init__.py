"""BookieBench v2 `stress` pack: transforms over instances of ANY family (see BOOKIEBENCH_V2.md).

A transform maps a source instance to a new one with the same world (variables' option indices, every step's joint,
gold, mart_var, perms and the original queries' logical content unchanged) and new surface text; some add queries,
whose exact answers always come from the joint via `bookiebench.sims.core.exact_answer`. Outputs are tagged with
`transform`, `pack="stress"`, `level="L4"`, `family` (= source family), `split="stress"`, `meta.source_id`,
`meta.source_split` and `meta.stress` (transform details).

Registry
  TRANSFORMS       name -> fn(instance, seed=0) for the deterministic transforms
  SCALING          fn(instance, pool, seed=0): composes independent sub-worlds (up to 4096 cells; >4 variables, so it
                   is validated with `scaling.validate_composite` instead of `core.validate`)
  LLM_TRANSFORMS   produced by `llm_run` with a local LLM (paraphrase, lang_de/es/hu/pt/zh, framing)
  FAMILIES         {} (this pack defines no world families)

Data: `python -m bookiebench.sims.packs.stress.build` (deterministic transforms) and `...stress.llm_run` (LLM rewrites)
write data/v2/stress/<transform>/<source family>.jsonl.
"""
from __future__ import annotations

from functools import partial

from . import bias, formats, longctx, queries, scaling

TRANSFORMS = {
    "base_rate": bias.base_rate,
    "conjunction": bias.conjunction,
    "anchoring": bias.anchoring,
    "negation": queries.negation,
    "disjunction": queries.disjunction,
    "nested": queries.nested,
    **{f"format_{f}": partial(formats.format_shift, fmt=f) for f in formats.FORMATS},
    **{f"long_{t // 1024}k": partial(longctx.long_context, target=t) for t in longctx.TARGETS},
}
SCALING = scaling.scaling
LLM_TRANSFORMS = ["paraphrase", "lang_de", "lang_es", "lang_hu", "lang_pt", "lang_zh", "framing"]
ALL_TRANSFORMS = list(TRANSFORMS) + ["scaling"] + LLM_TRANSFORMS
FAMILIES: dict = {}
