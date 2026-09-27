# Stress pack (release v1)

Each stress transform rewrites or embeds the 11 v1 source families: the 7 in `../test/` and the 4 in `../heldout/`,
300 instances each. The world, the joints, the queries and the gold stay the same, so the exact answers do not change.
Any score difference between a stressed instance and its source comes from the transform.

| transform | release format | kept / source |
|---|---|---|
| anchoring, base_rate, conjunction, disjunction, negation, nested | data | 3,300 / 3,300 each |
| format_chat, format_csv, format_email, format_log, format_markdown | data | 3,300 / 3,300 each |
| paraphrase | data (LLM, two judges) | 2,250 / 3,300 |
| lang_de | data (LLM, two judges) | 2,156 / 3,300 |
| lang_es | data (LLM, two judges) | 2,535 / 3,300 |
| lang_pt | data (LLM, two judges) | 2,451 / 3,300 |
| **lang_zh (BETA)** | data (LLM, two judges + rule filter) | 2,119 / 3,300 |
| long_4k, long_8k, long_16k, long_32k | **recipe** (`recipe.json`) | 3,300 / 3,300 each |
| scaling | **recipe** (`recipe.json`) | 3,300 / 3,300 |
| lang_hu, framing | excluded from v1 | see `manifest.json` → `excluded_v1` |

**LLM transforms.** An item is kept only if both judges pass every segment: prelude, evidence and queries. Judge 1
is the rewriting model, Qwen/Qwen3.8-27B-FP8 at temperature 0. Judge 2 is deepseek-v41-flash. Each item carries both
verdicts in `meta.stress.judge` and `meta.stress.judge2`. Stressed-vs-clean comparisons are unbiased only on
`manifest.json` → `transforms[t][family].kept_source_ids`. Score the clean source restricted to the same ids.

**lang_zh is BETA.** The upstream judge controls plant a query-negation flip in 100 items per transform. In Chinese,
judge 1 detects 0.64 of them and judge 2 detects 0.66. For de, es and pt the rates are 0.96-1.00, and paraphrase is
1.00. Source: `logs/stress_judge_controls_judge.json` and `logs/stress_judge_controls_judge2.json` upstream. A
rule-based double-negation filter removed 64 more lang_zh items (`logs/stress_finalize.json`,
`programmatic_dropped`). Residual meaning errors in kept lang_zh queries are therefore likely. Report lang_zh
separately and keep it out of pooled stress numbers.

**Recipes.** long_* embed third-party filler text: CNN/DailyMail, which is ids-only in the licence review, plus
WikiQA and 20 Newsgroups. That text is not redistributed. scaling is fully determined by the shipped sources and is
about 260 MB, so it is rebuilt instead of shipped. Rebuild both with:

```bash
pip install -e ".[stress]"                                       # transformers + tokenizers + pyarrow (tokenizer only, no torch)
bookiebench rebuild-stress --transforms scaling                  # no external data needed
# long_*: fetch the pinned filler datasets and tokenizer once (commands in recipe.json -> filler / tokenizer), then
HF_HUB_OFFLINE=1 bookiebench rebuild-stress --long
bookiebench rebuild-stress --long --sample 3 --check             # quick verification only
```

Every rebuilt instance is checked against its sha256 in `recipe.json`, and every full file against its file hash. The
token counts that place the filler depend on the tokenizer, so the rebuild pins it: Qwen/Qwen3.5-2B-Base at revision
b1485b2f. It was tested with transformers 5.17.0, tokenizers 0.23.2 and pyarrow 25.0.1. The rebuild writes to
`long_*/` and `scaling/` here; both are git-ignored.
