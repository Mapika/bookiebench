# Changelog

All notable changes to BookieBench are listed here. Versions follow semantic versioning once published.

## [Unreleased] - 0.1.0.dev0 (local only, not published)

### Added
- Project name **BookieBench** (package, CLI and PyPI name `bookiebench`). It was developed internally as
  "HonestBench", with the upstream package `honest`.
- A standalone package `bookiebench`, vendored from the upstream research code (`honest` → `bookiebench`). It
  includes the simulators (v1 families, the prior families, and the packs mechanics, programs, tables and stress),
  the realcoh v2 builder with `rebuild`, metrics, the runner core, and the logit / decider / julia / api runners
  plus `temper`, `leaderboard` and `release_subset`. It excludes the trained-heads model code and the review
  internals.
- CLI `bookiebench generate | rebuild-realcoh | run | score | compare`, plus model-free `uniform` and `oracle`
  runners.
- Data: the release eval/dev files (template version 2, seed 0; 37 files, 11,100 instances) and realcoh v2
  (25 sources × 200; `keep` sources in full text, `ids_only` as references + sha256 + rebuild), with
  `data/CHECKSUMS.sha256`.
- `scripts/make_hidden_test.py`, the hidden leaderboard edition generator: secret seed from env, commitment hash,
  dedupe against train and public eval, audit gates.
- README, DATASHEET, NOTICE, CITATION, and LEADERBOARD (placeholder).
- Tests: the upstream suite (minus trained-heads tests) plus `tests/test_release_package.py`, which covers shipped
  data integrity, oracle exactness on every shipped file, the CLI pipeline and the hidden-test generator.

### Licensing
- The code is Apache-2.0. The generated synthetic data is CC-BY-4.0 (`LICENSE-DATA`). realcoh sources keep their own
  licences (NOTICE.md).

### Changed (relative to upstream)
- Machine-specific defaults replaced by environment variables: `HF_HOME` (defaults to `~/.cache/huggingface`),
  `DECIDER_SRC`, `BOOKIEBENCH_DECIDER_PUBLIC` / `_PRIVATE`. `HONEST_METRICS_OWN_EXACT` was renamed to
  `BOOKIEBENCH_METRICS_OWN_EXACT`.

### Metrics gate (synced from upstream)
- `coh_valid` now gates on skill_prior ≥ 0.05 (skill when no prior tables are given) and sens ≥ 0.05. Reports
  record a `gates` marker. `compare` works without a manifest, grouping into realcoh / stress / other.

### Stress pack
- `data/release/stress/` holds 16 transforms as data, about 300 MB. They are the programmatic transforms,
  paraphrase and lang_de/es/pt/zh. The LLM items passed both judges (Qwen3.8-27B-FP8 and deepseek-v41-flash) on every
  segment.
- long_4k/8k/16k/32k and scaling ship as a hash-checked recipe (`recipe.json`). The new command
  `bookiebench rebuild-stress --long` / `--transforms scaling` regenerates them and verifies every instance and
  every file. Full rebuilds of long_4k, long_32k and scaling for urn and spam matched the upstream files byte for
  byte.
- lang_zh is marked BETA (manifest `beta`, README, DATASHEET, leaderboard footnote). lang_hu and framing are
  excluded from v1.

### Synced from upstream (metrics and groups)
- Groups: `in_family_v2` (13 pack dev families with train data), and `new_mechanics` (val plus 9 pack families held
  out of train permanently, `HOLDOUT_TRAIN`, 11 families). The train plan now has 22 files and 600k instances. The
  eval files are byte-identical.
- Metrics: `ref:prior` / `skill_prior` (the best of uniform_joint, label_prior and template_uj per family, with a
  uniform fallback for new_mechanics and procedural families), a label-keyed `ref:indep_joint`, the coherence gate
  skill ≥ 0.05 and sens ≥ 0.05, and group tables from the manifest.
- realcoh: e-mail masking (`[EMAIL]`) in every source, both code and data. Two files changed:
  - `ledgar.jsonl`, which is kept in full text;
  - `swe_issues.jsonl`, which is ids-only and has its hash updated.

  `rebuild-realcoh` of ledgar and swe_issues verifies against the new hashes.
- The leaderboard.
