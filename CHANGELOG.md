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

### Changed (relative to upstream)
- Machine-specific defaults replaced by environment variables: `HF_HOME` (defaults to `~/.cache/huggingface`),
  `DECIDER_SRC`, `BOOKIEBENCH_DECIDER_PUBLIC` / `_PRIVATE`. `HONEST_METRICS_OWN_EXACT` was renamed to
  `BOOKIEBENCH_METRICS_OWN_EXACT`.

### Pending before 0.1.0
- Resync of the metrics (ref:prior / skill_prior, label-keyed ref:indep_joint, stricter coherence gate, manifest
  group tables), the release plan and `sims_manifest.json` (group re-split: `in_family_v2`, and `new_mechanics`
  with 9 families held out of train).
- The stress pack data, after the judge-2 manifest.
- The leaderboard and the licence for generated data.
