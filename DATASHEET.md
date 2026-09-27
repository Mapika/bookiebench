# Datasheet for BookieBench (v0.1, pre-release)

This datasheet follows Gebru et al., "Datasheets for Datasets" (2018/2021). Numbers are read from
`data/release/sims_manifest.json` and `data/release/realcoh/manifest.json` in this repository. Items marked TODO
need sign-off or depend on pending upstream changes.

## Motivation

- **Purpose.** BookieBench measures whether a model's stated probabilities can be trusted. It asks whether the
  probabilities are coherent (not Dutch-bookable, invariant to option and evidence order, martingale in updating),
  calibrated against sampled outcomes, and informative. Informative means close to the exact Bayesian posterior
  where one exists (skill against the evidence-blind `uniform_joint`) and responsive to evidence (`sens`). The
  benchmark was built because existing probability outputs of LLM-based "System One" classifiers carry no
  guarantees about how answers relate to each other.
- **Creators / funding.** Created by Mark Marosi. Funding: TODO (to be stated before release).

## Composition

- **Instances.** An instance is one situation: a prelude (the stated model) plus evidence revealed in steps, typed
  variables with options, and a set of queries. Queries are marginals, conjunctions, negations, negated
  conjunctions and conditionals, plus stepped marginals for the martingale check. Simulator instances also carry the
  exact posterior joint after every step, sampled gold values, exchangeable evidence orders (`perms`) where they
  exist, and the exact law of the next evidence (`next_evidence`). realcoh instances have `"joint": null` and gold
  only where the source dataset labels a variable.
- **Shipped counts.**

  | part | files | instances | final-step queries |
  |---|---|---|---|
  | in_family (`test/`) | 7 | 2,100 | 17,789 |
  | prior (`test_prior/`) | 2 | 600 | 5,007 |
  | in_family_v2 (13 pack `dev/` families with train data) | 13 | 3,900 | 32,167 |
  | surface_transfer (`heldout/`) | 4 | 1,200 | 10,236 |
  | new_mechanics (`val/` + 9 pack `dev/` families never in train) | 11 | 3,300 | 27,393 |
  | realcoh | 25 sources | 5,000 | 110,000 (all queries) |
- **Not shipped but regenerable.** The train split is not shipped: 22 files, 600,000 instances, about 3.8 GB. Regenerate it with
  `bookiebench generate --train-only`. It is deterministic, and prefixes were verified byte-identical against the
  upstream build. Leaderboard test editions are generated from secret seeds (`scripts/make_hidden_test.py`).
- **Stress pack.** 21 transforms of the 11 v1 test/heldout families, 300 source instances each.
  - Shipped as data (about 300 MB): 11 programmatic transforms (3,300 items each), paraphrase (2,250), lang_de
    (2,156), lang_es (2,535), lang_pt (2,451) and **lang_zh, BETA** (2,119).
  - Shipped as a hash-checked recipe: long_4k/8k/16k/32k (third-party filler) and scaling (about 260 MB,
    deterministic).
  - Excluded from v1: lang_hu and framing. Details are in `data/release/stress/README.md`.
- **Sample or complete?** Simulator splits are samples from the generators' priors, after release selection (below).
  realcoh draws 200 states per source from the named datasets at pinned revisions (seed 0).
- **Labels.**
  - Simulators: the exact posterior (enumeration or closed form, never Monte Carlo) and gold outcomes sampled from
    the true process.
  - realcoh: gold where the dataset provides it; for example `news` and `support_chat` have none.
- **Missing information.** In the `ids_only` realcoh sources the evidence text is withheld (`null`), and a sha256
  of it is shipped for verification.
- **Relationships.**
  - Stress instances pair with source ids in `test/` and `heldout/`.
  - realcoh conditional pairs share a `link` tag.
  - Paraphrase variants are keyed by the original query id.
- **Splits.** See the README groups table. The public files are a dev set. The leaderboard uses hidden editions.
- **Errors, noise, redundancy.** From the internal reviews:
  - Exactness: the first re-derivation from the text alone matched 34/37 families. Its blocker findings (forensic
    label inversion, an unstated montyhall reliability) were addressed before the final exactness pass (upstream
    `review/final_exactness/`). In the release, the forensic label scan shows 0 contradictions.
  - The Dutch-book LP was checked against an independent dual.
  - Remaining known issues are listed under README "Known limitations": double-negation wording, questions
    answerable from the question alone (dice 0.119, alarm 0.115), surface-transfer heldout, text-baseline skill, contamination
    flags, judge-dependent stress filtering.
  - Near-duplicates: eval files were deduplicated against train and within each split (world-parameter key and
    digit-masked text key). Residual duplicate answer vectors can occur in families with small parameter grids.
- **Self-contained?** The simulator files are self-contained. The `ids_only` realcoh sources need the Hugging Face
  datasets at the pinned revisions (`data/release/realcoh/manifest.json` → `revisions`).
- **Confidential, offensive or personal data.**
  - Simulator data is synthetic, with invented names.
  - realcoh contains real text: news about real public figures (the forecast_news prelude carries a content note),
    system logs (ssh_logs is masked for IPs, hosts and users), legal and medical text.
  - E-mail addresses are masked as `[EMAIL]` in every realcoh source, including ids_only rebuilds (the masking
    lives in `build_v2`). This affected ledgar (SEC contract notice addresses) and swe_issues.
  - Synthetic e-mail addresses in the `format_email` stress transform use the reserved `example.net` domain.
  - Sources with personal data (mailing_list) or no licence plus jailbreak content (chat) were dropped.
- **Sub-populations.** Not applicable to the simulators. realcoh covers several languages: Spanish (belebele_es),
  French (eurlex_fr), Russian (med_ru) and English.

## Collection process

- **Simulators.** Generated by code (`bookiebench/sims`, template version 2), deterministic in (seed, family,
  split, index). Parameters are drawn as integers or percents, and the text states them exactly, so the exact
  posterior is computable from the text.
- **Release selection.**
  - Each instance takes the first of up to 6 seed attempts whose martingale posterior moves after step 0 and whose
    degenerate-query fraction (near-certain or uniform) is ≤ 10%.
  - Option order is shuffled per instance.
  - Eval instances matching a train instance or an earlier instance of the same split are redrawn.
  - Per-file audit rates are in `sims_manifest.json`.
- **realcoh.** Built offline from the local Hugging Face cache at pinned revisions (`bookiebench.realcoh.build_v2`).
  Questions come from per-source templates. Review fixes applied:
  - IP, host and user masking (ssh_logs)
  - a stripped alert label (bgl_logs)
  - masking of act-type words (eurlex_fr) and project names (swe_issues)
  - dropping items whose answer appears verbatim (belebele_es)
  - corrected symptom systems
  - gold withheld where it was biased (support_chat)
- **Stress.**
  - Programmatic transforms: formats, bias framings, long-context filler, scaling. They are deterministic in
    (seed, transform, source id).
  - LLM transforms: paraphrase and translation, rewritten by Qwen3.8-27B-FP8. An item is kept only if it passes
    programmatic checks (numbers and names preserved) and **both** judges on every segment. Judge 1 is the rewriting
    model at temperature 0; judge 2 is deepseek-v41-flash. Instance-level judge agreement is κ 0.34-0.47.
  - lang_zh additionally passes a rule-based double-negation filter, which removed 64 items. It is still marked
    **BETA**: both judges detect only 0.64/0.66 of planted query-negation flips in Chinese
    (`logs/stress_judge_controls_judge{,2}.json` upstream).
- **Timeframe.** Built September 2026. realcoh sources span their datasets' own collection periods.
- **Ethical review.** No human subjects were recruited. TODO: state any institutional review, if applicable.

## Preprocessing / cleaning / labelling

Described under Collection. Raw upstream datasets are not redistributed here. For `ids_only` sources, only
references and hashes are.

## Uses

- **Intended.**
  - Evaluating probability outputs of models: coherence, calibration, exact-posterior skill and evidence
    sensitivity.
  - Diagnosing where incoherence comes from, via the stress transforms and the query relation tags.
  - Tracking progress on a leaderboard with hidden editions.
- **Training.** The train split exists so models can be trained on these worlds. Report results on groups with no
  train data separately (`new_mechanics`). Treat `surface_transfer` as surface transfer only.
- **Not intended.**
  - Claims about real-world domain knowledge. The simulators use invented entities and stated parameters.
  - Medical, legal or financial decisions from any realcoh-derived output.
  - Coherence-only rankings. `uniform_joint` is perfectly coherent.
- **Risks.**
  - Contamination. 23/25 realcoh sources are likely in pretraining corpora, and 10 are in decider's training mixture
    (flags in the manifest and in each instance's `meta`).
  - Public dev files will leak into training over time. That is why the leaderboard uses hidden editions.

## Distribution

- **How.** TODO: not published yet. The candidate plan is a code repository plus a dataset hub. The data files in git
  total about 415 MB: 113 MB of eval/dev and realcoh, and 300 MB of stress text. Recipe-built stress (long_*,
  scaling) would add about 1.3 GB, which users build locally.
- **Licence.** Code is Apache-2.0. Generated synthetic data (simulators, our stress text) is CC-BY-4.0
  (`LICENSE-DATA`). realcoh text keeps each source's licence
  (`NOTICE.md`, `data/release/realcoh/NOTICE.md`). `ids_only` sources are rebuilt by users under the upstream terms.
- **Export controls / regulatory restrictions.** None known.

## Maintenance

- **Maintainer.** Mark Marosi.
- **Versioning.** `CHANGELOG.md`. Data integrity: `data/CHECKSUMS.sha256`. Generator determinism is covered by
  tests.
- **Leaderboard editions.** Each edition uses a fresh secret seed, a published commitment hash, and the secret
  revealed at retirement (README "Hidden test set").
- **Errata.** TODO: an issue tracker once published. Known issues are in README "Known limitations".
- **Contributions.** New families or packs must provide exact answers, pass `bookiebench.sims.core.validate`, score
  kl = 0 and dutch = 0 under the oracle, and ship their own tests.
