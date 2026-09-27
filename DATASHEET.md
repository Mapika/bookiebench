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
  | surface_transfer (`heldout/`) | 4 | 1,200 | 10,236 |
  | new_mechanics (`val/` + pack `dev/`) | 24 | 7,200 | 59,560 |
  | realcoh | 25 sources | 5,000 | 110,000 (all queries) |

  TODO(groups): upstream is re-splitting new_mechanics into `in_family_v2` (13 families) and `new_mechanics`
  (val + 9 families with no train data). Eval files are unchanged.
- **Not shipped but regenerable.** The train split is not shipped: 31 files, 780,000 instances. Regenerate it with
  `bookiebench generate --train-only`. It is deterministic, and prefixes were verified byte-identical against the
  upstream build. Leaderboard test editions are generated from secret seeds (`scripts/make_hidden_test.py`).
- **Pending.** The stress pack: 21 transforms of the 11 v1 test/heldout families (see
  `data/release/stress/TODO.md`).
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
- **Stress (pending).**
  - Programmatic transforms: formats, bias framings, long-context filler, scaling.
  - LLM transforms: paraphrase, translation. These are checked programmatically and by an LLM judge. A second judge
    is pending.
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
  train data separately (`new_mechanics` after the re-split). Treat `surface_transfer` as surface transfer only.
- **Not intended.**
  - Claims about real-world domain knowledge. The simulators use invented entities and stated parameters.
  - Medical, legal or financial decisions from any realcoh-derived output.
  - Coherence-only rankings. `uniform_joint` is perfectly coherent.
- **Risks.**
  - Contamination. 23/25 realcoh sources are likely in pretraining corpora, and 10 are in decider's training mixture
    (flags in the manifest and in each instance's `meta`).
  - Public dev files will leak into training over time. That is why the leaderboard uses hidden editions.

## Distribution

- **How.** TODO: not published yet. The candidate plan is a code repository plus a dataset hub. The data is about
  113 MB now; stress would add about 1.7 GB, mostly long-context.
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
