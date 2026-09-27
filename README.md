# BookieBench

<p align="center"><img src="docs/assets/bookiebench.gif" alt="Pixel-art racetrack: a robot prices RED 70% and NOT RED 50% (sum 120%); the bookie sells it both tickets and wins 0.20 whichever ball is drawn (a Dutch book); a coherent robot prices 60/40, exactly 3/5, and the bookie gets nothing; then the four checks: coherence, calibration, skill, sensitivity" width="800"></p>

**Can a model's probabilities be trusted?** BookieBench asks a model for many linked probabilities about one
situation: marginals, conjunctions, negations, conditionals, the same question after more evidence, and the same
question with the options reordered. It then checks four things:

1. **Coherence.** Can a bookie trading at the model's prices lock in a sure profit (a Dutch book)? Are the answers
   invariant to option order and evidence order? Do beliefs update as a martingale?
2. **Calibration.** Log score, top-label ECE and split-conformal coverage against sampled gold outcomes.
3. **Skill.** On the simulator splits every question has an exact answer (the posterior under the stated model,
   computed by enumeration or closed form). The score is `skill = 1 - KL / KL(uniform_joint)`, where
   `uniform_joint` is the uniform distribution over the joint outcome space. That predictor reads nothing and is
   perfectly coherent, so any model has to beat it.
4. **Evidence sensitivity** (`sens`). Does the model's belief move the way the exact posterior moves as evidence
   arrives? A model that ignores the evidence scores 0.

Coherence is cheap: `uniform_joint` has `dutch = 0`. So BookieBench reports coherence only together with skill (and
`dutch_inf` only for models with positive skill). "Coherent but uninformative" is never a win.

**Links:** code at https://github.com/Mapika/bookiebench, data at https://huggingface.co/datasets/Mapika/bookiebench
(the same files as `data/release/`), and model results in [LEADERBOARD.md](LEADERBOARD.md).

## What is in the box

| part | what | exact answers? |
|---|---|---|
| simulators (`bookiebench.sims`) | 15 hand-built and procedural families (urns, dice, diagnosis, alarms, sensors/HMMs, cards, witnesses, genetics, tracking, factory, weather, spam, hiring, random Bayes nets, random HMMs) | yes |
| pack `mechanics` | 17 closed-form worlds: blackjack, poker, minesweeper, Monty Hall, birthday, matching, reliability, tournaments, noisy channels, Bayesian search, epidemics, gauges, queues, inventory, forensic evidence, raters, capture-recapture | yes |
| pack `programs` | random probabilistic programs rendered as code (abstract and domain-flavoured) | yes (enumeration) |
| pack `tables` | CSV/JSON tables and log streams; questions under the empirical distribution | yes |
| pack `stress` | 21 transforms of the 11 v1 test/heldout families: base rate, conjunction/disjunction framing, anchoring, negation, nesting, formats (chat/CSV/email/log/markdown), long context (4k-32k tokens), many-world scaling, LLM paraphrase and translation (de/es/pt, zh **BETA**) | inherited |
| `realcoh` | 25 real-world sources x 200 states with linked question sets; no exact law, so coherence plus gold where the dataset has labels | no |

### Levels and groups

Every instance carries `level`, `pack`, `family` and `transform` tags.

| level | content |
|---|---|
| L0 | 1 variable, 1-2 evidence steps, stated numbers |
| L1 | v1 families (2-4 variables), Bayes updating |
| L2 | random BNs / HMMs, mechanics |
| L3 | programs and tables |
| L4 | stress transforms at high intensity, long context |
| R | realcoh |

Groups say what a model could have trained on (`data/release/sims_manifest.json`):

| group | files | instances | train data for the family? |
|---|---|---|---|
| `in_family` | `test/` (7 v1 families) | 2,100 | yes (unseen worlds) |
| `prior` | `test_prior/` (randbn, randhmm) | 600 | yes (unseen worlds) |
| `in_family_v2` | 13 pack dev families: blackjack, poker, minesweeper, birthday, matching, reliability, tournament, channel, gauge, inventory, prog_abstract, tab_pick, tab_big | 3,900 | yes (unseen worlds) |
| `surface_transfer` | `heldout/` (factory, weather, spam, hiring) | 1,200 | no, but they re-skin the v1 train mechanics: **not** new reasoning |
| `new_mechanics` | `val/` (genetics, tracking) + 9 pack dev families held out of train permanently: epidemic, forensic, raters, recapture, montyhall, search, queue, prog_domain, tab_stream | 3,300 | **no**: 11 families never in train |

`new_mechanics` is the transfer headline. The nine held-out pack families are
`bookiebench.sims.release.HOLDOUT_TRAIN`. Their train split is generated only in memory during a build, for dedupe
keys, and is never written.

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"            # numpy + scipy only; extras: realcoh, models, api, stress
python -m pytest -q                 # CPU, about 6 minutes

# 1. predictions from a model-free baseline (any runner writes the same format)
bookiebench run oracle  data/release/test data/release/heldout --out results/oracle
bookiebench run uniform data/release/test data/release/heldout --out results/uniform
# 2. score one model, or compare several (reference rows are added automatically)
bookiebench score   results/oracle --data data/release
bookiebench compare results/oracle results/uniform --data data/release
```

Regenerate data (deterministic in `--seed`; CPU; `--workers` processes):

```bash
bookiebench generate --out data/release --eval-only           # the eval/dev files shipped here (no train dedupe)
bookiebench generate --out data/release --train-only          # the train split (22 files, 600k instances, ~3.8 GB) + dedupe keys
bookiebench generate --out /tmp/smoke --n-scale 0.01          # a quick, small build of everything
```

The shipped eval files were built together with train (seed 0), and each one was deduplicated against train. To
reproduce them byte for byte, run `bookiebench generate --out data/release` (train first, then eval), not
`--eval-only`. Train instance *i* depends only on (seed, family, split, *i*), so any `--n-scale` build reproduces
a prefix of the full train file.

Stress transforms whose text is third-party (long_*) or large and fully determined (scaling) ship as a hash-checked
recipe (`data/release/stress/README.md`):

```bash
pip install -e ".[stress]"
bookiebench rebuild-stress --transforms scaling       # no external data
HF_HUB_OFFLINE=1 bookiebench rebuild-stress --long    # after fetching the pinned filler datasets + tokenizer
```

Real-world states whose licence does not allow redistribution ship as ids only. Rebuild their text from a local
Hugging Face cache at the pinned revisions (`pip install -e ".[realcoh]"`; see `data/release/realcoh/NOTICE.md`):

```bash
bookiebench rebuild-realcoh --release data/release/realcoh --out data/realcoh_full
```

## How to score a model

1. **Pick a runner.**

   | runner | for |
   |---|---|
   | `bookiebench run logit` | HF causal LMs (letter logits) |
   | `bookiebench run decider` | decider models |
   | `bookiebench run julia` | Julia-1 |
   | `bookiebench run api` | OpenAI-compatible endpoints, logprob or CoT-JSON mode |

   Or write your own scorer (a class with `name` and `score(rows)`; see `bookiebench/runners/core.py` and
   `bookiebench/runners/baselines.py`).
2. **Emit the variants.** The runner expands every instance into questions:

   | variant | what the runner asks |
   |---|---|
   | `base` | every query |
   | `optperm:<s>` | marginals with the options permuted, answers mapped back |
   | `evperm:<i>` | evidence in an exchangeable reorder |
   | `next:<k>:<j>` | the martingale query after each possible next piece of evidence |
   | `para:<i>` | paraphrased wording (realcoh) |

   Restrict them with `--variants`.
3. **Write one file per split and family.** Predictions go to `results/<model>/<split>__<family>.jsonl`, one line
   per (instance, variant): `{"id", "model", "variant", "answers": {qid: [p...]}}`. A marginal answer is a vector
   over the options in stored order. A yes/no answer is `[p_yes]`.
4. **Score.** Run `bookiebench score results/<model> --data data/release` for per-split and per-family tables
   (it also writes `report.json`). Run `bookiebench compare ...` for cross-model tables that include the
   evidence-blind reference rows `ref:uniform_joint` and, with `--train data/release/train`, `ref:indep_joint`.

Every instance of a split you submit for is scored, not only the ones you answered (see *missing answers* below).

## Metric definitions

Implemented in `bookiebench/metrics/core.py` and `bookiebench/metrics/dutch.py`:

| metric | definition |
|---|---|
| `kl`, `kl_marg`, `kl_bin` | mean KL(exact ‖ predicted) over base-variant queries at their own step: vector KL for marginals, Bernoulli KL for yes/no/conditional queries |
| `skill` (`skill_marg`, `skill_bin`) | mean over families of 1 − KL / KL_ref. KL_ref is the KL of `uniform_joint` on the same queries. ≤ 0 means no better than reading nothing |
| `skill_prior` | mean over families of 1 − KL / KL_prior, with KL_prior the best (lowest-KL) of `uniform_joint`, `ref:label_prior` and `ref:template_uj` on that family. The last two are fitted on train (`--train`, default `<data>/train`). **Uniform fallback:** families in `new_mechanics` and the procedural families randbn/randhmm (whose option words have random meanings) use `uniform_joint` only. ≤ 0 means no better than an evidence-blind train-fitted prior |
| `sens` | 1 − Σ‖Δmodel − Δexact‖₁ / Σ‖Δexact‖₁ over consecutive steps of the martingale variable. Exact tracking scores 1, ignoring the evidence scores 0 |
| `logscore`, `acc`, `ece` | against the sampled gold for final-step marginals whose variable has gold. `acc` splits credit among tied argmaxes (1/\|ties\|). `ece` is top-label with 15 bins |
| `dutch` | per instance, the maximum guaranteed bookie profit, with stakes in [−1, 1], over all final-step queries (LP over the joint cells; conditional bets are called off when the condition fails); mean over instances. Also reported: `dutch_per_bet`, the tolerance-aware `dutch@0.005` (prices read as ±δ intervals), and `dutch_frac_exploitable` (share of instances with `dutch@0.01` > 1e-6) |
| `coh_valid`, `dutch_inf` | `coh_valid` = 1 iff **skill_prior** ≥ 0.05 (plain `skill` only when no train is given for the prior refs) **and** sens ≥ 0.05. The sens condition is skipped where sens is undefined. Gating on skill_prior keeps evidence-light shortcuts out: the bag-of-words `tfidf_lr` has skill 0.1 but skill_prior 0.025. `dutch_inf` = `dutch` only when `coh_valid`, else "–". The gates are the `--skill-gate` / `--sens-gate` flags. Reports record which gate was used (`gates`), so a stale `report.json` is recomputed |
| `optperm`, `evperm`, `para` | mean total variation between base answers and option-permuted, evidence-permuted or paraphrased answers |
| `mart` | mean \|p_k − Σ_j P(e_j) p_{k+1}^{(j)}\|, with the expectation taken under the **exact** law of the next evidence |
| `kl_avgperm` | KL of the option-order-averaged answer. `kl − kl_avgperm` is the price of position bias |
| `cover@α`, `size@α` | split-conformal (LAC) coverage and mean set size on a hash-determined half of the instances, calibrated on the other half within each reported group |
| `answered_frac` | share of queries with a valid answer (a headline column) |

Reference predictors are computed internally by `bookiebench compare`, with no prediction files needed. All of
them are evidence-blind (`sens` = 0).

| reference | what it is |
|---|---|
| `ref:uniform_joint` | uniform over joint cells |
| `ref:indep_joint` | product of train-mean marginals keyed by option text, shrunk toward 1/K |
| `ref:label_prior`, `ref:template_uj` | train-fitted option-label and question-template lookups |
| `ref:prior` | the stronger of `ref:label_prior` and `ref:template_uj` per family |

Reports are broken down by group (from `sims_manifest.json`), by split and by family.

**Missing answers never help.** An answer counts as missing if it is absent, NaN, or has the wrong length.

| metric | how a missing answer is scored |
|---|---|
| `kl`, `logscore`, `acc`, `ece`, `cover`, `sens` | as the uniform answer (1/K per option, 0.5 for yes/no) |
| `dutch` | adversarially: the bookie may choose any price in [0, 1] for it. A missing bet becomes a sign-constrained variable in a small MILP (HiGHS), so `dutch` with an omitted answer is at least `dutch` under *any* answer to it |

An instance of a scored split with no `base` prediction counts as fully unanswered. The review tested this rule.
Exact marginals with every other answer omitted used to game `dutch` to 0. Under the adversarial rule that predictor
gets `dutch` ≈ 3.45 and `answered_frac` ≈ 0.48 (`review/final_exactness/gaming.out` upstream).

## Reference rows

The reference rows below were computed from the files in this repository with `bookiebench run {oracle,uniform}`
and `bookiebench compare --recompute --train <train>` over all 11,100 shipped simulator eval/dev instances, every
variant. The train split was regenerated with `bookiebench generate --train-only`. `uniform` is per-question uniform
(1/K, 0.5), which is incoherent. `ref:uniform_joint` is the coherent evidence-blind floor.

| group | model | n | skill | skill_prior | kl | sens | acc | logscore | dutch | cover@0.1 |
|---|---|---|---|---|---|---|---|---|---|---|
| overall | ref:uniform_joint | 11100 | 0.0000 | -0.1466 | 0.2175 | 0.0000 | 0.4018 | -0.9699 | 0.0000 | 0.9071 |
| overall | ref:indep_joint | 11100 | 0.0727 | 0.0073 | 0.1998 | 0.0000 | 0.4466 | -0.9464 | 0.0000 | 0.9022 |
| overall | ref:prior | 11100 | 0.0694 | -0.0018 | 0.2003 | -0.0183 | 0.4492 | -0.9458 | 0.0755 | 0.9031 |
| overall | oracle | 11100 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6771 | -0.6808 | 0.0000 | 0.9016 |
| overall | uniform | 11100 | -0.2978 | -0.4757 | 0.2712 | 0.0000 | 0.4018 | -0.9699 | 0.8200 | 0.9071 |
| in_family | ref:uniform_joint | 2100 | 0.0000 | -0.1214 | 0.2139 | 0.0000 | 0.4353 | -0.8580 | 0.0000 | 0.9691 |
| in_family | ref:prior | 2100 | 0.0638 | -0.0094 | 0.1929 | -0.0073 | 0.5081 | -0.8290 | 0.1070 | 0.8982 |
| in_family | oracle | 2100 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.7227 | -0.5776 | 0.0000 | 0.9078 |
| prior | ref:uniform_joint | 600 | 0.0000 | 0.0000 | 0.1711 | 0.0000 | 0.4342 | -0.8558 | 0.0000 | 0.9616 |
| prior | oracle | 600 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6912 | -0.6309 | 0.0000 | 0.8969 |
| in_family_v2 | ref:uniform_joint | 3900 | 0.0000 | -0.3480 | 0.2219 | 0.0000 | 0.3835 | -1.0403 | 0.0000 | 0.9113 |
| in_family_v2 | ref:indep_joint | 3900 | 0.1697 | 0.0260 | 0.1822 | 0.0000 | 0.4733 | -0.9885 | 0.0000 | 0.8975 |
| in_family_v2 | ref:prior | 3900 | 0.1597 | 0.0000 | 0.1841 | -0.0580 | 0.4757 | -0.9896 | 0.1436 | 0.9015 |
| in_family_v2 | oracle | 3900 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6572 | -0.7291 | 0.0000 | 0.8986 |
| surface_transfer | ref:uniform_joint | 1200 | 0.0000 | -0.0121 | 0.1633 | 0.0000 | 0.4437 | -0.8300 | 0.0000 | 1.0000 |
| surface_transfer | ref:prior | 1200 | 0.0113 | -0.0002 | 0.1607 | 0.0000 | 0.4571 | -0.8208 | 0.0447 | 0.8978 |
| surface_transfer | oracle | 1200 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6814 | -0.6522 | 0.0000 | 0.9098 |
| new_mechanics | ref:uniform_joint | 3300 | 0.0000 | 0.0000 | 0.2461 | 0.0000 | 0.3779 | -1.0403 | 0.0000 | 0.9467 |
| new_mechanics | oracle | 3300 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 0.6649 | -0.7152 | 0.0000 | 0.9112 |
| new_mechanics | uniform | 3300 | -0.3061 | -0.3061 | 0.3028 | 0.0000 | 0.3779 | -1.0403 | 0.8823 | 0.9467 |

On `prior` and `new_mechanics`, every train-fitted reference equals `ref:uniform_joint` because of the uniform
fallback. On `in_family_v2`, evidence-blind train-fitted references reach skill of about 0.16-0.17. That is why
`skill_prior` exists, and why the coherence gate is on skill_prior ≥ 0.05 together with sens ≥ 0.05. No reference
row passes the gate: the refs have skill_prior ≤ 0.026 and sens ≤ 0. Only the oracle's coherence is reported as
informative (`dutch_inf`). The rows were rerun after the gate change and are unchanged; the gate affects only
`coh_valid` / `dutch_inf`.

The oracle's `acc` and `logscore` are the ceiling that sampled gold allows: gold is a draw from the exact posterior,
so even perfect probabilities are "wrong" about a third of the time.

Model results are in [LEADERBOARD.md](LEADERBOARD.md). It scores every model on the same shared subset of the
shipped files, reports tempered (`-Tfit`, one fitted temperature per model, never fitted on eval data) and raw rows,
and gives per-group tables, realcoh split by decider-training overlap, and paired stress deltas.

## Hidden test set (leaderboard)

The public files are a dev set. Leaderboard editions are regenerated from a secret seed with
`scripts/make_hidden_test.py`:

```bash
export BOOKIEBENCH_SECRET_SEED="$(openssl rand -hex 32)"     # fresh per edition; keep offline; never commit or log it
python scripts/make_hidden_test.py --edition 2026-10 --out /secure/hb-hidden/2026-10 \
       --train-keys data/release/.train_keys                   # from `bookiebench generate --train-only`
```

Protocol:

- **A fresh seed for every edition.** The script reads the secret only from the environment. It derives a 63-bit
  seed from (secret, edition) that is disjoint from the public seed range, and refuses to write inside the repo.
- **The same plan as the public release.** It uses the same families, groups, sizes, template version, option
  shuffle, degeneracy cap and attempt budget (`bookiebench.sims.release.plan`). Only the seed differs.
- **Deduplication.** Instances are deduplicated against the public train split of each family (world-parameter key
  and digit-masked text key), against the public eval/dev file of the family, and within the edition. Families held
  out of train permanently (`HOLDOUT_TRAIN`) have no train file. For them, the public train split is regenerated in
  memory, for its dedupe keys only, the same way the public build does it. For a family that has a train file, the
  in-memory keys equal its keys file exactly (checked on birthday: 40,000 keys). Ids get the prefix
  `hidden<edition>-`, so predictions made on public files cannot be scored against a hidden edition by accident.
- **The same audits.** The script runs `validate`, the degenerate / uniform / still-posterior rates, the
  option-position excess mass, dedupe and fallback counts, and the exact-posterior calibration check. It gates on
  them and exits non-zero on a failure. The manifest records every rate.

  The heavier review audits (the label and wording scans, and the per-family solvers that re-derive answers from
  the text) are not shipped. Maintainers run them on each edition before it opens.
- **Commit, then reveal.** `hidden_manifest.json` stores `sha256("bookiebench-hidden:" + edition + ":" + secret)`,
  never the seed. Publish the commitment when an edition opens. Publish the secret when it is retired, so anyone
  can regenerate the edition and check it.
- **Coverage.** realcoh and stress are not regenerated by the script. realcoh needs new real-world states (a curation
  step). Stress can be applied to an edition afterwards with `bookiebench.sims.packs.stress.build`.

## Known limitations

These come from the internal exactness, fairness, shortcut and release audits of this release.

- **Double negation in the two-clause negated-conjunction template.** An example is "Is it false that both the
  opponent holds no clubs and the opponent does not hold a pair?". These questions are correct but hard to parse.
  The release scan counts 1,058 of 105,517 scanned queries. Per family, they reach up to 33% of negations (poker),
  28% (alarm, forensic) and 26% (spam).
- **Some questions can be answered from the question alone, without the state.** The measure is the share of
  queries whose exact answer is within TV 0.05 of the mean train answer for the same (question, options), counting
  only keys seen at least 5 times in train. On the release eval against the release train it is:

  | family | share |
  |---|---|
  | dice | 0.119 |
  | alarm | 0.115 |
  | cards | 0.052 |
  | every other file | ≤ 0.033 |

  Source: the `qonly` field of `review/release_audit/degenerate.json`, upstream.
- **`heldout` is surface transfer, not new mechanics.** The four held-out families re-skin the v1 train mechanics.
  Read `new_mechanics` (11 families with no train data) as the transfer headline.
- **Shallow text baselines have positive skill.** Evidence-blind or template-level predictors are not at zero. The
  shipped `ref:prior` reaches skill 0.160 on `in_family_v2` and 0.069 overall (reference rows above). In the
  shortcut review, which used the pre-re-split groups, `tfidf_joint` reached 0.115 and a template-keyed
  uniform-joint lookup 0.099 on the full eval. Use `skill_prior`, not only `skill`, for families with train data.
  The coherence gate uses `skill_prior` for this reason.
  `uniform_joint` is 0 by definition and the oracle 1. Compare models against these rows, not only against 0.
- **Overlap with decider training.** 10 realcoh sources are flagged `in_decider_train`: gold_news, bgl_logs,
  hdfs_logs, symptoms, tos, ledgar, climate, med_ru, code_defects, student_answers. They appear in decider's
  private training mixture, and many of their states occur verbatim in its training half. Examples: student_answers
  187/200, symptoms 165, med_ru 164, climate 159, gold_news 147. Report decider results on these sources separately.
  The leaderboard splits realcoh by this flag.
- **Personal data in real text.** realcoh is real text, and some of it names real people and organisations. Examples
  are news about public figures, contracts, issue threads and system logs. Mitigations:
  - E-mail addresses are masked as `[EMAIL]` in every realcoh source. `rebuild-realcoh` applies the same masking.
  - ssh_logs masks IPs, hosts and users.
  - Sources with personal data (mailing_list) were dropped.

  The e-mail addresses that appear in the synthetic `format_email` stress transform use the reserved
  `example.net` domain.
- **Pretraining contamination.** 23 of 25 realcoh sources are flagged `likely_pretraining` (all except symptoms and
  support_chat). Gold accuracy on them may be inflated. The coherence metrics need no gold and are much less affected.
- **Judge-based filtering of the LLM stress items.** Paraphrase and translation items are kept only if they pass
  programmatic number-preservation checks and **both** LLM judges on every segment. Judge 1 is the rewriting model,
  Qwen3.8-27B-FP8, at temperature 0. Judge 2 is deepseek-v41-flash. The kept sets therefore depend on the judges.

  | transform | kept / 3,300 | judge agreement (Cohen's κ) |
  |---|---|---|
  | paraphrase | 2,250 | 0.47 |
  | lang_de | 2,156 | 0.42 |
  | lang_es | 2,535 | 0.37 |
  | lang_pt | 2,451 | 0.34 |
  | lang_zh | 2,119 | 0.41 |

  Source: `logs/stress_finalize.json` upstream. Two transforms were excluded from v1:
  - `lang_hu`: only 48.6% of items survived judge 1, and a hand check found meaning errors in judge-clean items.
  - `framing`: 26.6% survived, and the judge caught only 73% of planted number changes.

  Stressed-vs-clean comparisons are unbiased only on the kept source ids in `data/release/stress/manifest.json`.
- **lang_zh is BETA.** Planted query-negation flips in Chinese are detected only 0.64 of the time by judge 1 and
  0.66 by judge 2. For de, es, pt and paraphrase the rates are 0.96-1.00 (`logs/stress_judge_controls_judge{,2}.json`
  upstream). A rule-based double-negation filter removed 64 more items. Report lang_zh separately, and do not pool it
  into headline stress numbers (`manifest.json` → `beta`).
- **Release selection.** Instances are redrawn when their questions are degenerate or their posterior never moves.
  The exact answers are unaffected, and a 1,000-3,000-instance-per-family check found no family with \|acc − conf\| > 0.03
  after the redraw. Gold-based accuracy and ECE are still measured on the selected subset. On the shipped
  300-instance files, the release audit flags five exact-posterior calibration gaps (heldout/weather,
  mechanics/dev/montyhall, inventory, recapture, tables/dev/tab_big, with \|gap\| between 0.030 and 0.049). A
  3,000-instance recheck of the same families shows \|gap\| ≤ 0.008 after release selection
  (`review/release_audit/calibration_{files,big}.txt` upstream), so the flags look like sampling noise in small
  files.
- **Stated numbers are the true parameters, and names are invented.** This is deliberate: real-world priors cannot
  stand in for the stated ones. It also means the simulators test reasoning from stated numbers, not world knowledge.

## Repository layout

```
bookiebench/            the package (sims + packs, realcoh, metrics, runners, cli)
data/release/           shipped eval/dev files, realcoh release, stress (data + recipe.json), manifests;
                        CHECKSUMS in data/CHECKSUMS.sha256
scripts/make_hidden_test.py   leaderboard edition generator
tools/                  checksums.py, sync_upstream.py (maintainers)
tests/                  CPU test suite
```

Verify the data with `python tools/checksums.py` (or `cd data && sha256sum -c CHECKSUMS.sha256`).

## Licence

- Code is Apache-2.0 ([LICENSE](LICENSE)).
- The generated synthetic data (simulator splits and the stress text we generated) is CC-BY-4.0
  ([LICENSE-DATA](LICENSE-DATA)).
- realcoh sources keep their own licences ([NOTICE.md](NOTICE.md), `data/release/realcoh/NOTICE.md`).
- The long-context stress filler is not shipped; it is rebuilt from the original datasets. Citation: [CITATION.cff](CITATION.cff).
