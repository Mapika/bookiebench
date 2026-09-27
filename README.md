# BookieBench

_Developed internally under the name "HonestBench"._

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

> **Status: pre-release, local only.** This repository has not been published. Open items are listed under
> [TODO before release](#todo-before-release).

## What is in the box

| part | what | exact answers? |
|---|---|---|
| simulators (`bookiebench.sims`) | 15 hand-built and procedural families (urns, dice, diagnosis, alarms, sensors/HMMs, cards, witnesses, genetics, tracking, factory, weather, spam, hiring, random Bayes nets, random HMMs) | yes |
| pack `mechanics` | 17 closed-form worlds: blackjack, poker, minesweeper, Monty Hall, birthday, matching, reliability, tournaments, noisy channels, Bayesian search, epidemics, gauges, queues, inventory, forensic evidence, raters, capture-recapture | yes |
| pack `programs` | random probabilistic programs rendered as code (abstract and domain-flavoured) | yes (enumeration) |
| pack `tables` | CSV/JSON tables and log streams; questions under the empirical distribution | yes |
| pack `stress` | transforms over v1 families: base rate, conjunction/disjunction framing, anchoring, negation, nesting, formats (chat/CSV/email/log/markdown), long context (4k-32k tokens), many-world scaling, LLM paraphrase and translation | inherited |
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

| group | files | instances |
|---|---|---|
| `in_family` | `test/` (7 v1 train families, unseen worlds) | 2,100 |
| `prior` | `test_prior/` (randbn, randhmm) | 600 |
| `surface_transfer` | `heldout/` (factory, weather, spam, hiring): re-skinned versions of the train mechanics, **not** new reasoning | 1,200 |
| `new_mechanics` | `val/` + `{mechanics,programs,tables}/dev/` | 7,200 |

> **TODO(groups): upstream is re-splitting these groups.** Nine v2 families (epidemic, forensic, raters, recapture,
> montyhall, search, queue, prog_domain, tab_stream) are being taken out of train and move to `new_mechanics` with
> `val/`. The other 13 v2 dev families become `in_family_v2`. The eval files stay byte-identical. Only
> `sims_manifest.json`, the train plan and the group tables change. Update this table after the next sync.

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
bookiebench generate --out data/release --train-only          # the 780k-instance train split (about 4.8 GB) + dedupe keys
bookiebench generate --out /tmp/smoke --n-scale 0.01          # a quick, small build of everything
```

The shipped eval files were built together with train (seed 0), and each one was deduplicated against train. To
reproduce them byte for byte, run `bookiebench generate --out data/release` (train first, then eval), not
`--eval-only`. Train instance *i* depends only on (seed, family, split, *i*), so any `--n-scale` build reproduces
a prefix of the full train file.

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
| `sens` | 1 − Σ‖Δmodel − Δexact‖₁ / Σ‖Δexact‖₁ over consecutive steps of the martingale variable. Exact tracking scores 1, ignoring the evidence scores 0 |
| `logscore`, `acc`, `ece` | against the sampled gold for final-step marginals whose variable has gold. `acc` splits credit among tied argmaxes (1/\|ties\|). `ece` is top-label with 15 bins |
| `dutch` | per instance, the maximum guaranteed bookie profit, with stakes in [−1, 1], over all final-step queries (LP over the joint cells; conditional bets are called off when the condition fails); mean over instances. Also reported: `dutch_per_bet`, the tolerance-aware `dutch@0.005` (prices read as ±δ intervals), and `dutch_frac_exploitable` (share of instances with `dutch@0.01` > 1e-6) |
| `coh_valid`, `dutch_inf` | `dutch_inf` = `dutch` only for models with skill > 1e-6, else "–" |
| `optperm`, `evperm`, `para` | mean total variation between base answers and option-permuted, evidence-permuted or paraphrased answers |
| `mart` | mean \|p_k − Σ_j P(e_j) p_{k+1}^{(j)}\|, with the expectation taken under the **exact** law of the next evidence |
| `kl_avgperm` | KL of the option-order-averaged answer. `kl − kl_avgperm` is the price of position bias |
| `cover@α`, `size@α` | split-conformal (LAC) coverage and mean set size on a hash-determined half of the instances, calibrated on the other half within each reported group |
| `answered_frac` | share of queries with a valid answer (a headline column) |

> **TODO(metrics): upstream is extending the metrics.** Coming: a train-fitted prior reference (`ref:prior`,
> `skill_prior`), a corrected label-keyed `ref:indep_joint`, a stricter coherence gate (skill ≥ 0.05 and sens ≥ 0.05
> instead of skill > 1e-6), and group tables read from the manifest. Update this table after the next sync.

**Missing answers never help.** An answer counts as missing if it is absent, NaN, or has the wrong length.

| metric | how a missing answer is scored |
|---|---|
| `kl`, `logscore`, `acc`, `ece`, `cover`, `sens` | as the uniform answer (1/K per option, 0.5 for yes/no) |
| `dutch` | adversarially: the bookie may choose any price in [0, 1] for it. A missing bet becomes a sign-constrained variable in a small MILP (HiGHS), so `dutch` with an omitted answer is at least `dutch` under *any* answer to it |

An instance of a scored split with no `base` prediction counts as fully unanswered. The review tested this rule.
Exact marginals with every other answer omitted used to game `dutch` to 0. Under the adversarial rule that predictor
gets `dutch` ≈ 3.45 and `answered_frac` ≈ 0.48 (`review/final_exactness/gaming.out` upstream).

## Reference rows

The reference rows below were computed from the files in this repository with `bookiebench run {oracle,uniform}` and
`bookiebench compare --recompute` over all 11,100 shipped simulator eval/dev instances (every variant). `uniform` is
per-question uniform (1/K, 0.5), which is incoherent. `ref:uniform_joint` is the coherent evidence-blind floor.

| split | model | n | skill | kl | kl_marg | kl_bin | sens | acc | logscore | dutch | cover@0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | ref:uniform_joint | 11100 | 0.0000 | 0.2175 | 0.2813 | 0.1570 | 0.0000 | 0.4018 | -0.9699 | 0.0000 | 0.9071 |
| overall | oracle | 11100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6771 | -0.6808 | 0.0000 | 0.9016 |
| overall | uniform | 11100 | -0.2978 | 0.2712 | 0.2813 | 0.2616 | 0.0000 | 0.4018 | -0.9699 | 0.8200 | 0.9071 |
| test | ref:uniform_joint | 2100 | 0.0000 | 0.2139 | 0.2603 | 0.1645 | 0.0000 | 0.4353 | -0.8580 | 0.0000 | 0.9691 |
| test | oracle | 2100 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7227 | -0.5776 | 0.0000 | 0.9078 |
| test_prior | ref:uniform_joint | 600 | 0.0000 | 0.1711 | 0.1932 | 0.1465 | 0.0000 | 0.4342 | -0.8558 | 0.0000 | 0.9616 |
| test_prior | oracle | 600 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6912 | -0.6309 | 0.0000 | 0.8969 |
| heldout | ref:uniform_joint | 1200 | 0.0000 | 0.1633 | 0.1901 | 0.1342 | 0.0000 | 0.4437 | -0.8300 | 0.0000 | 1.0000 |
| heldout | oracle | 1200 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6814 | -0.6522 | 0.0000 | 0.9098 |
| val | ref:uniform_joint | 600 | 0.0000 | 0.3240 | 0.4352 | 0.1996 | 0.0000 | 0.3777 | -1.0309 | 0.0000 | 0.9248 |
| val | oracle | 600 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6858 | -0.6273 | 0.0000 | 0.8866 |
| dev (packs) | ref:uniform_joint | 6600 | 0.0000 | 0.2235 | 0.3025 | 0.1558 | 0.0000 | 0.3812 | -1.0413 | 0.0000 | 0.9282 |
| dev (packs) | oracle | 6600 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6582 | -0.7322 | 0.0000 | 0.9063 |

The oracle's `acc` and `logscore` are the ceiling that sampled gold allows: gold is a draw from the exact posterior,
so even perfect probabilities are "wrong" about a third of the time. Model results are in [LEADERBOARD.md](LEADERBOARD.md).

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

  > **TODO(audits):** the heavier review audits (the label and wording scans, the per-family text re-derivation
  > solvers) live in the upstream review suite and are not shipped. Run them on each edition before opening it.
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
  Read `new_mechanics` (and, after the pending re-split, only the families with no train data) as the transfer
  headline.
- **Shallow text baselines have positive skill.** Evidence-blind or template-level predictors are not at zero. On
  the full release eval, `tfidf_joint` reaches skill 0.115 and a template-keyed uniform-joint lookup reaches 0.099.
  `uniform_joint` is 0 by definition and the oracle 1. Compare models against these rows, not only against 0.
- **Overlap with decider training.** 10 realcoh sources are flagged `in_decider_train`: gold_news, bgl_logs,
  hdfs_logs, symptoms, tos, ledgar, climate, med_ru, code_defects, student_answers. They appear in decider's
  private training mixture, and many of their states occur verbatim in its training half. Examples: student_answers
  187/200, symptoms 165, med_ru 164, climate 159, gold_news 147. Report decider results on these sources separately.
  The leaderboard splits realcoh by this flag.
- **Pretraining contamination.** 23 of 25 realcoh sources are flagged `likely_pretraining` (all except symptoms and
  support_chat). Gold accuracy on them may be inflated. The coherence metrics need no gold and are much less affected.
- **Judge-based filtering of the LLM stress items.** Paraphrase and translation items are kept only if they pass
  programmatic number-preservation checks and an LLM judge. The judge is the same model that did the rewriting, at
  temperature 0. The kept sets therefore depend on the judge. Kept fractions: lang_de 2,274 / 3,300, lang_es 2,665,
  lang_pt 2,590, lang_zh 2,195, paraphrase 2,428. Two transforms were excluded:
  - `lang_hu`: only 48.6% of items survived, and a hand check found meaning errors in judge-clean items.
  - `framing`: 26.6% survived, and the judge caught only 73% of planted number changes.

  Stressed-vs-clean comparisons are unbiased only on the kept source ids listed in the stress manifest.
  **TODO(stress): the DeepSeek second-judge pass is still running upstream. The final stress manifest will add
  judge-2 results, and the kept sets may shrink.**
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
data/release/           shipped eval/dev files, realcoh release, manifests; CHECKSUMS in data/CHECKSUMS.sha256
scripts/make_hidden_test.py   leaderboard edition generator
tools/                  checksums.py, sync_upstream.py (maintainers)
tests/                  CPU test suite
```

Verify the data with `python tools/checksums.py` (or `cd data && sha256sum -c CHECKSUMS.sha256`).

## TODO before release

- [ ] Resync `bookiebench/metrics`, `bookiebench/sims/release.py` and `data/release/sims_manifest.json` after the
      upstream group re-split and metrics changes (`python tools/sync_upstream.py --src <upstream> --data`). Then
      update the groups and metrics sections, rerun the reference rows, and regenerate checksums.
- [ ] Add `data/release/stress/` once the stress manifest has the judge-2 field (see `data/release/stress/TODO.md`).
- [ ] Fill `LEADERBOARD.md` from the upstream release leaderboard.
- [ ] Decide on hosting for the data (about 113 MB in git now; consider LFS or a dataset hub) (name check: `docs/NAME_CHECK.md`).

## Licence

Code is Apache-2.0 ([LICENSE](LICENSE)). Data licences differ per source; see [NOTICE.md](NOTICE.md) and
`data/release/realcoh/NOTICE.md`. Citation: [CITATION.cff](CITATION.cff).
