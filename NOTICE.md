# Notices and data licences

## Code

All code in `bookiebench/`, `scripts/`, `tools/` and `tests/` is licensed under the Apache License 2.0
(see `LICENSE`). Copyright 2026 Mark Marosi.

## Generated synthetic data: CC-BY-4.0

**Scope:** `data/release/{test,test_prior,heldout,val,mechanics,programs,tables}`, the stress text we generated under
`data/release/stress/`, and anything the generators produce. It is licensed under the Creative Commons Attribution 4.0
International licence (see `LICENSE-DATA`). Copyright 2026 Mark Marosi.

**Exceptions** (not covered by CC-BY-4.0):
- the realcoh sources, which keep their own licences (below);
- third-party filler text in the long-context stress transforms. That text is not shipped, and users rebuild it
  from the original datasets.


These files are produced by the generators in this repository from a seed. They contain no third-party text.
People, places, diseases, patterns and organisations in them are invented, or drawn from the generator's own lexicon
(`bookiebench/sims/lexicon.py`).


## realcoh (`data/release/realcoh/`)

Each realcoh source keeps the licence of its upstream dataset. The authoritative per-source table, with pinned
dataset revisions, is `data/release/realcoh/NOTICE.md` and `data/release/realcoh/manifest.json`. It is reproduced
here:

| source | dataset | licence | release mode |
|---|---|---|---|
| belebele_es | facebook/belebele (spa_Latn) | cc-by-sa-4.0 | keep |
| bgl_logs | logfit-project/BGL | LogHub licence (research use), re-published | ids_only |
| climate | tdiggelm/climate_fever | not declared | ids_only |
| code_defects | google/code_x_glue_cc_defect_detection | c-uda (CodeXGLUE); code LGPL-2.1+ (FFmpeg) / GPL-2.0 (QEMU) | keep |
| eurlex_fr | coastalcph/multi_eurlex (fr) | cc-by-sa-4.0 | keep |
| finance_news | abisee/cnn_dailymail | apache-2.0 (dataset scripts); article text copyright CNN / Daily Mail | ids_only |
| forecast_news | abisee/cnn_dailymail | as above | ids_only |
| gold_news | ChanceFocus/flare-headlines | not declared | ids_only |
| hdfs_logs | logfit-project/hdfsv1-grouped-labeled | LogHub licence (research use), re-published | ids_only |
| ledgar | coastalcph/lex_glue (ledgar) | cc-by-4.0 | keep |
| math_problems | DigitalLearningGmbH/MATH-lighteval | mit | keep |
| med_ru | AlucardV/medical-specialty-classification | not declared | ids_only |
| mmlu_pro | TIGER-Lab/MMLU-Pro | mit | keep |
| news | abisee/cnn_dailymail | as above | ids_only |
| reviews | HuggingFaceFW/fineweb-edu | odc-by-1.0 (dataset); page text copyright of each site | ids_only |
| sci_claims | copenlu/scientific-exaggeration-detection | gpl-3.0 | ids_only |
| sports | abisee/cnn_dailymail | as above | ids_only |
| ssh_logs | bolu61/loghub_2 | LogHub licence (research use); IPs, hosts and users masked | ids_only |
| stackexchange | HuggingFaceFW/fineweb-edu | odc-by-1.0 (dataset); page text copyright of each site | ids_only |
| student_answers | nkazi/SciEntsBank | cc-by-4.0 | keep |
| support_chat | Salesforce/APIGen-MT-5k | cc-by-nc-4.0 | ids_only |
| swe_issues | princeton-nlp/SWE-bench_Verified | mit (dataset); issue text by GitHub users | ids_only |
| symptoms | gretelai/symptom_to_diagnosis | apache-2.0 | keep |
| thunderbird_logs | logfit-project/Thunderbird | LogHub licence (research use), re-published | ids_only |
| tos | coastalcph/lex_glue (unfair_tos) | cc-by-4.0 | keep |

- A `keep` source ships its text in full, under the source licence. The CC-BY-SA sources (belebele_es, eurlex_fr)
  are share-alike.
- An `ids_only` source ships no text. It ships the source reference (dataset, revision, file, row/id), a sha256 of
  the evidence text, and the generated questions. The text is reconstructed locally with
  `bookiebench rebuild-realcoh`, under the terms of the upstream dataset, which each user obtains themselves.
- **code_defects**: the functions come from FFmpeg (LGPL-2.1-or-later) and QEMU (GPL-2.0); see
  <https://ffmpeg.org/legal.html> and <https://wiki.qemu.org/License>. They are redistributed via CodeXGLUE
  (Devign) under C-UDA.
- Dropped before release: `chat` (ShareGPT: no licence, jailbreak prompts), `mailing_list` (personal data),
  `job_ads` and `recipes` (not the intended kind of page).

## Stress pack (not shipped yet)

When `data/release/stress/` is added, note the following. The `long_*` transforms embed filler text from CNN/DailyMail
(validation split), microsoft/wiki_qa and SetFit/20_newsgroups (comp/sci/rec/forsale groups). Their licences must
be added here, or the long-context transforms shipped as a rebuild-only download. The stress LLM transforms were
produced with a local open-weights model.

`bookiebench/sims/packs/stress/common_words.txt.gz` is a list of lower-case English word types seen at least 3 times
in 60k CNN/DailyMail training articles. It is a vocabulary list, with no running text.
