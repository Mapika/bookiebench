"""Per-source provenance for realcoh v2: licence, release mode, decider-training overlap, pretraining exposure.

    release_mode   keep      the release ships the full state text
                   ids_only  the release ships ids + meta.source_ref + a text hash + the questions; the text is rebuilt
                             locally from the HF cache (`python -m bookiebench.realcoh.build_v2 rebuild`)
    in_decider_train   the source dataset is used for TRAINING in decider-public's released mixture
                       (decider-public/decider/data/{mixture,tasks_*,teacher_*,core}.py) or in the private mixture_v2
                       (decider/broad_corpus/public/*.py, merged by broad_corpus/merge/tasks_v2.py). Checked on 2026-09-27.
                       `decider_note` quotes the task name(s) and the measured state overlap with its training half
                       (review/fairness/REPORT.md, R2). Held-out-only use (eval split) is recorded as False with a note.
    likely_pretraining the text (or the data it was derived from) was public on the web before 2024 and is likely in
                       general LLM pretraining corpora. A judgement, not a measurement: gold accuracy on these sources may
                       be inflated by memorisation; coherence metrics are much less affected.

Report decider results on in_decider_train sources separately (or exclude them) for any decider comparison.
"""
from __future__ import annotations

import os
import re

DECIDER_PUBLIC = os.environ.get("BOOKIEBENCH_DECIDER_PUBLIC", "")  # optional audit
DECIDER_PRIVATE = os.environ.get("BOOKIEBENCH_DECIDER_PRIVATE", "")  # maintainers only


def P(dataset, license, release_mode, in_decider_train, decider_note, likely_pretraining, pretraining_note, notice=None):
    return dict(dataset=dataset, license=license, release_mode=release_mode, in_decider_train=in_decider_train,
                decider_note=decider_note, likely_pretraining=likely_pretraining, pretraining_note=pretraining_note,
                notice=notice)


_CNN = ("abisee/cnn_dailymail", "apache-2.0 (dataset scripts); article text copyright CNN / Daily Mail", "ids_only")
_FW = ("HuggingFaceFW/fineweb-edu", "odc-by-1.0 (dataset); page text copyright of each site", "ids_only")
PROVENANCE = {
    "news": P(*_CNN, False, "not referenced", True, "CNN/DM articles 2007-2015 on the open web"),
    "finance_news": P(*_CNN, False, "not referenced", True, "CNN/DM articles on the open web"),
    "sports": P(*_CNN, False, "not referenced", True, "CNN/DM articles on the open web"),
    "forecast_news": P(*_CNN, False, "not referenced", True,
                       "2012-2014 Daily Mail articles: later events are likely known to models (hindsight, not forecasting)"),
    "gold_news": P("ChanceFocus/flare-headlines", "not declared (FLARE / gold-news headlines)", "ids_only", True,
                   "private mixture_v2 task fin_headlines; 147/200 states found in its training half", True, "news headlines"),
    "bgl_logs": P("logfit-project/BGL", "LogHub licence (research use), re-published", "ids_only", True,
                  "private mixture_v2 tasks bgl_anomaly, bgl_severity; ~52/200 states overlap", True, "LogHub is public since 2020"),
    "thunderbird_logs": P("logfit-project/Thunderbird", "LogHub licence (research use), re-published", "ids_only", False,
                          "not referenced", True, "LogHub is public since 2020"),
    "hdfs_logs": P("logfit-project/hdfsv1-grouped-labeled", "LogHub licence (research use), re-published", "ids_only", True,
                   "private mixture_v2 task hdfs_blocks (overlap not measurable: templated text)", True, "LogHub HDFS_v1"),
    "ssh_logs": P("bolu61/loghub_2", "LogHub licence (research use); IPs, hosts and users masked", "ids_only", False,
                  "loghub only mentioned in a docstring of decider/broad_corpus/public/logs.py", True, "LogHub OpenSSH"),
    "symptoms": P("gretelai/symptom_to_diagnosis", "apache-2.0", "keep", True,
                  "private mixture_v2 task symptom_diagnosis; 164/200 states found in its training half", False,
                  "synthetic (Gretel), 2023"),
    "tos": P("coastalcph/lex_glue (unfair_tos)", "cc-by-4.0", "keep", True,
             "private mixture_v2 task unfair_tos; 10-14/200 states overlap", True, "public Terms of Service"),
    "ledgar": P("coastalcph/lex_glue (ledgar)", "cc-by-4.0", "keep", True,
                "private mixture_v2 task ledgar; 31/200 states overlap", True, "SEC EDGAR filings"),
    "climate": P("tdiggelm/climate_fever", "not declared", "ids_only", True,
                 "private mixture_v2 task climate_fever; 159/200 states overlap", True, "claims + Wikipedia sentences"),
    "support_chat": P("Salesforce/APIGen-MT-5k", "cc-by-nc-4.0", "ids_only", False, "not referenced", False,
                      "synthetic dialogues released 2025"),
    "med_ru": P("AlucardV/medical-specialty-classification", "not declared", "ids_only", True,
                "private mixture_v2 task medical_specialty_ru; 164/200 states overlap", True, "Russian medical Q&A sites"),
    "sci_claims": P("copenlu/scientific-exaggeration-detection", "gpl-3.0", "ids_only", False, "not referenced", True,
                    "EurekAlert press releases and PubMed abstracts"),
    "code_defects": P("google/code_x_glue_cc_defect_detection", "c-uda (CodeXGLUE); code LGPL-2.1+ (FFmpeg) / GPL-2.0 (QEMU)",
                      "keep", True, "private mixture_v2 task code_defect; 14/200 states overlap", True, "GitHub code",
                      notice="Functions are from FFmpeg (LGPL-2.1-or-later) and QEMU (GPL-2.0); see https://ffmpeg.org/legal.html "
                             "and https://wiki.qemu.org/License. Redistributed via CodeXGLUE (Devign) under C-UDA."),
    "swe_issues": P("princeton-nlp/SWE-bench_Verified", "mit (dataset); issue text by GitHub users", "ids_only", False,
                    "not referenced", True, "public GitHub issues"),
    "stackexchange": P(*_FW, False, "not referenced", True, "Stack Exchange pages (cc-by-sa) crawled in CommonCrawl"),
    "reviews": P(*_FW, False, "not referenced", True, "CommonCrawl pages (Goodreads cross-posts)"),
    "math_problems": P("DigitalLearningGmbH/MATH-lighteval", "mit", "keep", False, "not referenced", True,
                       "MATH (2021) and AoPS solutions are widely trained on"),
    "mmlu_pro": P("TIGER-Lab/MMLU-Pro", "mit", "keep", False,
                  "not referenced; its ori_mmlu items (cais/mmlu, in decider-public's mixture) are excluded", True,
                  "MMLU-Pro (2024) test set"),
    "student_answers": P("nkazi/SciEntsBank", "cc-by-4.0", "keep", True,
                         "private mixture_v2 task scientsbank; 187/200 states found in its training half", True,
                         "SemEval-2013 task 7"),
    "eurlex_fr": P("coastalcph/multi_eurlex (fr)", "cc-by-sa-4.0", "keep", False, "not referenced", True, "EUR-Lex"),
    "belebele_es": P("facebook/belebele (spa_Latn)", "cc-by-sa-4.0", "keep", False,
                     "private mixture_v2 uses belebele as a HELD-OUT eval task only (not trained on)", True,
                     "FLORES passages from Wikimedia projects"),
}
DROPPED = {"mailing_list": "PII (names, emails) in public mailing-list archives (R6)",
           "chat": "ShareGPT: no licence, jailbreak prompts (R7)",
           "job_ads": "fineweb-edu has almost no real job postings (R10)",
           "recipes": "even the tightened filter mostly finds nutrition pages with a stray recipe (R10)"}

# repo ids grepped in the decider code bases (both), for the consistency test
DATASET_KEYS = {k: re.split(r" \(", v["dataset"])[0] for k, v in PROVENANCE.items()}
_ALIASES = {"coastalcph/lex_glue": {"tos": "unfair_tos", "ledgar": "ledgar"}}


def decider_references(source, roots=(DECIDER_PUBLIC, DECIDER_PRIVATE)):
    """Files in the decider code bases that mention this source's dataset (or its lex_glue config)."""
    key = DATASET_KEYS[source]
    pat = _ALIASES.get(key, {}).get(source, key)
    hits = []
    for root in roots:
        for dp, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".py"):
                    p = os.path.join(dp, f)
                    try:
                        if pat.lower() in open(p, errors="ignore").read().lower():
                            hits.append(p)
                    except OSError:
                        pass
    return hits
