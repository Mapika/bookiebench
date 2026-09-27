"""realcoh v2 sources: 25 real-world state sources from the local HF cache (HF_HUB_OFFLINE, no network, CPU only).

Every loader is `fn(rng, n) -> [(text, gold, source_ref), ...]`; source_ref = {dataset, revision (HF snapshot), file,
row(s)/id} locates the state in the cache (`meta.source_ref`). Provenance, licences, release modes and decider-training
overlap are in `bookiebench.realcoh.provenance` (and in the manifest). NOTE: several datasets ARE in decider's private
mixture_v2 (decider/broad_corpus), see PROVENANCE[...]["in_decider_train"]; "not in decider's mixture" only holds for
the others.

    source           dataset (split)                                 state                          gold
    news             abisee/cnn_dailymail (test)                      article lead                   -
    gold_news        ChanceFocus/flare-headlines (test)               headline                       price, time, comparison
    bgl_logs         logfit-project/BGL                               6 log lines (label stripped)   failure, subsystem
    symptoms         gretelai/symptom_to_diagnosis                    patient description            system, infection (corrected map)
    tos              coastalcph/lex_glue unfair_tos                   ToS clause                     fairness, clause_type
    climate          tdiggelm/climate_fever (test)                    claim + evidence               verdict
    support_chat     Salesforce/APIGen-MT-5k                          first customer message         - (simulated dialogues; R9)
    ledgar           coastalcph/lex_glue ledgar (test)                contract provision             provision
    med_ru           AlucardV/medical-specialty-classification        Russian patient question       specialty
    sci_claims       copenlu/scientific-exaggeration-detection        press release vs abstract      exaggeration, both strengths
    code_defects     google/code_x_glue_cc_defect_detection (test)    C function                     vulnerable, project
    swe_issues       princeton-nlp/SWE-bench_Verified                 GitHub issue (names masked)    repo, difficulty
    thunderbird_logs logfit-project/Thunderbird                       6 log lines                    alert
    hdfs_logs        logfit-project/hdfsv1-grouped-labeled (eval)     one block's log session        anomaly
    ssh_logs         bolu61/loghub_2 openssh.txt                      8 sshd lines (IPs/users masked) -
    finance_news     abisee/cnn_dailymail (train file 0, business)    article lead                   -
    sports           abisee/cnn_dailymail (train file 0, sports)      article lead                   -
    stackexchange    HuggingFaceFW/fineweb-edu (SE urls)              Q&A page                       community (from the url)
    reviews          HuggingFaceFW/fineweb-edu ("My rating: k of 5")  book review, rating masked     verdict (from the rating)
    math_problems    DigitalLearningGmbH/MATH-lighteval (test)        competition problem            subject, level, integer answer
    mmlu_pro         TIGER-Lab/MMLU-Pro (test, non-MMLU items)        exam question + options        category, answer
    student_answers  nkazi/SciEntsBank (train)                        question, reference, answer    grade (5-way)
    eurlex_fr        coastalcph/multi_eurlex fr (test)                French EU act, act words masked act type (CELEX id)
    belebele_es      facebook/belebele spa_Latn                       Spanish passage + question     answer, source site
    forecast_news    abisee/cnn_dailymail (train file 1, dated DM)    dated article lead             -  (linked forecasts)

Multilingual states: med_ru (Russian), eurlex_fr (French), belebele_es (Spanish); the questions stay in English.
Dropped after review: mailing_list (PII), chat/ShareGPT (no licence, jailbreaks), job_ads (no real postings), recipes (mostly nutrition pages with a stray recipe, R10).
"""
from __future__ import annotations

import collections
import glob
import json
import os
import re

from . import sources as v1
from .queries_v2 import T_FORECAST

_files = v1._files
_pd = v1._pd
HUB = v1.HUB
ref_of = v1.ref_of


def _join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def V2(name, options, clauses, questions, cum=None, coarse=None):
    """A variable spec. A coarse variable's option texts and clauses name their fine members explicitly, so the
    coarse = sum-of-members constraint holds by definition (review R5)."""
    assert len(options) == len(clauses), name
    assert len(questions) >= 3, name
    v = dict(name=name, options=list(options), clauses=list(clauses), questions=list(questions))
    if cum is not None:
        assert len(cum) == len(options) - 1, name
        v["cum"] = list(cum)
    if coarse is not None:
        assert len(coarse["map"]) == len(options) and len(coarse["options"]) == len(coarse["clauses"]), name
        assert set(coarse["map"]) == set(range(len(coarse["options"]))), name
        mem = [[options[i] for i, m in enumerate(coarse["map"]) if m == c] for c in range(len(coarse["options"]))]
        c = dict(coarse)
        c["options"] = [f"{o} ({_join(m)})" for o, m in zip(coarse["options"], mem)]
        c["clauses"] = [f"{cl}, that is: {_join(m)}" for cl, m in zip(coarse["clauses"], mem)]
        v["coarse"] = c
    return v


def C(name, options, clauses, mapping):
    return dict(name=name, options=list(options), clauses=list(clauses), map=list(mapping))


def ext(var, extra_questions=(), cum=None, coarse=None):
    """A v1 variable, extended to the v2 spec."""
    return V2(var["name"], var["options"], var["clauses"], list(var["questions"]) + list(extra_questions), cum, coarse)


def _clip(text, n=1400):
    text = text.strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    k = max(cut.rfind("\n"), cut.rfind(". "))
    return (cut[:k + 1] if k > n * 0.6 else cut).rstrip() + " [...]"


def _v1(fn, **kw):
    def load(rng, n):
        return fn(rng, n, ref=True, **kw)
    load.__name__ = fn.__name__
    return load


# ======================================================================================= the six v1 sources, extended
_N, _G, _L, _S, _T, _K = v1.NEWS_VARS, v1.GOLD_VARS, v1.LOG_VARS, v1.SYM_VARS, v1.TOS_VARS, v1.CLIM_VARS
NEWS_VARS = [
    ext(_N[0], ["Under which heading would you file this story?"]),
    ext(_N[1], ["How would you describe the mood of this story?"],
        cum=["the tone of the article is negative", "the tone of the article is not positive"]),
    ext(_N[2], ["In which country or region is this story set?"]),
]
GOLD_VARS = [
    ext(_G[0], ["What price movement for gold does the headline report?"],
        coarse=C("price_move", ["mentions the price", "does not mention the price"],
                 ["the headline talks about the gold price", "the headline does not talk about the gold price"], [0, 0, 0, 1])),
    ext(_G[1], ["Is the headline reporting news or making a forecast?"]),
    ext(_G[2], ["Does the headline mention gold alongside another investment?"]),
]
LOG_VARS = [
    ext(_L[0], ["Is this excerpt evidence of a real error, or just normal operation?"]),
    ext(_L[1], ["Which component logged most of these lines?"],
        coarse=C("layer", ["software", "hardware"], ["the messages come from software", "the messages come from hardware"],
                 [0, 0, 1, 0])),
    ext(_L[2], ["What response does this excerpt call for?"],
        cum=["no action is needed", "nobody needs to be paged right away"]),
]
SYM_VARS = [
    ext(_S[0], ["Where in the body is the problem most likely located?"],
        coarse=C("organ_group", ["an internal organ", "the skin, joints or muscles", "the whole body"],
                 ["the problem is in an internal organ", "the problem is in the skin, joints or muscles",
                  "the problem affects the whole body or the nervous system"], [0, 0, 1, 1, 0, 0, 2])),
    ext(_S[1], ["Is an infection the likely cause of these symptoms?"]),
    ext(_S[2], ["How urgent is this case?"],
        cum=["the patient can manage this with self-care at home", "the patient does not need emergency care right now"]),
]
# review R11: allergy and drug reactions are systemic (not skin-only); infection status of jaundice and peptic ulcer
# (hepatitis, H. pylori) and the organ system of typhoid are genuinely ambiguous, so those golds are withheld.
_DX2 = dict(v1._DX)
_DX2.update({"allergy": (6, 0), "drug reaction": (6, 0), "jaundice": (0, None), "peptic ulcer disease": (0, None),
             "typhoid": (None, 1)})
TOS_VARS = [
    ext(_T[0], ["Is this clause balanced, or does it put the user at a disadvantage?"]),
    ext(_T[1], ["What kind of term is this?"],
        coarse=C("topic_group", ["the company's powers", "liability and disputes", "other"],
                 ["the clause is about the company's powers", "the clause is about liability or how disputes are settled",
                  "the clause is about something else"], [1, 0, 0, 0, 1, 2, 2])),
    ext(_T[2], ["Which side does this clause favour?"],
        cum=["the clause mainly protects the company", "the clause does not mainly protect the user"]),
]
CLIM_VARS = [
    ext(_K[0], ["Is the claim right, wrong, or impossible to judge from this evidence?"]),
    ext(_K[1], ["Which aspect of the climate does the claim concern?"],
        coarse=C("aspect", ["physical observations", "causes and science", "policy and society"],
                 ["the claim is about observed physical changes", "the claim is about causes or climate science",
                  "the claim is about policy, economics or energy"], [0, 0, 1, 0, 1, 2])),
    ext(_K[2], ["Would a climate-change denier agree with this claim?"]),
]


# ============================================================================================== customer support chat
SUPPORT_VARS = [
    V2("request",
       ["cancel a booking or order", "change a booking or order", "return or exchange delivered items",
        "make a new booking", "only get information"],
       ["the customer wants to cancel a booking or order", "the customer wants to change an existing booking or order",
        "the customer wants to return or exchange items they received", "the customer wants to make a new booking",
        "the customer only wants information and no change to anything"],
       ["What does the customer want the agent to do?", "What is this customer's request, in short?",
        "Which kind of action is this customer asking for?"],
       coarse=C("change_kind", ["reverse a purchase", "modify a purchase", "no existing purchase touched"],
                ["the customer wants to reverse a purchase", "the customer wants to modify an existing booking or order",
                 "the customer does not want to touch an existing purchase"], [0, 1, 0, 2, 2])),
    V2("outcome", ["the agent carries it out", "the agent cannot do it and says so", "the agent hands over to a human"],
       ["the agent ends up carrying out a change for the customer", "the agent ends up making no change to the account",
        "the agent ends up transferring the customer to a human agent"],
       ["How will this support conversation most likely end?", "What will the support agent end up doing?",
        "Will the customer get what they asked for?"]),
    V2("length", ["at most 3 customer messages", "4 or 5 customer messages", "6 or more customer messages"],
       ["the conversation takes at most 3 messages from the customer", "the conversation takes 4 or 5 messages from the customer",
        "the conversation takes 6 or more messages from the customer"],
       ["How many messages will the customer send before the conversation ends?",
        "How long will this support conversation be, counted in customer messages?",
        "How many turns will the customer need to get this sorted?"],
       cum=["the customer needs at most 3 messages", "the customer needs at most 5 messages"]),
]


def support_chat(rng, n):
    """Gold withheld (review R9): the dialogues are simulated, the tool-call-derived outcome is near-constant and
    the request type is only observed when a write succeeded (a biased subset)."""
    f = _files("Salesforce/APIGen-MT-5k", "apigen-mt_5k.json")[0]
    d = json.load(open(f))
    out, seen = [], set()
    idx = list(range(len(d)))
    rng.shuffle(idx)
    for i in idx:
        conv = d[i]["conversations"]
        first = conv[0]["value"].strip() if conv and conv[0]["from"] == "human" else ""
        if not 60 <= len(first) <= 900 or first in seen:
            continue
        seen.add(first)
        out.append((first, {}, ref_of(f, row=i, turn=0)))
        if len(out) == n:
            break
    return out


# ============================================================================================================ LEDGAR
_LEDGAR = {  # label id -> fine option
    47: 0, 56: 0, 94: 0, 21: 0, 82: 0,        # governing law / jurisdiction / venue
    6: 1, 96: 1,                              # arbitration, jury waiver
    38: 2, 52: 2,                             # entire agreement / integration
    79: 3,                                    # severability
    65: 4,                                    # notices
    26: 5,                                    # counterparts
    48: 6, 90: 6,                             # headings / titles
    49: 7, 50: 7,                             # indemnification
    20: 8,                                    # confidentiality
    87: 9, 86: 9,                             # taxes / withholding
    41: 10, 25: 10, 42: 10,                   # expenses / costs / fees
    88: 11,                                   # terminations
}
LEDGAR_VARS = [
    V2("provision",
       ["governing law or jurisdiction", "arbitration or jury-trial waiver", "entire agreement", "severability",
        "notices", "counterparts", "headings", "indemnification", "confidentiality", "taxes and withholding",
        "expenses and fees", "termination"],
       ["the provision says which law or courts govern the contract", "the provision is about arbitration or waiving a jury trial",
        "the provision says the contract is the entire agreement", "the provision says invalid parts can be severed",
        "the provision says how notices must be given", "the provision lets the contract be signed in counterparts",
        "the provision says headings do not affect interpretation", "the provision is an indemnification clause",
        "the provision is about keeping information confidential", "the provision is about taxes or tax withholding",
        "the provision is about who pays expenses or fees", "the provision is about ending the agreement"],
       ["What kind of provision is this?", "Which standard contract clause is this?",
        "How would a lawyer label this section of the contract?"],
       coarse=C("provision_group", ["disputes and applicable law", "boilerplate about the document", "money and risk",
                                    "confidentiality and termination"],
                ["the provision deals with disputes or which law applies", "the provision is boilerplate about the contract document",
                 "the provision deals with money or risk", "the provision is about confidentiality or ending the agreement"],
                [0, 0, 1, 1, 1, 1, 1, 2, 3, 2, 2, 3])),
    V2("favours", ["mainly one party", "both parties equally"],
       ["the provision mainly benefits one of the parties", "the provision treats both parties the same"],
       ["Does this provision favour one party, or is it neutral between them?", "Is this clause balanced between the parties?",
        "Would either side gain more from this clause than the other?"]),
    V2("negotiated", ["standard boilerplate", "commonly adjusted", "heavily negotiated"],
       ["the provision is standard boilerplate that is rarely negotiated", "the provision is the kind that is commonly adjusted a little",
        "the provision is the kind that is heavily negotiated"],
       ["How much would lawyers usually negotiate a clause like this?", "Is this boilerplate or a negotiated term?",
        "How often do parties fight over clauses like this one?"],
       cum=["the provision is standard boilerplate", "the provision is not the kind that is heavily negotiated"]),
]


def ledgar(rng, n):
    pd = _pd()
    f = _files("coastalcph/lex_glue", "ledgar/test-*.parquet")[0]
    df = pd.read_parquet(f).assign(_row=lambda d: range(len(d)))
    df = df[df.label.isin(list(_LEDGAR)) & df.text.str.len().between(80, 1200)].drop_duplicates("text").reset_index(drop=True)
    by = collections.defaultdict(list)
    for i, l in enumerate(df.label):
        by[_LEDGAR[int(l)]].append(i)
    per = max(1, n // len(by))
    idx = []
    for k in sorted(by):
        idx += rng.sample(by[k], min(per, len(by[k])))
    chosen = set(idx)
    idx += rng.sample([i for i in range(len(df)) if i not in chosen], n - len(idx))
    rng.shuffle(idx)
    return [(df.text.iloc[i].strip(), {"provision": _LEDGAR[int(df.label.iloc[i])]}, ref_of(f, row=int(df._row.iloc[i])))
            for i in idx]


# ================================================================================================= medical, Russian
_SPEC_RU = ["Гинеколог", "Уролог", "Проктолог", "Гастроэнтеролог", "Терапевт", "Невролог", "Дерматолог", "Травматолог",
            "Отоларинголог", "Офтальмолог", "Стоматолог", "Хирург"]
MEDRU_VARS = [
    V2("specialty",
       ["gynaecologist", "urologist", "proctologist", "gastroenterologist", "general practitioner", "neurologist",
        "dermatologist", "trauma surgeon", "ear, nose and throat doctor", "ophthalmologist", "dentist", "general surgeon"],
       ["the patient should see a gynaecologist", "the patient should see a urologist", "the patient should see a proctologist",
        "the patient should see a gastroenterologist", "the patient should see a general practitioner",
        "the patient should see a neurologist", "the patient should see a dermatologist", "the patient should see a trauma surgeon",
        "the patient should see an ear, nose and throat doctor", "the patient should see an ophthalmologist",
        "the patient should see a dentist", "the patient should see a general surgeon"],
       ["Which specialist should this patient see?", "To which kind of doctor should this question be forwarded?",
        "Which medical specialty is responsible for this complaint?"],
       coarse=C("specialty_group", ["pelvic and digestive organs", "general medicine and nerves", "surface, senses and teeth",
                                    "surgery and injuries"],
                ["the complaint belongs to a pelvic or digestive specialist", "the complaint belongs to general medicine or neurology",
                 "the complaint belongs to a skin, eye, ENT or dental specialist", "the complaint needs a surgeon"],
                [0, 0, 0, 0, 1, 1, 2, 3, 2, 2, 2, 3])),
    V2("urgency", ["can wait for a routine appointment", "should be seen within days", "needs urgent care today"],
       ["the problem can wait for a routine appointment", "the patient should be seen within a few days",
        "the patient needs urgent care today"],
       ["How urgent is this patient's problem?", "How quickly should this person see a doctor?",
        "Can this wait, or does the patient need help soon?"],
       cum=["the problem can wait for a routine appointment", "the patient does not need urgent care today"]),
    V2("chronic", ["no, it is a new problem", "yes, it has lasted a long time"],
       ["the problem started recently", "the problem has been going on for a long time"],
       ["Is this a long-standing problem?", "Has the patient had this complaint for a long time?",
        "Is the complaint chronic rather than new?"]),
]


def med_ru(rng, n):
    pd = _pd()
    f = _files("AlucardV/medical-specialty-classification", "data/*.parquet")[0]
    df = pd.read_parquet(f).assign(_row=lambda d: range(len(d)))
    df = df[df.spec10.isin(_SPEC_RU)].drop_duplicates("text").reset_index(drop=True)
    idx = rng.sample(range(len(df)), n)
    return [(df.text.iloc[i].strip(), {"specialty": _SPEC_RU.index(df.spec10.iloc[i])}, ref_of(f, row=int(df._row.iloc[i])))
            for i in idx]


# ===================================================================================== science: press vs abstract
_STRENGTH = ["no relationship", "a correlation", "a conditional causal link", "a direct causal link"]
SCI_VARS = [
    V2("exaggeration", ["same strength", "the press release exaggerates", "the press release downplays"],
       ["the press release states the finding as strongly as the abstract", "the press release overstates the finding",
        "the press release understates the finding"],
       ["Compared with the abstract, how does the press release state the finding?",
        "Does the press release exaggerate, downplay or faithfully report the paper's conclusion?",
        "Is the press release's claim stronger, weaker or equal to the paper's?"]),
    V2("press_strength", _STRENGTH,
       ["the press release claims no relationship", "the press release claims only a correlation",
        "the press release claims a causal link under some conditions", "the press release claims a direct causal link"],
       ["How strong a relationship does the press release claim?", "What kind of link does the press release sentence assert?",
        "Does the press release describe correlation or causation?"],
       cum=["the press release claims no relationship", "the press release claims at most a correlation",
            "the press release does not claim a direct causal link"]),
    V2("abstract_strength", _STRENGTH,
       ["the abstract claims no relationship", "the abstract claims only a correlation",
        "the abstract claims a causal link under some conditions", "the abstract claims a direct causal link"],
       ["How strong a relationship does the abstract's conclusion claim?", "What kind of link does the paper's abstract assert?",
        "Does the abstract describe correlation or causation?"],
       cum=["the abstract claims no relationship", "the abstract claims at most a correlation",
            "the abstract does not claim a direct causal link"]),
]
_EXAG = {"same": 0, "exaggerates": 1, "downplays": 2}


def sci_claims(rng, n):
    rows = []
    for f in _files("copenlu/scientific-exaggeration-detection", "*.jsonl"):
        rows += [(f, j, json.loads(l)) for j, l in enumerate(open(f))]
    seen, uniq = set(), []
    for f, j, r in rows:
        k = r["press_release_conclusion"].strip()
        if k not in seen:
            seen.add(k); uniq.append((f, j, r))
    idx = rng.sample(range(len(uniq)), n)
    out = []
    for i in idx:
        f, j, r = uniq[i]
        t = (f"Press release: {r['press_release_conclusion'].strip()}\n"
             f"Abstract conclusion: {r['abstract_conclusion'].strip()}")
        out.append((t, {"exaggeration": _EXAG[r["exaggeration_label"]], "press_strength": int(r["press_release_strength"]),
                        "abstract_strength": int(r["abstract_strength"])},
                    ref_of(f, row=j, id=str(r.get("original_file_id")))))
    return out


# ====================================================================================================== code defects
CODE_VARS = [
    V2("vulnerable", ["no, it looks safe", "yes, it has a security flaw"],
       ["the function is free of security vulnerabilities", "the function contains a security vulnerability"],
       ["Does this C function contain a security vulnerability?", "Would a security audit flag this function as vulnerable?",
        "Is there an exploitable bug (e.g. a buffer overflow or use-after-free) in this code?"]),
    V2("project", ["FFmpeg", "QEMU"],
       ["the function comes from FFmpeg", "the function comes from QEMU"],
       ["Which open-source project is this function from?", "Is this code from FFmpeg or from QEMU?",
        "Which code base does this function belong to?"]),
    V2("review_time", ["under 5 minutes", "5 to 30 minutes", "more than 30 minutes"],
       ["reviewing the function takes under 5 minutes", "reviewing the function takes 5 to 30 minutes",
        "reviewing the function takes more than 30 minutes"],
       ["How long would a careful code review of this function take?", "How much reviewer time does this function need?",
        "Is this a quick review or a long one?"],
       cum=["a careful review takes under 5 minutes", "a careful review takes at most 30 minutes"]),
]


def code_defects(rng, n):
    pd = _pd()
    f = _files("google/code_x_glue_cc_defect_detection", "data/test-*.parquet")[0]
    df = pd.read_parquet(f)
    df["func"] = df.func.str.replace(r"\n\s*\n", "\n", regex=True)
    df = df[df.func.str.len().between(200, 1500)].drop_duplicates("func").reset_index(drop=True)
    pos = [i for i in range(len(df)) if df.target.iloc[i]]
    neg = [i for i in range(len(df)) if not df.target.iloc[i]]
    idx = rng.sample(pos, n // 2) + rng.sample(neg, n - n // 2)
    rng.shuffle(idx)
    return [(df.func.iloc[i].strip(), {"vulnerable": int(df.target.iloc[i]), "project": int(df.project.iloc[i] == "qemu")},
             ref_of(f, id=int(df.id.iloc[i]), commit=str(df.commit_id.iloc[i]))) for i in idx]


# ====================================================================================================== GitHub issues
_REPOS = ["django/django", "sympy/sympy", "sphinx-doc/sphinx", "matplotlib/matplotlib", "scikit-learn/scikit-learn",
          "astropy/astropy", "pydata/xarray", "pytest-dev/pytest", "pylint-dev/pylint", "psf/requests"]
SWE_VARS = [
    V2("repo", ["Django", "SymPy", "Sphinx", "Matplotlib", "scikit-learn", "Astropy", "xarray", "pytest", "Pylint", "Requests"],
       ["the issue was filed against Django", "the issue was filed against SymPy", "the issue was filed against Sphinx",
        "the issue was filed against Matplotlib", "the issue was filed against scikit-learn", "the issue was filed against Astropy",
        "the issue was filed against xarray", "the issue was filed against pytest", "the issue was filed against Pylint",
        "the issue was filed against Requests"],
       ["Which project's issue tracker is this from?", "Which Python library is this issue about?",
        "In which repository was this issue opened?"],
       coarse=C("repo_group", ["web and HTTP", "scientific computing", "developer tools and docs"],
                ["the issue is about a web or HTTP library", "the issue is about a scientific-computing library",
                 "the issue is about a developer tool or documentation tool"], [0, 1, 2, 1, 1, 1, 1, 2, 2, 0])),
    V2("difficulty", ["under 15 minutes", "15 minutes to 1 hour", "1 to 4 hours", "more than 4 hours"],
       ["an experienced engineer fixes it in under 15 minutes", "an experienced engineer needs 15 minutes to an hour",
        "an experienced engineer needs 1 to 4 hours", "an experienced engineer needs more than 4 hours"],
       ["How long would an experienced engineer need to fix this issue?", "How hard is this issue to resolve, in working time?",
        "Is this a quick fix or a long job?"],
       cum=["the fix takes under 15 minutes", "the fix takes at most an hour", "the fix takes at most 4 hours"]),
    V2("kind", ["a bug", "a feature request or improvement"],
       ["the issue reports a bug", "the issue asks for a new feature or an improvement"],
       ["Is this issue a bug report or a feature request?", "Does the issue describe broken behaviour or ask for something new?",
        "Is something broken here, or is a change being requested?"]),
]
_DIFF = {"<15 min fix": 0, "15 min - 1 hour": 1, "1-4 hours": 2, ">4 hours": 3}
# review R12: the repo can be read off the text; project and package names (and their usual aliases) are masked
_SWE_MASK = re.compile(r"(?i)\b(django|sympy|sphinx(?:-doc)?|matplotlib|mpl_toolkits|mpl|pyplot|plt|sklearn|scikit-learn|"
                       r"scikit_learn|astropy|xarray|pytest(?:-dev)?|pylint(?:-dev)?|psf)\b")
_SWE_REQ = re.compile(r"(?i)\b(import requests|requests(?=\.[a-z_])|python-requests|psf/requests)\b")


def swe_mask(text):
    return _SWE_REQ.sub("[project]", _SWE_MASK.sub("[project]", text))


def swe_issues(rng, n):
    pd = _pd()
    f = _files("princeton-nlp/SWE-bench_Verified", "data/test-*.parquet")[0]
    df = pd.read_parquet(f)
    df = df[df.repo.isin(_REPOS)].reset_index(drop=True)
    df["ps"] = df.problem_statement.str.strip().map(swe_mask)
    short = [i for i in range(len(df)) if 120 <= len(df.ps.iloc[i]) <= 2200]
    longer = [i for i in range(len(df)) if len(df.ps.iloc[i]) > 2200]
    idx = rng.sample(short, min(n, len(short)))
    idx += rng.sample(longer, n - len(idx))
    rng.shuffle(idx)
    return [(_clip(df.ps.iloc[i], 2200), {"repo": _REPOS.index(df.repo.iloc[i]), "difficulty": _DIFF[df.difficulty.iloc[i]]},
             ref_of(f, id=str(df.instance_id.iloc[i]))) for i in idx]


# ================================================================================================= Thunderbird logs
TB_VARS = [
    ext(v1.LOG_VARS[0], ["Is there an alert-worthy event in these lines?"]),
    V2("source", ["the kernel", "a system daemon or service", "a user job or login"],
       ["most of the lines come from the kernel", "most of the lines come from a system daemon or service",
        "most of the lines come from a user job or login session"],
       ["What produced most of these log lines?", "Which kind of process wrote most of this log excerpt?",
        "Are these messages mainly from the kernel, a daemon, or user activity?"]),
    ext(v1.LOG_VARS[2], ["What should an operator do after reading these lines?"],
        cum=["no action is needed", "nobody needs to be paged right away"]),
]


def thunderbird_logs(rng, n, width=6):
    pd = _pd()
    import numpy as np
    cols = ["date", "location", "month", "day", "time", "component", "pid", "content", "anomaly"]
    f = _files("logfit-project/Thunderbird", "data/train-00000-*.parquet")[0]
    df = pd.read_parquet(f, columns=cols)
    an = np.flatnonzero(df.anomaly.to_numpy()).tolist()
    out, seen = [], set()
    want = {"anom": n // 2, "norm": n - n // 2}
    tries = 0
    while sum(want.values()) and tries < 100000:
        tries += 1
        s = rng.choice(an) if (rng.random() < 0.5 and want["anom"]) else rng.randrange(len(df) - width)
        s = max(0, min(s - rng.randrange(width), len(df) - width))
        if s in seen:
            continue
        w = df.iloc[s:s + width]
        k = "anom" if w.anomaly.any() else "norm"
        if not want[k]:
            continue
        seen.add(s); want[k] -= 1
        lines = [f"{r.date} {r.location} {r.month} {r.day} {r.time} {r.component}[{r.pid}]: {r.content}"
                 for r in w.itertuples(index=False)]
        out.append(("\n".join(lines), {"failure": int(w.anomaly.any())}, ref_of(f, row=int(s), n_rows=width)))
    rng.shuffle(out)
    return out


# ======================================================================================================= HDFS logs
HDFS_VARS = [
    V2("anomaly", ["no, the block was handled normally", "yes, something went wrong"],
       ["the block was handled normally", "something went wrong with this block"],
       ["Does this block's log session show an anomaly?", "Did something go wrong while handling this HDFS block?",
        "Is this a normal block lifecycle or a problematic one?"]),
    V2("stage", ["writing or receiving the block", "replicating or transferring the block", "deleting the block"],
       ["the session is mostly about writing or receiving the block", "the session is mostly about replicating or transferring the block",
        "the session is mostly about deleting the block"],
       ["What is happening to the block in most of these lines?", "Which stage of the block's life does this session mostly show?",
        "Is the block mostly being written, copied, or deleted here?"]),
    V2("data_loss", ["very unlikely", "possible", "likely"],
       ["data loss for this block is very unlikely", "data loss for this block is possible", "data loss for this block is likely"],
       ["How likely is it that this block's data is lost?", "Should the operator worry about losing this block's data?",
        "What is the risk of data loss for this block?"],
       cum=["data loss is very unlikely", "data loss is at most possible, not likely"]),
]


def hdfs_logs(rng, n, lo=19, hi=27):
    pd = _pd()
    f = _files("logfit-project/hdfsv1-grouped-labeled", "data/eval-*.parquet")[0]
    df = pd.read_parquet(f)
    df = df[df.num_sentences.between(lo, hi)].drop_duplicates("text").reset_index(drop=True)  # both labels occur here
    pos = [i for i in range(len(df)) if df.anomaly.iloc[i]]
    neg = [i for i in range(len(df)) if not df.anomaly.iloc[i]]
    idx = rng.sample(pos, n // 2) + rng.sample(neg, n - n // 2)
    rng.shuffle(idx)
    return [(f"Block session {df.start_timestamp.iloc[i]} - {df.end_timestamp.iloc[i]}\n" + df.text.iloc[i].strip(),
             {"anomaly": int(df.anomaly.iloc[i])}, ref_of(f, id=str(df.block_id.iloc[i]))) for i in idx]


# ========================================================================================================= ssh logs
SSH_VARS = [
    V2("activity", ["a password-guessing attack", "a scan for invalid user names", "normal logins by real users",
                    "harmless connection noise"],
       ["the lines show a password-guessing attack", "the lines show someone probing for invalid user names",
        "the lines show normal logins by real users", "the lines show only harmless connection noise"],
       ["What activity do these SSH log lines show?", "What is going on on this server, judging by the sshd log?",
        "How would a security analyst describe these lines?"],
       coarse=C("hostile", ["hostile", "benign"], ["the lines show hostile activity", "the lines show benign activity"],
                [0, 0, 1, 1])),
    V2("severity", ["informational", "low", "high"],
       ["the lines are purely informational", "the lines deserve a low-severity alert", "the lines deserve a high-severity alert"],
       ["What severity should an intrusion-detection system give these lines?", "How serious is this security event?",
        "How would you rate the severity of this sshd activity?"],
       cum=["the lines are purely informational", "the lines do not deserve a high-severity alert"]),
    V2("breach", ["no", "yes"],
       ["no attacker got into the server", "an attacker managed to log in"],
       ["Did an attacker succeed in logging in?", "Do these lines show a successful break-in?",
        "Was the server compromised according to this excerpt?"]),
]
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HOST = re.compile(r"(getaddrinfo for |mapping checking getaddrinfo for |\bfor )([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?= \[)")
_USER = re.compile(r"(\b(?:[Ii]nvalid user|[Ii]llegal user|user|for|by) (?!(?:invalid|illegal|user)\b)|\b(?:ruser|user)=)([A-Za-z0-9._-]+)")
_DOMAIN = re.compile(r"\b(?=[A-Za-z0-9.-]*[A-Za-z])[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.(?:com|net|org|edu|gov|info|biz|[a-z]{2})\b")
_KEEP_USER = {"root", "invalid", "unknown", "from", "user", "illegal"}


def ssh_mask(lines):
    """Mask IPs, host names and user names (review R8), consistently within a window (<IP1>, <HOST1>, <USER1>...), so
    repeated sources and targets stay visible. 'root' is kept: it is not personal data and matters for the task."""
    tabs = {"IP": {}, "HOST": {}, "USER": {}}

    def tag(kind, x):
        t = tabs[kind]
        t.setdefault(x, f"<{kind}{len(t) + 1}>")
        return t[x]
    out = []
    for ln in lines:
        ln = re.sub(r"^(\S+ +\S+ \S+) \S+ ", r"\1 <SERVER> ", ln)
        ln = re.sub(r"rhost=(\S+)", lambda m: "rhost=" + (tag("IP", m.group(1)) if _IP.fullmatch(m.group(1)) else tag("HOST", m.group(1))), ln)
        ln = _HOST.sub(lambda m: m.group(1) + tag("HOST", m.group(2)), ln)
        ln = _DOMAIN.sub(lambda m: tag("HOST", m.group(0)), ln)
        ln = _IP.sub(lambda m: tag("IP", m.group(0)), ln)
        ln = _USER.sub(lambda m: m.group(0) if m.group(2).lower() in _KEEP_USER or m.group(2).startswith("<")
                       else m.group(1) + tag("USER", m.group(2)), ln)
        out.append(ln)
    return out


def ssh_logs(rng, n, width=8):
    f = _files("bolu61/loghub_2", "data/openssh.txt")[0]
    lines = open(f, encoding="utf-8", errors="replace").read().splitlines()
    out, seen = [], set()
    while len(out) < n:
        s = rng.randrange(len(lines) - width)
        w = ssh_mask(lines[s:s + width])
        key = tuple(re.sub(r"^\S+ +\S+ \S+ ", "", x) for x in w)
        if s in seen or key in seen:
            continue
        seen.add(s); seen.add(key)
        out.append(("\n".join(w), {}, ref_of(f, line=s, n_lines=width)))
    return out


# ============================================================================================== CNN finance / sports
_FIN = r"\b(shares|stocks?|investors|Dow|Nasdaq|FTSE|profits?|revenues?|economy|economic|central bank|Federal Reserve|interest rates?|inflation|earnings|GDP|market)\b"
_SPORT = r"\b(Premier League|Champions League|NBA|NFL|Wimbledon|goals?|scored|striker|midfielder|quarterback|tennis|golf|Formula One|F1|cricket|rugby|match|coach|season|tournament|championship|touchdown|innings)\b"
_CNN = {}


def _cnn(split, k=0):
    key = (split, k)
    if key not in _CNN:
        pd = _pd()
        f = _files("abisee/cnn_dailymail", f"3.0.0/{split}-*.parquet")[k]
        df = pd.read_parquet(f, columns=["article", "id"]).assign(_row=lambda d: range(len(d)))
        df["lead"] = df.article.map(v1._lead)
        df = df[df.lead.str.len().between(250, 1000)].drop_duplicates("lead").reset_index(drop=True)
        _CNN[key] = (f, df)
    return _CNN[key]


def _cnn_filter(rng, n, pattern, k, other=None, k_other=99):
    f, df = _cnn("train")
    c = df.lead.str.count(pattern)
    m = c >= k
    if other is not None:
        m &= df.lead.str.count(other) < k_other
    idx = [i for i in range(len(df)) if m.iloc[i]]
    return [(df.lead.iloc[i], {}, ref_of(f, row=int(df._row.iloc[i]), id=str(df.id.iloc[i]))) for i in rng.sample(idx, n)]


FIN_VARS = [
    V2("direction", ["good news for investors", "mixed or neutral", "bad news for investors"],
       ["the news is good for investors", "the news is mixed or neutral for investors", "the news is bad for investors"],
       ["Is this story good or bad news for investors?", "How would markets likely read this news?",
        "Is the economic news here positive, negative or mixed?"],
       cum=["the news is good for investors", "the news is not bad for investors"]),
    V2("sector", ["banking and finance", "energy and commodities", "technology", "retail and consumer goods",
                  "transport and manufacturing", "government, trade and the wider economy"],
       ["the story is mainly about banks or finance", "the story is mainly about energy or commodities",
        "the story is mainly about technology companies", "the story is mainly about retail or consumer goods",
        "the story is mainly about transport or manufacturing", "the story is mainly about government policy, trade or the whole economy"],
       ["Which part of the economy is this story mainly about?", "Which sector does this business story cover?",
        "What industry or area of the economy is at the centre of this story?"],
       coarse=C("scope", ["specific industries", "the economy as a whole"],
                ["the story is about specific industries", "the story is about the economy as a whole"], [0, 0, 0, 0, 0, 1])),
    V2("horizon", ["within days", "within months", "over years"],
       ["the effects will be felt within days", "the effects will be felt within months", "the effects will play out over years"],
       ["Over what time frame will the effects of this news be felt?", "How soon will this news affect people's finances?",
        "Is this a short-term or a long-term story?"],
       cum=["the effects will be felt within days", "the effects will be felt within months at most"]),
]
SPORT_VARS = [
    V2("sport", ["football (soccer)", "American football", "basketball", "tennis", "golf", "motor racing",
                 "cricket or rugby", "another sport"],
       ["the story is about football (soccer)", "the story is about American football", "the story is about basketball",
        "the story is about tennis", "the story is about golf", "the story is about motor racing",
        "the story is about cricket or rugby", "the story is about another sport"],
       ["Which sport is this story about?", "What sport does this report cover?", "Which sport's section would this run in?"],
       coarse=C("football_code", ["a football code", "any other sport"],
                ["the story is about a football code", "the story is about a sport that is not a football code"],
                [0, 0, 1, 1, 1, 1, 1, 1])),
    V2("result", ["a win", "a loss or setback", "no result reported"],
       ["the story reports a win for its main team or athlete", "the story reports a loss or setback for its main team or athlete",
        "the story reports no result"],
       ["What result does the story report for its main team or athlete?", "Did the main team or athlete win, lose, or neither?",
        "Is this a victory story, a defeat story, or neither?"]),
    V2("offfield", ["no", "yes"],
       ["the story is about what happened on the field", "the story is mainly about something off the field (injury, transfer, scandal)"],
       ["Is the story mainly about something off the field, like an injury, a transfer or a scandal?",
        "Is this more about off-field news than about a game?", "Does the story focus on events away from the competition itself?"]),
]


def finance_news(rng, n):
    return _cnn_filter(rng, n, _FIN, 4, _SPORT, 3)


def sports(rng, n):
    return _cnn_filter(rng, n, _SPORT, 5, _FIN, 2)


# ===================================================================================================== fineweb-edu
_SE = {  # site -> fine group
    "physics": 0, "astronomy": 0, "space": 0, "earthscience": 0, "chemistry": 0,
    "math": 1, "mathoverflow": 1, "stats": 1, "cs": 1, "crypto": 1, "cstheory": 1,
    "programmers": 2, "softwareengineering": 2, "unix": 2, "security": 2, "gamedev": 2, "codegolf": 2, "gis": 2,
    "retrocomputing": 2, "superuser": 2, "serverfault": 2, "askubuntu": 2, "dba": 2, "tex": 2,
    "electronics": 3, "engineering": 3, "aviation": 3, "diy": 3,
    "biology": 4, "medicalsciences": 4, "psychology": 4, "fitness": 4,
    "english": 5, "ell": 5, "japanese": 5, "chinese": 5, "german": 5, "spanish": 5, "linguistics": 5, "french": 5,
    "history": 6, "politics": 6, "law": 6, "economics": 6, "skeptics": 6,
    "christianity": 7, "hermeneutics": 7, "judaism": 7, "philosophy": 7, "islam": 7, "buddhism": 7,
    "cooking": 8, "photo": 8, "music": 8, "parenting": 8, "worldbuilding": 8, "gardening": 8, "outdoors": 8, "travel": 8,
    "scifi": 8, "movies": 8, "boardgames": 8, "rpg": 8, "writing": 8,
}
SE_VARS = [
    V2("community", ["physical sciences", "mathematics and theoretical CS", "software and computing", "engineering and electronics",
                     "life sciences and health", "languages and linguistics", "history, politics and society",
                     "religion and philosophy", "hobbies and everyday life"],
       ["the page is from a physical-sciences community", "the page is from a mathematics or theoretical computer-science community",
        "the page is from a software or computing community", "the page is from an engineering or electronics community",
        "the page is from a life-sciences or health community", "the page is from a language or linguistics community",
        "the page is from a history, politics or society community", "the page is from a religion or philosophy community",
        "the page is from a hobby or everyday-life community"],
       ["Which kind of Q&A community is this page from?", "On which sort of Stack Exchange site would this question be posted?",
        "What subject community does this discussion belong to?"],
       coarse=C("area", ["science, maths and technology", "humanities", "everyday life"],
                ["the page is from a science, maths or technology community", "the page is from a humanities community",
                 "the page is from an everyday-life community"], [0, 0, 0, 0, 0, 1, 1, 1, 2])),
    V2("answer_quality", ["poor", "adequate", "excellent"],
       ["the best answer on the page is poor", "the best answer on the page is adequate", "the best answer on the page is excellent"],
       ["How good is the best answer on this page?", "How well is the question answered here?",
        "Would you rate the answer here as poor, adequate or excellent?"],
       cum=["the best answer is poor", "the best answer is at most adequate"]),
    V2("qtype", ["asks for facts or an explanation", "asks for advice or a recommendation", "asks for opinions"],
       ["the question asks for facts or an explanation", "the question asks for advice or a recommendation",
        "the question asks for opinions"],
       ["What kind of question is being asked?", "Is the asker looking for facts, advice or opinions?",
        "What sort of answer does the question want?"]),
]
RECIPE_VARS = [
    V2("course", ["a main dish", "a side dish, salad or soup", "a dessert or sweet baked good", "bread or savoury baking",
                  "a snack, drink or sauce"],
       ["the recipe is for a main dish", "the recipe is for a side dish, salad or soup", "the recipe is for a dessert or sweet baked good",
        "the recipe is for bread or savoury baking", "the recipe is for a snack, drink or sauce"],
       ["What kind of dish does this recipe make?", "Which part of a meal is this recipe for?", "How would you classify this dish?"]),
    V2("time", ["under 30 minutes", "30 to 60 minutes", "1 to 2 hours", "more than 2 hours"],
       ["making it takes under 30 minutes", "making it takes 30 to 60 minutes", "making it takes 1 to 2 hours",
        "making it takes more than 2 hours"],
       ["How long does this recipe take from start to finish?", "How much total time does this dish need?",
        "Is this a quick recipe or a slow one?"],
       cum=["it takes under 30 minutes", "it takes at most an hour", "it takes at most two hours"]),
    V2("vegetarian", ["no", "yes"], ["the dish contains meat or fish", "the dish is vegetarian"],
       ["Is this dish vegetarian?", "Could a vegetarian eat this dish as written?", "Is the recipe free of meat and fish?"]),
]
REVIEW_VARS = [
    V2("verdict", ["positive (4 or 5 stars)", "mixed (3 stars)", "negative (1 or 2 stars)"],
       ["the reviewer gave it 4 or 5 stars out of 5", "the reviewer gave it 3 stars out of 5", "the reviewer gave it 1 or 2 stars out of 5"],
       ["What star rating (out of 5) did the reviewer give?", "Did the reviewer like the book?",
        "Is this a positive, mixed or negative review?"],
       cum=["the rating is 4 or 5 stars", "the rating is at least 3 stars"]),
    V2("genre", ["fiction", "non-fiction", "a children's or young-adult book"],
       ["the book is fiction for adults", "the book is non-fiction for adults", "the book is for children or young adults"],
       ["What kind of book is being reviewed?", "Is this book fiction, non-fiction, or for young readers?",
        "Which shelf would this book go on?"]),
    V2("recommend", ["no", "yes"],
       ["the reviewer would not recommend the book", "the reviewer would recommend the book to others"],
       ["Would the reviewer recommend this book?", "Does the reviewer suggest others read it?",
        "Is the reviewer telling people to read this book?"]),
]
_QL = re.compile(r"(?im)^\s*[-*•]?\s*(\d+(\s?\d/\d|/\d|\.\d)?|½|¼|¾|⅓)\s*(cups?|c\.|tbsp\.?|tablespoons?|tsp\.?|teaspoons?|"
                 r"grams|g|kg|oz\.?|ounces?|lbs?\.?|pounds?|ml|cloves?|pinch|large|medium|small|cans?)\b")
_RATING = re.compile(r"My rating: ([1-5]) of 5 stars")
_RATING_MASK = re.compile(r"(?i)(my rating:\s*[1-5] of 5 stars|\b[1-5](\.5)? (?:out )?of 5 stars\b|\b[1-5](\.5)?\s*/\s*5\b)")
_FW_MEMO: dict = {}
FW_KINDS = ("stackexchange", "reviews")


def _recipe_start(text):
    """Offset of an 'Ingredients' mention followed within 600 characters by >= 4 ingredient lines, or None."""
    for m in re.finditer(r"(?i)ingredients", text):
        if len(_QL.findall(text[m.start():m.start() + 600])) >= 4:
            return m.start()
    return None


def _fw_classify(url, text):
    if re.search(r"https?://([a-z0-9]+)\.stackexchange\.com/questions/\d+", url):
        site = re.search(r"https?://([a-z0-9]+)\.stackexchange\.com", url).group(1)
        return ("stackexchange", {"community": _SE[site]}) if site in _SE else (None, None)
    m = _RATING.search(text)
    if m:
        k = int(m.group(1))
        return "reviews", {"verdict": 0 if k >= 4 else (1 if k == 3 else 2)}
    return None, None


def _fw_text(kind, text):
    if kind == "recipes":
        k = _recipe_start(text)
        k = text.rfind("\n", 0, max(0, k - 300)) + 1 if k > 300 else 0
        title = text.split("\n", 1)[0][:150]
        return (title + "\n[...]\n" if k > 0 else "") + _clip(text[k:], 1300)
    if kind == "reviews":                                # the rating is the gold: mask every stated rating
        return _clip(re.sub(r"\n{2,}", "\n", _RATING_MASK.sub("[rating removed]", text)), 1500)
    return _clip(text, 1500)


def _fineweb(kind, n):
    """Candidates of `kind`, mined deterministically from the sorted fineweb-edu sample files (10BT, then 100BT)."""
    quota = max(n + 40, int(1.3 * n))
    if kind in _FW_MEMO and len(_FW_MEMO[kind]) >= quota:
        return _FW_MEMO[kind]
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    got = {k: [] for k in FW_KINDS}
    root = os.path.join(v1.snapshot_dir("HuggingFaceFW/fineweb-edu"), "sample")
    fs = sorted(glob.glob(os.path.join(root, "10BT", "*.parquet"))) + sorted(glob.glob(os.path.join(root, "100BT", "*.parquet")))
    seen = set()
    pre_url, pre_text = r"stackexchange\.com/questions/", r"My rating: [1-5] of 5 stars"
    for f in fs:
        pf = pq.ParquetFile(f)
        off = 0
        for rg in range(pf.num_row_groups):
            nrows = pf.metadata.row_group(rg).num_rows
            if all(len(got[k]) >= quota for k in FW_KINDS):
                break
            t = pf.read_row_group(rg, columns=["id", "url", "text", "token_count"])
            m = pc.and_(pc.less_equal(t["token_count"], 1500),
                        pc.or_(pc.match_substring_regex(t["url"], pre_url), pc.match_substring_regex(t["text"], pre_text)))
            rows = np.flatnonzero(m.to_numpy(zero_copy_only=False))
            s = t.take(rows).to_pydict()
            for j, doc, url, text in zip(rows, s["id"], s["url"], s["text"]):
                k, g = _fw_classify(url, text)
                if k is None or len(got[k]) >= quota or len(text) < 200:
                    continue
                body = _fw_text(k, text)
                if body in seen:
                    continue
                seen.add(body)
                got[k].append((body, g, ref_of(f, row=int(off + j), id=doc, url=url)))
            off += nrows
        if all(len(got[k]) >= quota for k in FW_KINDS):
            break
    _FW_MEMO.update(got)
    return got[kind]


def _fw_source(kind):
    def fn(rng, n):
        cands = _fineweb(kind, n)
        if len(cands) < n:
            raise RuntimeError(f"fineweb-edu: only {len(cands)} {kind} candidates")
        return [cands[i] for i in sorted(rng.sample(range(len(cands)), n))]
    fn.__name__ = kind
    return fn


# ============================================================================================================== MATH
_MTYPES = ["Prealgebra", "Algebra", "Intermediate Algebra", "Number Theory", "Counting & Probability", "Geometry", "Precalculus"]
MATH_VARS = [
    V2("subject", ["prealgebra", "algebra", "intermediate algebra", "number theory", "counting and probability", "geometry",
                   "precalculus"],
       ["the problem is a prealgebra problem", "the problem is an algebra problem", "the problem is an intermediate-algebra problem",
        "the problem is a number-theory problem", "the problem is a counting-and-probability problem", "the problem is a geometry problem",
        "the problem is a precalculus problem"],
       ["Which subject area is this competition problem from?", "Under which topic would this problem be filed?",
        "What branch of mathematics does this problem belong to?"],
       coarse=C("branch", ["algebra and arithmetic", "discrete mathematics", "geometry and trigonometry"],
                ["the problem is from an algebra or arithmetic category", "the problem is from a discrete-mathematics category",
                 "the problem is from a geometry or trigonometry category"], [0, 0, 0, 1, 1, 2, 2])),
    V2("level", ["level 1 (easiest)", "level 2", "level 3", "level 4", "level 5 (hardest)"],
       ["the problem is at difficulty level 1", "the problem is at difficulty level 2", "the problem is at difficulty level 3",
        "the problem is at difficulty level 4", "the problem is at difficulty level 5"],
       ["How difficult is this problem on the 1-5 competition scale?", "What difficulty level (1 easiest, 5 hardest) is this problem?",
        "How hard would students find this problem, from level 1 to level 5?"],
       cum=["the difficulty is level 1", "the difficulty is at most level 2", "the difficulty is at most level 3",
            "the difficulty is at most level 4"]),
    V2("integer", ["no", "yes"], ["the final answer is not an integer", "the final answer is an integer"],
       ["Is the final answer to this problem an integer?", "Will the answer come out as a whole number?",
        "Is the solution's final answer a whole number?"]),
]


def _boxed(sol):
    k = sol.rfind("\\boxed")
    if k < 0:
        return None
    i = sol.find("{", k)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(sol):
        depth += {"{": 1, "}": -1}.get(sol[j], 0)
        if depth == 0:
            return sol[i + 1:j]
        j += 1
    return None


def answer_is_integer(ans):
    """True / False for a MATH boxed answer, None when it cannot be decided safely (review R12: the plain -?\\d+ check
    missed units, thousands separators, degrees, '\\$', 'x=' and exact fractions)."""
    if ans is None:
        return None
    s = ans.strip()
    m = re.fullmatch(r"\\(?:text|mbox|textbf)\{\s*([^{}]*?)\s*\}", s)
    if m:                                                   # a pure text answer: a letter choice, a name, a word or a number
        return bool(re.fullmatch(r"[-+]?\d+", m.group(1)))
    s = re.sub(r"\\(text|mbox|textbf|mathrm)\{\s*(?:[a-zA-Z]+\.?\s*)+(?:\^\{?\d\}?)?\}", "", s)   # unit words
    for a, b in (("\\!", ""), ("{,}", ""), ("\\,", ""), ("\\$", ""), ("$", ""), ("\\%", ""), ("%", ""),
                 ("^\\circ", ""), ("^{\\circ}", ""), ("\\circ", ""), ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), (" ", "")):
        s = s.replace(a, b)
    s = re.sub(r"^[a-zA-Z]=", "", s)
    if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    if re.fullmatch(r"[-+]?\d+(\.0+)?", s):
        return True
    m = re.fullmatch(r"([-+]?)\\frac\{?(\d+)\}?\{?(\d+)\}?", s)
    if m:
        return int(m.group(3)) != 0 and int(m.group(2)) % int(m.group(3)) == 0
    if re.fullmatch(r"\d+\^\{?\d+\}?", s):
        return True
    if not s or re.search(r"\\text|\\mbox", s):
        return None
    return False


def math_problems(rng, n):
    pd = _pd()
    f = _files("DigitalLearningGmbH/MATH-lighteval", "data/test-*.parquet")[0]
    df = pd.read_parquet(f).assign(_row=lambda d: range(len(d)))
    df = df[df.type.isin(_MTYPES) & df.level.isin([f"Level {i}" for i in range(1, 6)]) & df.problem.str.len().between(40, 900)]
    df = df.drop_duplicates("problem").reset_index(drop=True)
    idx = rng.sample(range(len(df)), n)
    out = []
    for i in idx:
        r = df.iloc[i]
        g = {"subject": _MTYPES.index(r.type), "level": int(r.level[-1]) - 1}
        z = answer_is_integer(_boxed(r.solution))
        if z is not None:
            g["integer"] = int(z)
        out.append((r.problem.strip(), g, ref_of(f, row=int(r._row))))
    return out


# =========================================================================================================== MMLU-Pro
# only the items MMLU-Pro added (stemez, theoremQA, scibench): the ori_mmlu items are cais/mmlu questions, which decider
# trained on; the categories that are purely ori_mmlu (health, history, law, other, philosophy) therefore disappear.
_MMLU = ["math", "physics", "chemistry", "engineering", "computer science", "biology", "psychology", "economics", "business"]
MMLU_VARS = [
    V2("category", ["mathematics", "physics", "chemistry", "engineering", "computer science", "biology", "psychology",
                    "economics", "business"],
       ["the question is from a mathematics exam", "the question is from a physics exam", "the question is from a chemistry exam",
        "the question is from an engineering exam", "the question is from a computer-science exam", "the question is from a biology exam",
        "the question is from a psychology exam", "the question is from an economics exam", "the question is from a business exam"],
       ["Which exam subject is this question from?", "Which subject category does this test question belong to?",
        "In which course would this exam question appear?"],
       coarse=C("faculty", ["natural sciences and mathematics", "engineering and computing", "social sciences and business"],
                ["the question is from a natural-science or mathematics exam", "the question is from an engineering or computing exam",
                 "the question is from a social-science or business exam"], [0, 0, 0, 1, 1, 0, 2, 2, 2])),
    V2("answer", list("ABCDEFGHIJ"), [f"the correct answer is option {c}" for c in "ABCDEFGHIJ"],
       ["Which option is the correct answer?", "Which letter should a student circle?", "What is the right choice among the options?"]),
]


def mmlu_pro(rng, n):
    pd = _pd()
    f = _files("TIGER-Lab/MMLU-Pro", "data/test-*.parquet")[0]
    df = pd.read_parquet(f)
    df = df[(df.options.map(len) == 10) & ~df.src.str.startswith("ori_mmlu") & df.category.isin(_MMLU)].reset_index(drop=True)
    df["state"] = [q.strip() + "\n" + "\n".join(f"{c}. {o}" for c, o in zip("ABCDEFGHIJ", opts))
                   for q, opts in zip(df.question, df.options)]
    df = df[df.state.str.len() <= 1600].drop_duplicates("state").reset_index(drop=True)
    by = collections.defaultdict(list)
    for i, c in enumerate(df.category):
        by[c].append(i)
    per = n // len(_MMLU)
    idx = []
    for c in _MMLU:
        idx += rng.sample(by[c], min(per, len(by[c])))
    rest = sorted(set(range(len(df))) - set(idx))
    idx += rng.sample(rest, n - len(idx))
    rng.shuffle(idx)
    return [(df.state.iloc[i], {"category": _MMLU.index(df.category.iloc[i]), "answer": int(df.answer_index.iloc[i])},
             ref_of(f, id=int(df.question_id.iloc[i]), src=str(df.src.iloc[i]))) for i in idx]


# ===================================================================================================== SciEntsBank
STUDENT_VARS = [
    V2("grade", ["correct", "contradicts the reference", "partially correct but incomplete", "irrelevant", "off-topic"],
       ["the student's answer is correct", "the student's answer contradicts the reference answer",
        "the student's answer is partially correct but incomplete", "the student's answer is irrelevant to the question",
        "the student's answer is off-topic (not about the subject at all)"],
       ["How should this student answer be graded?", "Which grade does this answer deserve compared with the reference?",
        "How does the student's answer compare with the reference answer?"],
       # the dataset's own 3-way scheme: correct / contradictory / incorrect (partial, irrelevant, non-domain)
       coarse=C("grade3", ["correct", "contradictory", "incorrect"],
                ["the answer counts as correct in the 3-way scheme", "the answer counts as contradictory in the 3-way scheme",
                 "the answer counts as incorrect in the 3-way scheme"], [0, 1, 2, 2, 2])),
    V2("misconception", ["no", "yes"], ["the answer shows no scientific misconception", "the answer shows a scientific misconception"],
       ["Does the student's answer reveal a scientific misconception?", "Is there a misunderstanding of the science in this answer?",
        "Does the student hold a wrong idea about the science involved?"]),
    V2("feedback", ["none needed", "a short hint", "re-teaching the idea"],
       ["the student needs no feedback", "a short hint is enough feedback", "the idea needs to be taught again"],
       ["What feedback does this student need?", "How much follow-up teaching does this answer call for?",
        "Should the teacher do nothing, give a hint, or re-teach?"],
       cum=["no feedback is needed", "at most a short hint is needed"]),
]


def student_answers(rng, n):
    pd = _pd()
    f = _files("nkazi/SciEntsBank", "data/train-*.parquet")[0]
    df = pd.read_parquet(f)
    df = df.drop_duplicates(["question", "student_answer"]).reset_index(drop=True)
    by = collections.defaultdict(list)
    for i, l in enumerate(df.label):
        by[int(l)].append(i)
    idx = rng.sample(by[4], min(len(by[4]), n // 10))   # the rare off-topic grade
    per = (n - len(idx)) // 4
    for l in (1, 2, 3):
        idx += rng.sample(by[l], per)
    idx += rng.sample(by[0], n - len(idx))
    rng.shuffle(idx)
    out = []
    for i in idx:
        r = df.iloc[i]
        t = f"Question: {r.question.strip()}\nReference answer: {r.reference_answer.strip()}\nStudent answer: {r.student_answer.strip()}"
        out.append((t, {"grade": int(r.label)}, ref_of(f, id=str(r.id))))
    return out


# ======================================================================================================= EUR-Lex, fr
EURLEX_VARS = [
    V2("act_type", ["a regulation", "a directive", "a decision"],
       ["the act is a regulation", "the act is a directive", "the act is a decision"],
       ["What type of EU legal act is this?", "Is this act a regulation, a directive or a decision?",
        "Which kind of EU legislation is this text?"]),
    V2("domain", ["agriculture, fisheries and food", "trade, customs and the internal market", "environment, energy and transport",
                  "finance, economy and competition", "health, social affairs and justice", "external relations and institutions"],
       ["the act is about agriculture, fisheries or food", "the act is about trade, customs or the internal market",
        "the act is about the environment, energy or transport", "the act is about finance, the economy or competition",
        "the act is about health, social affairs or justice", "the act is about external relations or the EU institutions"],
       ["Which policy area does this act concern?", "What is this piece of EU law mainly about?",
        "Under which policy heading would this act be filed?"]),
    V2("reach", ["a few named parties", "one sector or product group", "the general public across the EU"],
       ["the act directly affects only a few named parties", "the act affects one sector or product group",
        "the act affects the general public across the EU"],
       ["How widely does this act apply?", "Who is directly affected by this act?", "How broad is the reach of this legal act?"],
       cum=["the act affects only a few named parties", "the act affects at most one sector"]),
]
_CELEX = {"R": 0, "L": 1, "D": 2}
_ACT_WORDS = re.compile(r"(?i)\b(r[èe]glements?|directives?|d[ée]cisions?)\b")   # review R12: the type is in the text


def eurlex_fr(rng, n):
    pd = _pd()
    f = _files("coastalcph/multi_eurlex", "fr/test/0000.parquet")[0]
    df = pd.read_parquet(f)
    df["t"] = df.celex_id.str.extract(r"^3\d{4}([RLD])")[0]
    df = df[df.t.notna()].reset_index(drop=True)
    by = collections.defaultdict(list)
    for i, t in enumerate(df.t):
        by[t].append(i)
    idx = []
    for t, share in (("R", 0.4), ("L", 0.2), ("D", 0.4)):
        idx += rng.sample(by[t], min(len(by[t]), int(n * share)))
    idx += rng.sample(sorted(set(range(len(df))) - set(idx)), n - len(idx))
    rng.shuffle(idx)
    return [(_clip(_ACT_WORDS.sub("[ACTE]", df.text.iloc[i]), 1300), {"act_type": _CELEX[df.t.iloc[i]]},
             ref_of(f, id=str(df.celex_id.iloc[i]))) for i in idx]


# ===================================================================================================== Belebele, es
_BSITE = {"wikinews": 0, "wikivoyage": 1, "wikibooks": 2, "wikipedia": 3}
BELE_VARS = [
    V2("answer", ["answer 1", "answer 2", "answer 3", "answer 4"],
       ["answer 1 is the correct one", "answer 2 is the correct one", "answer 3 is the correct one", "answer 4 is the correct one"],
       ["Which of the four answers is correct?", "Which numbered answer should be chosen?", "What is the right answer to the question?"]),
    V2("site", ["a news site (Wikinews)", "a travel guide (Wikivoyage)", "a how-to or textbook (Wikibooks)", "an encyclopedia (Wikipedia)"],
       ["the passage comes from a news site", "the passage comes from a travel guide", "the passage comes from a how-to book or textbook",
        "the passage comes from an encyclopedia"],
       ["Where was this passage originally published?", "What kind of source is the passage taken from?",
        "Which type of website does this passage come from?"],
       coarse=C("genre", ["news", "reference or guide"], ["the passage is news", "the passage is reference or guide material"], [0, 1, 1, 1])),
    V2("difficulty", ["easy", "moderate", "hard"],
       ["the question is easy for a fluent reader", "the question is moderately hard for a fluent reader",
        "the question is hard for a fluent reader"],
       ["How hard is this question for a fluent reader?", "How difficult is this reading-comprehension item?",
        "Would a fluent reader find this question easy, moderate or hard?"],
       cum=["the question is easy", "the question is at most moderately hard"]),
]


def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def belebele_es(rng, n):
    f = _files("facebook/belebele", "data/spa_Latn.jsonl")[0]
    rows = [json.loads(l) for l in open(f)]
    # review R12: drop items whose correct answer appears verbatim in the passage (answer read off the text)
    ok = [j for j, r in enumerate(rows) if _norm(r[f"mc_answer{r['correct_answer_num']}"]) not in _norm(r["flores_passage"])]
    idx = rng.sample(ok, n)
    out = []
    for i in idx:
        r = rows[i]
        t = (f"Pasaje: {r['flores_passage'].strip()}\nPregunta: {r['question'].strip()}\n"
             + "\n".join(f"Respuesta {k}: {r[f'mc_answer{k}'].strip()}" for k in range(1, 5)))
        g = {"answer": int(r["correct_answer_num"]) - 1}
        m = re.search(r"//[a-z]+\.(wikinews|wikivoyage|wikibooks|wikipedia)\.org", r["link"])
        if m:
            g["site"] = _BSITE[m.group(1)]
        out.append((t, g, ref_of(f, row=i, link=r["link"], question_number=int(r["question_number"]))))
    return out


# ====================================================================================== linked forecasts (Daily Mail)
FORECAST_VARS = [
    V2("response", ["yes", "no"],
       ["a person or organisation named in the article publicly responds (a statement, interview or press release) within a week",
        "nobody named in the article publicly responds within a week"],
       ["Within a week of publication, will a person or organisation named in the article publicly respond to it?",
        "Will someone named in this story issue a public statement about it within the next seven days?",
        "Will any of the people or organisations in the article comment publicly on it within a week?"]),
    V2("action", ["yes", "no"],
       ["an official decision or formal action connected to this story (a vote, ruling, fine, resignation, formal investigation or policy change) is reported within three months",
        "no official decision or formal action connected to this story is reported within three months"],
       ["Within three months, will an official decision or formal action connected to this story (a vote, ruling, fine, resignation, formal investigation or policy change) be reported?",
        "Will this story lead to a formal decision or official action within the next three months?",
        "Will any authority, court, company board or parliament act formally on this within three months?"]),
    V2("followup", ["within a day", "after 1 to 7 days", "after 1 to 4 weeks", "after 1 to 12 months", "not within a year"],
       ["the next major news story about this event appears within a day of publication",
        "the next major news story about this event appears between one and seven days after publication",
        "the next major news story about this event appears between one and four weeks after publication",
        "the next major news story about this event appears between one and twelve months after publication",
        "no major news story about this event appears within a year of publication"],
       ["When will the next major news story about this same event be published?",
        "How soon will major outlets report on this event again?", "When will this event next make major news?"],
       cum=["a major follow-up story appears within a day", "a major follow-up story appears within a week",
            "a major follow-up story appears within a month", "a major follow-up story appears within a year"]),
]
_DATE = re.compile(r"PUBLISHED:\s*\.\s*(?:\d{1,2}:\d{2} (?:EST|GMT|BST), )?(\d{1,2} [A-Z][a-z]+ \d{4})")
# review R12: no private individuals or minors. Keep stories about public actors, drop crime/court/death/minor stories.
_FC_PUBLIC = re.compile(r"\b(minister|MP|MPs|president|prime minister|government|council|company|firm|chief executive|club|"
                        r"league|party|officials?|spokesman|spokeswoman|university|bank|regulator|parliament|senator|mayor)\b")
_FC_DROP = re.compile(r"(?i)\b(child(ren)?|boys?|girls?|bab(y|ies)|toddlers?|teen(ager)?s?|pupils?|schoolboy|schoolgirl|sons?|"
                      r"daughters?|\d{1,2}-year-old|aged \d{1,2}|police|arrest(ed)?|charged|court|trial|jailed|prison|murder(ed)?|"
                      r"kill(ed|ing)?|rape[ds]?|sex(ual)?|abuse[ds]?|assault(ed)?|victims?|stabbed|shot|accused|convicted|"
                      r"sentenced|inquest|died|death|dead|suicide)\b")


def forecast_news(rng, n):
    """Daily Mail articles (cnn_dailymail train file 1) with a PUBLISHED date, about public actors only."""
    f, df = _cnn("train", 1)
    raw = _pd().read_parquet(f, columns=["article"]).article
    dates = raw.iloc[df._row.to_numpy()].str.extract(_DATE)[0].to_numpy()
    ok = [i for i in range(len(df)) if isinstance(dates[i], str) and _FC_PUBLIC.search(df.lead.iloc[i])
          and not _FC_DROP.search(df.lead.iloc[i])]
    idx = rng.sample(ok, n)
    return [(f"Published: {dates[i]}\n{df.lead.iloc[i]}", {}, ref_of(f, row=int(df._row.iloc[i]), id=str(df.id.iloc[i])))
            for i in idx]


# ============================================================================================================= registry
# name -> (loader, variables, prelude, options)
_FC_PRELUDE = ("A news article (lead paragraphs, with its publication date). Content note: real news about real public "
               "figures and organisations. Answer as a forecaster on the publication date, using only what was knowable "
               "then:")
SOURCES_V2 = {
    "news": (_v1(v1.news), NEWS_VARS, "A news article from CNN or the Daily Mail (lead paragraphs):", {}),
    "gold_news": (_v1(v1.gold_news), GOLD_VARS, v1.SOURCES["gold_news"][2], {}),
    "bgl_logs": (_v1(v1.bgl_logs, strip_label=True), LOG_VARS, v1.SOURCES["bgl_logs"][2], {}),
    "symptoms": (_v1(v1.symptoms, dx=_DX2), SYM_VARS, v1.SOURCES["symptoms"][2], {}),
    "tos": (_v1(v1.tos), TOS_VARS, v1.SOURCES["tos"][2], {}),
    "climate": (_v1(v1.climate), CLIM_VARS, v1.SOURCES["climate"][2], {}),
    "support_chat": (support_chat, SUPPORT_VARS, "The first message a customer sends to an airline or online-shop support agent (simulated dialogue):", {}),
    "ledgar": (ledgar, LEDGAR_VARS, "A provision from a commercial contract filed with the US SEC:", {}),
    "med_ru": (med_ru, MEDRU_VARS, "A patient's question to an online medical service (in Russian):", {}),
    "sci_claims": (sci_claims, SCI_VARS, "The main claim of a university press release and the conclusion of the paper it reports on:", {}),
    "code_defects": (code_defects, CODE_VARS, "A C function from an open-source project:", {}),
    "swe_issues": (swe_issues, SWE_VARS, "An issue from the GitHub tracker of a popular Python library (project names masked as [project]):", {}),
    "thunderbird_logs": (thunderbird_logs, TB_VARS, "Six consecutive lines from the system log of the Thunderbird supercomputer:", {}),
    "hdfs_logs": (hdfs_logs, HDFS_VARS, "The log lines of one data block in a Hadoop (HDFS) cluster (numbers and addresses masked):", {}),
    "ssh_logs": (ssh_logs, SSH_VARS, "Eight consecutive lines from the sshd log of an internet-facing server (addresses, host and user names masked):", {}),
    "finance_news": (finance_news, FIN_VARS, "A business and economy news story (lead paragraphs):", {}),
    "sports": (sports, SPORT_VARS, "A sports news story (lead paragraphs):", {}),
    "stackexchange": (_fw_source("stackexchange"), SE_VARS, "A question-and-answer page from a Stack Exchange community:", {}),
    "reviews": (_fw_source("reviews"), REVIEW_VARS, "A book review from a blog (the reviewer's star rating removed):", {}),
    "math_problems": (math_problems, MATH_VARS, "A problem from a high-school mathematics competition:", {}),
    "mmlu_pro": (mmlu_pro, MMLU_VARS, "A multiple-choice exam question with ten options:", {}),
    "student_answers": (student_answers, STUDENT_VARS, "A school science question, the teacher's reference answer and a student's answer:", {}),
    "eurlex_fr": (eurlex_fr, EURLEX_VARS, "The beginning of an EU legal act (in French; the words naming the type of act are masked as [ACTE]):", {}),
    "belebele_es": (belebele_es, BELE_VARS, "A reading-comprehension item in Spanish (passage, question and four answers):", {}),
    "forecast_news": (forecast_news, FORECAST_VARS, _FC_PRELUDE, {"templates": T_FORECAST, "forecast": True}),
}
