"""Real-world states for the coherence benchmark (realcoh), from public datasets in the local HF cache.

Every source yields (state text, gold dict) pairs and declares its typed variables. None of these six datasets is in
decider-public's released mixture (decider-public/decider/data/{mixture,tasks*,teacher*}.py), BUT gold_news, bgl_logs,
symptoms, tos and climate ARE in the private mixture_v2 (decider/broad_corpus, merged by broad_corpus/merge/tasks_v2.py),
and many of their states occur verbatim in its training half (review/fairness/REPORT.md, R2). Report decider results on
them as contaminated; see bookiebench.realcoh.provenance for per-source flags.

    news      abisee/cnn_dailymail (test)                  article lead           no gold
    gold_news ChanceFocus/flare-headlines (test)           gold-market headline   gold: price, time, comparison
    bgl_logs  logfit-project/BGL                           6 consecutive lines    gold: failure, subsystem
    symptoms  gretelai/symptom_to_diagnosis (train+test)   patient description    gold: body system, infection
    tos       coastalcph/lex_glue unfair_tos (all splits)  Terms-of-Service clause gold: fairness (+ type if unfair)
    climate   tdiggelm/climate_fever (test)                claim + evidence       gold: verdict

A variable is {"name", "options", "clauses", "questions"}:
    options   short answer texts shown as Choice options
    clauses   one declarative clause per option, used inside yes/no questions ("the gold price is going up")
    questions >= 2 phrasings of the marginal question (the first is the base query, the rest are paraphrases)
"""
import glob
import os
import random
import re

HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


# HF snapshot revisions every realcoh build reads (the ones data/realcoh, data/v2/realcoh and the release were built
# from). Pinned so that `rebuild` gives outsiders byte-identical text; fetch a missing one with
#   huggingface-cli download <repo> --repo-type dataset --revision <rev>
PINNED = {
    "AlucardV/medical-specialty-classification": "48dc0101e5533ee9aa6e079f60494d0d5a9f1422",
    "ChanceFocus/flare-headlines": "d8a188f96866372227e1ef9f6fd6230040d3fe75",
    "DigitalLearningGmbH/MATH-lighteval": "0530c78699ea5e8eb5530600900e1f328b48acad",
    "HuggingFaceFW/fineweb-edu": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
    "Salesforce/APIGen-MT-5k": "abc4a517d67c541f85f6470cbd8fd3186b36830e",
    "TIGER-Lab/MMLU-Pro": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "abisee/cnn_dailymail": "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
    "bolu61/loghub_2": "4a98d3eb30522891b340609d17fa34709a1d44d2",
    "coastalcph/lex_glue": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
    "coastalcph/multi_eurlex": "6278c7994cd86297e823becad7fdf897faaea174",
    "copenlu/scientific-exaggeration-detection": "544352335e47b53a991ed4157a396453fdb25d49",
    "facebook/belebele": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2",
    "google/code_x_glue_cc_defect_detection": "69bd48c03223c2104342acd9a807caf61ac3efb8",
    "gretelai/symptom_to_diagnosis": "722cfb0e11f8ae37339c7f573b5e10429b94df49",
    "logfit-project/BGL": "09a26419fc644a2f96e96d18080397f0e212c8a6",
    "logfit-project/Thunderbird": "6b4a6401591e85a534c401a49355553631133bd0",
    "logfit-project/hdfsv1-grouped-labeled": "ef9f7a1d3ebb763dbbd7c264d703df008b114443",
    "nkazi/SciEntsBank": "abaadf77345c5d68b73b630131a8ae164a45f3ab",
    "princeton-nlp/SWE-bench_Verified": "c104f840cc67f8b6eec6f759ebc8b2693d585d4a",
    "tdiggelm/climate_fever": "ae61ccb9320a78109a246414139ff3a2bd677b8b",
}


def snapshot_dir(repo):
    rev = PINNED.get(repo, "*")
    return os.path.join(HUB, "datasets--" + repo.replace("/", "--"), "snapshots", rev)


def _files(repo, pattern):
    fs = sorted(glob.glob(os.path.join(snapshot_dir(repo), pattern)))
    if not fs:
        rev = PINNED.get(repo)
        hint = f" (pinned revision {rev}: huggingface-cli download {repo} --repo-type dataset --revision {rev})" if rev else ""
        raise FileNotFoundError(f"{repo}: {pattern} not in the HF cache{hint}")
    return fs


def _pd():
    import pandas as pd
    return pd


def _lead(text, max_chars=900):
    """The first sentences of an article, cut at a sentence end, without the '(CNN)' dateline."""
    text = re.sub(r"^.{0,80}?\(CNN\)\s*(--\s*)?", "", text.strip())
    text = re.sub(r"^By \. .*? \. (PUBLISHED: .*? \. \| \. UPDATED: .*? \. )?", "", text)
    out = ""
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(out) + len(s) > max_chars and out:
            break
        out += (" " if out else "") + s
    return out


def ref_of(path, **row):
    """Provenance of a state: dataset repo, snapshot revision, file inside the snapshot, and the row id(s)."""
    parts = path.split(os.sep)
    k = parts.index("snapshots")
    repo = parts[k - 1][len("datasets--"):].replace("--", "/")
    return dict(dataset=repo, revision=parts[k + 1], file="/".join(parts[k + 2:]), **row)


def V(name, options, clauses, questions):
    assert len(options) == len(clauses)
    return dict(name=name, options=list(options), clauses=list(clauses), questions=list(questions))


# ---------------------------------------------------------------------------------------------------------------- news
NEWS_VARS = [
    V("topic",
      ["politics and government", "crime and justice", "business and the economy", "sports",
       "entertainment and celebrities", "science, health or the environment", "war, terrorism or international conflict"],
      ["the article is mainly about politics and government", "the article is mainly about crime and justice",
       "the article is mainly about business and the economy", "the article is mainly about sports",
       "the article is mainly about entertainment or celebrities", "the article is mainly about science, health or the environment",
       "the article is mainly about war, terrorism or international conflict"],
      ["What is this news article mainly about?", "Which section of a newspaper would this story most likely run in?",
       "How would you classify the main subject of this article?"]),
    V("tone", ["mostly negative", "neutral", "mostly positive"],
      ["the overall tone of the article is negative", "the article is written in a neutral tone",
       "the overall tone of the article is positive"],
      ["What is the overall tone of this article?", "Is the news in this article good, bad or neither?"]),
    V("place", ["the United States", "the United Kingdom", "somewhere else"],
      ["the story takes place mainly in the United States", "the story takes place mainly in the United Kingdom",
       "the story takes place mainly outside the US and the UK"],
      ["Where does this story mainly take place?", "In which part of the world is this news story set?"]),
]


def news(rng, n, ref=False):
    pd = _pd()
    f = _files("abisee/cnn_dailymail", "3.0.0/test-*.parquet")[0]
    df = pd.read_parquet(f)
    idx = rng.sample(range(len(df)), n * 2)
    out = []
    for i in idx:
        t = _lead(df.article.iloc[i])
        if 200 <= len(t) <= 1000:
            out.append((t, {}, ref_of(f, row=int(i), id=str(df.id.iloc[i]))) if ref else (t, {}))
        if len(out) == n:
            break
    return out


# ------------------------------------------------------------------------------------------------------- gold headlines
GOLD_VARS = [
    V("price",
      ["the gold price is rising", "the gold price is falling", "the gold price is stable",
       "the headline does not mention the gold price"],
      ["the headline says the gold price is going up", "the headline says the gold price is going down",
       "the headline says the gold price is holding steady", "the headline says nothing about the price of gold"],
      ["What does this headline say about the price of gold?", "According to the headline, which way is gold moving?"]),
    V("time", ["past or present events", "the future"],
      ["the headline is about something that has already happened", "the headline is about what may happen in the future"],
      ["Is this headline about the past and present, or about the future?", "Does the headline report what happened or look ahead?"]),
    V("comparison", ["no", "yes"],
      ["the headline does not compare gold with any other asset", "the headline compares gold with another asset"],
      ["Does the headline compare gold with another asset, such as stocks, the dollar or silver?",
       "Is gold being set against some other investment in this headline?"]),
]


def gold_news(rng, n, ref=False):
    pd = _pd()
    f = _files("ChanceFocus/flare-headlines", "data/test-*.parquet")[0]
    df = pd.read_parquet(f)
    df["text"] = df["query"].str.extract(r"Text:\s*(.*?)\s*Answer:\s*$", flags=re.S)[0].str.strip()
    rows_of = df.reset_index().groupby("text")["index"].apply(lambda x: sorted(int(i) for i in x)).to_dict()
    lab = df.pivot_table(index="text", columns="label_type", values="answer", aggfunc="first")
    keys = sorted(lab.index)
    rng.shuffle(keys)
    out = []
    for t in keys:
        r = lab.loc[t]
        if r.isna().any() or not 25 <= len(t) <= 300:
            continue
        yes = lambda k: r[k] == "Yes"
        dirs = [yes("Direction Up"), yes("Direction Down"), yes("Direction Constant")]
        if not yes("Price or Not") and not any(dirs):
            price = 3
        elif sum(dirs) == 1:
            price = dirs.index(True)
        else:
            price = None
        g = {"time": 1 if (yes("FutureNews") or yes("FuturePrice")) else 0, "comparison": int(yes("Asset Comparision"))}
        if price is not None:
            g["price"] = price
        out.append((t, g, ref_of(f, rows=rows_of[t])) if ref else (t, g))
        if len(out) == n:
            break
    return out


# ----------------------------------------------------------------------------------------------------------- BGL logs
LOG_VARS = [
    V("failure", ["no, these are routine messages", "yes, something failed"],
      ["these log lines are only routine messages", "these log lines report a real failure"],
      ["Do these log lines show that something actually failed?", "Is there a genuine fault in this log excerpt, or is it routine noise?"]),
    V("subsystem",
      ["the compute-node kernel", "an application or job (e.g. ciod)", "hardware, link cards or node discovery",
       "the control system (MMCS, CMCS, BGL master)"],
      ["the messages come from the compute-node kernel", "the messages come from an application or job",
       "the messages come from hardware, link cards or node discovery", "the messages come from the control system"],
      ["Which part of the supercomputer produced most of these messages?", "Where do these log messages mostly originate?"]),
    V("action", ["no action needed", "look into it within the day", "page the on-call engineer now"],
      ["no action is needed", "someone should look into this within the day", "the on-call engineer should be paged right away"],
      ["What should the operations team do about this log excerpt?", "How urgently does this excerpt need a human?"]),
]
_SUB = {"KERNEL": 0, "APP": 1, "HARDWARE": 2, "LINKCARD": 2, "DISCOVERY": 2, "MMCS": 3, "CMCS": 3, "BGLMASTER": 3}


def bgl_logs(rng, n, width=6, ref=False, strip_label=True):
    """strip_label: drop the leading LogHub alert-label column ("-" or the alert category), which leaks `failure`
    (review R1). Stripped by default since 2026-09-27 (v1 and v2); False reproduces the original leaky v1 file."""
    pd = _pd()
    cols = ["label", "timestamp", "date", "node", "time", "node_repeat", "type", "component", "level", "content", "anomaly"]
    f = _files("logfit-project/BGL", "data/train-00000-*.parquet")[0]
    df = pd.read_parquet(f, columns=cols)
    import numpy as np
    an = df.anomaly.to_numpy(); comp = df.component.to_numpy()
    anom_idx = np.flatnonzero(an).tolist()
    rare = np.flatnonzero(comp != "KERNEL").tolist()           # non-kernel windows are rare: oversample them
    out, seen = [], set()
    want = {"anom": n // 2, "norm": n - n // 2}
    tries = 0
    while sum(want.values()) and tries < 100000:
        tries += 1
        u = rng.random()
        if u < 0.25:
            s = rng.choice(rare)
        elif u < 0.6 and want["anom"]:
            s = rng.choice(anom_idx)
        else:
            s = rng.randrange(len(df) - width)
        s = max(0, min(s - rng.randrange(width), len(df) - width))
        if s in seen:
            continue
        w = df.iloc[s:s + width]
        k = "anom" if w.anomaly.any() else "norm"
        if not want[k]:
            continue
        seen.add(s); want[k] -= 1
        shown = cols[1:-1] if strip_label else cols[:-1]
        lines = [" ".join(str(x) for x in row) for row in w[shown].itertuples(index=False)]
        subs = [_SUB.get(c) for c in w.component]
        g = {"failure": int(w.anomaly.any())}
        top = max(set(subs), key=subs.count)
        if top is not None and subs.count(top) > width / 2:
            g["subsystem"] = top
        out.append(("\n".join(lines), g, ref_of(f, row=int(s), n_rows=width)) if ref else ("\n".join(lines), g))
    rng.shuffle(out)
    return out


# ------------------------------------------------------------------------------------------------------------ symptoms
SYM_VARS = [
    V("system",
      ["the digestive system or liver", "the lungs and airways", "the skin", "the joints, bones or muscles",
       "the urinary tract", "the heart and blood vessels", "the whole body or the nervous system"],
      ["the problem is in the digestive system or the liver", "the problem is in the lungs or airways",
       "the problem is in the skin", "the problem is in the joints, bones or muscles", "the problem is in the urinary tract",
       "the problem is in the heart or blood vessels", "the problem affects the whole body or the nervous system"],
      ["Which part of the body is most likely affected?", "Which organ system does this complaint point to?"]),
    V("infection", ["no", "yes"],
      ["the illness is not caused by an infection", "the illness is caused by an infection"],
      ["Is this most likely caused by an infection?", "Would you expect a virus, bacterium, fungus or parasite to be behind this?"]),
    V("care", ["self-care at home", "see a doctor within a few days", "seek emergency care now"],
      ["the patient can manage this with self-care at home", "the patient should see a doctor within a few days",
       "the patient should seek emergency care right now"],
      ["What level of care does this person need?", "How soon should this patient get medical help?"]),
]
_DX = {  # diagnosis -> (system, infectious)
    "peptic ulcer disease": (0, 0), "gastroesophageal reflux disease": (0, 0), "jaundice": (0, 0), "typhoid": (0, 1),
    "bronchial asthma": (1, 0), "pneumonia": (1, 1), "common cold": (1, 1), "impetigo": (2, 1), "psoriasis": (2, 0),
    "fungal infection": (2, 1), "chicken pox": (2, 1), "drug reaction": (2, 0), "allergy": (2, 0), "arthritis": (3, 0),
    "cervical spondylosis": (3, 0), "urinary tract infection": (4, 1), "varicose veins": (5, 0), "hypertension": (5, 0),
    "migraine": (6, 0), "diabetes": (6, 0), "dengue": (6, 1), "malaria": (6, 1),
}


def symptoms(rng, n, ref=False, dx=None):
    """dx: optional diagnosis -> (system or None, infectious or None) override (realcoh v2, review R11)."""
    pd = _pd()
    dx = dx or _DX
    fs = _files("gretelai/symptom_to_diagnosis", "*.jsonl")
    df = pd.concat([pd.read_json(f, lines=True).assign(_f=f, _row=lambda d: range(len(d))) for f in fs], ignore_index=True)
    df = df.drop_duplicates("input_text").reset_index(drop=True)
    idx = rng.sample(range(len(df)), n)
    out = []
    for i in idx:
        s, inf = dx[df.output_text.iloc[i]]
        g = {k: v for k, v in (("system", s), ("infection", inf)) if v is not None}
        t = df.input_text.iloc[i].strip()
        out.append((t, g, ref_of(df._f.iloc[i], row=int(df._row.iloc[i]))) if ref else (t, g))
    return out


# ----------------------------------------------------------------------------------------------------------------- ToS
TOS_VARS = [
    V("fairness", ["fair to the user", "potentially unfair to the user"],
      ["the clause is fair to the user", "the clause is potentially unfair to the user"],
      ["Is this Terms-of-Service clause fair to the user?", "Would a consumer-protection lawyer flag this clause as unfair?"]),
    V("clause_type",
      ["liability and damages", "termination or suspension of the account", "changes to the terms or the service",
       "removal of user content", "governing law, jurisdiction or arbitration", "agreeing by using the service",
       "something else"],
      ["the clause limits the company's liability or damages", "the clause is about terminating or suspending accounts",
       "the clause lets the company change the terms or the service", "the clause is about removing user content",
       "the clause sets the governing law, the courts or arbitration", "the clause says that using the service means accepting the terms",
       "the clause is about something else"],
      ["What is this clause mainly about?", "Which kind of provision is this?"]),
    V("benefits", ["mainly the company", "both sides about equally", "mainly the user"],
      ["the clause mainly protects the company", "the clause protects both sides about equally", "the clause mainly protects the user"],
      ["Whose interests does this clause mainly protect?", "Who does this clause favour?"]),
]
_TOS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 4, 6: 4, 7: 4}


def tos(rng, n, ref=False):
    pd = _pd()
    fs = _files("coastalcph/lex_glue", "unfair_tos/*.parquet")
    df = pd.concat([pd.read_parquet(f).assign(_f=f, _row=lambda d: range(len(d))) for f in fs], ignore_index=True)
    df = df[df.text.str.len().between(120, 900)].drop_duplicates("text").reset_index(drop=True)
    unfair = [i for i in range(len(df)) if len(df.labels.iloc[i])]
    fair = [i for i in range(len(df)) if not len(df.labels.iloc[i])]
    idx = rng.sample(unfair, min(len(unfair), n // 2)); idx += rng.sample(fair, n - len(idx)); rng.shuffle(idx)
    out = []
    for i in idx:
        labs = list(df.labels.iloc[i])
        g = {"fairness": int(bool(labs))}
        types = {_TOS[int(x)] for x in labs}
        if len(types) == 1:
            g["clause_type"] = types.pop()
        t = re.sub(r"\s+([,.;:)])", r"\1", df.text.iloc[i]).replace("( ", "(").strip()
        out.append((t, g, ref_of(df._f.iloc[i], row=int(df._row.iloc[i]))) if ref else (t, g))
    return out


# ------------------------------------------------------------------------------------------------------------- climate
CLIM_VARS = [
    V("verdict", ["supported by the evidence", "refuted by the evidence", "not enough information to decide"],
      ["the evidence supports the claim", "the evidence contradicts the claim", "the evidence is not enough to judge the claim"],
      ["Given the evidence, what is the status of the claim?", "Does the evidence back the claim, contradict it, or leave it open?"]),
    V("topic",
      ["temperatures and warming", "ice, glaciers and sea level", "carbon dioxide and emissions",
       "extreme weather and natural disasters", "climate science and models in general", "policy, economics or energy"],
      ["the claim is about temperatures and warming", "the claim is about ice, glaciers or sea level",
       "the claim is about carbon dioxide and emissions", "the claim is about extreme weather or natural disasters",
       "the claim is about climate science or climate models in general", "the claim is about policy, economics or energy"],
      ["What is the claim mainly about?", "Which climate topic does this claim concern?"]),
    V("skeptic", ["no", "yes"],
      ["the claim does not downplay climate change", "the claim downplays or denies climate change"],
      ["Does the claim downplay or deny climate change?", "Is this the kind of claim a climate skeptic would make?"]),
]


def climate(rng, n, ref=False):
    pd = _pd()
    f = _files("tdiggelm/climate_fever", "data/test-*.parquet")[0]
    df = pd.read_parquet(f).assign(_row=lambda d: range(len(d)))
    df = df[df.claim_label.isin([0, 1, 2])].reset_index(drop=True)
    idx = rng.sample(range(len(df)), n)
    out = []
    for i in idx:
        r = df.iloc[i]
        ev = [e["evidence"].strip().strip('"') for e in r.evidences][:3]
        t = f"Claim: {r.claim.strip()}\nEvidence:\n" + "\n".join(f"- {e}" for e in ev)
        g = {"verdict": int(r.claim_label)}
        out.append((t, g, ref_of(f, row=int(r._row), claim_id=str(r.claim_id))) if ref else (t, g))
    return out


SOURCES = {
    "news": (news, NEWS_VARS, "A news article from CNN (lead paragraphs):"),
    "gold_news": (gold_news, GOLD_VARS, "A news headline about the gold commodity market:"),
    "bgl_logs": (bgl_logs, LOG_VARS, "Six consecutive lines from the system log of the BlueGene/L supercomputer:"),
    "symptoms": (symptoms, SYM_VARS, "A patient describes their symptoms:"),
    "tos": (tos, TOS_VARS, "A clause from an online service's Terms of Service:"),
    "climate": (climate, CLIM_VARS, "A claim about climate, with evidence sentences retrieved from Wikipedia:"),
}
