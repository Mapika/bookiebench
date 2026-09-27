"""Bias-trap transforms: base-rate neglect, conjunction fallacy (Linda), anchoring. (Gain/loss framing needs a
rewriter and lives in `llm.py`.)

None of them changes the world. Every added sentence is either number-free and explicitly uninformed ("X has not seen
any of the figures"), or an irrelevant number (anchoring), so the stated model and hence the exact posterior are
unchanged.
"""
from __future__ import annotations

import json
import re

from bookiebench.sims.common import FIRST_NAMES, pick

from .common import Clauses, add_query, is_json_prelude, join_units, map_evidence, start, transform_rng, units

# ----------------------------------------------------------------------------------------------------------------------
# base-rate neglect: vivid, number-free narrative up front, an uninformed confident lure, vivid evidence wording, and
# the base-rate sentence moved to a footnote at the end
# ----------------------------------------------------------------------------------------------------------------------

VIVID_OPEN = [
    "It had been a long, tense morning, and everyone involved remembered afterwards exactly how it felt when the "
    "first report came in: the room went quiet, somebody put down a coffee cup, and all eyes turned to the file.",
    "What follows was written up in a hurry. The people on the spot were rattled, the phones kept ringing, and more "
    "than one of them later said that the details seemed to leap off the page.",
    "Nobody who was there will forget the scene. The lights flickered, a door banged somewhere down the corridor, "
    "and the atmosphere was thick with the feeling that something important was about to be decided.",
    "The case drew a small crowd. People leaned over each other's shoulders to read the notes, voices were raised, "
    "and a sense of drama hung over every new detail as it arrived.",
]
LURE = [
    "{N}, who has not seen any of the figures or observations in this file, is absolutely convinced that {c}. "
    "\"It has all the hallmarks,\" {N} keeps saying.",
    "Before anything was written down, {N} (who has no access to the figures or the observations) declared with total "
    "confidence that {c}, describing it as \"the most obvious case I have ever seen\".",
    "{N}, an onlooker who knows none of the numbers or findings below, has a vivid gut feeling that {c}, and tells "
    "everyone so in dramatic terms.",
]
FOOTNOTE = [
    "Footnote from the statistical appendix: {s}",
    "(Buried in the small print at the end of the file: {s})",
    "Appendix, general background figures: {s}",
    "P.S. from the records office, easy to overlook: {s}",
]
VIVID_EV = [
    "Everyone in the room gasped at this.",
    "It was a striking moment that people would talk about for days.",
    "The report was delivered in a breathless, dramatic voice.",
    "Somebody underlined it twice in red ink.",
    "It made a vivid impression on all who heard it.",
]


BASE_RATE_CUE = re.compile(
    r"\b(overall|of all|in general|base rate|prior|population|usually|typically|normally|background:|"
    r"on any given|chosen with probabilit|picked with probabilit|makes? \d+|comes? from|belong to|"
    r"of (the )?(city's |town's )?(patients|tickets|emails|messages|batches|applicants|candidates|taxis|cars|days|"
    r"nights|cases|residents|visitors)|\(probability \d)|"
    r"\bis [\w\s'-]{1,40} with probability [\d.]+( ?%| percent)?,? (or|[\w\s'-]{1,40} with probability)", re.I)


def _fresh_name(rng, text):
    cands = [n for n in FIRST_NAMES if n not in text]
    return pick(rng, cands or FIRST_NAMES)


def _all_text(inst):
    return inst["prelude"] + " ".join(st["evidence"] for st in inst["steps"]) + " ".join(q["text"] for q in inst["queries"])


def base_rate(inst, seed: int = 0):
    rng = transform_rng(inst, "base_rate", seed)
    out = start(inst, "base_rate")
    info = out["meta"]["stress"]
    cl = Clauses(inst)
    pre = inst["prelude"]
    buried = None
    if is_json_prelude(pre):
        us = units(pre)
        body = json.loads(us[-1]["text"])
        keys = list(body)
        key = next((k for k in keys[1:] if any(w in k.lower() for w in ("base", "prior", "rate", "share", "p(")) and
                    any(ch.isdigit() for ch in json.dumps(body[k]))), None)
        if key is not None:
            val = body.pop(key)
            body[key] = val  # moved to the end of the object
            buried = key
        indent = 2 if "\n" in us[-1]["text"] else None
        us[-1]["text"] = json.dumps(body, indent=indent, ensure_ascii=False)
        core = join_units(us)
    else:
        us = units(pre)
        idx = next((i for i, u in enumerate(us) if (u["movable"] or (i == 0 and len(us) > 2)) and
                    not u["text"].lstrip().startswith("- ") and any(ch.isdigit() for ch in u["text"]) and
                    BASE_RATE_CUE.search(u["text"])), None)
        tail = ""
        if idx is not None:
            buried = us[idx]["text"]
            rest = us[:idx] + us[idx + 1:]
            core = join_units(rest)
            tail = "\n" + pick(rng, FOOTNOTE).format(s=buried)
        else:
            core = pre
        core = core + tail
    name = _fresh_name(rng, _all_text(inst))
    mv = inst.get("mart_var") or inst["variables"][0]["name"]
    nopt = len(next(v for v in inst["variables"] if v["name"] == mv)["options"])
    lure_opt = int(rng.integers(nopt))
    lure = pick(rng, LURE).format(N=name, c=cl.clause(mv, [lure_opt]))
    out["prelude"] = pick(rng, VIVID_OPEN) + " " + lure + "\n" + core
    vivid_steps = []
    choice = {k: (pick(rng, VIVID_EV) if rng.random() < 0.6 else None) for k in range(len(inst["steps"]))}

    def fn(t, k):
        return t + " " + choice[k] if choice[k] else t

    map_evidence(out, fn)
    vivid_steps = [k for k, v in choice.items() if v]
    info.update({"buried": buried, "lure": lure, "lure_var": mv, "lure_option": lure_opt, "vivid_steps": vivid_steps,
                 "native_clauses": cl.native})
    return out


# ----------------------------------------------------------------------------------------------------------------------
# conjunction fallacy (Linda): a representative, uninformed description of one outcome + conjunction/component pairs
# ----------------------------------------------------------------------------------------------------------------------

DESCR = [
    "People who have followed the situation describe it as a textbook picture of a case in which {c}: every detail "
    "they have heard about seems to fit that story perfectly. (Their impression is not based on any of the figures or "
    "observations given here.)",
    "{N}, a long-time observer, has written a colourful profile of the situation. It reads exactly like the classic "
    "story in which {c}; everything in the profile is typical of it. (The profile does not use any of the numbers or "
    "observations reported here.)",
    "Anyone reading the background notes gets a strong sense that this is the kind of case where {c}; the atmosphere, "
    "the people and the setting are all just what one would imagine for it. (These notes contain no information "
    "beyond the stated figures and observations.)",
]
COMP = ["Is it the case that {b}?", "How likely is it that {b}?", "What is the probability that {b}?"]
CONJ = ["Is it the case that {b} and {a}?", "How likely is it that {b} and that {a}?",
        "What is the probability that {b}, and also that {a}?"]


def conjunction(inst, seed: int = 0, n_pairs: int = 3):
    rng = transform_rng(inst, "conjunction", seed)
    out = start(inst, "conjunction")
    cl = Clauses(inst)
    V = inst["variables"]
    ai = int(rng.integers(len(V)))
    A = V[ai]["name"]
    a = int(rng.integers(len(V[ai]["options"])))
    name = _fresh_name(rng, _all_text(inst))
    out["prelude"] = inst["prelude"] + "\n" + pick(rng, DESCR).format(c=cl.clause(A, [a]), N=name)
    others = [i for i in range(len(V)) if i != ai]
    pairs = []
    tried = set()
    for _ in range(30):
        if len(pairs) >= n_pairs:
            break
        bi = int(pick(rng, others))
        b = int(rng.integers(len(V[bi]["options"])))
        if (bi, b) in tried:
            continue
        tried.add((bi, b))
        B = V[bi]["name"]
        cb, ca = cl.clause(B, [b]), cl.clause(A, [a])
        qc = add_query(out, {"kind": "noul", "event": {B: [b]}, "text": pick(rng, COMP).format(b=cb)})
        qj = add_query(out, {"kind": "noul", "event": {A: [a], B: [b]} if ai < bi else {B: [b], A: [a]},
                             "text": pick(rng, CONJ).format(b=cb, a=ca)})
        pairs.append({"conj": qj["id"], "component": qc["id"]})
    out["meta"]["stress"].update({"representative": {"var": A, "option": a}, "pairs": pairs,
                                  "native_clauses": cl.native})
    return out


# ----------------------------------------------------------------------------------------------------------------------
# anchoring: an irrelevant number right before every question
# ----------------------------------------------------------------------------------------------------------------------

ANCHOR = [
    "(A random number generator in the next room just printed {n}.) {q}",
    "Before you answer: a wheel of fortune at a nearby fair has just stopped on {n}. {q}",
    "Someone who has not read any of the above guesses \"about {n}%\". {q}",
    "Unrelated reference number: {n}. {q}",
    "The last four digits of the case file's barcode are {n4}. {q}",
    "A passer-by shouts out the number {n} for no particular reason. {q}",
]


def anchoring(inst, seed: int = 0):
    rng = transform_rng(inst, "anchoring", seed)
    out = start(inst, "anchoring")
    anchors = {}
    for q in out["queries"]:
        high = bool(rng.random() < 0.5)
        n = int(rng.integers(85, 100)) if high else int(rng.integers(1, 16))
        n4 = f"{n:02d}{int(rng.integers(10, 100))}"
        q["text"] = pick(rng, ANCHOR).format(n=n, n4=n4, q=q["text"])
        anchors[q["id"]] = {"value": n, "high": high}
    out["meta"]["stress"]["anchors"] = anchors
    return out
