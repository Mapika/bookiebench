"""Nuisance augmentation (family-agnostic post-processing of an instance).

Changes only surface text; the posterior (joints, queries, gold, next_evidence probabilities) is untouched:
  * irrelevant distractor sentences with numbers (some look like probabilities about unrelated things),
  * shuffling the order of self-contained stated-parameter sentences (prose) or top-level keys (JSON),
  * redundant restatements of stated parameters,
  * distractor clauses appended to some evidence steps (the same clause is appended to every next_evidence
    alternative of that step, so it carries no information).
Uses its own rng, so the underlying instance is identical to the un-augmented one.
"""
from __future__ import annotations

import copy
import json
import re

from .common import COMPANIES, FIRST_NAMES, TOWNS, pick

DISTRACT = [
    "{N}'s cousin owns {n} bicycles.",
    "The building's lobby was repainted {n} years ago.",
    "In an unrelated staff survey, {p}% of respondents said they prefer tea to coffee.",
    "The nearest post office is {d} km away.",
    "A bakery in {T} sells about {nn} loaves a day.",
    "The car park has {nn} spaces, {n} of them reserved.",
    "About {p}% of {T} residents own a cat, which has no bearing on anything here.",
    "{N} once scored {nn} points in a board game.",
    "The {T} library holds roughly {nnn} books.",
    "The office thermostat is set to {t} degrees.",
    "On an ordinary weekday there is a {p}% chance that the {T} train is delayed; this is irrelevant here.",
    "{C} reported {p}% growth in its stationery budget last year.",
    "A nearby radio station plays {n} songs an hour.",
    "{N} has visited {T} {n} times.",
    "The local football club has won {n} of its last {nn} friendly matches.",
    "Roughly {p} out of 100 tourists in {T} arrive by ferry.",
    "The printer on the second floor jams about once every {n} days.",
    "{C}'s cafeteria serves soup on {n} days out of 7.",
    "A crossword in yesterday's paper had {nn} clues.",
    "There is a {p} percent chance of drizzle in {T} tomorrow, unrelated to this problem.",
]
EV_DISTRACT = [
    "(Meanwhile, {N} orders {n} coffees.)",
    "Separately, the clock in the hall shows {t} minutes past the hour.",
    "({N} mentions that {p}% of the snacks are gone.)",
    "Unrelatedly, a delivery van with {n} parcels drives past.",
    "Someone notes that {T} has {nn} street lamps.",
]
RESTATE = ["To restate: {s}", "As noted above, {s}", "Recall that {s}", "(Repeating for clarity: {s})"]
_DEPENDENT = re.compile(r"^(It |Its |It's |They |Their |Them |This |These |Otherwise|After a|After an|Nobody|"
                        r"Independently|Each |All |Every |Out of sight|Its result|That )")


_AVOID = {"text": ""}


def _fresh(rng, pool):
    cands = [x for x in pool if x not in _AVOID["text"]] or pool
    return pick(rng, cands)


def _fill(rng, tpl):
    return tpl.format(N=_fresh(rng, FIRST_NAMES), T=_fresh(rng, TOWNS), C=_fresh(rng, COMPANIES),
                      n=int(rng.integers(2, 12)), nn=int(rng.integers(12, 90)), nnn=int(rng.integers(1000, 90000)),
                      p=int(rng.integers(3, 97)), d=round(float(rng.uniform(0.2, 9.5)), 1), t=int(rng.integers(15, 26)))


def _split_units(text):
    """Split prose into order-independent units. Returns list of (unit_text, movable, sep_before)."""
    text = text.replace("Dr. ", "Dr§ ")
    lines = text.split("\n")
    blocks = []  # group bullet lines with their header
    for ln in lines:
        if ln.startswith("- ") and blocks:
            blocks[-1] += "\n" + ln
        else:
            blocks.append(ln)
    units = []
    for b in blocks:
        if "\n- " in b:
            units.append([b, "\n"])
            continue
        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", b)
        for j, s in enumerate(sents):
            if units and _DEPENDENT.match(s):
                units[-1][0] += " " + s
            else:
                units.append([s, " " if j else "\n"])
    out = []
    for i, (u, sep) in enumerate(units):
        u = u.replace("Dr§ ", "Dr. ")
        movable = i > 0 and bool(re.search(r"\d", u)) and not re.search(r"\b(then|that (urn|jar|bag|box|tin|"
                                                                          r"bucket|deck|drum)|them)\b", u)
        out.append((u, movable, sep))
    return out


_LOWER_OK = {"The", "A", "An", "When", "If", "Given", "Each", "In", "At", "On", "Among", "Under", "There", "Overall",
             "About", "Out", "Behind", "Without", "During", "For", "Readings", "Stations"}


def _restate(rng, s):
    w = s.split(" ", 1)[0]
    if w in _LOWER_OK:
        s = s[0].lower() + s[1:]
    return pick(rng, RESTATE).format(s=s)


def _augment_prose(rng, text, cfg):
    units = _split_units(text)
    texts = [u for u, _, _ in units]
    seps = [s for _, _, s in units]
    mov = [i for i, (_, m, _) in enumerate(units) if m]
    if cfg["shuffle"] and len(mov) >= 2:
        perm = [mov[int(i)] for i in rng.permutation(len(mov))]
        new = list(texts)
        for dst, src in zip(mov, perm):
            new[dst] = texts[src]
        texts = new
    multiline = "\n" in text
    pieces = [texts[0]] + [("\n" if ("\n" in t or sep == "\n") else " ") + t for t, sep in zip(texts[1:], seps[1:])]
    extra = [_restate(rng, texts[pick(rng, mov)]) for _ in range(cfg["n_restate"]) if mov]
    extra = [e for e in extra if "\n" not in e]
    for e in extra + [_fill(rng, pick(rng, DISTRACT)) for _ in range(cfg["n_distract"])]:
        pos = int(rng.integers(1, len(pieces) + 1))
        pieces.insert(pos, ("\n" if multiline and rng.random() < 0.5 else " ") + e)
    return "".join(pieces)


def _augment_json(rng, text, cfg):
    i = text.index("{")
    head, body = text[:i], text[i:]
    obj = json.loads(body)
    items = list(obj.items())
    if cfg["shuffle"]:
        items = [items[int(j)] for j in rng.permutation(len(items))]
    for k in range(cfg["n_distract"]):
        pos = int(rng.integers(0, len(items) + 1))
        items.insert(pos, (pick(rng, ["aside", "note", "trivia", "unrelated"]) + f"_{k + 1}",
                           _fill(rng, pick(rng, DISTRACT))))
    d = dict(items)
    if cfg["n_restate"] and len(obj) > 1:
        key = pick(rng, [k for k in obj])
        d[f"repeat of {key}"] = obj[key]
    indent = 2 if "\n" in body else None
    return head + json.dumps(d, indent=indent, ensure_ascii=False)


def augment(instance, rng, strength=None):
    """Return a nuisance-augmented deep copy of `instance`."""
    inst = copy.deepcopy(instance)
    strength = strength if strength is not None else float(rng.uniform(0.3, 1.0))
    cfg = {"shuffle": bool(rng.random() < 0.8), "n_distract": int(rng.integers(1, 2 + int(4 * strength))),
           "n_restate": int(rng.integers(0, 2 + int(strength)))}
    pre = inst["prelude"]
    _AVOID["text"] = pre + " ".join(st["evidence"] for st in inst["steps"]) + " ".join(q["text"] for q in inst["queries"])
    is_json = False
    if "{" in pre:
        try:
            json.loads(pre[pre.index("{"):])
            is_json = True
        except ValueError:
            pass
    inst["prelude"] = _augment_json(rng, pre, cfg) if is_json else _augment_prose(rng, pre, cfg)
    n_ev = 0
    for k, st in enumerate(inst["steps"]):
        if rng.random() < 0.3 * strength + 0.1:
            d = _fill(rng, pick(rng, EV_DISTRACT))
            st["evidence"] = st["evidence"] + " " + d
            if k > 0:
                for a in inst["steps"][k - 1]["next_evidence"]:
                    a["evidence"] = a["evidence"] + " " + d
            n_ev += 1
    inst.setdefault("meta", {})["nuisance"] = {"shuffle": cfg["shuffle"], "distractors": cfg["n_distract"],
                                               "restatements": cfg["n_restate"], "evidence_distractors": n_ev,
                                               "json": is_json}
    return inst
