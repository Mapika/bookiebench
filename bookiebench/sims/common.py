"""Shared text/number helpers for the world families (surface randomization)."""
from __future__ import annotations

import json

import numpy as np

FIRST_NAMES = [
    "Maya", "Liam", "Aisha", "Tomas", "Priya", "Jonah", "Elena", "Kwame", "Sofia", "Hiro", "Nadia", "Oscar",
    "Leila", "Mateo", "Ingrid", "Ravi", "Chloe", "Dmitri", "Amara", "Felix", "Yuki", "Bruno", "Zara", "Tariq",
    "Greta", "Omar", "Lucia", "Soren", "Imani", "Pavel", "Noor", "Emeka", "Hana", "Viktor", "Rosa", "Kofi",
    "Anika", "Diego", "Freya", "Malik", "Ines", "Arjun", "Mira", "Stefan", "Talia", "Wen", "Beatriz", "Idris",
    "Lena", "Marco", "Esther", "Kenji", "Olga", "Samir", "Clara", "Ade", "Petra", "Jin", "Farah", "Hugo",
]
SURNAMES = [
    "Okafor", "Lindqvist", "Moreau", "Tanaka", "Castillo", "Novak", "Haddad", "Brennan", "Kowalski", "Mensah",
    "Varga", "Ruiz", "Petrov", "Ashford", "Nakamura", "Osei", "Fischer", "Delgado", "Iyer", "Horvath",
    "Quinn", "Sato", "Abara", "Kerr", "Lund", "Mbeki", "Rossi", "Silva", "Takacs", "Weber",
]
TOWNS = [
    "Millbrook", "Easton Vale", "Harrowgate", "Pinecrest", "Saltmarsh", "Redfield", "Oakhaven", "Brightwater",
    "Coldspring", "Larkhill", "Stonebridge", "Westmere", "Fernhollow", "Ashby Cross", "Kestrel Bay",
]
COMPANIES = [
    "Norvik", "Altamira", "Pellucid", "Brightline", "Quorra", "Halvard", "Zentek", "Marlow & Finch", "Octavo",
    "Sableworks", "Kestrel", "Vantor", "Luma", "Corvid", "Ironleaf", "Tessellate",
]
COLORS = ["red", "blue", "green", "yellow", "white", "black", "orange", "purple"]
LETTERS = "ABCDEFGH"


def pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def sample(rng, seq, k):
    idx = rng.choice(len(seq), size=k, replace=False)
    return [seq[int(i)] for i in idx]


def person(rng):
    return pick(rng, FIRST_NAMES)


def people(rng, k):
    return sample(rng, FIRST_NAMES, k)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def join_list(items, conj="and"):
    items = list(items)
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} " + items[-1]


def composition(rng, n, total, min_each=0):
    """Random integer composition of `total` into n parts, each >= min_each."""
    rest = total - n * min_each
    assert rest >= 0
    w = rng.dirichlet(np.ones(n))
    parts = np.floor(w * rest).astype(int)
    for i in rng.choice(n, size=rest - parts.sum(), replace=True):
        parts[i] += 1
    return [int(p) + min_each for p in parts]


def pct_row(rng, n, lo=1, step=1, total=100, conc=1.0):
    """Random probability row (in integer percent, multiples of `step`, each >= lo) summing to `total`."""
    units = total // step
    lo_u = max(1, -(-lo // step)) if lo > 0 else 0
    rest = units - n * lo_u
    w = rng.dirichlet(np.full(n, conc))
    parts = np.floor(w * rest).astype(int)
    for i in rng.choice(n, size=rest - parts.sum(), replace=True):
        parts[i] += 1
    return [int((p + lo_u) * step) for p in parts]


def rand_pct(rng, lo, hi, step=5):
    vals = list(range(lo, hi + 1, step))
    return pick(rng, vals)


class Fmt:
    """Per-instance number formatting style for probabilities given in integer percent."""

    STYLES = ["pct", "percent", "decimal", "pct"]

    def __init__(self, rng):
        self.style = pick(rng, self.STYLES)

    def p(self, pct: int) -> str:
        if self.style == "pct":
            return f"{pct}%"
        if self.style == "percent":
            return f"{pct} percent"
        if pct == 100:
            return "1"
        if pct == 0:
            return "0"
        return f"{pct / 100:g}"


def json_block(rng, obj, label_choices=("state", "world", "setup", "context")):
    lab = pick(rng, list(label_choices))
    indent = pick(rng, [None, None, 2])
    body = json.dumps(obj, indent=indent, ensure_ascii=False)
    form = pick(rng, ["{lab}: {body}", "{lab} = {body}", "{body}"])
    return form.format(lab=lab.upper() if rng.random() < 0.5 else lab, body=body)


def json_line(obj):
    return json.dumps(obj, ensure_ascii=False)


def prob01(pct: int) -> float:
    return pct / 100.0


def a_an(phrase: str) -> str:
    w = phrase.strip()
    first = w.split()[0].lower() if w else ""
    vowel = first[:1] in "aeiou" or first.startswith(("8", "11", "18"))
    return ("an " if vowel else "a ") + w


# ----------------------------------------------------------------------------------------------------------------------
# template versions: tv=1 reproduces the pre-release (internal) data byte for byte; tv=2 is the release default
# (fairness-review fixes F2-F13). Families read `get_tv()` for text-only fixes that never change rng consumption.
# ----------------------------------------------------------------------------------------------------------------------
import contextlib as _contextlib

DEFAULT_TV = 2
_TV = [DEFAULT_TV]


def get_tv() -> int:
    return _TV[-1]


@_contextlib.contextmanager
def template_version(tv):
    """Temporarily set the template version (None = leave unchanged)."""
    if tv is None:
        yield get_tv()
        return
    if int(tv) not in (1, 2):
        raise ValueError(f"unknown template version {tv}")
    _TV.append(int(tv))
    try:
        yield int(tv)
    finally:
        _TV.pop()
