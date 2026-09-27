"""Shared helpers for the procedural families: exact rational CPT rows and many number/layout renderings."""
from __future__ import annotations

from fractions import Fraction
from math import gcd

import numpy as np

from .common import cap, join_list, pick

DENOMS = [4, 5, 6, 8, 10, 12, 20, 20, 100, 100, 100]
ALPHAS = [0.15, 0.3, 0.6, 1.0, 1.0, 2.0, 4.0, 8.0]


def round_counts(w, D):
    """Largest-remainder rounding of weights w (sum 1) to integer counts summing to D."""
    raw = np.asarray(w, float) * D
    c = np.floor(raw).astype(int)
    rem = D - c.sum()
    order = np.argsort(-(raw - c))
    for i in order[:rem]:
        c[i] += 1
    return [int(x) for x in c]


def rand_row(rng, n, D, alpha):
    return round_counts(rng.dirichlet(np.full(n, alpha)), D)


class Row:
    """A probability row stored as integer counts over a denominator (exact rationals)."""

    def __init__(self, counts, D):
        self.counts = list(counts)
        self.D = D

    @property
    def probs(self):
        return np.array(self.counts, float) / self.D


def _exact_pct(c, D):
    f = Fraction(100 * c, D)
    if f.denominator in (1, 2, 4):
        v = float(f)
        return f"{v:g}"
    return None


class NumStyle:
    """How probabilities are written. One or two styles per instance; each row picks one."""

    ALL = ["pct", "decimal", "frac", "ratio", "odds", "percent_words"]

    def __init__(self, rng):
        k = int(rng.integers(1, 3))
        self.styles = list(rng.choice(self.ALL, size=k, replace=False))
        self.rng = rng

    def usable(self, row: Row):
        out = []
        for s in self.styles:
            if s in ("pct", "decimal", "percent_words") and any(_exact_pct(c, row.D) is None for c in row.counts):
                continue
            if s == "odds" and len(row.counts) != 2:
                continue
            out.append(s)
        return out or ["frac"]

    def num(self, c, D, style):
        if style == "pct":
            return f"{_exact_pct(c, D)}%"
        if style == "percent_words":
            return f"{_exact_pct(c, D)} percent"
        if style == "decimal":
            return f"{c / D:g}" if _exact_pct(c, D) is not None else f"{c}/{D}"
        f = Fraction(c, D)
        if style == "frac":
            if rng_bool(self.rng) or not f.numerator:
                return f"{c} out of {D}"
            return f"{f.numerator} out of {f.denominator}"
        return f"{c}/{D}"

    def row_text(self, row: Row, states, subject, future=False):
        """e.g. 'the X is low with probability 30%, or high with probability 70%'."""
        style = pick(self.rng, self.usable(row))
        verb = "will be" if future else "is"
        if is_whether(subject):
            subject = f"the outcome for {subject}"
        if style == "ratio":
            g = 0
            for c in row.counts:
                g = gcd(g, c)
            parts = [str(c // g) for c in row.counts]
            return (f"{subject} {verb} " + join_list(states, "or") + " in the ratio " + " : ".join(parts))
        if style == "odds":
            a, b = row.counts
            g = gcd(a, b) or 1
            return f"the odds that {subject} {verb} {states[0]} rather than {states[1]} are {a // g} to {b // g}"
        items = [f"{s} with probability {self.num(c, row.D, style)}" for s, c in zip(states, row.counts)]
        if style == "frac":
            items = [f"{s} in {self.num(c, row.D, 'frac')} cases" for s, c in zip(states, row.counts)]
        return f"{subject} {verb} " + join_list(items, "or")

    def cells(self, row: Row, states):
        """Compact 'state: number' cells for tables/JSON."""
        style = pick(self.rng, [s for s in self.usable(row) if s not in ("ratio", "odds")] or ["frac"])
        if style == "frac":
            return {s: f"{c}/{row.D}" for s, c in zip(states, row.counts)}
        return {s: self.num(c, row.D, style) for s, c in zip(states, row.counts)}

    def scalar(self, pct):
        """A single probability given in integer percent."""
        style = pick(self.rng, self.styles)
        if style in ("pct", "ratio", "odds"):
            return f"{pct}%"
        if style == "percent_words":
            return f"{pct} percent"
        if style == "decimal":
            return f"{pct / 100:g}"
        f = Fraction(pct, 100)
        if pct == 0:
            return "0%"
        return f"{f.numerator} in {f.denominator}"


def rng_bool(rng, p=0.5):
    return bool(rng.random() < p)


# ---------------------------------------------------------------------------------------------------------------------
# phrasing of concepts
# ---------------------------------------------------------------------------------------------------------------------

def is_whether(np_: str) -> bool:
    return np_.startswith("whether ")


def statement(np_: str, v: str, future=False) -> str:
    if is_whether(np_):
        return f"the outcome for {np_} {'will be' if future else 'is'} '{v}'"
    return f"{np_} {'will be' if future else 'is'} {v}"


def questions(np_: str, future=False):
    if is_whether(np_):
        return [f"What is the outcome for {np_}?", f"Regarding {np_}, which outcome holds?"]
    if future:
        return [f"What will {np_} be?", f"Which value will {np_} take?"]
    return [f"What is {np_}?", f"Which best describes {np_}?", f"What value does {np_} take?"]


def cap_first(s):
    return cap(s)
