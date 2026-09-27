"""Shared helpers for the `mechanics` pack (BookieBench v2).

Every family module exposes
    NAME: str
    make_world(rng, level: int, scale: int = 1) -> bookiebench.sims.world.World
    simulate(world, rng) -> (assignment tuple over the declared variables, list of observations)
`simulate` re-runs the *physical* story (shuffle a deck, place mines, run an epidemic, ...) from `world.params`
without using the enumerated likelihoods; tests use it for an independent Monte Carlo check of the exact joints.

Levels: 0 = small config, 2 variables, 1-2 evidence steps; 1 = moderate; 2 = full mechanics (3-4 variables, longer
evidence, latent nuisance causes). `scale` > 1 enlarges configurations (joints may then exceed 256 cells).
"""
from __future__ import annotations

import zlib

import numpy as np

from bookiebench.sims.common import Fmt, a_an, cap, join_list, json_block, json_line, people, person, pick, sample  # noqa: F401
from bookiebench.sims.lexicon import pseudo  # noqa: F401
from bookiebench.sims.world import Var, World  # noqa: F401

PACK = "mechanics"
SPLIT_CODE = {"train": 21, "dev": 22, "test": 23, "heldout": 24}
LEVEL_P = (0.25, 0.35, 0.40)


def instance_rng(family: str, split: str, idx: int, seed: int = 0, scale: int = 1):
    return np.random.default_rng([int(seed), zlib.crc32(("mechanics/" + family).encode()), SPLIT_CODE[split], int(idx),
                                  int(scale)])


def draw_level(rng) -> int:
    return int(rng.choice(3, p=LEVEL_P))


def n_steps(rng, level: int, lo2: int = 3, hi2: int = 6) -> int:
    """Evidence length by level: L0 1-2, L1 2-4, L2 lo2..hi2."""
    if level == 0:
        return int(rng.integers(1, 3))
    if level == 1:
        return int(rng.integers(2, 5))
    return int(rng.integers(lo2, hi2 + 1))


def is_json(rng, p: float = 0.3) -> bool:
    return bool(rng.random() < p)


def per_step(rng, T: int, tpls: list):
    """One template per evidence position (fixed upfront so rendering alternatives is deterministic)."""
    return [pick(rng, tpls) for _ in range(T)]


def num_word(n: int) -> str:
    w = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"]
    return w[n] if 0 <= n < len(w) else str(n)


def pct_str(fmt: Fmt, x) -> str:
    """Format a percentage that may be non-integer (e.g. 0.5%)."""
    if float(x) == int(x):
        return fmt.p(int(x))
    if fmt.style == "decimal":
        return f"{float(x) / 100:g}"
    if fmt.style == "percent":
        return f"{float(x):g} percent"
    return f"{float(x):g}%"


def round_pcts(w, total=100, lo=1):
    """Round positive weights to integers >= lo summing to `total` (largest remainder)."""
    w = np.asarray(w, float)
    w = w / w.sum()
    n = len(w)
    rest = total - lo * n
    raw = w * rest
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: rest - base.sum()]:
        base[i] += 1
    return [int(b) + lo for b in base]


def ints_row(rng, n, total=100, lo=1, conc=1.0):
    w = rng.dirichlet(np.full(n, conc))
    return round_pcts(w, total, lo)


def distinct_names(rng, k):
    out = []
    while len(out) < k:
        nm = pseudo(rng)
        if nm not in out and len(nm) >= 4:
            out.append(nm)
    return out


def mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, mart_var, exchangeable, style, theme,
             params, **meta):
    prior = np.asarray(prior, float)
    proj = np.asarray(proj, int)
    keep = prior > 0
    if not keep.all():
        # drop impossible latent rows; the likelihood callbacks index rows through params["rows"]
        raise AssertionError("mk_world expects strictly positive prior rows; filter before calling")
    m = {"style": style, "theme": theme}
    m.update(meta)
    return World(variables=variables, prior=prior / prior.sum(), proj=proj, T=T, alternatives=alternatives, lik=lik,
                 render=render, prelude=prelude, mart_var=mart_var, exchangeable=exchangeable, meta=m, params=params)


def art(phrase: str) -> str:
    """'a'/'an' + phrase, by sound (handles 'hour', 'unit', 'one', digits 8/11/18)."""
    w = phrase.strip()
    lw = w.lower()
    an = (lw[:1] in "aeiou" and not lw.startswith(("one", "uni", "use", "eu", "once"))) or \
        lw.startswith(("hour", "honest", "8", "11 ", "18 ", "11-", "18-"))
    return ("an " if an else "a ") + w


def repeat_tags(keys):
    """For evidence items that repeat the same noisy test, a distinct label per occurrence ("A", "B", ...);
    '' for keys occurring once. Labels are names, not order words, so texts stay natural under perms."""
    from collections import Counter
    tot, seen, out = Counter(keys), Counter(), []
    for k in keys:
        out.append("" if tot[k] == 1 else "ABCDEFGH"[seen[k]])
        seen[k] += 1
    return out


def tag_text(text: str, label: str) -> str:
    """Attach a repeat label ("run A") to an evidence text (prose: before the final period; JSON: extra key)."""
    if not label:
        return text
    if text.startswith("{"):
        import json as _json
        d = _json.loads(text)
        d["label"] = label
        return _json.dumps(d, ensure_ascii=False)
    return f"{text[:-1]} ({label}){text[-1]}" if text[-1] in ".!?" else f"{text} ({label})"
