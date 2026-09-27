"""Generic exact-inference machinery shared by all world families.

A family builds a `World`: an enumerated latent space Z (rows), a prior over Z, a projection Z -> the declared
variables, and a sequence of T observations. Observation k takes a value from `alternatives(k, past)` (which may
depend on the realised past but never on z) with likelihood `lik(o, past)[z]` (a vector over Z; for every z the
likelihoods over the alternatives sum to 1).

Everything else is generic and exact:
  * gold: z ~ prior, then o_k ~ lik(., past)[z] sequentially  (the true generative process)
  * posterior after prefix k: prior * prod_j lik(o_j, o_<j), normalised, projected onto the variables
  * next_evidence at step k: the alternatives for step k+1 with P(o | prefix k) = post_k . lik(o, prefix k)
    (the martingale identity holds by construction)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import re

from .common import a_an, get_tv, pick, template_version

ROUND = 10


@dataclass
class Var:
    name: str
    options: list[str]
    questions: list[str]  # paraphrases of the marginal question
    clause: Any  # "... {opt} ..." template, or a list with one full clause per option
    questions_at: Any = None  # optional k -> [questions] for step-k martingale queries (tense-correct, tv>=2)

    def says(self, idxs) -> str:
        idxs = list(idxs)
        if isinstance(self.clause, list):
            if len(idxs) == 1:
                return self.clause[idxs[0]]
            return "either " + " or ".join(self.clause[i] for i in idxs)
        if len(idxs) == 1:
            return self.clause.format(opt=self.options[idxs[0]])
        if get_tv() >= 2:
            m = re.search(r"\b(a|an) \{opt\}", self.clause)
            if m:  # "show a {opt}" -> "show either a 2 or a 6" (F9)
                alts = "either " + " or ".join(a_an(self.options[i]) for i in idxs)
                return self.clause[:m.start()] + alts + self.clause[m.end():]
        return self.clause.format(opt="either " + " or ".join(self.options[i] for i in idxs))


@dataclass
class World:
    variables: list[Var]
    prior: np.ndarray  # (n,)
    proj: np.ndarray  # (n, n_vars) int
    T: int
    alternatives: Callable[[int, list], list]
    lik: Callable[[Any, list], np.ndarray]
    render: Callable[[int, Any], str]
    prelude: str
    mart_var: str
    exchangeable: bool
    meta: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)  # raw generative parameters (not serialized; used by tests)

    @property
    def shape(self):
        return tuple(len(v.options) for v in self.variables)

    def project(self, post: np.ndarray) -> np.ndarray:
        J = np.zeros(self.shape)
        np.add.at(J, tuple(self.proj.T), post)
        return J

    def posterior(self, obs, prior=None) -> np.ndarray:
        post = (self.prior if prior is None else prior).astype(float).copy()
        for k, o in enumerate(obs):
            post = post * self.lik(o, list(obs[:k]))
            s = post.sum()
            if s <= 0:
                raise ValueError("evidence has probability zero")
            post /= s
        return post

    def sample(self, rng):
        z = int(rng.choice(len(self.prior), p=self.prior / self.prior.sum()))
        obs = []
        for k in range(self.T):
            alts = self.alternatives(k, obs)
            w = np.array([self.lik(a, obs)[z] for a in alts], dtype=float)
            assert abs(w.sum() - 1) < 1e-9, f"likelihoods over alternatives sum to {w.sum()} at step {k}"
            obs.append(alts[int(rng.choice(len(alts), p=w / w.sum()))])
        return z, obs


# ----------------------------------------------------------------------------------------------------------------------
# queries
# ----------------------------------------------------------------------------------------------------------------------

CONJ2 = [
    "Is it the case that {a} and {b}?",
    "Is it true both that {a} and that {b}?",
    "What is the probability that {a} and {b}?",
    "How likely is it that {a} and, at the same time, {b}?",
    "Is it true that {a} and also that {b}?",
]
CONJ3 = [
    "Is it true that {a}, that {b}, and that {c}?",
    "What is the probability that {a}, {b}, and {c}?",
    "How likely is it that all of the following hold: {a}; {b}; {c}?",
]
NEG1 = [
    "Is it false that {a}?",
    "Is it not the case that {a}?",
    "How likely is it that it is NOT true that {a}?",
    "Is it untrue that {a}?",
]
NEG2 = [
    "Is it false that both {a} and {b}?",
    "Is it not the case that {a} and {b}?",
    "How likely is it that it is not true that {a} and {b}?",
]
COND = [
    "If {g}, is it the case that {a}?",
    "Suppose {g}. How likely is it then that {a}?",
    "Assuming {g}, what is the probability that {a}?",
    "Given that {g}, is it true that {a}?",
    "In the case where {g}, how likely is it that {a}?",
]


def _rand_event(rng, world: World, var_ids, allow_multi=True):
    ev = {}
    for vi in var_ids:
        v = world.variables[vi]
        n = len(v.options)
        if allow_multi and n >= 3 and rng.random() < 0.25:
            idx = sorted(int(i) for i in rng.choice(n, size=2, replace=False))
        else:
            idx = [int(rng.integers(n))]
        ev[vi] = idx
    return ev


def _ev_names(world, ev):
    return {world.variables[vi].name: idx for vi, idx in ev.items()}


def make_queries(world: World, J_final: np.ndarray, rng, tv: int | None = None) -> list[dict]:
    """The SPEC query set. tv=1: legacy templates; tv=2 (release): fairness-review fixes."""
    tv = get_tv() if tv is None else int(tv)
    with template_version(tv):
        return (_make_queries_v2 if tv >= 2 else _make_queries_v1)(world, J_final, rng)


def _make_queries_v1(world: World, J_final: np.ndarray, rng) -> list[dict]:
    V = world.variables
    nv = len(V)
    qs: list[dict] = []

    seen = set()

    def key(q):
        return (q["kind"], q.get("neg", False), tuple(sorted((k, tuple(v)) for k, v in q.get("event", {}).items())),
                tuple(sorted((k, tuple(v)) for k, v in q.get("given", {}).items())))

    def add(q):
        if q["kind"] != "marginal":
            kk = key(q)
            if kk in seen:
                return False
            seen.add(kk)
        q["id"] = f"q{len(qs)}"
        qs.append(q)
        return True

    for v in V:
        add({"kind": "marginal", "var": v.name, "text": pick(rng, v.questions), "options": list(v.options)})

    # conjunctions (>= 2)
    n_conj, made = int(rng.integers(2, 4)), 0
    for _ in range(50):
        if made >= n_conj:
            break
        k = 3 if (nv >= 3 and rng.random() < 0.3) else 2
        vids = sorted(int(i) for i in rng.choice(nv, size=k, replace=False))
        ev = _rand_event(rng, world, vids)
        cl = [V[vi].says(ev[vi]) for vi in vids]
        if k == 2:
            text = pick(rng, CONJ2).format(a=cl[0], b=cl[1])
        else:
            text = pick(rng, CONJ3).format(a=cl[0], b=cl[1], c=cl[2])
        made += add({"kind": "noul", "event": _ev_names(world, ev), "text": text})
    assert made >= 2

    # negations (>= 1)
    n_neg, made = int(rng.integers(1, 3)), 0
    for _ in range(50):
        if made >= n_neg:
            break
        if rng.random() < 0.6:
            vi = int(rng.integers(nv))
            ev = _rand_event(rng, world, [vi])
            text = pick(rng, NEG1).format(a=V[vi].says(ev[vi]))
        else:
            vids = sorted(int(i) for i in rng.choice(nv, size=2, replace=False))
            ev = _rand_event(rng, world, vids, allow_multi=False)
            text = pick(rng, NEG2).format(a=V[vids[0]].says(ev[vids[0]]), b=V[vids[1]].says(ev[vids[1]]))
        made += add({"kind": "noul", "event": _ev_names(world, ev), "neg": True, "text": text})
    assert made >= 1

    # conditionals (>= 1); the conditioning event must have positive probability at the final step
    n_cond = int(rng.integers(1, 3))
    made = 0
    for _ in range(50):
        if made >= n_cond:
            break
        a, g = (int(i) for i in rng.choice(nv, size=2, replace=False))
        ev = _rand_event(rng, world, [a], allow_multi=False)
        marg_g = J_final.sum(axis=tuple(i for i in range(nv) if i != g))
        ok = [i for i in range(len(V[g].options)) if marg_g[i] > 1e-6]
        if not ok:
            continue
        gi = [int(pick(rng, ok))]
        text = pick(rng, COND).format(g=V[g].says(gi), a=V[a].says(ev[a]))
        made += add({"kind": "cond", "event": _ev_names(world, ev), "given": {V[g].name: gi}, "text": text})
    assert made >= 1

    mv = next(v for v in V if v.name == world.mart_var)
    for k in range(world.T):
        add({"kind": "marginal", "var": mv.name, "step": k, "text": pick(rng, mv.questions), "options": list(mv.options)})
    return qs


# ----------------------------------------------------------------------------------------------------------------------
# tv=2 (release) templates: every yes/no query is a polar question (F5); conjunctions are fenced ("both that ... and
# that ...", "all of the following") and put a multi-option "either" clause last (F2); negated conjunctions always
# say "both" (F3); no "If X, is it the case that Y?" (F13); marginal questions never name an entity absent from the
# prelude (F8).
# ----------------------------------------------------------------------------------------------------------------------

CONJ2_V2 = [
    "Is it true both that {a} and that {b}?",
    "Is it the case both that {a} and that {b}?",
    "Do both of the following hold: {a}; {b}?",
]
CONJ3_V2 = [
    "Is it true that {a}, that {b}, and that {c}?",
    "Do all of the following hold: {a}; {b}; {c}?",
    "Is it the case that all of the following hold: {a}; {b}; {c}?",
]
NEG1_V2 = [
    "Is it false that {a}?",
    "Is it not the case that {a}?",
    "Is it untrue that {a}?",
]
NEG2_V2 = [
    "Is it false that both {a} and {b}?",
    "Is it not the case that both {a} and {b}?",
    "Is it untrue that both {a} and {b}?",
]
COND_V2 = [
    "Given that {g}, is it true that {a}?",
    "Assuming {g}, is it the case that {a}?",
    "Suppose {g}. Is it then true that {a}?",
    "In the case where {g}, does it hold that {a}?",
]
_NEGWORD = re.compile(r"\b(not|no|never|neither|nobody|none|untrue|false)\b|n't\b", re.I)
_NAME = re.compile(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b")


_FALSE_OPT = re.compile(r"""[`'"]?false[`'"]?""", re.I)


def false_to_true(var: "Var", idxs):
    """tv2 double-negation guard for boolean options: a single boolean-False option of a two-valued variable is
    replaced by its other (True) option, so a negated query never reads "not (x is False)". Deterministic (no rng
    draw), so worlds and all other queries are unchanged; a no-op for every variable without a False option."""
    if len(idxs) == 1 and len(var.options) == 2 and _FALSE_OPT.fullmatch(str(var.options[idxs[0]]).strip()):
        return [1 - idxs[0]]
    return idxs


def names_absent(text: str, context: str) -> set:
    """Capitalised non-initial words of `text` that never occur in `context`."""
    return {w for w in _NAME.findall(text) if w not in context}


def visible_questions(var: Var, context: str) -> list[str]:
    ok = [q for q in var.questions if not names_absent(q, context)]
    return ok or sorted(var.questions, key=lambda q: len(names_absent(q, context)))[:1]


def either_last(pairs):
    """[(vi, idxs)] ordered so that multi-option (either-) clauses come last (stable)."""
    return sorted(pairs, key=lambda p: len(p[1]) > 1)


COND_GUARD_V2 = 1e-3  # P(given) floor for conditionals (rounding-safe)
CERTAIN_EPS = 1e-9
DEGEN_EPS = 0.01  # yes/no answers outside [0.01, 0.99] are redrawn when possible
UNIFORM_EPS = 0.01  # ... and so are answers within 0.01 of 1/2


def _event_mask(shape, ev):
    m = np.ones(shape, dtype=bool)
    for vi, idx in ev.items():
        v = np.zeros(shape[vi], dtype=bool)
        v[list(idx)] = True
        sh = [1] * len(shape)
        sh[vi] = shape[vi]
        m &= v.reshape(sh)
    return m


def _make_queries_v2(world: World, J_final: np.ndarray, rng) -> list[dict]:
    V = world.variables
    nv = len(V)
    J = J_final / J_final.sum()
    shape = J.shape
    qs: list[dict] = []
    seen = set()
    ctx = world.prelude

    def key(q):
        return (q["kind"], q.get("neg", False), tuple(sorted((k, tuple(v)) for k, v in q.get("event", {}).items())),
                tuple(sorted((k, tuple(v)) for k, v in q.get("given", {}).items())))

    def add(q):
        if q["kind"] != "marginal":
            kk = key(q)
            if kk in seen:
                return False
            seen.add(kk)
        q["id"] = f"q{len(qs)}"
        qs.append(q)
        return True

    def certain(p):  # degenerate: near-certain (#11 / shortcuts #4) or exactly uniform
        return p < DEGEN_EPS or p > 1 - DEGEN_EPS or abs(p - 0.5) < UNIFORM_EPS

    for v in V:
        add({"kind": "marginal", "var": v.name, "text": pick(rng, visible_questions(v, ctx)), "options": list(v.options)})

    # each query family: first try to draw non-certain events (#11), then accept anything to meet the SPEC minimum
    def fill(n_want, n_min, draw):
        made = 0
        for attempt in range(90):
            if made >= n_want or (attempt >= 60 and made >= n_min):
                break
            q, p = draw()
            if q is None or (attempt < 60 and certain(p)):
                continue
            made += add(q)
        assert made >= n_min

    def draw_conj():
        k = 3 if (nv >= 3 and rng.random() < 0.3) else 2
        vids = sorted(int(i) for i in rng.choice(nv, size=k, replace=False))
        ev = _rand_event(rng, world, vids)
        cl = [V[vi].says(idx) for vi, idx in either_last([(vi, ev[vi]) for vi in vids])]
        tpl = pick(rng, CONJ2_V2 if k == 2 else CONJ3_V2)
        p = float(J[_event_mask(shape, ev)].sum())
        return {"kind": "noul", "event": _ev_names(world, ev), "text": tpl.format(a=cl[0], b=cl[1], c=cl[-1])}, p

    def draw_neg():
        if rng.random() < 0.6:
            vi = int(rng.integers(nv))
            ev = _rand_event(rng, world, [vi])
            for _ in range(3):  # prefer a positively phrased clause, avoiding double negation
                if not _NEGWORD.search(V[vi].says(ev[vi])):
                    break
                ev = _rand_event(rng, world, [vi])
            ev[vi] = false_to_true(V[vi], ev[vi])
            text = pick(rng, NEG1_V2).format(a=V[vi].says(ev[vi]))
        else:
            vids = sorted(int(i) for i in rng.choice(nv, size=2, replace=False))
            ev = _rand_event(rng, world, vids, allow_multi=False)
            ev = {vi: false_to_true(V[vi], idx) for vi, idx in ev.items()}
            text = pick(rng, NEG2_V2).format(a=V[vids[0]].says(ev[vids[0]]), b=V[vids[1]].says(ev[vids[1]]))
        p = 1 - float(J[_event_mask(shape, ev)].sum())
        return {"kind": "noul", "event": _ev_names(world, ev), "neg": True, "text": text}, p

    def draw_cond():
        a, g = (int(i) for i in rng.choice(nv, size=2, replace=False))
        ev = _rand_event(rng, world, [a], allow_multi=False)
        marg_g = J.sum(axis=tuple(i for i in range(nv) if i != g))
        ok = [i for i in range(len(V[g].options)) if marg_g[i] > COND_GUARD_V2]
        if not ok:
            return None, 0.5
        gi = [int(pick(rng, ok))]
        gm = _event_mask(shape, {g: gi})
        p = float(J[gm & _event_mask(shape, ev)].sum() / J[gm].sum())
        text = pick(rng, COND_V2).format(g=V[g].says(gi), a=V[a].says(ev[a]))
        return {"kind": "cond", "event": _ev_names(world, ev), "given": {V[g].name: gi}, "text": text}, p

    fill(int(rng.integers(2, 4)), 2, draw_conj)
    fill(int(rng.integers(1, 3)), 1, draw_neg)
    fill(int(rng.integers(1, 3)), 1, draw_cond)

    # martingale marginals at steps 0..T-2; the final step is the (unstepped) final marginal (#9: no duplicate)
    mv = next(v for v in V if v.name == world.mart_var)
    for k in range(world.T - 1):
        qk = mv.questions_at(k) if mv.questions_at is not None else None
        add({"kind": "marginal", "var": mv.name, "step": k,
             "text": pick(rng, qk or visible_questions(mv, ctx)), "options": list(mv.options)})
    return qs


# ----------------------------------------------------------------------------------------------------------------------
# instance assembly
# ----------------------------------------------------------------------------------------------------------------------

def _r(a: np.ndarray, decimals: int = ROUND):
    return np.round(a, decimals).tolist()


def build_instance(world: World, rng, *, family: str, split: str, idx: int, return_trace: bool = False,
                   decimals: int = ROUND, tv: int | None = None):
    z, obs = world.sample(rng)
    names = [v.name for v in world.variables]
    mv_axis = names.index(world.mart_var)
    other = lambda J: tuple(i for i in range(J.ndim) if i != mv_axis)  # noqa: E731

    steps = []
    post = world.prior.astype(float) / world.prior.sum()
    Js = []
    for k in range(world.T):
        post = post * world.lik(obs[k], obs[:k])
        post = post / post.sum()
        J = world.project(post)
        Js.append(J)
        st = {"evidence": world.render(k, obs[k]), "joint": _r(J, decimals)}
        if k < world.T - 1:
            past = obs[: k + 1]
            alts = []
            for a in world.alternatives(k + 1, past):
                l = world.lik(a, past)
                pa = float(post @ l)
                if pa <= 1e-12:
                    continue
                pa_post = world.project(post * l / pa).sum(axis=other(J))
                alts.append({"evidence": world.render(k + 1, a), "prob": pa, "mart_post": _r(pa_post, decimals)})
            tot = sum(a["prob"] for a in alts)
            for a in alts:
                a["prob"] = round(a["prob"] / tot, decimals)
            st["next_evidence"] = alts
        steps.append(st)

    inst = {
        "id": f"{family}-{split}-{idx:06d}",
        "family": family,
        "split": split,
        "variables": [{"name": v.name, "options": list(v.options)} for v in world.variables],
        "prelude": world.prelude,
        "steps": steps,
        "gold": {n: int(g) for n, g in zip(names, world.proj[z])},
        "mart_var": world.mart_var,
    }
    if world.exchangeable and world.T >= 2:
        perms, seen = [], {tuple(range(world.T))}
        for _ in range(20):
            p = tuple(int(i) for i in rng.permutation(world.T))
            if p not in seen:
                seen.add(p)
                perms.append(list(p))
            if len(perms) >= 3:
                break
        inst["perms"] = perms
    tv = get_tv() if tv is None else int(tv)
    inst["queries"] = make_queries(world, Js[-1], rng, tv=tv)
    inst["meta"] = dict(world.meta)
    if tv >= 2:
        inst["meta"]["tv"] = tv
    if return_trace:
        return inst, {"z": z, "obs": obs}
    return inst
