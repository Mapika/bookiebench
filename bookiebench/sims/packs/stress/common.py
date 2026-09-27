"""Shared machinery for the stress transforms.

A transform takes a source instance (any family, SPEC section 1) and returns a new instance that
  * keeps `variables` (option *indices*), every `steps[k]["joint"]`, `gold`, `mart_var`, `perms` and the original
    queries' events/kinds unchanged (text fields may change),
  * carries `transform`, `pack="stress"`, `level`, `family` (= source family) and `meta.source_id`,
  * may add queries (`meta.stress.added_qids`); their answers are always computed from the joint via
    `bookiebench.sims.core.exact_answer`, never stated.

Text edits to evidence go through `map_evidence`, which applies the same map to the realised evidence and to every
`next_evidence` alternative, so the realised text stays one of the alternatives.
"""
from __future__ import annotations

import copy
import re
import zlib
from functools import lru_cache

import numpy as np

from bookiebench.sims.core import exact_answer, joint_array, event_mask, var_names

PACK = "stress"
LEVEL = "L4"


# ----------------------------------------------------------------------------------------------------------------------
# rng / tagging
# ----------------------------------------------------------------------------------------------------------------------

def transform_rng(inst, transform: str, seed: int = 0, extra: int = 0):
    """Deterministic rng in (seed, transform, source id)."""
    return np.random.default_rng([int(seed), zlib.crc32(transform.encode()), zlib.crc32(inst["id"].encode()),
                                  int(extra)])


def source_id(inst) -> str:
    return inst.get("meta", {}).get("source_id", inst["id"])


def start(inst, transform: str, **info) -> dict:
    """Deep copy of `inst` tagged as the output of `transform`."""
    out = copy.deepcopy(inst)
    src = inst["id"]
    out["id"] = f"{src}~{transform}"
    out["transform"] = transform
    out["pack"] = PACK
    out["level"] = LEVEL
    out["family"] = inst["family"]
    meta = out.setdefault("meta", {})
    meta["source_id"] = inst.get("meta", {}).get("source_id", src)
    meta["source_split"] = inst.get("meta", {}).get("source_split", inst.get("split"))
    meta["stress"] = {"transform": transform, **info}
    out["split"] = "stress"
    return out


# ----------------------------------------------------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------------------------------------------------

def map_evidence(inst, fn):
    """Apply fn(text, k) to the evidence of step k and to all next_evidence alternatives that describe step k."""
    for k, st in enumerate(inst["steps"]):
        st["evidence"] = fn(st["evidence"], k)
        if k > 0 and inst["steps"][k - 1].get("next_evidence"):
            for a in inst["steps"][k - 1]["next_evidence"]:
                a["evidence"] = fn(a["evidence"], k)
    return inst


def evidence_texts(inst) -> list[str]:
    """All distinct evidence strings (realised + alternatives), in first-seen order."""
    seen, out = set(), []
    for k, st in enumerate(inst["steps"]):
        cand = [st["evidence"]] + [a["evidence"] for a in st.get("next_evidence") or []]
        for t in cand:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def replace_evidence_texts(inst, mapping: dict):
    for st in inst["steps"]:
        st["evidence"] = mapping.get(st["evidence"], st["evidence"])
        for a in st.get("next_evidence") or []:
            a["evidence"] = mapping.get(a["evidence"], a["evidence"])
    return inst


_ABBR = re.compile(r"\b(Dr|Mr|Mrs|Ms|St|vs|e\.g|i\.e|No)\. ")


def units(text: str) -> list[dict]:
    """Split a prelude into self-contained units {text, movable, sep} (sentences with their dependent follow-ups;
    bullet blocks and a JSON tail are kept whole; `sep` is the separator before the unit). `join_units` inverts it
    (up to whitespace)."""
    from bookiebench.sims.nuisance import _split_units  # the same unit splitter nuisance uses for safe reordering
    j = _json_start(text)
    if j is not None:
        head, body = text[:j], text[j:]
        h = head.strip()
        if h and len(h) <= 40 and not re.search(r"[.!?]$", h):  # a label such as "STATE =" or "CONTEXT:"
            return [{"text": body, "movable": False, "sep": "", "json": True, "prefix": head}]
        out = units(head.rstrip()) if h else []
        gap = head[len(head.rstrip()):] or ("" if not out else " ")
        return out + [{"text": body, "movable": False, "sep": gap if out else "", "json": True}]
    out = []
    for u, m, sep in _split_units(text):
        out.append({"text": u, "movable": m, "sep": sep if out else ""})
    return out


def join_units(us) -> str:
    return "".join((u["sep"] if i else "") + u.get("prefix", "") + u["text"] for i, u in enumerate(us))


def split_units(text: str) -> list[str]:
    return [u["text"] for u in units(text)]


def _json_start(text: str):
    import json
    i = text.find("{")
    if i < 0:
        return None
    try:
        json.loads(text[i:])
        return i
    except ValueError:
        return None


def is_json_prelude(text: str) -> bool:
    return _json_start(text) is not None


NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> list[str]:
    """Multiset (sorted list) of number tokens, with ',' decimal/grouping separators normalised to '.'."""
    return sorted(t.replace(",", ".") for t in NUM_RE.findall(text))


# ----------------------------------------------------------------------------------------------------------------------
# event clauses ("Tariq has Ilsen syndrome") for building new queries
# ----------------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=4096)
def _regen_world(family: str, split: str, idx: int, seed: int = 0, tv=None):
    from bookiebench.sims import FAMILIES, make_world
    from bookiebench.sims.common import template_version
    if family not in FAMILIES:
        return None
    try:
        with template_version(tv):
            world, _ = make_world(family, split, idx, seed)
    except Exception:  # noqa: BLE001
        return None
    return world


def _world_for(inst):
    """(World, option_perm) of a registry instance (for its clause templates), or (None, None).

    Handles release instances: the world is regenerated with seed = meta.release_attempt (base seed 0) and template
    version meta.tv, and options may be shuffled (meta.option_perm[var][new] = canonical index). The world is only
    used when its prelude and (permuted) options match the instance exactly."""
    sid = source_id(inst)
    m = re.match(r"^([a-z_]+)-([a-z_]+)-(\d+)$", sid)
    if not m:
        return None, None
    fam, split, idx = m.group(1), m.group(2), int(m.group(3))
    split = "test" if split == "robust" else split
    meta = inst.get("meta") or {}
    w = _regen_world(fam, split, idx, int(meta.get("release_attempt", 0)), meta.get("tv"))
    if w is None or [v.name for v in w.variables] != var_names(inst):
        return None, None
    perm = meta.get("option_perm") or {}
    for v, iv in zip(w.variables, inst["variables"]):
        p = perm.get(v.name, list(range(len(v.options))))
        if [v.options[i] for i in p] != iv["options"]:
            return None, None
    src_pre = meta.get("source_prelude")
    if w.prelude != (src_pre if src_pre is not None else inst["prelude"]):
        return None, None
    return w, perm


class Clauses:
    """clause(var, idxs) -> declarative clause stating var in idxs; neg_clause -> its negation."""

    def __init__(self, inst):
        self.inst = inst
        self.world, self.perm = _world_for(inst)
        self.names = var_names(inst)
        self.opts = {v["name"]: v["options"] for v in inst["variables"]}
        self.qtext = {}
        for q in inst["queries"]:
            if q["kind"] == "marginal" and "step" not in q and q["var"] not in self.qtext:
                self.qtext[q["var"]] = q["text"].strip()

    @property
    def native(self) -> bool:
        return self.world is not None

    def clause(self, var: str, idxs) -> str:
        idxs = [int(i) for i in idxs]
        if self.world is not None:
            v = self.world.variables[self.names.index(var)]
            p = (self.perm or {}).get(var)
            return v.says(sorted(p[i] for i in idxs) if p else idxs)
        opts = self.opts[var]
        q = self.qtext.get(var, var.replace("_", " ") + "?")
        val = " or ".join(f'"{opts[i]}"' for i in idxs)
        return f'the answer to "{q}" is {"either " if len(idxs) > 1 else ""}{val}'

    def not_clause(self, var: str, idxs) -> str:
        """A positive clause for the complement set (var not in idxs)."""
        comp = [i for i in range(len(self.opts[var])) if i not in set(int(x) for x in idxs)]
        return self.clause(var, comp)


# ----------------------------------------------------------------------------------------------------------------------
# queries
# ----------------------------------------------------------------------------------------------------------------------

def next_qid(inst) -> str:
    used = {q["id"] for q in inst["queries"]}
    i = len(inst["queries"])
    while f"q{i}" in used:
        i += 1
    return f"q{i}"


def add_query(inst, q: dict) -> dict:
    """Append query q (id assigned) after checking it is answerable; records it in meta.stress.added_qids and stores
    its exact answer in meta.stress.added_exact (for audit; consumers recompute via exact_answer)."""
    q = dict(q)
    q["id"] = next_qid(inst)
    if q["kind"] == "cond":
        J = joint_array(inst, q.get("step"))
        pg = float(J[event_mask(inst, q["given"])].sum())
        if pg <= 1e-6:
            raise ValueError("conditioning event has ~zero probability")
    ans = exact_answer(inst, q)
    inst["queries"].append(q)
    st = inst["meta"]["stress"]
    st.setdefault("added_qids", []).append(q["id"])
    st.setdefault("added_exact", {})[q["id"]] = [round(float(a), 10) for a in ans]
    return q


def positive_prob_options(inst, var: str, eps: float = 1e-6):
    J = joint_array(inst)
    ax = var_names(inst).index(var)
    m = J.sum(axis=tuple(i for i in range(J.ndim) if i != ax))
    return [i for i, p in enumerate(m) if p > eps]


def invariance_signature(inst) -> dict:
    """What a transform must not change (used by tests): variables' option counts, joints, gold, mart_var, perms and
    the (kind, event, given, neg, var, step) of every original query."""
    qs = [(q["id"], q["kind"], q.get("var"), q.get("step"), repr(sorted((q.get("event") or {}).items())),
           repr(sorted((q.get("given") or {}).items())), bool(q.get("neg"))) for q in inst["queries"]]
    return {"shape": [len(v["options"]) for v in inst["variables"]], "names": var_names(inst),
            "joints": [st["joint"] for st in inst["steps"]], "gold": inst["gold"], "mart_var": inst.get("mart_var"),
            "perms": inst.get("perms"), "queries": qs,
            "ne_probs": [[a["prob"] for a in st.get("next_evidence") or []] for st in inst["steps"]]}
