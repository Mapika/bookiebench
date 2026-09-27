"""Evidence-blind reference predictors, computed internally (no prediction files needed).

- uniform_joint: the uniform distribution over the joint cells; every query is answered from it. Needs nothing.
- indep_joint:   the product over variables of a train-mean marginal. Marginals are keyed by option TEXT, not
                 position, because release instances shuffle options (meta.option_perm): the table holds the mean exact
                 final-step probability of (family, variable name, option text, K), shrunk hierarchically (SHRINK pseudo-
                 observations) toward (family, text, K) and that toward 1/K, because many option texts are rare
                 names seen a handful of times; unseen keys fall back to (family, option text, K) within the same family, then to 1/K,
                 and the vector is renormalised. There is deliberately no cross-family text fallback: on procedural
                 families (randbn/randhmm) whose option words are reused with random meanings it made indep_joint
                 worse than uniform; cross-family label transfer is what ref:label_prior measures.
- label_prior / template_uj (as in review/final_shortcuts/newcheats.py), fitted on train:
    label_prior  marginal: option text -> train-mean K*p, keyed (text, K), fallback text alone, smoothed toward 1;
                 yes/no: the uniform_joint answer.
    template_uj  marginal: (question template, option text) -> train-mean K*p (fallback label_prior); yes/no:
                 0.5 * train-mean p_yes of the template + 0.5 * uniform_joint answer (fallback uniform_joint).
  `ref:prior` is the stronger of the two per family. Procedural families (PRIOR_FALLBACK_FAMILIES = bookiebench.sims
  PRIOR_FAMILIES: randbn, randhmm) answer uniform_joint in all train-fitted refs, as do families in
  PRIOR_FALLBACK_GROUPS (new_mechanics: mechanics
  without train data) get uniform_joint instead.

All of them are evidence-blind (sens = 0), and uniform_joint / indep_joint are coherent by construction (dutch = 0).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .exact import exact_answer, shape_of

PRIOR_FALLBACK_GROUPS = {"new_mechanics"}
try:  # procedural families: option words carry random meanings, so the true train-mean marginal is 1/K
    from bookiebench.sims import PRIOR_FAMILIES as _PF
    PRIOR_FALLBACK_FAMILIES = set(_PF)
except Exception:  # noqa: BLE001
    PRIOR_FALLBACK_FAMILIES = {"randbn", "randhmm"}


def uses_uniform_fallback(inst: dict, groups: dict | None = None, group: str | None = None) -> bool:
    """True when every train-fitted reference (indep_joint, label_prior, template_uj) must answer with uniform_joint:
    the family is procedural (PRIOR_FALLBACK_FAMILIES: fitted means are pure estimator noise around 1/K) or its group
    has no train data (PRIOR_FALLBACK_GROUPS)."""
    if inst.get("family") in PRIOR_FALLBACK_FAMILIES:
        return True
    if group is None and groups:
        group = groups.get(f"{inst.get('split')}/{inst.get('family')}")
    return group in PRIOR_FALLBACK_GROUPS
SHRINK = 10.0  # pseudo-observations for indep_joint's text-keyed marginals (many option texts are rare names)
PRIOR_NAMES = ("label_prior", "template_uj")


def with_joint(inst: dict, joint: np.ndarray) -> dict:
    """Copy of `inst` whose every step has `joint` (so step-k queries are answered from it too)."""
    j = np.asarray(joint, dtype=float).tolist()
    return dict(inst, steps=[dict(s, joint=j) for s in inst["steps"]])


def uniform_joint(inst: dict) -> np.ndarray:
    shp = shape_of(inst)
    return np.full(shp, 1.0 / int(np.prod(shp)))


def _options(inst: dict, q: dict) -> list[str]:
    if q.get("options"):
        return list(q["options"])
    for v in inst["variables"]:
        if v["name"] == q["var"]:
            return list(v["options"])
    return []


def answers_from_joint(inst: dict, joint: np.ndarray) -> dict:
    fake = with_joint(inst, joint)
    out = {}
    for q in inst["queries"]:
        a = exact_answer(fake, q)
        if q["kind"] == "cond" and not np.all(np.isfinite(a)):
            a = np.array([0.5])  # P(given) = 0 under this joint: every price is coherent, the bet is never on
        if np.all(np.isfinite(a)):
            out[q["id"]] = a.tolist()
    return out


def _train_files(train: str | Path) -> list[Path]:
    train = Path(train)
    if train.is_file():
        return [train]
    return sorted(train.rglob("*.jsonl")) if train.is_dir() else []


def _head(path: Path, n: int):
    with open(path) as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            if line.strip():
                yield json.loads(line)


def _final_joint(inst: dict):
    if not inst.get("steps") or inst["steps"][-1].get("joint") is None:
        return None
    j = np.asarray(inst["steps"][-1]["joint"], dtype=float).reshape(shape_of(inst))
    return j / j.sum()


# ----------------------------------------------------------------------------- indep_joint
class IndepTable:
    def __init__(self):
        self.fam: dict[tuple[str, str, str, int], float] = {}  # (family, var name, option text, K) -> mean prob
        self.text: dict[tuple[str, str, int], float] = {}  # (family, option text, K) -> mean prob
        self.source = None

    def marginal(self, family: str, var: dict) -> np.ndarray:
        opts = var["options"]
        k = len(opts)
        m = []
        for o in opts:
            p = self.fam.get((family, var["name"], o, k))
            if p is None:
                p = self.text.get((family, o, k), 1.0 / k)
            m.append(p)
        m = np.clip(np.asarray(m, dtype=float), 1e-6, None)
        return m / m.sum()

    def joint(self, inst: dict) -> np.ndarray:
        J = None
        for v in inst["variables"]:
            m = self.marginal(inst.get("family", "?"), v)
            J = m if J is None else np.multiply.outer(J, m)
        return J


def fit_indep(train: str | Path, n_per_file: int = 1500) -> IndepTable | None:
    """Fit the indep_joint table from train files (None if `train` does not exist)."""
    files = _train_files(train)
    if not files:
        return None
    fam, txt = defaultdict(list), defaultdict(list)
    for f in files:
        for inst in _head(f, n_per_file):
            j = _final_joint(inst)
            if j is None:
                continue
            for ax, v in enumerate(inst["variables"]):
                m = j.sum(axis=tuple(a for a in range(j.ndim) if a != ax))
                for o, p in zip(v["options"], m):
                    fam[(inst.get("family", "?"), v["name"], o, m.size)].append((p, m.size))
                    txt[(inst.get("family", "?"), o, m.size)].append((p, m.size))
    # hierarchical shrinkage with SHRINK pseudo-observations: (family, text, K) -> 1/K, (family, var, text) -> the former
    t = IndepTable()
    t.text = {k: (sum(p for p, _ in v) + SHRINK / v[0][1]) / (len(v) + SHRINK) for k, v in txt.items()}
    for (f, name, o, k), v in fam.items():
        base = t.text[(f, o, k)]
        t.fam[(f, name, o, k)] = (sum(p for p, _ in v) + SHRINK * base) / (len(v) + SHRINK)
    t.source = str(train)
    return t


# ----------------------------------------------------------------------------- label / template priors
_DIG = re.compile(r"\d+(?:\.\d+)?")
_QUOT = re.compile(r"'[^']*'|\"[^\"]*\"")
_CAP = re.compile(r"(?<=[\w,;:] )[A-Z][a-z]+")


def template(text: str) -> str:
    t = _QUOT.sub("Q", text or "")
    t = _CAP.sub("X", t)
    t = _DIG.sub("#", t)
    return t.lower().strip()


class PriorTables:
    """Train tables for label_prior and template_uj (same statistics as review/final_shortcuts/newcheats.py)."""

    def __init__(self, lab=None, labk=None, tq=None, tb=None, source=None):
        self.lab = lab or {}  # (text, K) -> smoothed mean K*p
        self.labk = labk or {}  # text -> smoothed mean K*p
        self.tq = tq or {}  # (template, text) -> smoothed mean K*p
        self.tb = tb or {}  # template -> mean p_yes
        self.source = source

    def opt_score(self, o: str, K: int) -> float:
        o = o.lower()
        return self.lab.get((o, K), self.labk.get(o, 1.0))

    def to_json(self) -> dict:
        enc = lambda d: [[list(k) if isinstance(k, tuple) else k, v] for k, v in d.items()]  # noqa: E731
        return {"lab": enc(self.lab), "labk": enc(self.labk), "tq": enc(self.tq), "tb": enc(self.tb),
                "source": self.source}

    @classmethod
    def from_json(cls, d: dict) -> "PriorTables":
        dec = lambda xs: {tuple(k) if isinstance(k, list) else k: v for k, v in xs}  # noqa: E731
        return cls(dec(d["lab"]), dec(d["labk"]), dec(d["tq"]), dec(d["tb"]), d.get("source"))


def _cache_path(files: list[Path], n: int) -> Path:
    h = hashlib.sha256()
    for f in files:
        st = f.stat()
        h.update(f"{f.resolve()}:{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(f"n={n};v=1".encode())
    root = Path(os.environ.get("HONEST_METRICS_CACHE", Path.home() / ".cache" / "honest_metrics"))
    return root / f"prior_{h.hexdigest()[:16]}.json"


def fit_prior(train: str | Path, n_per_file: int = 2000, cache: bool = True) -> PriorTables | None:
    """Fit label/template tables on the first `n_per_file` instances of every train file (None if no train).
    Cached under $HONEST_METRICS_CACHE (default ~/.cache/honest_metrics), keyed by file stats."""
    files = _train_files(train)
    if not files:
        return None
    cp = _cache_path(files, n_per_file)
    if cache and cp.exists():
        try:
            return PriorTables.from_json(json.loads(cp.read_text()))
        except Exception:  # noqa: BLE001 - corrupt cache: refit
            pass
    lab, labk, tq, tb = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for f in files:
        for inst in _head(f, n_per_file):
            if _final_joint(inst) is None:
                continue
            for q in inst["queries"]:
                try:
                    ex = exact_answer(inst, q)
                except Exception:  # noqa: BLE001
                    continue
                if q["kind"] == "marginal":
                    opts = _options(inst, q)
                    K = len(opts)
                    if ex.size != K:
                        continue
                    t = template(q.get("text", ""))
                    for j, o in enumerate(opts):
                        o = o.lower()
                        lab[(o, K)].append(K * ex[j])
                        labk[o].append(K * ex[j])
                        tq[(t, o)].append(K * ex[j])
                elif np.isfinite(ex[0]):
                    tb[template(q.get("text", ""))].append(float(ex[0]))
    sm = lambda d, c=1: {k: (float(np.mean(v)) * len(v) + c) / (len(v) + c) for k, v in d.items()}  # noqa: E731
    T = PriorTables(sm(lab), sm(labk), sm(tq, 2), {k: float(np.mean(v)) for k, v in tb.items()}, str(train))
    if cache:
        try:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(T.to_json()))
        except OSError:
            pass
    return T


def prior_answers(inst: dict, name: str, T: PriorTables) -> dict:
    """Answers of label_prior / template_uj for one instance."""
    uj = answers_from_joint(inst, uniform_joint(inst))
    out = {}
    for q in inst["queries"]:
        if q["kind"] == "marginal":
            opts = _options(inst, q)
            K = len(opts)
            if name == "template_uj":
                t = template(q.get("text", ""))
                w = [T.tq.get((t, o.lower()), T.opt_score(o, K)) for o in opts]
            else:
                w = [T.opt_score(o, K) for o in opts]
            w = np.clip(np.asarray(w, float), 1e-3, None)
            out[q["id"]] = (w / w.sum()).tolist()
        else:
            p = uj.get(q["id"], [0.5])[0]
            if name == "template_uj":
                v = T.tb.get(template(q.get("text", "")))
                if v is not None:
                    p = 0.5 * v + 0.5 * p
            out[q["id"]] = [float(p)]
    return out


# ----------------------------------------------------------------------------- prediction rows
def ref_predictions(instances: dict[str, dict], name: str, table=None, groups: dict | None = None) -> dict:
    """{id: {"base": prediction}} for reference `name` in {"uniform_joint", "indep_joint", "label_prior",
    "template_uj"} (base variant only). `table`: IndepTable or PriorTables. For the train-fitted refs, instances with
    `uses_uniform_fallback` (procedural families, or groups in PRIOR_FALLBACK_GROUPS via `groups`: "split/family" ->
    group) get uniform_joint answers."""
    out = {}
    for iid, inst in instances.items():
        try:
            if name == "uniform_joint":
                ans = answers_from_joint(inst, uniform_joint(inst))
            elif name in ("indep_joint",) + PRIOR_NAMES and uses_uniform_fallback(inst, groups):
                ans = answers_from_joint(inst, uniform_joint(inst))
            elif name == "indep_joint":
                ans = answers_from_joint(inst, table.joint(inst))
            elif name in PRIOR_NAMES:
                ans = prior_answers(inst, name, table)
            else:
                raise ValueError(name)
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - malformed instance: reference abstains
            ans = {}
        out[iid] = {"base": {"id": iid, "model": name, "variant": "base", "answers": ans}}
    return out
