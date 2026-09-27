"""BookieBench metrics (SPEC §3).

Pipeline: `evaluate(instances, predictions)` -> per-instance records; `aggregate(records)` -> metric dict;
`build_report(records)` -> {"overall", "by_split", "by_family"}.

Conventions
- `kl`: mean over all answered base-variant queries (at their own step) of KL(exact || predicted); marginal vectors are
  clipped at EPS and renormalised, Bernoulli answers clipped into [EPS, 1-EPS]. `kl_marg` / `kl_bin` split it.
- `logscore` (mean log p(gold), higher is better), `acc`, `ece` (15 equal-width bins, top-label), `cover@a`/`size@a`:
  final-step marginal queries whose variable has a gold value.
- Missing answers (absent, NaN, wrong length) never help. kl/logscore/acc/ece/cover score them as the uniform answer
  (1/K per option, 0.5 for noul/cond); `answered_frac` reports the answered share (a headline column); instances of an
  evaluated split with no base prediction count as fully missing (`n_missing_inst`).
- `acc` gives fractional credit 1/|argmax set| on ties (expected accuracy under random tie-breaking), so a uniform or
  omitted answer earns 1/K, not an arbitrary index-0 hit.
- `dutch`: LP/MILP over final-step queries with missing prices adversarial, see dutch.py; `dutch_norm` = profit /
  #queries, `dutch_per_bet` = profit / #bets, `dutch@d` = tolerance-aware (+-d price intervals), `dutch_ans` = answered
  prices only (pre-fix, gameable diagnostic); `dutch_frac_exploitable` = share of instances with dutch@0.01 > 1e-6.
- `optperm` / `evperm`: mean total variation between base and permuted-variant answers (evperm: final-step queries only).
- `para`: mean total variation between base answers and the same query asked with paraphrased wording (variants
  "para:<i>", realcoh). Instances with "joint": null (realcoh) skip kl/mart; dutch, optperm, para and gold scores apply.
- `mart`: mean abs entry-wise difference between the model's p_k and sum_j prob_j * p_{k+1}^{(j)}.
- `skill` = mean over families of 1 - KL/KL_ref, where KL_ref is the KL of the evidence-blind uniform_joint reference
  (uniform over joint cells, every query answered from it) on the same queries, computed analytically per instance.
  `skill_marg` / `skill_bin` do the same for kl_marg / kl_bin. skill <= 0 means no better than reading nothing.
- `coh_valid` = 1 if skill > 1e-6 else 0; `dutch_inf` = dutch when coh_valid, else None ("–"): coherence is only
  informative for a model that uses the evidence (uniform_joint is perfectly coherent).
- `sens` (evidence sensitivity) = 1 - sum_k |dm_k - de_k|_1 / sum_k |de_k|_1 over consecutive steps of the mart_var
  marginal (dm = model change, de = exact change), pooled over the group: 1 for the oracle, 0 for a predictor that
  ignores the evidence, < 0 for changes worse than none. Missing step answers count as uniform.
- `kl_avgperm`: KL of the option-order-averaged answer (mean of base and all optperm:* answers), over instances that
  have optperm variants; removes option-position bias, so kl - kl_avgperm is the price of order bias.
- `cover@a`: split conformal with LAC score 1 - p_gold; instances go to calibration / test by sha256(id) parity;
  calibration is done within each reported group (so per-family rows are group-conditional).
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np

from .dutch import dutch_book
from .exact import exact_answer, is_final, joint_at, query_step
from .refs import uniform_joint, with_joint

EPS = 1e-6
ALPHAS = (0.1, 0.2)
ECE_BINS = 15

EXPLOIT_DELTA = 0.01
SKILL_EPS = 1e-6  # skill must exceed this to count as informative (uniform_joint scores 0 up to rounding)
DUTCH_KEYS = ("dutch", "dutch_norm", "dutch_per_bet", "dutch@0.005", "dutch@0.01", "dutch_ans")

METRIC_COLUMNS = [
    "answered_frac", "skill", "kl", "kl_marg", "kl_bin", "kl_avgperm", "sens", "logscore", "acc", "ece",
    "coh_valid", "dutch_inf", "dutch", "dutch_per_bet", "dutch@0.005", "dutch_frac_exploitable",
    "optperm", "evperm", "mart", "para",
    "cover@0.1", "size@0.1", "cover@0.2", "size@0.2",
]


# ----------------------------------------------------------------------------- elementary pieces
def _norm_vec(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float).ravel(), EPS, None)
    return p / p.sum()


def kl_vec(exact, pred) -> float:
    e = np.asarray(exact, dtype=float).ravel()
    q = _norm_vec(pred)
    m = e > 0
    return max(0.0, float(np.sum(e[m] * (np.log(e[m]) - np.log(q[m])))))


def kl_bern(p_exact: float, p_pred: float) -> float:
    q = float(np.clip(p_pred, EPS, 1 - EPS))
    out = 0.0
    if p_exact > 0:
        out += p_exact * (math.log(p_exact) - math.log(q))
    if p_exact < 1:
        out += (1 - p_exact) * (math.log(1 - p_exact) - math.log(1 - q))
    return max(out, 0.0)


def tv(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size == 1:  # Bernoulli [p_yes]
        return float(abs(a[0] - b[0]))
    return 0.5 * float(np.abs(_norm_vec(a) - _norm_vec(b)).sum()) if a.size == b.size else float("nan")


def ece_top_label(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> float:
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return float("nan")
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(correct[m].mean() - conf[m].mean())
    return float(tot / conf.size)


def is_calibration(inst_id: str) -> bool:
    """Deterministic 50/50 split by hashed instance id."""
    return int(hashlib.sha256(inst_id.encode()).hexdigest(), 16) % 2 == 0


def conformal(items: list[tuple[str, np.ndarray, int]], alpha: float) -> tuple[float, float]:
    """Split-conformal LAC. items = (instance id, probs, gold). Returns (coverage, mean set size) on the test half."""
    cal = [1.0 - p[g] for iid, p, g in items if is_calibration(iid)]
    test = [(p, g) for iid, p, g in items if not is_calibration(iid)]
    if not cal or not test:
        return float("nan"), float("nan")
    cal = np.sort(np.asarray(cal))
    n = cal.size
    k = math.ceil((n + 1) * (1 - alpha))
    qhat = math.inf if k > n else float(cal[k - 1])
    cov, size = [], []
    for p, g in test:
        s = (1.0 - p) <= qhat + 1e-12
        cov.append(bool(s[g]))
        size.append(int(s.sum()))
    return float(np.mean(cov)), float(np.mean(size))


# ----------------------------------------------------------------------------- per-instance evaluation
def _answer(answers: dict, qid: str):
    a = answers.get(qid)
    if a is None:
        return None
    a = np.atleast_1d(np.asarray(a, dtype=float)).ravel()
    return a if a.size and np.all(np.isfinite(a)) else None


def _fill(q: dict, a, inst: dict):
    """(answer with missing -> uniform, answered flag), or None for an unknown query kind."""
    if q["kind"] == "marginal":
        K = len(q.get("options") or _var_options(inst, q["var"]))
        if a is not None and a.size == K:
            return a, 1
        return np.full(K, 1.0 / K), 0  # missing -> uniform
    if a is not None:
        return a[:1], 1
    return np.array([0.5]), 0  # missing -> uniform Bernoulli


def has_exact_joint(inst: dict) -> bool:
    """False for instances whose law is unknown ("joint": null, the realcoh benchmark)."""
    return bool(inst.get("steps")) and all(s.get("joint") is not None for s in inst["steps"])


def _var_options(inst: dict, name: str) -> list:
    for v in inst["variables"]:
        if v["name"] == name:
            return v["options"]
    return []


def evaluate_instance(inst: dict, variants: dict[str, dict] | None) -> dict:
    """variants: variant name -> prediction dict. A missing `base` variant scores the instance as fully unanswered."""
    variants = variants or {}
    base = variants.get("base")
    ans = (base or {}).get("answers") or {}
    rec = {
        "id": inst["id"], "family": inst.get("family", "?"), "split": inst.get("split", "?"),
        "kl_marg": [], "kl_bin": [], "marg": [], "optperm": [], "evperm": [], "mart": [], "para": [],
        "kl_ref_marg": [], "kl_ref_bin": [], "kl_avgperm": [], "sens_num": 0.0, "sens_den": 0.0,
        "n_queries": len(inst["queries"]), "n_answered": 0, "pred_missing": base is None,
    }
    gold = inst.get("gold") or {}
    qby = {q["id"]: q for q in inst["queries"]}
    has_joint = has_exact_joint(inst)
    uj = with_joint(inst, uniform_joint(inst)) if has_joint else None
    optperms = [v.get("answers") or {} for k, v in variants.items() if k.startswith("optperm:")]
    filled: dict[str, np.ndarray] = {}
    exact: dict[str, np.ndarray] = {}
    for q in inst["queries"]:
        a = _fill(q, _answer(ans, q["id"]), inst)
        if a is None:
            continue
        a, answered = a
        rec["n_answered"] += answered
        filled[q["id"]] = a
        if q["kind"] == "marginal":
            if has_joint:  # realcoh: unknown law -> no kl; gold-based scores and coherence still apply
                ex = exact_answer(inst, q)
                if ex.size == a.size:
                    exact[q["id"]] = ex
                    rec["kl_marg"].append(kl_vec(ex, a))
                    rec["kl_ref_marg"].append(kl_vec(ex, exact_answer(uj, q)))
            if is_final(inst, q) and q["var"] in gold and gold[q["var"]] is not None:
                rec["marg"].append(_norm_vec(a).tolist() + [int(gold[q["var"]])])
        else:
            if has_joint:
                ex = exact_answer(inst, q)
                if np.isfinite(ex[0]):
                    exact[q["id"]] = ex
                    rec["kl_bin"].append(kl_bern(float(ex[0]), float(a[0])))
                    rec["kl_ref_bin"].append(kl_bern(float(ex[0]), float(exact_answer(uj, q)[0])))
        if optperms and q["id"] in exact:
            alts = [_answer(o, q["id"]) for o in optperms]
            alts = [x for x in alts if x is not None and x.size == a.size]
            avg = np.mean([_norm_vec(a) if a.size > 1 else a] + [_norm_vec(x) if x.size > 1 else x for x in alts], 0)
            ex = exact[q["id"]]
            rec["kl_avgperm"].append(kl_vec(ex, avg) if q["kind"] == "marginal" else kl_bern(float(ex[0]), float(avg[0])))

    # evidence sensitivity on the mart_var marginal across steps
    mv = inst.get("mart_var")
    if mv is not None and has_joint:
        by_step = {}
        for q in inst["queries"]:
            if q["kind"] == "marginal" and q.get("var") == mv and q["id"] in exact:
                by_step.setdefault(query_step(inst, q), q["id"])
        ks = sorted(by_step)
        for k0, k1 in zip(ks, ks[1:]):
            de = exact[by_step[k1]] - exact[by_step[k0]]
            dm = _norm_vec(filled[by_step[k1]]) - _norm_vec(filled[by_step[k0]])
            rec["sens_num"] += float(np.abs(dm - de).sum())
            rec["sens_den"] += float(np.abs(de).sum())

    d = dutch_book(inst, ans)
    rec.update({k: d[k] for k in DUTCH_KEYS})
    rec.update(dutch_n_queries=d["n_queries"], dutch_n_bets=d["n_bets"], dutch_n_missing_bets=d["n_missing_bets"])

    # permutation variance
    for vname, pred in variants.items():
        kind = vname.split(":", 1)[0]
        if kind not in ("optperm", "evperm", "para"):
            continue
        for qid, a2 in (pred.get("answers") or {}).items():
            q = qby.get(qid)
            a1 = _answer(ans, qid)
            a2 = _answer(pred.get("answers"), qid)
            if q is None or a1 is None or a2 is None or a1.size != a2.size:
                continue
            if kind == "evperm" and not is_final(inst, q):
                continue  # step-k prefixes differ between evidence orders
            d = tv(a1, a2)
            if np.isfinite(d):
                rec[kind].append(d)

    # martingale
    mv = inst.get("mart_var")
    if mv is not None:
        mq = {query_step(inst, q): q for q in inst["queries"] if q["kind"] == "marginal" and q.get("var") == mv}
        for k, step in enumerate(inst["steps"]):
            alts = step.get("next_evidence")
            if not alts or k not in mq:
                continue
            q = mq[k]
            pk = _answer(ans, q["id"])
            if pk is None:
                continue
            probs = np.array([float(al["prob"]) for al in alts])
            nexts = []
            for j in range(len(alts)):
                v = variants.get(f"next:{k}:{j}")
                a = None
                if v is not None:
                    va = v.get("answers") or {}
                    a = _answer(va, q["id"])
                    if a is None and len(va) == 1:
                        a = _answer(va, next(iter(va)))
                nexts.append(a)
            if any(a is None or a.size != pk.size for a in nexts) or probs.sum() <= 0:
                continue
            probs = probs / probs.sum()
            expected = sum(w * _norm_vec(a) for w, a in zip(probs, nexts))
            rec["mart"].append(float(np.mean(np.abs(_norm_vec(pk) - expected))))

    # joint head (optional)
    if base is not None and base.get("joint") is not None and has_joint:
        try:
            ej = joint_at(inst).ravel()
            pj = np.asarray(base["joint"], dtype=float).ravel()
            if pj.size == ej.size:
                rec["kl_joint"] = kl_vec(ej, pj)
        except Exception:  # noqa: BLE001
            pass
    return rec


def evaluate(instances: dict[str, dict], predictions: dict[str, dict[str, dict]],
             splits=None) -> tuple[list[dict], dict]:
    """Iterate over the dataset, not over predicted ids.

    instances: id -> instance; predictions: id -> variant -> prediction. `splits`: the splits to score; by default
    every split in which the model predicted at least one instance. Every instance of those splits is scored, and
    those without a base prediction count as fully missing. Returns (records, coverage info)."""
    if splits is None:
        splits = {instances[i].get("split", "?") for i in predictions if i in instances}
    splits = set(splits)
    recs = []
    n_missing_inst = 0
    for iid, inst in instances.items():
        if inst.get("split", "?") not in splits:
            continue
        r = evaluate_instance(inst, predictions.get(iid))
        n_missing_inst += r["pred_missing"]
        recs.append(r)
    info = {"n_pred_ids": len(predictions),
            "n_pred_without_instance": sum(i not in instances for i in predictions),
            "n_evaluated": len(recs), "n_missing_inst": n_missing_inst, "splits": sorted(splits)}
    return recs, info


# ----------------------------------------------------------------------------- aggregation
def _tie_credit(p: np.ndarray, g: int) -> float:
    top = np.flatnonzero(p >= p.max() - 1e-12)
    return 1.0 / top.size if g in top else 0.0


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else x


def aggregate(recs: list[dict]) -> dict:
    out: dict = {"n": len(recs)}
    klm = [x for r in recs for x in r["kl_marg"]]
    klb = [x for r in recs for x in r["kl_bin"]]
    out["kl"] = _mean(klm + klb)
    out["kl_marg"] = _mean(klm)
    out["kl_bin"] = _mean(klb)
    out["kl_avgperm"] = _mean([x for r in recs for x in r.get("kl_avgperm", [])])
    fams = defaultdict(list)
    for r in recs:
        fams[(r["split"], r["family"])].append(r)
    for suffix, keys in (("", ("marg", "bin")), ("_marg", ("marg",)), ("_bin", ("bin",))):
        sk = []
        for fr in fams.values():
            m = [x for r in fr for k in keys for x in r[f"kl_{k}"]]
            ref = [x for r in fr for k in keys for x in r.get(f"kl_ref_{k}", [])]
            if m and ref and np.mean(ref) > 1e-12:
                sk.append(1.0 - np.mean(m) / np.mean(ref))
        out[f"skill{suffix}"] = float(np.mean(sk)) if sk else None
    den = sum(r.get("sens_den", 0.0) for r in recs)
    out["sens"] = 1.0 - sum(r.get("sens_num", 0.0) for r in recs) / den if den > 1e-12 else None
    items = [(r["id"], np.asarray(m[:-1]), int(m[-1])) for r in recs for m in r["marg"]]
    if items:
        pg = np.array([p[g] for _, p, g in items])
        out["logscore"] = float(np.mean(np.log(np.clip(pg, EPS, 1))))
        conf = np.array([p.max() for _, p, _ in items])
        corr = np.array([_tie_credit(p, g) for _, p, g in items])
        out["acc"] = float(corr.mean())
        out["ece"] = ece_top_label(conf, corr)
    else:
        out["logscore"] = out["acc"] = out["ece"] = None
    for k in DUTCH_KEYS:
        out[k] = _mean([r.get(k) for r in recs])
    key = f"dutch@{EXPLOIT_DELTA}"
    out["dutch_frac_exploitable"] = _mean([float(r[key] > 1e-6) for r in recs if r.get(key) is not None])
    out["n_missing_inst"] = sum(bool(r.get("pred_missing")) for r in recs)
    out["coh_valid"] = None if out["skill"] is None else float(out["skill"] > SKILL_EPS)
    out["dutch_inf"] = out["dutch"] if out["coh_valid"] else None
    for k in ("optperm", "evperm", "mart", "para"):
        out[k] = _mean([x for r in recs for x in r[k]])
    out["kl_joint"] = _mean([r.get("kl_joint") for r in recs])
    for a in ALPHAS:
        c, s = conformal(items, a) if items else (None, None)
        out[f"cover@{a}"] = _nan_none(c)
        out[f"size@{a}"] = _nan_none(s)
    out["answered_frac"] = _mean([r["n_answered"] / r["n_queries"] for r in recs if r["n_queries"]])
    return out


def build_report(recs: list[dict]) -> dict:
    by_split, by_family = defaultdict(list), defaultdict(list)
    for r in recs:
        by_split[r["split"]].append(r)
        by_family[f"{r['split']}/{r['family']}"].append(r)
    return {
        "overall": aggregate(recs),
        "by_split": {k: aggregate(v) for k, v in sorted(by_split.items())},
        "by_family": {k: aggregate(v) for k, v in sorted(by_family.items())},
    }


def markdown_table(rows: list[tuple[str, dict]], columns=None, label: str = "group") -> str:
    columns = columns or ["n"] + METRIC_COLUMNS
    head = f"| {label} | " + " | ".join(columns) + " |"
    sep = "|---" * (len(columns) + 1) + "|"
    lines = [head, sep]
    for name, m in rows:
        cells = []
        for c in columns:
            v = m.get(c)
            if v is None:
                cells.append("–")
            elif c == "n":
                cells.append(str(v))
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
