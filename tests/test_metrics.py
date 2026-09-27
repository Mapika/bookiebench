"""Tests for bookiebench.metrics: Dutch-book LP, calibration/conformal pieces, synthetic model orderings, CLIs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bookiebench.metrics import (
    build_report, conformal, dutch_book, ece_top_label, evaluate, event_mask, exact_answer, own_exact_answer,
)
from bookiebench.metrics.dutch import Bet, max_dutch_book

ROOT = Path(__file__).resolve().parents[1]


# ============================================================================ fixtures
def _joint_instance(joint, names=("A", "B", "C"), queries=None, iid="t-0"):
    joint = np.asarray(joint, dtype=float)
    variables = [{"name": n, "options": [f"{n}{k}" for k in range(s)]} for n, s in zip(names, joint.shape)]
    return {"id": iid, "family": "toy", "split": "test", "variables": variables,
            "steps": [{"evidence": "e", "joint": joint.tolist()}], "prelude": "", "gold": {},
            "queries": queries or []}


def _rich_queries(inst):
    qs = [{"id": f"m{i}", "kind": "marginal", "var": v["name"], "options": v["options"]}
          for i, v in enumerate(inst["variables"])]
    qs += [
        {"id": "c0", "kind": "noul", "event": {"A": [0], "B": [1]}},
        {"id": "c1", "kind": "noul", "event": {"A": [1, 2], "C": [0]}},
        {"id": "n0", "kind": "noul", "event": {"A": [0]}, "neg": True},
        {"id": "n1", "kind": "noul", "event": {"B": [0], "C": [1]}, "neg": True},
        {"id": "k0", "kind": "cond", "event": {"B": [0]}, "given": {"A": [1]}},
        {"id": "k1", "kind": "cond", "event": {"C": [1]}, "given": {"A": [0, 2], "B": [1]}},
        {"id": "k2", "kind": "cond", "event": {"A": [2]}, "given": {"C": [0]}, "neg": True},
    ]
    return qs


def _exact_answers(inst):
    return {q["id"]: exact_answer(inst, q).tolist() for q in inst["queries"]}


def make_world(rng, iid, family, split, T=4, H=3):
    """Latent H with iid binary observations; variables (H, next). Exact joints, next_evidence laws, perms."""
    prior = rng.dirichlet(np.ones(H) * 2)
    L = rng.uniform(0.1, 0.9, size=H)  # P(x=1 | h)
    lik = np.stack([1 - L, L], axis=1)  # H x 2
    h_true = rng.choice(H, p=prior)
    obs = [int(rng.random() < L[h_true]) for _ in range(T)]
    next_true = int(rng.random() < L[h_true])

    def post(xs):
        p = prior.copy()
        for x in xs:
            p = p * lik[:, x]
        return p / p.sum()

    steps = []
    for k in range(T):
        pk = post(obs[: k + 1])
        step = {"evidence": f"obs {obs[k]}", "joint": (pk[:, None] * lik).tolist()}
        if k < T - 1:
            pred = pk @ lik  # law of the next observation
            step["next_evidence"] = [{"evidence": f"obs {x}", "prob": float(pred[x])} for x in (0, 1)]
        steps.append(step)
    variables = [{"name": "H", "options": [f"h{i}" for i in range(H)]}, {"name": "next", "options": ["0", "1"]}]
    queries = [
        {"id": "qH", "kind": "marginal", "var": "H", "options": variables[0]["options"]},
        {"id": "qN", "kind": "marginal", "var": "next", "options": ["0", "1"]},
        {"id": "c0", "kind": "noul", "event": {"H": [0], "next": [1]}},
        {"id": "c1", "kind": "noul", "event": {"H": [1, 2], "next": [0]}},
        {"id": "n0", "kind": "noul", "event": {"H": [0]}, "neg": True},
        {"id": "k0", "kind": "cond", "event": {"next": [1]}, "given": {"H": [1]}},
    ]
    queries += [{"id": f"s{k}", "kind": "marginal", "var": "H", "step": k, "options": variables[0]["options"]}
                for k in range(T - 1)]
    return {"id": iid, "family": family, "split": split, "variables": variables, "steps": steps, "prelude": "p",
            "gold": {"H": int(h_true), "next": next_true}, "queries": queries, "mart_var": "H",
            "perms": [list(rng.permutation(T)) for _ in range(2)], "_obs": obs, "_prior": prior, "_lik": lik}


def _uniform_inst(inst):
    u = dict(inst)
    shp = np.asarray(inst["steps"][-1]["joint"]).shape
    u["steps"] = [{"joint": (np.ones(shp) / np.prod(shp)).tolist()} for _ in inst["steps"]]
    return u


def predictions_for(inst, model, rng, n_optperm=2):
    """Prediction lines for one world under 'oracle' | 'uniform' | 'noisy'."""
    ref = _uniform_inst(inst) if model == "uniform" else inst

    def answer(q, prefix_post=None, noise=True):
        if prefix_post is not None:
            a = np.asarray(prefix_post)
        else:
            a = exact_answer(ref, q)
        if model == "noisy" and noise:
            if q["kind"] == "marginal":
                z = np.log(np.clip(a, 1e-9, None)) + rng.normal(0, 0.6, size=a.size)
                a = np.exp(z) / np.exp(z).sum()  # each vector sums to 1, but queries are mutually inconsistent
            else:
                a = np.clip(a + rng.normal(0, 0.15, size=1), 0.01, 0.99)
        return [float(x) for x in np.atleast_1d(a)]

    lines = [{"id": inst["id"], "model": model, "variant": "base",
              "answers": {q["id"]: answer(q) for q in inst["queries"]}}]
    for s in range(n_optperm):
        lines.append({"id": inst["id"], "model": model, "variant": f"optperm:{s}",
                      "answers": {q["id"]: answer(q) for q in inst["queries"]}})
    for i in range(len(inst["perms"])):
        lines.append({"id": inst["id"], "model": model, "variant": f"evperm:{i}",
                      "answers": {q["id"]: answer(q) for q in inst["queries"] if "step" not in q}})
    # martingale re-asks: H posterior after prefix k + alternative j
    prior, lik, obs = inst["_prior"], inst["_lik"], inst["_obs"]
    for k in range(len(inst["steps"]) - 1):
        for j in (0, 1):
            p = prior.copy()
            for x in obs[: k + 1] + [j]:
                p = p * lik[:, x]
            p = p / p.sum()
            if model == "uniform":
                p = np.ones_like(p) / p.size
            q = next(q for q in inst["queries"] if q["id"] == f"s{k}")
            lines.append({"id": inst["id"], "model": model, "variant": f"next:{k}:{j}",
                          "answers": {q["id"]: answer(q, prefix_post=p)}})
    return lines


def _world_set(n=240, seed=0):
    rng = np.random.default_rng(seed)
    worlds = []
    for i in range(n):
        fam, split = [("urnA", "test"), ("urnB", "test"), ("urnC", "heldout")][i % 3]
        worlds.append(make_world(rng, f"{fam}-{i:05d}", fam, split))
    return worlds


def _eval_model(worlds, model, seed=1):
    rng = np.random.default_rng(seed)
    preds = {}
    for w in worlds:
        for line in predictions_for(w, model, rng):
            preds.setdefault(line["id"], {})[line["variant"]] = line
    recs, _ = evaluate({w["id"]: w for w in worlds}, preds)
    return build_report(recs)


@pytest.fixture(scope="module")
def worlds():
    return _world_set()


@pytest.fixture(scope="module")
def reports(worlds):
    return {m: _eval_model(worlds, m) for m in ("oracle", "uniform", "noisy")}


# ============================================================================ exact answers
def test_own_exact_answer_matches_brute_force():
    rng = np.random.default_rng(3)
    j = rng.dirichlet(np.ones(12)).reshape(3, 2, 2)
    inst = _joint_instance(j)
    inst["queries"] = _rich_queries(inst)
    assert np.allclose(own_exact_answer(inst, inst["queries"][0]), j.sum(axis=(1, 2)))
    assert np.isclose(own_exact_answer(inst, {"kind": "noul", "event": {"A": [0], "B": [1]}})[0], j[0, 1].sum())
    assert np.isclose(own_exact_answer(inst, {"kind": "noul", "event": {"A": [0]}, "neg": True})[0], 1 - j[0].sum())
    p = own_exact_answer(inst, {"kind": "cond", "event": {"B": [0]}, "given": {"A": [1]}})[0]
    assert np.isclose(p, j[1, 0].sum() / j[1].sum())
    assert event_mask(inst, {"A": [0, 2], "C": [1]}).sum() == 2 * 2 * 1


# ============================================================================ Dutch book
@pytest.mark.parametrize("seed", range(25))
def test_dutch_zero_for_exact_prices_from_any_joint(seed):
    rng = np.random.default_rng(seed)
    shape = [(3, 2, 2), (4, 4, 2), (3, 2, 3), (3, 3, 3)][seed % 4]
    alpha = [0.05, 0.3, 1.0, 5.0][seed % 4]
    j = rng.dirichlet(np.ones(int(np.prod(shape))) * alpha).reshape(shape)
    if seed % 5 == 0:
        j.ravel()[rng.choice(j.size, size=j.size // 3, replace=False)] = 0  # zeros in the joint
        j /= j.sum()
    inst = _joint_instance(j)
    inst["queries"] = [q for q in _rich_queries(inst) if np.isfinite(exact_answer(inst, q)[0])]
    r = dutch_book(inst, _exact_answers(inst))
    assert r["n_queries"] == len(inst["queries"])
    assert r["dutch"] < 1e-7


def test_dutch_negation_sum_1_2_analytic():
    inst = _joint_instance(np.full((2, 2), 0.25), names=("A", "B"))
    inst["queries"] = [{"id": "a", "kind": "noul", "event": {"A": [0]}},
                       {"id": "na", "kind": "noul", "event": {"A": [0]}, "neg": True}]
    r = dutch_book(inst, {"a": [0.7], "na": [0.5]})
    # sell both unit bets: receive 1.2, pay exactly 1 in every outcome
    assert r["dutch"] == pytest.approx(0.2, abs=1e-9)
    assert r["dutch_norm"] == pytest.approx(0.1, abs=1e-9)
    # coherent pair -> 0
    assert dutch_book(inst, {"a": [0.7], "na": [0.3]})["dutch"] < 1e-9


def test_dutch_marginal_vector_not_summing_to_one():
    inst = _joint_instance(np.full((3, 2), 1 / 6), names=("A", "B"))
    inst["queries"] = [{"id": "m", "kind": "marginal", "var": "A"}]
    assert dutch_book(inst, {"m": [0.5, 0.5, 0.3]})["dutch"] == pytest.approx(0.3, abs=1e-9)
    assert dutch_book(inst, {"m": [0.2, 0.3, 0.3]})["dutch"] == pytest.approx(0.2, abs=1e-9)
    assert dutch_book(inst, {"m": [0.2, 0.5, 0.3]})["dutch"] < 1e-9


def test_dutch_conjunction_above_conjunct_exploitable():
    inst = _joint_instance(np.full((2, 2), 0.25), names=("A", "B"))
    inst["queries"] = [{"id": "a", "kind": "noul", "event": {"A": [0]}},
                       {"id": "ab", "kind": "noul", "event": {"A": [0], "B": [0]}}]
    r = dutch_book(inst, {"a": [0.3], "ab": [0.5]})
    # sell A&B at .5, buy A at .3: profit .2 + 1_A - 1_{AB} >= .2
    assert r["dutch"] == pytest.approx(0.2, abs=1e-9)
    assert dutch_book(inst, {"a": [0.5], "ab": [0.3]})["dutch"] < 1e-9


def test_dutch_inconsistent_conditional_exploitable():
    inst = _joint_instance(np.full((2, 2), 0.25), names=("A", "B"))
    inst["queries"] = [{"id": "mA", "kind": "marginal", "var": "A"},
                       {"id": "mB", "kind": "marginal", "var": "B"},
                       {"id": "ab", "kind": "noul", "event": {"A": [0], "B": [0]}},
                       {"id": "a_b", "kind": "cond", "event": {"A": [0]}, "given": {"B": [0]}}]
    coherent = {"mA": [0.5, 0.5], "mB": [0.5, 0.5], "ab": [0.25], "a_b": [0.5]}
    assert dutch_book(inst, coherent)["dutch"] < 1e-9
    bad = dict(coherent, a_b=[0.9])  # marginals and conjunction coherent, P(A|B) should be .25/.5 = .5
    r = dutch_book(inst, bad)
    assert r["dutch"] > 1e-3
    # without the conjunction query, P(A|B)=.9 is still extendable to a joint -> not exploitable (de Finetti sanity).
    # (Behaviour change, phase 2: merely *omitting* the answer to "ab" no longer works -- the missing price is
    # adversarial -- so the query itself is removed from the instance here.)
    no_conj_inst = dict(inst, queries=[q for q in inst["queries"] if q["id"] != "ab"])
    no_conj = {k: v for k, v in bad.items() if k != "ab"}
    assert dutch_book(no_conj_inst, no_conj)["dutch"] < 1e-9
    assert dutch_book(inst, no_conj)["dutch"] > 0.1


def test_dutch_conditional_called_off_analytic():
    # prices P(A|B)=.9, P(A&B)=.1, P(B)=.5: coherence requires P(A|B)=.1/.5=.2, so the book is exploitable
    W = [Bet(np.array([1, 0, 1, 0], bool), np.array([1, 1, 0, 0], bool), 0.9),  # A|B, B = cells 0,1; A = cells 0,2
         Bet(np.array([1, 0, 0, 0], bool), np.ones(4, bool), 0.1),              # A&B
         Bet(np.array([1, 1, 0, 0], bool), np.ones(4, bool), 0.5)]              # B
    profit, s = max_dutch_book(W)
    assert profit > 0.05
    # guaranteed profit really is attained in every outcome
    G = np.stack([b.given for b in W]).astype(float)
    E = np.stack([b.event for b in W]).astype(float)
    pay = (np.array([b.price for b in W])[:, None] * G - E * G).T @ s
    assert pay.min() == pytest.approx(profit, abs=1e-7)


def test_dutch_skips_non_final_step_queries():
    inst = _joint_instance(np.full((2, 2), 0.25), names=("A", "B"))
    inst["steps"] = [dict(inst["steps"][0]), dict(inst["steps"][0])]
    inst["queries"] = [{"id": "a0", "kind": "marginal", "var": "A", "step": 0},
                       {"id": "a1", "kind": "marginal", "var": "A"}]
    r = dutch_book(inst, {"a0": [0.9, 0.9], "a1": [0.5, 0.5]})
    assert r["dutch"] < 1e-9 and r["n_queries"] == 1


# ============================================================================ calibration / conformal pieces
def test_ece_hand_computed():
    conf = np.array([0.95, 0.95, 0.55, 0.55])
    corr = np.array([1, 1, 1, 0])
    # bin of .95: |1-.95|=.05 (weight .5); bin of .55: |.5-.55|=.05 (weight .5)
    assert ece_top_label(conf, corr) == pytest.approx(0.05)


def test_conformal_uniform_scores_nominal_coverage():
    rng = np.random.default_rng(0)
    items = []
    for i in range(4000):
        p = rng.dirichlet(np.ones(4))
        items.append((f"x{i}", p, int(rng.choice(4, p=p))))
    for a in (0.1, 0.2):
        cov, size = conformal(items, a)
        assert abs(cov - (1 - a)) < 0.03
        assert 1 <= size <= 4


# ============================================================================ synthetic model orderings
def test_oracle_is_perfect(reports):
    o = reports["oracle"]["overall"]
    assert o["n"] == 240
    assert o["kl"] < 1e-9
    assert o["dutch"] < 1e-7
    assert o["optperm"] < 1e-12 and o["evperm"] < 1e-12
    assert o["mart"] < 1e-9
    assert o["answered_frac"] == 1.0
    for a in (0.1, 0.2):
        assert abs(o[f"cover@{a}"] - (1 - a)) < 0.1


def test_uniform_coherent_but_uninformative(reports):
    o, u = reports["oracle"]["overall"], reports["uniform"]["overall"]
    assert u["dutch"] < 1e-7
    assert u["optperm"] < 1e-12 and u["mart"] < 1e-12
    assert u["logscore"] < o["logscore"]
    assert u["kl"] > o["kl"] + 0.01
    assert u["size@0.1"] > o["size@0.1"]


def test_noisy_incoherent_is_exploitable(reports):
    o, n = reports["oracle"]["overall"], reports["noisy"]["overall"]
    assert n["dutch"] > 0.01 and n["dutch_frac_exploitable"] > 0.9
    assert n["dutch_norm"] == pytest.approx(n["dutch_norm"]) and n["dutch_norm"] > 0
    assert n["kl"] > o["kl"] + 0.01
    assert n["optperm"] > 0.01 and n["evperm"] > 0.01 and n["mart"] > 0.01
    assert n["logscore"] < o["logscore"]


def test_breakdowns(reports):
    r = reports["oracle"]
    assert set(r["by_split"]) == {"test", "heldout"}
    assert set(r["by_family"]) == {"test/urnA", "test/urnB", "heldout/urnC"}
    assert sum(v["n"] for v in r["by_family"].values()) == r["overall"]["n"]
    assert r["by_split"]["heldout"]["n"] == 80


# ============================================================================ CLIs
def _write_fixture_tree(tmp_path, worlds):
    data = tmp_path / "data"
    for w in worlds:
        d = data / w["split"]
        d.mkdir(parents=True, exist_ok=True)
        clean = {k: v for k, v in w.items() if not k.startswith("_")}
        clean["perms"] = [[int(x) for x in p] for p in clean["perms"]]
        with open(d / f"{w['family']}.jsonl", "a") as f:
            f.write(json.dumps(clean) + "\n")
    rng = np.random.default_rng(5)
    for model in ("oracle", "noisy"):
        out = tmp_path / "results" / model
        out.mkdir(parents=True)
        for w in worlds:
            with open(out / f"{w['split']}__{w['family']}.jsonl", "a") as f:
                for line in predictions_for(w, model, rng):
                    f.write(json.dumps(line) + "\n")
    return data, tmp_path / "results"


def test_report_and_compare_cli(tmp_path, worlds):
    data, results = _write_fixture_tree(tmp_path, worlds[:60])
    py = sys.executable
    for model in ("oracle", "noisy"):
        p = subprocess.run([py, "-m", "bookiebench.metrics.report", str(results / model), "--data", str(data)],
                           cwd=ROOT, capture_output=True, text=True, check=True)
        assert "| overall |" in p.stdout and "split:heldout" in p.stdout and "heldout/urnC" in p.stdout
        rep = json.loads((results / model / "report.json").read_text())
        assert rep["overall"]["n"] == 60
    assert json.loads((results / "oracle" / "report.json").read_text())["overall"]["dutch"] < 1e-7
    p = subprocess.run([py, "-m", "bookiebench.metrics.compare", *map(str, sorted(results.iterdir()))],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    assert "| oracle |" in p.stdout and "| noisy |" in p.stdout and "## split: heldout" in p.stdout


# ============================================================================ phase 2: gaming regressions
# Scenarios from review/exactness/dutch/probe_gaming.py, run through the full pipeline on the synthetic worlds.
def _scenario_report(worlds, f):
    recs, _ = evaluate({w["id"]: w for w in worlds}, {w["id"]: {"base": {"answers": f(w)}} for w in worlds})
    return build_report(recs)["overall"]


def _exact_marg_nan_rest(w):
    return {q["id"]: (exact_answer(w, q).tolist() if q["kind"] == "marginal" else [float("nan")])
            for q in w["queries"]}


SCENARIOS = {
    "exact": lambda w: _exact_answers(w),
    "marg_nan_rest": _exact_marg_nan_rest,
    "marg_omit_rest": lambda w: {q["id"]: exact_answer(w, q).tolist() for q in w["queries"] if q["kind"] == "marginal"},
    "all_nan": lambda w: {q["id"]: [float("nan")] for q in w["queries"]},
    "wrong_len_nan": lambda w: {q["id"]: ([0.3] if q["kind"] == "marginal" else [float("nan")]) for q in w["queries"]},
    "half": lambda w: {q["id"]: ([0.5] * len(q["options"]) if q["kind"] == "marginal" else [0.5]) for q in w["queries"]},
    "uniform": lambda w: _exact_answers(_uniform_inst(w)),
}


@pytest.fixture(scope="module")
def scen(worlds):
    ws = worlds[:90]
    return {k: _scenario_report(ws, f) for k, f in SCENARIOS.items()}


def test_gaming_exact_is_clean(scen):
    e = scen["exact"]
    assert e["kl"] < 1e-9 and e["dutch"] < 1e-7 and e["answered_frac"] == 1.0
    assert e["dutch_frac_exploitable"] == 0.0


@pytest.mark.parametrize("name", ["marg_nan_rest", "marg_omit_rest"])
def test_gaming_omission_does_not_help(scen, name):
    r, e, u = scen[name], scen["exact"], scen["uniform"]
    assert r["answered_frac"] < 1.0
    assert r["dutch_ans"] < 1e-7  # the old, gameable number: omission looked perfectly coherent
    assert r["dutch"] > 0.5 and r["dutch_frac_exploitable"] == 1.0
    assert r["kl"] > e["kl"] + 0.01  # omitted noul/cond scored as 0.5
    assert r["dutch"] > u["dutch"] and r["dutch"] > scen["half"]["dutch"]


def test_gaming_nan_equals_omission(scen):
    for k in ("dutch", "kl", "answered_frac", "logscore"):
        assert scen["marg_nan_rest"][k] == pytest.approx(scen["marg_omit_rest"][k])


def test_gaming_all_missing_is_worst(scen):
    a, w = scen["all_nan"], scen["wrong_len_nan"]
    assert a["answered_frac"] == 0.0 and w["answered_frac"] == 0.0
    assert a["dutch"] == pytest.approx(w["dutch"]) and a["kl"] == pytest.approx(w["kl"])
    assert a["dutch"] >= max(v["dutch"] for v in scen.values()) - 1e-9
    # missing is scored exactly like the uniform answer for kl / logscore / acc
    assert a["kl"] == pytest.approx(scen["half"]["kl"])
    assert a["logscore"] == pytest.approx(scen["half"]["logscore"])
    assert a["acc"] == pytest.approx(scen["half"]["acc"])


def test_gaming_half_everywhere_and_uniform(scen):
    h, u = scen["half"], scen["uniform"]
    assert h["answered_frac"] == 1.0 and h["dutch"] > 0.1  # 0.5 on a 3-way marginal sums to 1.5
    assert u["dutch"] < 1e-7 and u["kl"] > 0.01
    # tie credit: a uniform 3-way / 2-way answer earns 1/K accuracy, not an index-0 hit
    assert u["acc"] == pytest.approx(np.mean([1 / 3, 1 / 2]), abs=1e-9)


def test_omission_weakly_dominated_by_any_answer():
    rng = np.random.default_rng(11)
    for seed in range(15):
        j = np.random.default_rng(seed).dirichlet(np.ones(12)).reshape(3, 2, 2)
        inst = _joint_instance(j)
        inst["queries"] = _rich_queries(inst)
        ans = {k: [float(np.clip(x + rng.normal(0, 0.1), 0, 1)) for x in v] for k, v in _exact_answers(inst).items()}
        drop = inst["queries"][rng.integers(len(inst["queries"]))]["id"]
        omitted = {k: v for k, v in ans.items() if k != drop}
        d_omit = dutch_book(inst, omitted)["dutch"]
        for _ in range(5):
            alt = dict(omitted)
            alt[drop] = list(rng.dirichlet(np.ones(len(ans[drop])))) if len(ans[drop]) > 1 else [float(rng.random())]
            assert d_omit >= dutch_book(inst, alt)["dutch"] - 1e-7


def test_missing_instances_count(worlds):
    ws = worlds[:30]
    preds = {w["id"]: {"base": {"answers": _exact_answers(w)}} for w in ws[:20]}
    recs, info = evaluate({w["id"]: w for w in ws}, preds)
    o = build_report(recs)["overall"]
    assert info["n_evaluated"] == 30 and info["n_missing_inst"] == 10 and o["n_missing_inst"] == 10
    assert o["answered_frac"] == pytest.approx(20 / 30)
    assert o["dutch"] > 0.1 and o["kl"] > 0.01


# ============================================================================ phase 2: normalisation, tolerance, dedup
def test_dutch_per_bet_and_tolerance():
    inst = _joint_instance(np.full((2, 2), 0.25), names=("A", "B"))
    inst["queries"] = [{"id": "a", "kind": "noul", "event": {"A": [0]}},
                       {"id": "na", "kind": "noul", "event": {"A": [0]}, "neg": True},
                       {"id": "m", "kind": "marginal", "var": "B"}]
    r = dutch_book(inst, {"a": [0.7], "na": [0.5], "m": [0.5, 0.5]})
    assert r["n_bets"] == 4 and r["n_queries"] == 3
    assert r["dutch_per_bet"] == pytest.approx(0.2 / 4)
    assert r["dutch@0.005"] == pytest.approx(0.19) and r["dutch@0.01"] == pytest.approx(0.18)
    assert dutch_book(inst, {"a": [0.7], "na": [0.3], "m": [0.5, 0.5]})["dutch@0.01"] == 0.0


def test_rounding_absorbed_by_tolerance(worlds):
    rounded = 0
    for w in worlds[:60]:
        ans = {k: [round(x, 2) for x in v] for k, v in _exact_answers(w).items()}
        r = dutch_book(w, ans)
        rounded += r["dutch"] > 1e-9
        assert r["dutch@0.005"] < 1e-9
    assert rounded > 0  # raw dutch does flag 2-decimal rounding; the tolerant variant does not


def test_paraphrase_dup_and_structural_duplicates_excluded():
    inst = _joint_instance(np.full((2, 2), 0.25), names=("A", "B"))
    inst["steps"] = [dict(inst["steps"][0]), dict(inst["steps"][0])]
    inst["queries"] = [{"id": "a", "kind": "noul", "event": {"A": [0]}},
                       {"id": "na", "kind": "noul", "event": {"A": [0]}, "neg": True},
                       {"id": "a2", "kind": "noul", "event": {"A": [0]}, "rel": "paraphrase-dup"},
                       {"id": "m", "kind": "marginal", "var": "B"},
                       {"id": "m_final", "kind": "marginal", "var": "B", "step": 1}]  # final-step mart duplicate
    ans = {"a": [0.7], "na": [0.5], "a2": [0.7], "m": [0.5, 0.5], "m_final": [0.5, 0.5]}
    r = dutch_book(inst, ans)
    assert r["n_queries"] == 3 and r["dutch"] == pytest.approx(0.2)  # a2 would double the stake budget to 0.4
    # a missing tagged duplicate is not penalised either
    assert dutch_book(inst, {k: v for k, v in ans.items() if k != "a2"})["dutch"] == pytest.approx(0.2)


# ============================================================================ phase 3: skill, refs, sens, kl_avgperm
def test_skill_and_informative_coherence(reports):
    o, u, n = (reports[m]["overall"] for m in ("oracle", "uniform", "noisy"))
    assert o["skill"] == pytest.approx(1.0) and o["skill_marg"] == pytest.approx(1.0)
    assert abs(u["skill"]) < 1e-9  # the uniform fixture model *is* uniform_joint
    assert 0 < n["skill"] < 1
    assert o["coh_valid"] == 1.0 and u["coh_valid"] == 0.0
    assert o["dutch_inf"] == pytest.approx(o["dutch"]) and u["dutch_inf"] is None
    assert o["kl_marg"] is not None and o["kl_bin"] is not None
    for fam in reports["oracle"]["by_family"].values():
        assert fam["skill"] == pytest.approx(1.0)


def test_sens(reports):
    o, u, n = (reports[m]["overall"] for m in ("oracle", "uniform", "noisy"))
    assert o["sens"] == pytest.approx(1.0)
    assert u["sens"] == pytest.approx(0.0)
    assert n["sens"] < 0.9


def test_kl_avgperm(reports):
    o, n = reports["oracle"]["overall"], reports["noisy"]["overall"]
    assert o["kl_avgperm"] < 1e-9
    assert n["kl_avgperm"] < n["kl"]  # averaging independent option-order draws removes part of the error


def test_kl_avgperm_removes_position_bias():
    rng = np.random.default_rng(4)
    j = rng.dirichlet(np.ones(12)).reshape(3, 2, 2)
    inst = _joint_instance(j)
    inst["queries"] = [q for q in _rich_queries(inst) if q["kind"] == "marginal"]
    ex = _exact_answers(inst)

    def biased(perm):  # +0.3 on whatever option is shown last, mapped back to canonical order
        out = {}
        for k, v in ex.items():
            v = np.array(v)
            v[perm(len(v))[-1]] += 0.3
            out[k] = (v / v.sum()).tolist()
        return out
    variants = {"base": {"answers": biased(lambda K: list(range(K)))}}
    for s in range(12):
        variants[f"optperm:{s}"] = {"answers": biased(lambda K, s=s: list(np.random.default_rng(s).permutation(K)))}
    recs, _ = evaluate({inst["id"]: inst}, {inst["id"]: variants})
    o = build_report(recs)["overall"]
    assert o["kl_avgperm"] < 0.5 * o["kl"]


def test_reference_predictors(worlds, tmp_path):
    from bookiebench.metrics.refs import fit_indep, ref_predictions
    ws = {w["id"]: w for w in worlds[:60]}
    train = tmp_path / "train"
    train.mkdir()
    with open(train / "urn.jsonl", "w") as f:
        for w in worlds[60:]:
            f.write(json.dumps(dict({k: v for k, v in w.items() if not k.startswith("_")}, perms=[])) + "\n")
    table = fit_indep(train)
    assert fit_indep(tmp_path / "nope") is None
    for name in ("uniform_joint", "indep_joint"):
        recs, _ = evaluate(ws, ref_predictions(ws, name, table))
        o = build_report(recs)["overall"]
        assert o["answered_frac"] == 1.0 and o["dutch"] < 1e-7  # coherent by construction ...
        assert o["sens"] == pytest.approx(0.0)  # ... and evidence-blind
        if name == "uniform_joint":
            assert abs(o["skill"]) < 1e-9 and o["coh_valid"] == 0.0
        else:
            assert o["skill"] < 0.5


def test_compare_includes_reference_rows(tmp_path, worlds):
    data, results = _write_fixture_tree(tmp_path, worlds[:45])
    train = tmp_path / "train"
    train.mkdir()
    with open(train / "urnA.jsonl", "w") as f:
        for w in worlds[100:160]:
            f.write(json.dumps(dict({k: v for k, v in w.items() if not k.startswith("_")}, perms=[])) + "\n")
    py = sys.executable
    p = subprocess.run([py, "-m", "bookiebench.metrics.compare", str(results / "oracle"), str(results / "noisy"),
                        "--data", str(data), "--train", str(train), "--ref", str(results / "noisy")],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    out = p.stdout
    assert "| ref:uniform_joint |" in out and "| ref:indep_joint |" in out and "| xref:noisy |" in out
    assert "informative coherence" in out
    inf = out.split("informative coherence, overall")[1].split("##")[0]
    assert "| oracle |" in inf and "ref:uniform_joint" not in inf
    p = subprocess.run([py, "-m", "bookiebench.metrics.compare", str(results / "oracle"), "--data", str(data),
                        "--train", str(tmp_path / "missing")], cwd=ROOT, capture_output=True, text=True, check=True)
    assert "ref:indep_joint, ref:prior and skill_prior skipped" in p.stdout  # phase 4 wording


# ============================================================================ phase 4: text-keyed refs, prior, gate, groups
COLOURS = ["red", "green", "blue"]


def _shuffled_world(rng, iid, split="test", family="skew"):
    """Two variables whose options are shuffled per instance (like meta.option_perm); the colour law is skewed by
    option TEXT (red >> green >> blue), so a position-keyed prior washes out while a text-keyed one does not."""
    pc = rng.dirichlet(np.array([14.0, 4.0, 2.0]))  # by text
    ps = rng.dirichlet([3.0, 3.0])
    perm = rng.permutation(3)
    opts = [COLOURS[i] for i in perm]
    j = np.outer(pc[perm], ps)
    red = opts.index("red")
    return {"id": iid, "family": family, "split": split, "meta": {"option_perm": perm.tolist()},
            "variables": [{"name": "colour", "options": opts}, {"name": "size", "options": ["small", "large"]}],
            "steps": [{"evidence": "e", "joint": j.tolist()}], "prelude": "",
            "gold": {"colour": int(rng.choice(3, p=pc[perm])), "size": int(rng.choice(2, p=ps))},
            "queries": [{"id": "q0", "kind": "marginal", "var": "colour", "options": opts, "text": "Which colour is it?"},
                        {"id": "q1", "kind": "marginal", "var": "size", "options": ["small", "large"],
                         "text": "Which size is it?"},
                        {"id": "q2", "kind": "noul", "event": {"colour": [red], "size": [0]},
                         "text": "Is it red and small?"},
                        {"id": "q3", "kind": "noul", "event": {"colour": [red]}, "neg": True,
                         "text": "Is it not red?"}]}


@pytest.fixture(scope="module")
def shuffled(tmp_path_factory):
    rng = np.random.default_rng(21)
    root = tmp_path_factory.mktemp("shuf")
    train = root / "train"
    train.mkdir()
    with open(train / "skew.jsonl", "w") as f:
        for i in range(400):
            f.write(json.dumps(_shuffled_world(rng, f"skew-train-{i}", "train")) + "\n")
    evals = {w["id"]: w for w in (_shuffled_world(rng, f"skew-test-{i}") for i in range(200))}
    return train, evals


def _ref_overall(evals, name, table, prior=None, groups=None):
    from bookiebench.metrics.refs import ref_predictions
    recs, _ = evaluate(evals, ref_predictions(evals, name, table, groups), prior=prior, groups=groups)
    return build_report(recs)["overall"]


def test_indep_joint_keyed_by_option_text(shuffled):
    from bookiebench.metrics.refs import IndepTable, fit_indep
    train, evals = shuffled
    t = fit_indep(train)
    m = t.marginal("skew", {"name": "colour", "options": ["blue", "red", "green"]})
    assert m[1] > 0.55 and m[0] < 0.2  # follows the text, whatever its position
    o = _ref_overall(evals, "indep_joint", t)
    assert o["skill"] > 0.2 and o["dutch"] < 1e-7
    # the old position-keyed table: average marginal per position is ~flat under shuffling -> no skill
    pos = IndepTable()
    pos.fam = {("skew", "colour", c, 3): 1 / 3 for c in COLOURS}
    assert _ref_overall(evals, "indep_joint", pos)["skill"] < o["skill"] - 0.2


def test_prior_refs_and_skill_prior(shuffled):
    from bookiebench.metrics.refs import fit_prior
    train, evals = shuffled
    T = fit_prior(train, cache=False)
    lab = _ref_overall(evals, "label_prior", T, prior=T)
    tmp = _ref_overall(evals, "template_uj", T, prior=T)
    uj = _ref_overall(evals, "uniform_joint", None, prior=T)
    assert lab["skill_marg"] > 0.2 and tmp["skill"] > 0
    # skill_prior is measured against the best of uniform_joint / label / template, so the best prior scores ~0
    assert max(lab["skill_prior"], tmp["skill_prior"]) == pytest.approx(0.0, abs=1e-9)
    assert uj["skill_prior"] < 0 and uj["skill"] == pytest.approx(0.0)
    # exact answers: skill_prior = 1; noisy answers: skill_prior < skill
    recs, _ = evaluate(evals, {i: {"base": {"answers": _exact_answers(w)}} for i, w in evals.items()}, prior=T)
    assert build_report(recs)["overall"]["skill_prior"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noisy = {i: {"base": {"answers": {k: list(np.clip(np.array(v) + rng.normal(0, .05, len(v)), .01, .99))
                                      for k, v in _exact_answers(w).items()}}} for i, w in evals.items()}
    o = build_report(evaluate(evals, noisy, prior=T)[0])["overall"]
    assert o["skill_prior"] < o["skill"]
    # new_mechanics: the prior falls back to uniform_joint, so skill_prior == skill
    groups = {"test/skew": "new_mechanics"}
    o2 = build_report(evaluate(evals, noisy, prior=T, groups=groups)[0])["overall"]
    assert o2["skill_prior"] == pytest.approx(o2["skill"])
    fb = _ref_overall(evals, "label_prior", T, prior=T, groups=groups)
    assert fb["skill"] == pytest.approx(0.0)


def test_coherence_gate_needs_skill_and_sens(worlds):
    from bookiebench.metrics import aggregate
    from bookiebench.metrics.core import evaluate_instance
    ws = worlds[:60]
    # exact final-step answers but the step-k martingale marginals frozen at the final answer: skill high, sens ~0
    recs = []
    for w in ws:
        ans = _exact_answers(w)
        fin = ans["qH"]
        for q in w["queries"]:
            if q.get("step") is not None:
                ans[q["id"]] = fin
        recs.append(evaluate_instance(w, {"base": {"answers": ans}}))
    o = aggregate(recs)
    assert o["skill"] > 0.3 and o["sens"] < 0.05
    assert o["coh_valid"] == 0.0 and o["dutch_inf"] is None
    lax = aggregate(recs, skill_gate=0.05, sens_gate=-10)
    assert lax["coh_valid"] == 1.0 and lax["dutch_inf"] == pytest.approx(lax["dutch"])
    strict = aggregate(recs, skill_gate=0.999, sens_gate=-10)
    assert strict["coh_valid"] == 0.0


def test_groups_from_manifest(tmp_path, worlds):
    data, results = _write_fixture_tree(tmp_path, worlds[:45])
    man = {"groups": {}, "files": {
        "test/urnA.jsonl": {"split": "test", "family": "urnA", "group": "in_family"},
        "test/urnB.jsonl": {"split": "test", "family": "urnB", "group": "in_family"},
        "heldout/urnC.jsonl": {"split": "heldout", "family": "urnC", "group": "surface_transfer"}}}
    (data / "sims_manifest.json").write_text(json.dumps(man))
    from bookiebench.metrics.loading import load_groups
    assert load_groups(data)["heldout/urnC"] == "surface_transfer"
    from bookiebench.metrics.report import run as report_run
    rep = report_run(results / "oracle", data, write=False, train=tmp_path / "none")
    assert set(rep["by_group"]) == {"in_family", "surface_transfer"}
    assert rep["by_group"]["in_family"]["n"] == 30
    p = subprocess.run([sys.executable, "-m", "bookiebench.metrics.compare", str(results / "oracle"), str(results / "noisy"),
                        "--data", str(data), "--train", str(tmp_path / "none"), "--recompute"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    assert "## group: in_family" in p.stdout and "## group: surface_transfer" in p.stdout
    assert "informative coherence, group: in_family" in p.stdout


def test_procedural_families_fall_back_to_uniform(shuffled):
    """randbn / randhmm option words have random meanings: every train-fitted ref must answer uniform_joint there."""
    from bookiebench.metrics.refs import PRIOR_FALLBACK_FAMILIES, fit_indep, fit_prior
    assert {"randbn", "randhmm"} <= PRIOR_FALLBACK_FAMILIES
    train, evals = shuffled
    # relabel the skewed family as a procedural one, in train and eval alike
    tr2 = train.parent / "train_proc"
    tr2.mkdir(exist_ok=True)
    with open(tr2 / "randbn.jsonl", "w") as f:
        for line in open(train / "skew.jsonl"):
            f.write(json.dumps(dict(json.loads(line), family="randbn")) + "\n")
    ev = {i: dict(w, family="randbn") for i, w in evals.items()}
    ti, T = fit_indep(tr2), fit_prior(tr2, cache=False)
    for name, table in (("indep_joint", ti), ("label_prior", T), ("template_uj", T)):
        o = _ref_overall(ev, name, table, prior=T)
        assert o["skill"] == pytest.approx(0.0, abs=1e-12) and o["skill_prior"] == pytest.approx(0.0, abs=1e-12)
    # a model's skill_prior on procedural families is measured against uniform_joint, i.e. equals skill
    rng = np.random.default_rng(1)
    noisy = {i: {"base": {"answers": {k: list(np.clip(np.array(v) + rng.normal(0, .05, len(v)), .01, .99))
                                      for k, v in _exact_answers(w).items()}}} for i, w in ev.items()}
    o = build_report(evaluate(ev, noisy, prior=T)[0])["overall"]
    assert o["skill_prior"] == pytest.approx(o["skill"])
    # the same data under a non-procedural family name does get a real prior
    assert _ref_overall(evals, "indep_joint", fit_indep(train))["skill"] > 0.2


def test_gate_uses_skill_prior(shuffled):
    """A train-prior cheater beats uniform_joint (skill > 0.05) but not the train prior (skill_prior < 0.05): not valid."""
    from bookiebench.metrics.refs import fit_prior, prior_answers
    train, evals = shuffled
    T = fit_prior(train, cache=False)
    rng = np.random.default_rng(3)
    cheat = {}
    for i, w in evals.items():
        ans = prior_answers(w, "label_prior", T)
        cheat[i] = {"base": {"answers": {k: list(np.clip(np.array(v) + rng.normal(0, .005, len(v)), .001, .999))
                                         for k, v in ans.items()}}}
    o = build_report(evaluate(evals, cheat, prior=T)[0])["overall"]
    assert o["skill"] > 0.05 and o["skill_prior"] < 0.05
    assert o["coh_valid"] == 0.0 and o["dutch_inf"] is None
    # without prior tables the gate falls back to skill, which this cheater passes (no mart_var -> sens skipped)
    o2 = build_report(evaluate(evals, cheat)[0])["overall"]
    assert o2["skill_prior"] is None and o2["coh_valid"] == 1.0
    # a model that reads the evidence clears the skill_prior gate
    ex = {i: {"base": {"answers": _exact_answers(w)}} for i, w in evals.items()}
    assert build_report(evaluate(evals, ex, prior=T)[0])["overall"]["coh_valid"] == 1.0


def test_compare_without_manifest_and_old_reports(tmp_path, worlds):
    data, results = _write_fixture_tree(tmp_path, worlds[:30])
    assert not (data / "sims_manifest.json").exists()
    # an old-format report.json without by_group / gates / skill_prior must not break compare
    old = {"model": "noisy", "overall": {"n": 1}, "by_split": {}, "by_family": {}}
    (results / "noisy" / "report.json").write_text(json.dumps(old))
    p = subprocess.run([sys.executable, "-m", "bookiebench.metrics.compare", str(results / "oracle"), str(results / "noisy"),
                        "--data", str(data), "--train", str(tmp_path / "none")],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    assert "## group: other" in p.stdout and "| noisy |" in p.stdout
    assert "coherence gate" in p.stdout and "skill_prior >= 0.05" in p.stdout
    rep = json.loads((results / "noisy" / "report.json").read_text())
    assert set(rep["by_group"]) == {"other"} and rep["gates"]["on"] == "skill_prior"
    # render tolerates a report that still lacks by_group
    from bookiebench.metrics.compare import render
    assert "## overall" in render([old, rep])
