"""realcoh: query generation, joint=null metrics path, para variant."""
import json
import os
import random

import numpy as np
import pytest

from bookiebench.metrics.core import build_report, evaluate
from bookiebench.metrics.exact import event_mask, own_exact_answer
from bookiebench.realcoh.build import make_queries
from bookiebench.realcoh.sources import SOURCES
from bookiebench.runners import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def toy(name="tos", seed=0, gold=None):
    _, variables, prelude = SOURCES[name]
    qs, para = make_queries(random.Random(seed), variables)
    return {"id": f"realcoh-{name}-{seed:06d}", "family": name, "split": "realcoh",
            "variables": [{"name": v["name"], "options": v["options"]} for v in variables],
            "steps": [{"evidence": "some clause text", "joint": None}], "prelude": prelude,
            "gold": gold or {}, "queries": qs, "paraphrases": para}


def answers_from_joint(inst, J):
    """Coherent prices: every answer computed from one joint J."""
    tmp = dict(inst, steps=[{"evidence": "", "joint": J.tolist()}])
    return {q["id"]: own_exact_answer(tmp, q) for q in inst["queries"]}


@pytest.mark.parametrize("name", list(SOURCES))
def test_query_set(name):
    inst = toy(name)
    kinds = [q["kind"] for q in inst["queries"]]
    assert kinds.count("marginal") == len(inst["variables"])
    assert sum(1 for q in inst["queries"] if q["kind"] == "noul" and not q.get("neg")) >= 2
    assert any(q.get("neg") for q in inst["queries"]) and kinds.count("cond") >= 1
    for q in inst["queries"]:
        assert q["text"].endswith("?") and "{" not in q["text"]
        if q["kind"] != "marginal":
            assert event_mask(inst, q["event"]).any()
    assert all(len(v) >= 1 for v in inst["paraphrases"].values())
    shape = [len(v["options"]) for v in inst["variables"]]
    assert np.prod(shape) <= 256 and 2 <= len(shape) <= 4


def test_coherent_prices_no_dutch_book_and_no_kl():
    inst = toy(gold={"fairness": 1})
    rng = np.random.default_rng(0)
    J = rng.dirichlet(np.ones(int(np.prod([len(v["options"]) for v in inst["variables"]])))).reshape(
        [len(v["options"]) for v in inst["variables"]])
    ans = answers_from_joint(inst, J)
    preds = {inst["id"]: {"base": {"answers": ans}, "para:0": {"answers": {k: ans[k] for k in inst["paraphrases"]}}}}
    recs, _ = evaluate({inst["id"]: inst}, preds)
    o = build_report(recs)["overall"]
    assert o["kl"] is None and o["mart"] is None
    assert o["dutch"] < 1e-7 and o["para"] < 1e-12
    assert o["acc"] is not None and o["logscore"] is not None


def test_incoherent_prices_are_booked_and_para_measured():
    inst = toy()
    ans = {q["id"]: ([0.9] if q["kind"] != "marginal" else list(np.ones(len(q["options"])) / len(q["options"])))
           for q in inst["queries"]}
    alt = {k: list(np.eye(len(inst["queries"][int(k[1:])]["options"]))[0]) for k in inst["paraphrases"]}
    recs, _ = evaluate({inst["id"]: inst}, {inst["id"]: {"base": {"answers": ans}, "para:0": {"answers": alt}}})
    o = build_report(recs)["overall"]
    assert o["dutch"] > 0.05 and o["para"] > 0.1
    assert o["acc"] is None                                     # no gold


def test_runner_plan_para_variant():
    inst = toy()
    P = core.plan_instances([inst])
    vs = {v for _, v in P.order}
    assert {"base", "optperm:0", "optperm:1", "para:0"} <= vs and not any(v.startswith(("evperm", "next")) for v in vs)
    para = P.slots[(inst["id"], "para:0")]
    assert {q for q, _, _ in para} == set(inst["paraphrases"])
    k = para[0][1]
    assert P.rows[k].question == inst["paraphrases"][para[0][0]][0]
    assert P.rows[k].state == inst["prelude"] + "\n" + "some clause text"


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "realcoh", "tos.jsonl")), reason="data/realcoh not built")
def test_built_files():
    for name in SOURCES:
        rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "realcoh", f"{name}.jsonl"))]
        assert len(rows) == 200 and len({r["id"] for r in rows}) == 200
        assert len({r["steps"][0]["evidence"] for r in rows}) == 200
        for r in rows:
            assert r["steps"][0]["joint"] is None
            for v, g in r["gold"].items():
                assert 0 <= g < len(next(x for x in r["variables"] if x["name"] == v)["options"])
