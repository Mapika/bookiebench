"""Runner core tests (CPU, no model): variant expansion, permutation mapping, output format per SPEC §2."""
import json
from pathlib import Path

import pytest

from bookiebench.runners import core

FIX = Path(__file__).parent / "fixtures" / "urn.jsonl"


class OracleScorer:
    """Answers from the exact joint of the instance whose rendered state matches, so correct mapping gives exact answers."""
    name = "oracle"

    def __init__(self, insts):
        self.by_state = {}
        for inst in insts:
            n = len(inst["steps"])
            for k in range(n):
                self.by_state[core.render_state(inst, upto=k)] = (inst, k)
            for order in inst.get("perms", []):
                self.by_state[core.render_state(inst, order=order)] = (inst, n - 1)
        self.q = {(inst["id"], q["text"]): q for inst in insts for q in inst["queries"]}
        self.calls = 0

    def score(self, rows):
        self.calls += 1
        out = []
        for r in rows:
            inst, k = self.by_state.get(r.state, (None, None))
            if inst is None:                                   # next:<k>:<j> state: not in the table
                out.append([1.0 / len(r.options)] * len(r.options) if r.kind == "choice" else [0.5]); continue
            q = self.q[(inst["id"], r.question)]
            ex = exact(inst, dict(q, step=k))
            if r.kind == "choice":
                opts = core.query_options(inst, q)
                out.append([ex[opts.index(o)] for o in r.options])
            else:
                out.append(ex)
        return out


def cells(inst):
    import itertools
    sizes = [len(v["options"]) for v in inst["variables"]]
    return list(itertools.product(*[range(s) for s in sizes]))


def exact(inst, q):
    """Local reference (bookiebench.sims.core.exact_answer is the canonical one)."""
    import numpy as np
    J = np.asarray(inst["steps"][core.query_step(inst, q)]["joint"], dtype=float)
    names = [v["name"] for v in inst["variables"]]

    def mass(ev):
        s = 0.0
        for c in cells(inst):
            if all(c[names.index(v)] in ok for v, ok in ev.items()):
                s += J[c]
        return s
    if q["kind"] == "marginal":
        i = names.index(q["var"]); ax = tuple(a for a in range(len(names)) if a != i)
        return list(J.sum(axis=ax))
    if q["kind"] == "noul":
        p = mass(q["event"]); return [1 - p if q.get("neg") else p]
    return [mass({**q["event"], **q["given"]}) / mass(q["given"])]


@pytest.fixture
def insts():
    return core.load_instances(FIX)


def test_fixture_joint_sums(insts):
    import numpy as np
    for inst in insts:
        for s in inst["steps"]:
            assert abs(np.sum(s["joint"]) - 1) < 1e-9


def test_variants(insts):
    P = core.plan_instances(insts)
    vs = {v for _, v in P.order}
    assert {"base", "optperm:0", "optperm:1", "evperm:0", "evperm:1", "next:0:0", "next:0:1", "next:1:1"} <= vs
    inst = insts[0]
    base = P.slots[(inst["id"], "base")]
    assert [q for q, _, _ in base] == [q["id"] for q in inst["queries"]]
    # optperm: marginal queries only, never the identity
    op = P.slots[(inst["id"], "optperm:0")]
    assert {q for q, _, _ in op} == {q["id"] for q in inst["queries"] if q["kind"] == "marginal"}
    assert all(perm != sorted(perm) for _, _, perm in op)
    # evperm: only final-step queries
    ev = {q for q, _, _ in P.slots[(inst["id"], "evperm:0")]}
    assert "m0" not in ev and "q0" in ev
    # next: the martingale query of that step
    assert [q for q, _, _ in P.slots[(inst["id"], "next:1:0")]] == ["m1"]
    k = P.slots[(inst["id"], "next:1:0")][0][1]
    assert P.rows[k].state.endswith(inst["steps"][1]["next_evidence"][0]["evidence"])
    assert P.rows[k].state.count("\n") == 3                   # prelude + 2 evidence + alternative


def test_dedup(insts):
    P = core.plan_instances(insts)
    assert len(P.rows) == len(set(P.rows)) < sum(len(v) for v in P.slots.values())


def test_oracle_roundtrip(tmp_path, insts):
    sc = OracleScorer(insts)
    st = core.run_file(sc, FIX, tmp_path)
    assert sc.calls == 1
    lines = [json.loads(l) for l in open(st["out"])]
    assert Path(st["out"]).name == "fixture__urn.jsonl"
    by = {(l["id"], l["variant"]): l for l in lines}
    for inst in insts:
        for var in ("base", "optperm:0", "optperm:1", "evperm:0", "evperm:1"):
            ans = by[(inst["id"], var)]["answers"]
            for q in inst["queries"]:
                if q["id"] in ans:
                    assert ans[q["id"]] == pytest.approx(exact(inst, q), abs=1e-9), (var, q["id"])
    l = lines[0]
    assert set(l) == {"id", "model", "variant", "answers"} and l["model"] == "oracle"
    assert all(len(v) == 1 for k, v in l["answers"].items() if k in ("q2", "q3", "q4", "q5"))


def test_seeded_perm_stable():
    assert core.seeded_perm(5, 0, "a") == core.seeded_perm(5, 0, "a")
    assert core.seeded_perm(2, 0, "a") == [1, 0]


def test_prompt_letters():
    from bookiebench.runners.logit_runner import prompt_for
    r = core.Row("S", "Q?", ("x", "y", "z"), "choice")
    assert prompt_for(r).endswith("Options:\n(A) x\n(B) y\n(C) z\nAnswer: (")
    assert "(A) yes\n(B) no" in prompt_for(core.Row("S", "Q?", ("yes", "no"), "noul"))


def test_temper_scale():
    from bookiebench.runners.temper import rescale
    assert rescale([0.5, 0.5], 2.0) == pytest.approx([0.5, 0.5])
    p = rescale([0.8, 0.2], 2.0)
    assert p[0] < 0.8 and sum(p) == pytest.approx(1)
    assert rescale([0.8], 0.5)[0] > 0.8


def test_api_letter_readout_and_json_tail():
    import math
    from bookiebench.runners.api_runner import APIScorer, _json_tail
    sc = APIScorer("http://localhost:0/v1", "m", "t")
    d = {"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": "(B", "logprob": math.log(0.5)}, {"token": "A", "logprob": math.log(0.3)},
        {"token": "The", "logprob": math.log(0.1)}, {"token": " B", "logprob": math.log(0.05)}]}]}}]}
    p = sc._letters(d, 3)                                     # C missing: leftover 1 - 0.95 = 0.05
    assert p == pytest.approx([0.3 / 0.9, 0.55 / 0.9, 0.05 / 0.9]) and sc.stats["missing_letter_rows"] == 1
    assert _json_tail('think {"x": 1} ... final: {"r0": [0.2, 0.8], "r1": 0.3}') == {"r0": [0.2, 0.8], "r1": 0.3}
    assert _json_tail("no json here") is None


def test_missing_answers_are_omitted(tmp_path):
    class Half:
        name = "half"
        def score(self, rows):
            return [None if i % 2 else ([1.0 / len(r.options)] * len(r.options) if r.kind == "choice" else [0.5])
                    for i, r in enumerate(rows)]
    st = core.run_file(Half(), FIX, tmp_path)
    lines = [json.loads(l) for l in open(st["out"])]
    assert any(len(l["answers"]) < 9 for l in lines if l["variant"] == "base")
