"""realcoh v2: linked query sets, their Dutch-book semantics, the v1 regression, and the built data/v2/realcoh files."""
import hashlib
import json
import os
import random

import numpy as np
import pytest

from bookiebench.metrics.core import build_report, evaluate
from bookiebench.metrics.dutch import bets_for_instance, dutch_book, max_dutch_book
from bookiebench.metrics.exact import event_mask, own_exact_answer
from bookiebench.realcoh.build_v2 import instance_variables
from bookiebench.realcoh.provenance import DECIDER_PRIVATE, DECIDER_PUBLIC, DROPPED, PROVENANCE, decider_references
from bookiebench.realcoh.queries_v2 import make_queries_v2
from bookiebench.realcoh.sources import PINNED
from bookiebench.realcoh.sources_v2 import SOURCES_V2
from bookiebench.runners import core

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2DIR = os.path.join(ROOT, "data", "v2", "realcoh")
HAVE_CACHE = os.path.isdir(os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))

V1_SHA = {  # data/realcoh must stay byte-identical
    "bgl_logs": "d93cd2e67a97779ec30c0d2460e4d5a0a4cb1f755ff0b7392ea332a44b0e4b17",  # label column stripped (R1)
    "climate": "21faf8c9c428190a7565021a2e257e622202063210adbce3a96077df29e1516f",
    "gold_news": "1608ba3e93a0cb7984da8e35833634c53c2d470c3e2b6cd21e60e52787a70b38",
    "news": "88bbc7a610f430e7676208b9ff1fbce9a3d971f734441442aaf4c5791705a3cc",
    "symptoms": "057d32bcde1af0c8f64c49076436b9cd0f6ce83f77356ea14d07e26b9c09eb10",
    "tos": "15bbeb3259e838f07228a8856e1d2df82e70d32aa7da26ee0f97725f0f1ad1f1",
}


def toy(name, seed=0):
    _, variables, prelude, opts = SOURCES_V2[name]
    qs, para = make_queries_v2(random.Random(seed), variables, opts.get("templates"), opts.get("forecast", False))
    return {"id": f"realcoh2-{name}-{seed:06d}", "family": name, "split": "realcoh2", "level": "R", "pack": "realcoh",
            "transform": "none", "variables": instance_variables(variables),
            "steps": [{"evidence": "some text", "joint": None}], "prelude": prelude, "gold": {}, "queries": qs,
            "paraphrases": para}


def shape(inst):
    return [len(v["options"]) for v in inst["variables"]]


def rand_joint(inst, seed=0, alpha=1.0):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(int(np.prod(shape(inst))), alpha)).reshape(shape(inst))


def answers_from_joint(inst, J):
    tmp = dict(inst, steps=[{"evidence": "", "joint": J.tolist()}])
    return {q["id"]: own_exact_answer(tmp, q) for q in inst["queries"]}


def P(J, inst, event, neg=False):
    return float(J[event_mask(inst, event, neg)].sum())


def test_registry_breadth():
    assert len(SOURCES_V2) >= 24
    assert {"med_ru", "eurlex_fr", "belebele_es"} <= set(SOURCES_V2)          # >= 3 non-English sources
    assert "forecast_news" in SOURCES_V2
    assert not set(DROPPED) & set(SOURCES_V2)                                # R6, R7, R10


def test_provenance_complete():
    assert set(PROVENANCE) == set(SOURCES_V2)
    keep = {"symptoms", "tos", "ledgar", "math_problems", "mmlu_pro", "student_answers", "eurlex_fr", "belebele_es",
            "code_defects"}                                                   # the review's licence table
    for k, p in PROVENANCE.items():
        assert p["release_mode"] == ("keep" if k in keep else "ids_only"), k
        assert isinstance(p["in_decider_train"], bool) and isinstance(p["likely_pretraining"], bool)
        assert p["license"] and p["dataset"] and p["decider_note"] and p["pretraining_note"]
    for k in ("bgl_logs", "hdfs_logs", "symptoms", "med_ru", "ledgar", "tos", "gold_news", "climate", "student_answers",
              "code_defects"):                                                # R2
        assert PROVENANCE[k]["in_decider_train"], k


@pytest.mark.skipif(not (os.path.isdir(DECIDER_PUBLIC) and os.path.isdir(DECIDER_PRIVATE)), reason="decider code bases absent")
def test_provenance_matches_decider_code():
    for k, p in PROVENANCE.items():
        hits = decider_references(k)
        if p["in_decider_train"]:
            assert hits, k
        elif k != "belebele_es":                                              # belebele: held-out eval task only
            assert not hits, (k, hits)


def test_coarse_options_name_their_members():
    for name, (_, variables, _, _) in SOURCES_V2.items():
        for v in variables:
            if v.get("coarse"):
                c = v["coarse"]
                for ci, (o, cl) in enumerate(zip(c["options"], c["clauses"])):
                    for i, m in enumerate(c["map"]):
                        if m == ci:
                            assert v["options"][i] in o and v["options"][i] in cl, (name, o, v["options"][i])


def test_review_fixes_in_specs():
    fin = dict((v["name"], v) for v in SOURCES_V2["finance_news"][1])["direction"]
    assert fin["options"][1].startswith("mixed") and "not bad" in fin["cum"][1]                # R3
    fu = dict((v["name"], v) for v in SOURCES_V2["forecast_news"][1])["followup"]
    assert not set(fu["clauses"]) & set(fu["cum"]) and "between one and seven days" in fu["clauses"][1]   # R4
    from bookiebench.realcoh.sources_v2 import _DX2
    assert _DX2["allergy"][0] == 6 and _DX2["drug reaction"][0] == 6 and _DX2["jaundice"][1] is None  # R11
    from bookiebench.realcoh.sources_v2 import answer_is_integer as z                               # R12
    assert [z(x) for x in ["4", "5\\text{ cm}", "1{,}000", "\\$25", "x=3", "\\frac{8}{4}", "2^{10}", "90^\\circ", "\\text{13}"]] == [True] * 9
    assert [z(x) for x in ["\\frac{1}{2}", "\\text{(C)}", "2\\sqrt{3}", "(1,2)", "0.5"]] == [False] * 5


def test_ssh_masking():
    from bookiebench.realcoh.sources_v2 import ssh_mask
    out = ssh_mask(["Dec 10 06:55:46 LabSZ sshd[24200]: reverse mapping checking getaddrinfo for ns.example.com [173.234.31.186] failed - POSSIBLE BREAK-IN ATTEMPT!",
                    "Dec 10 06:55:48 LabSZ sshd[24200]: Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2",
                    "Dec 10 06:55:49 LabSZ sshd[24201]: pam_unix(sshd:auth): authentication failure; ruser= rhost=5.36.59.76.dsl.example.om  user=root",
                    "Dec 10 09:32:20 LabSZ sshd[24680]: Accepted password for fztu from 119.137.62.142 port 49116 ssh2"])
    txt = "\n".join(out)
    for leak in ("173.234", "example", "webmaster", "fztu", "LabSZ", "119.137", "5.36"):
        assert leak not in txt, leak
    assert "<IP1>" in out[0] and "<IP1>" in out[1] and "user=root" in out[2]                  # consistent tags, root kept


@pytest.mark.parametrize("name", list(SOURCES_V2))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_query_set(name, seed):
    inst = toy(name, seed)
    assert 2 <= len(inst["variables"]) <= 4 and np.prod(shape(inst)) <= 256
    rels = [q["rel"] for q in inst["queries"]]
    assert rels.count("marginal") == len(inst["variables"])
    for r in ("conj", "cond", "cond_rev", "cond_neg", "neg", "neg_conj", "disj"):
        assert r in rels, r
    if len(inst["variables"]) >= 3:
        assert "conj3" in rels
    _, variables, _, opts = SOURCES_V2[name]
    ncoarse = sum(len(v["coarse"]["options"]) for v in variables if v.get("coarse"))
    ncum = sum(len(v.get("cum") or []) for v in variables)
    assert rels.count("coarse") == ncoarse and rels.count("cum") == ncum
    assert ncoarse + ncum > 0, "every source has a granularity or a monotone constraint"
    ids = [q["id"] for q in inst["queries"]]
    assert len(set(ids)) == len(ids)
    full = int(np.prod(shape(inst)))
    for q in inst["queries"]:
        assert q["text"].endswith("?") and "{" not in q["text"], q
        alts = inst["paraphrases"][q["id"]]
        assert len(alts) >= (2 if q["kind"] == "marginal" else 1) and all(a != q["text"] for a in alts)
        if q["kind"] != "marginal":
            m = event_mask(inst, q["event"], bool(q.get("neg")))
            assert 0 < m.sum() < full, q                      # never trivially certain or impossible
        if q["kind"] == "cond":
            assert set(q["event"]).isdisjoint(q["given"])
    # both directions of each Bayes pair are present, with the conjunction
    links = {}
    for q in inst["queries"]:
        if "link" in q:
            links.setdefault(q["link"], set()).add(q["rel"])
    assert len(links) >= 2 and all({"conj", "cond", "cond_rev"} <= s for s in links.values())


@pytest.mark.parametrize("name", ["tos", "swe_issues", "forecast_news", "math_problems", "ledgar"])
def test_linked_semantics_exact(name):
    inst = toy(name, 3)
    J = rand_joint(inst, 1)
    ans = answers_from_joint(inst, J)
    qby = {q["id"]: q for q in inst["queries"]}
    vi = {v["name"]: i for i, v in enumerate(inst["variables"])}
    marg = {q["var"]: np.asarray(ans[q["id"]]) for q in inst["queries"] if q["kind"] == "marginal"}
    for qid, q in qby.items():
        a = ans[qid][0] if q["kind"] != "marginal" else None
        if q["rel"] == "coarse":                             # coarse option = sum of its fine options
            (v, allowed), = q["event"].items()
            assert abs(a - marg[v][allowed].sum()) < 1e-9
        if q["rel"] == "disj":                               # not(comp A and comp B) = P(A or B)
            (va, ca), (vb, cb) = q["event"].items()
            A = event_mask(inst, {va: [i for i in range(len(marg[va])) if i not in ca]})
            B = event_mask(inst, {vb: [i for i in range(len(marg[vb])) if i not in cb]})
            assert abs(a - J[A | B].sum()) < 1e-9
            assert a >= max(J[A].sum(), J[B].sum()) - 1e-12
        if q["rel"] == "cum":
            (v, allowed), = q["event"].items()
            assert abs(a - marg[v][: len(allowed)].sum()) < 1e-9
    # monotone: cum thresholds of one variable are non-decreasing
    cums = {}
    for q in inst["queries"]:
        if q["rel"] == "cum":
            (v, allowed), = q["event"].items()
            cums.setdefault(v, []).append((len(allowed), ans[q["id"]][0]))
    for v, xs in cums.items():
        ps = [p for _, p in sorted(xs)]
        assert all(x <= y + 1e-12 for x, y in zip(ps, ps[1:]))
    # Bayes: P(A|B) P(B) = P(B|A) P(A) = P(A and B)
    for link in {q["link"] for q in inst["queries"] if "link" in q}:
        grp = {q["rel"]: q for q in inst["queries"] if q.get("link") == link}
        c, r, j = grp["cond"], grp["cond_rev"], grp["conj"]
        pA, pB = P(J, inst, c["event"]), P(J, inst, c["given"])
        assert abs(ans[c["id"]][0] * pB - ans[j["id"]][0]) < 1e-9
        assert abs(ans[r["id"]][0] * pA - ans[j["id"]][0]) < 1e-9
        assert r["event"] == c["given"] and r["given"] == c["event"]
    assert set(vi)


@pytest.mark.parametrize("name", list(SOURCES_V2))
def test_coherent_prices_have_no_dutch_book(name):
    inst = toy(name, 5)
    ans = answers_from_joint(inst, rand_joint(inst, 2, alpha=0.5))
    assert dutch_book(inst, ans)["dutch"] < 1e-7


def _perturb(inst, ans, rel, f):
    out = dict(ans)
    hit = False
    for q in inst["queries"]:
        if q["rel"] == rel:
            out[q["id"]] = [float(np.clip(f(ans[q["id"]][0]), 0, 1))]
            hit = True
            break
    assert hit, rel
    return out


@pytest.mark.parametrize("name,rel,f", [
    ("swe_issues", "coarse", lambda p: p + 0.25 if p < 0.7 else p - 0.25),     # coarse != sum of fine
    ("tos", "disj", lambda p: 0.0),                                           # P(A or B) < P(A)
    ("forecast_news", "fc_event", lambda p: 1 - p),                          # yes/no event vs its marginal
    ("ledgar", "cond", lambda p: 1 - p),                                      # Bayes-inconsistent conditional
    ("sci_claims", "neg", lambda p: 1 - p),                                   # negation != complement
])
def test_single_violations_are_booked(name, rel, f):
    inst = toy(name, 7)
    ans = answers_from_joint(inst, rand_joint(inst, 3, alpha=2.0))
    assert dutch_book(inst, _perturb(inst, ans, rel, f))["dutch"] > 1e-3


def test_monotone_violation_is_booked():
    inst = toy("swe_issues", 1)
    ans = answers_from_joint(inst, rand_joint(inst, 4, alpha=2.0))
    cum = [q for q in inst["queries"] if q["rel"] == "cum"]
    lo, hi = sorted(cum, key=lambda q: len(q["event"]["difficulty"]))[:2]
    bad = dict(ans)
    bad[lo["id"]], bad[hi["id"]] = [0.8], [0.3]                  # P(fix < 15 min) > P(fix < 1 hour)
    assert dutch_book(inst, bad)["dutch"] > 0.1
    # the violation alone, on just the two threshold bets, is already a Dutch book of size 0.5
    sub = {lo["id"]: [0.8], hi["id"]: [0.3]}
    bets, _ = bets_for_instance(inst, sub)
    assert abs(max_dutch_book(bets)[0] - 0.5) < 1e-6


def test_metrics_pipeline_and_para_variants():
    inst = toy("support_chat", 2)
    inst["gold"] = {"outcome": 0, "length": 1}
    J = rand_joint(inst, 5)
    ans = answers_from_joint(inst, J)
    para = {k: ans[k] for k in inst["paraphrases"]}
    recs, _ = evaluate({inst["id"]: inst}, {inst["id"]: {"base": {"answers": ans}, "para:0": {"answers": para}}})
    o = build_report(recs)["overall"]
    assert o["kl"] is None and o["dutch"] < 1e-7 and o["para"] < 1e-12 and o["acc"] is not None
    P_ = core.plan_instances([inst])
    p0 = {q for q, _, _ in P_.slots[(inst["id"], "para:0")]}
    p1 = {q for q, _, _ in P_.slots[(inst["id"], "para:1")]}
    assert p0 == set(inst["paraphrases"])                       # every query has a first paraphrase
    assert p1 == {q["id"] for q in inst["queries"] if q["kind"] == "marginal"}


def test_v1_data_unchanged():
    for name, sha in V1_SHA.items():
        p = os.path.join(ROOT, "data", "realcoh", f"{name}.jsonl")
        if not os.path.exists(p):
            pytest.skip("data/realcoh not built")
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == sha, name


@pytest.mark.skipif(not HAVE_CACHE or not os.path.exists(os.path.join(ROOT, "data", "realcoh", "tos.jsonl")),
                    reason="needs the HF cache and data/realcoh")
@pytest.mark.parametrize("name", ["tos", "climate", "symptoms", "gold_news", "bgl_logs"])
def test_v1_builder_reproduces_v1_files(name):
    from bookiebench.realcoh.build import build_source
    rows = [json.dumps(r, ensure_ascii=False) for r in build_source(name)]
    assert rows == open(os.path.join(ROOT, "data", "realcoh", f"{name}.jsonl")).read().splitlines()


@pytest.mark.skipif(not HAVE_CACHE or not os.path.exists(os.path.join(V2DIR, "manifest.json")),
                    reason="needs the HF cache and data/v2/realcoh")
@pytest.mark.parametrize("name", ["sci_claims", "ledgar", "forecast_news"])
def test_v2_builder_is_deterministic(name):
    from bookiebench.realcoh.build_v2 import build_source
    rows = [json.dumps(r, ensure_ascii=False) for r in build_source(name)]
    assert rows == open(os.path.join(V2DIR, f"{name}.jsonl")).read().splitlines()


@pytest.mark.skipif(not os.path.exists(os.path.join(V2DIR, "manifest.json")), reason="data/v2/realcoh not built")
def test_built_v2_files():
    man = json.load(open(os.path.join(V2DIR, "manifest.json")))
    assert set(man["sources"]) == set(SOURCES_V2)
    texts = set()
    for name in SOURCES_V2:
        rows = [json.loads(l) for l in open(os.path.join(V2DIR, f"{name}.jsonl"))]
        assert len(rows) == 200 and len({r["id"] for r in rows}) == 200
        ev = [r["steps"][0]["evidence"] for r in rows]
        assert len(set(ev)) == 200
        texts |= set(ev)
        for r in rows:
            ref = r["meta"]["source_ref"]
            assert ref["dataset"] and ref["revision"] and ref["file"] and len(ref) > 3, ref
            assert ref["revision"] == PINNED[ref["dataset"]], ref                 # rebuilds read the pinned snapshot
            assert r["meta"]["in_decider_train"] == PROVENANCE[name]["in_decider_train"]
            assert r["steps"][0]["joint"] is None and r["level"] == "R" and r["pack"] == "realcoh"
            assert r["family"] == name and r["split"] == "realcoh2"
            for v, g in r["gold"].items():
                assert 0 <= g < len(next(x for x in r["variables"] if x["name"] == v)["options"])
            for v, g in r["meta"]["gold_coarse"].items():
                assert any(x.get("coarse", {}).get("name") == v and 0 <= g < len(x["coarse"]["options"]) for x in r["variables"])
        assert hashlib.sha256(open(os.path.join(V2DIR, f"{name}.jsonl"), "rb").read()).hexdigest() == man["sources"][name]["sha256"]
    assert len(texts) == 200 * len(SOURCES_V2)                   # no state is shared between sources


def _built(name):
    p = os.path.join(V2DIR, f"{name}.jsonl")
    if not os.path.exists(p):
        pytest.skip("data/v2/realcoh not built")
    return [json.loads(l) for l in open(p)]


def test_built_leak_fixes():
    import re
    for r in _built("bgl_logs"):                                    # R1: lines start with the unix timestamp, no label
        assert all(re.match(r"\d{10} ", ln) for ln in r["steps"][0]["evidence"].split("\n"))
    for r in _built("ssh_logs"):                                    # R8
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", r["steps"][0]["evidence"])
    for r in _built("swe_issues"):                                  # R12
        assert not re.search(r"(?i)\b(django|sympy|matplotlib|sklearn|astropy|xarray|pytest|pylint)\b", r["steps"][0]["evidence"])
    for r in _built("eurlex_fr"):
        assert not re.search(r"(?i)\b(r[èe]glements?|directives?|d[ée]cisions?)\b", r["steps"][0]["evidence"])
    for r in _built("reviews"):
        assert "My rating" not in r["steps"][0]["evidence"] and "verdict" in r["gold"]
    for r in _built("forecast_news"):                               # R12: dated, public actors only
        assert r["steps"][0]["evidence"].startswith("Published: ")
    assert all(not r["gold"] for r in _built("support_chat"))        # R9


@pytest.mark.skipif(not HAVE_CACHE, reason="needs the HF cache")
def test_release_and_rebuild_roundtrip(tmp_path):
    from bookiebench.realcoh.build_v2 import build_source, rebuild, write_release
    built = {k: build_source(k, 12) for k in ("sci_claims", "tos")}
    rel = tmp_path / "release"
    write_release(str(rel), built)
    man = json.load(open(rel / "manifest.json"))
    man.update(n=12, seed=0)
    json.dump(man, open(rel / "manifest.json", "w"))
    ids = [json.loads(l) for l in open(rel / "sci_claims.jsonl")]
    assert all(r["steps"][0]["evidence"] is None and r["meta"]["evidence_sha256"] for r in ids)   # ids_only
    assert [json.loads(l) for l in open(rel / "tos.jsonl")] == built["tos"]                       # keep
    out = tmp_path / "full"
    assert rebuild(str(rel), str(out)) == {"sci_claims": 12, "tos": 12}
    assert [json.loads(l) for l in open(out / "sci_claims.jsonl")] == json.loads(json.dumps(built["sci_claims"]))
