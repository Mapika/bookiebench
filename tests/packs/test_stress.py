"""Tests for the BookieBench v2 `stress` pack (bookiebench/sims/packs/stress): invariance of the world under every
transform, exactness of added queries, and the programmatic checks used for LLM rewrites."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pytest

from bookiebench.sims import HELDOUT_FAMILIES, TRAIN_FAMILIES, generate
from bookiebench.sims.core import (cell_assignment, exact_answer, iter_jsonl, joint_array, joint_shape, state_text,
                              validate, var_names)
from bookiebench.sims.packs.stress import SCALING, TRANSFORMS, bias, formats, llm, longctx, queries, scaling
from bookiebench.sims.packs.stress.common import invariance_signature, numbers, units

ROOT = Path(__file__).resolve().parents[2]
FAMS = [(f, "test") for f in TRAIN_FAMILIES] + [(f, "heldout") for f in HELDOUT_FAMILIES]


def _src(fam, split, i):
    return generate(fam, split, i)


SOURCES = [_src(f, s, i) for f, s in FAMS for i in (0, 1, 2)]
FAST = [n for n in TRANSFORMS if not n.startswith("long_")]


def _orig_signature(inst, n_orig):
    sig = invariance_signature(inst)
    sig["queries"] = sig["queries"][:n_orig]
    return sig


def assert_invariant(src, out):
    n = len(src["queries"])
    assert _orig_signature(out, n) == invariance_signature(src)
    assert [q["id"] for q in out["queries"][:n]] == [q["id"] for q in src["queries"]]
    for q0, q1 in zip(src["queries"], out["queries"]):
        assert exact_answer(src, q0) == exact_answer(out, q1)
    assert out["transform"] and out["pack"] == "stress" and out["level"] == "L4"
    assert out["family"] == src["family"] and out["meta"]["source_id"] == src["id"]


# ----------------------------------------------------------------------------------------------------------------------
# an independent evaluator for the logical forms recorded with added queries
# ----------------------------------------------------------------------------------------------------------------------

def _pyexpr(form):
    e = re.sub(r"(\w+) in comp\((\d+)\)", r"(c['\1']!=\2)", form)
    e = re.sub(r"(\w+) in \[([\d, ]+)\]", r"(c['\1'] in [\2])", e)
    e = re.sub(r"(\w+)!=(\d+)", r"(c['\1']!=\2)", e)
    e = re.sub(r"(?<![!'])\b(\w+)=(\d+)", r"(c['\1']==\2)", e)
    return e


def form_prob(inst, form):
    if " = " in form:
        form = form.split(" = ")[-1]
    J = joint_array(inst).ravel()
    cells = [cell_assignment(inst, k) for k in range(J.size)]
    if form.startswith("P("):
        body = form[2:-1]
        ev, given = body.split(" | ")
        given = re.sub(r", (?=[A-Za-z_])", " and ", given)
        fe, fg = _pyexpr(ev), _pyexpr(given)
        pg = sum(p for p, c in zip(J, cells) if eval(fg))  # noqa: S307
        return sum(p for p, c in zip(J, cells) if eval(fg) and eval(fe)) / pg  # noqa: S307
    fe = _pyexpr(form)
    return sum(p for p, c in zip(J, cells) if eval(fe))  # noqa: S307


# ----------------------------------------------------------------------------------------------------------------------
# deterministic transforms
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", FAST)
def test_transform_invariance(name):
    fn = TRANSFORMS[name]
    for src in SOURCES:
        out = fn(src)
        validate(out)
        assert_invariant(src, out)
        a, b = fn(src), fn(src)
        assert json.dumps(a) == json.dumps(b), "not deterministic"
        st = out["meta"]["stress"]
        for qid in st.get("added_qids", []):
            q = next(x for x in out["queries"] if x["id"] == qid)
            assert np.allclose(exact_answer(out, q), st["added_exact"][qid], atol=1e-9)
            assert "neg" not in q or q["kind"] == "noul"


@pytest.mark.parametrize("name", ["negation", "disjunction", "nested"])
def test_added_query_forms_match_events(name):
    """The logical form behind each added text (recorded in meta.stress.forms) evaluates, by independent brute force
    over the cells, to the same probability as the stored event."""
    n_checked = 0
    for src in SOURCES:
        out = TRANSFORMS[name](src)
        forms = out["meta"]["stress"]["forms"]
        assert len(forms) >= (4 if name == "negation" else 3)
        for q in out["queries"]:
            if q["id"] in forms:
                assert abs(form_prob(out, forms[q["id"]]) - exact_answer(out, q)[0]) < 1e-9, (q, forms[q["id"]])
                n_checked += 1
    assert n_checked > 50


def test_negation_texts_are_negation_heavy():
    out = queries.negation(SOURCES[0])
    added = [q for q in out["queries"] if q["id"] in out["meta"]["stress"]["added_qids"]]
    assert all(len(re.findall(r"\b(not|false|untrue|neither|nor|wrong|deny)\b", q["text"])) >= 1 for q in added)
    assert sum(len(re.findall(r"\b(not|false|untrue|neither|wrong)\b", q["text"])) >= 2 for q in added) >= 3


def test_conjunction_pairs():
    for src in SOURCES:
        out = bias.conjunction(src)
        pairs = out["meta"]["stress"]["pairs"]
        assert len(pairs) >= 2
        rep = out["meta"]["stress"]["representative"]
        qs = {q["id"]: q for q in out["queries"]}
        for p in pairs:
            qc, qj = qs[p["component"]], qs[p["conj"]]
            assert set(qj["event"]) == set(qc["event"]) | {rep["var"]}
            assert qj["event"][rep["var"]] == [rep["option"]]
            assert exact_answer(out, qj)[0] <= exact_answer(out, qc)[0] + 1e-12
        assert "not based on" in out["prelude"] or "does not use" in out["prelude"] or "no information" in out["prelude"]


def test_base_rate_keeps_every_number():
    for src in SOURCES:
        out = bias.base_rate(src)
        # the base rate is buried, never dropped; the lure adds no number beyond its outcome clause (e.g. a clock time)
        assert numbers(out["prelude"]) == sorted(numbers(src["prelude"]) + numbers(out["meta"]["stress"]["lure"]))
        buried = out["meta"]["stress"]["buried"]
        if buried and not buried.startswith(("base", "P(", "prior")) and "{" not in src["prelude"]:
            assert out["prelude"].rstrip().endswith(buried.rstrip()) or buried.rstrip(")") in out["prelude"]


def test_anchoring_only_prefixes_numbers():
    for src in SOURCES:
        out = bias.anchoring(src)
        for q0, q1 in zip(src["queries"], out["queries"]):
            assert q1["text"].endswith(q0["text"]) and q1["text"] != q0["text"]
            assert str(out["meta"]["stress"]["anchors"][q0["id"]]["value"]) in q1["text"]


@pytest.mark.parametrize("fmt", formats.FORMATS)
def test_formats_preserve_content(fmt):
    for src in SOURCES:
        out = formats.format_shift(src, fmt)
        # every number of the prelude survives (CSV moves them into the parameter table)
        a = numbers(src["prelude"])
        b = numbers(out["prelude"].replace(",", " , ") if fmt == "csv" else out["prelude"])
        for x in (set(a) if fmt != "csv" else {y for y in a if "." not in y}):
            assert b.count(x) >= a.count(x), (src["id"], fmt, x)
        # evidence is wrapped by a step-only wrapper, identically for all alternatives
        for k, st in enumerate(out["steps"]):
            if k and out["steps"][k - 1].get("next_evidence"):
                srcs = [x["evidence"] for x in src["steps"][k - 1]["next_evidence"]]
                outs = [x["evidence"] for x in out["steps"][k - 1]["next_evidence"]]
                shells = {o.replace(s_, "\x00") for s_, o in zip(srcs, outs) if s_ in o}
                if fmt != "csv":
                    assert len(shells) == 1, (fmt, shells)


def test_format_json_prelude_roundtrip():
    src = next(s for s in SOURCES if s["meta"]["style"] == "json")
    for fmt in formats.FORMATS:
        out = formats.format_shift(src, fmt)
        validate(out)
        assert_invariant(src, out)


# ----------------------------------------------------------------------------------------------------------------------
# long context
# ----------------------------------------------------------------------------------------------------------------------

def _tokenizer_available():
    try:
        longctx.tokenizer()
        return Path(longctx.FILES["news"]).exists()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _tokenizer_available(), reason="Qwen3.5 tokenizer / filler datasets not available")
@pytest.mark.parametrize("target", [4096, 16384])
def test_long_context(target):
    for src in SOURCES[::6]:
        out = longctx.long_context(src, target)
        validate(out)
        assert_invariant(src, out)
        n = out["meta"]["stress"]["n_tokens"]
        assert 0.9 * target <= n <= 1.06 * target, n
        assert n == longctx.n_tokens(state_text(out))
        pos = [out["prelude"].find(u["text"]) for u in units(src["prelude"])]
        assert all(p >= 0 for p in pos) and pos == sorted(pos), "world units missing or out of order"
        for k, st in enumerate(src["steps"]):
            assert out["steps"][k]["evidence"].endswith(st["evidence"])


# ----------------------------------------------------------------------------------------------------------------------
# scaling
# ----------------------------------------------------------------------------------------------------------------------

def test_scaling_composite_exact_and_fast():
    from bookiebench.metrics.dutch import dutch_book
    pool = SOURCES
    for j, src in enumerate(SOURCES[:12]):
        out = SCALING(src, pool, tier_idx=j % 3)
        scaling.validate_composite(out)
        st = out["meta"]["stress"]
        parts = [next(p for p in pool if p["id"] == sid) for sid in st["source_ids"]]
        assert st["n_cells"] == int(np.prod(joint_shape(out))) <= 4096
        # the composite joint is the outer product of the sub-worlds' joints at every step
        for k in range(len(out["steps"])):
            J = None
            for p in parts:
                Jp = joint_array(p, min(k, len(p["steps"]) - 1))
                J = Jp if J is None else np.multiply.outer(J, Jp)
            assert np.allclose(joint_array(out, k), J, atol=1e-10)
        # sub-world queries keep their exact answers
        for L, p in zip("ABCDEFGH", parts):
            for q in p["queries"]:
                q2 = next(x for x in out["queries"] if x["id"] == f"{L}.{q['id']}")
                assert np.allclose(exact_answer(out, q2), exact_answer(p, q), atol=1e-9)
        assert out["gold"] == {f"{L}.{k}": v for L, p in zip("ABCDEFGH", parts) for k, v in p["gold"].items()}
        # coherent (exact) prices admit no Dutch book, and the LP is fast
        ans = {q["id"]: exact_answer(out, q) for q in out["queries"]}
        t0 = time.time()
        r = dutch_book(out, ans)
        assert time.time() - t0 < 5.0
        assert r["dutch"] < 1e-6


def test_scaling_reaches_4096_cells():
    pool = [_src(f, s, i) for f, s in FAMS for i in range(10)]
    got = [SCALING(p, pool, tier_idx=2)["meta"]["stress"]["n_cells"] for p in pool[:30]]
    assert max(got) > 2048 and max(got) <= 4096


# ----------------------------------------------------------------------------------------------------------------------
# LLM-rewrite checks (no model needed)
# ----------------------------------------------------------------------------------------------------------------------

MED = SOURCES[6]  # medical-test-000000


def test_check_text_accepts_and_rejects():
    ctx = {"options": llm.all_options(MED)}
    s = "The chance of a persistent cough is 35% with Ilsen syndrome, 15% with Morrow syndrome, and 5% otherwise."
    good = "With Ilsen syndrome a persistent cough occurs 35% of the time; with Morrow syndrome, 15%; otherwise 5%."
    assert llm.check_text(s, good, "en", ctx) == []
    assert "numbers" in llm.check_text(s, good.replace("35%", "36%"), "en", ctx)
    assert "numbers" in llm.check_text(s, good + " About 2% have both.", "en", ctx)
    assert any(r.startswith("entity") for r in llm.check_text(s, good.replace("Ilsen", "Ilson"), "en", ctx))
    assert "number_words" in llm.check_text("The die shows five.", "The die shows five, twice.", "en", ctx)
    q = "Is it untrue that Tariq has Morrow syndrome?"
    assert llm.check_text(q, "Is it false that Tariq has Morrow syndrome?", "en", ctx, is_question=True) == []
    assert "negation" in llm.check_text(q, "Does Tariq have Morrow syndrome?", "en", ctx, is_question=True)
    assert "question" in llm.check_text(q, "It is false that Tariq has Morrow syndrome.", "en", ctx, is_question=True)
    de = {"options": llm.all_options(MED), "opt_map": {"Ilsen syndrome": "Ilsen-Syndrom",
                                                       "Morrow syndrome": "Morrow-Syndrom"},
          "exempt": llm.option_exempt_words(MED)}
    ok_de = ("Die Wahrscheinlichkeit eines anhaltenden Hustens beträgt 35 % beim Ilsen-Syndrom, 15 % beim "
             "Morrow-Syndrom und sonst 5 %.")
    assert llm.check_text(s, ok_de, "de", de) == []
    assert "numbers" in llm.check_text(s, ok_de.replace("35", "3,5"), "de", de)
    assert "not_translated" in llm.check_text(s, good, "de", de)
    assert llm.check_text("Dr. Varga is examining Mateo.", "Dr. Varga Mateót vizsgálja.", "hu", {}) == []


def test_complement_match():
    s = "The chance of a persistent cough is 35% with Ilsen syndrome, 15% with Morrow syndrome, and 5% otherwise."
    t = "The chance of no persistent cough is 65% with Ilsen syndrome, 85% with Morrow syndrome, and 95% otherwise."
    assert llm.complement_match(s, t) == (True, 3)
    assert llm.complement_match(s, t.replace("65%", "66%"))[0] is False
    assert llm.complement_match("There are 3 decks and 20% are red.", "There are 7 decks and 80% are not red.")[0] is False


def test_translation_assembly_keeps_world():
    tr = {v["name"]: [f"{o} (de)" for o in v["options"]] for v in MED["variables"]}
    evs = llm.evidence_texts(MED)
    out = llm.assemble(MED, "lang_de", "Vorspann.", {e: f"[de] {e}" for e in evs},
                       [q["text"] for q in MED["queries"]], opt_tr=tr, info={"lang": "de"})
    validate(out)
    assert_invariant(MED, out)
    for q in out["queries"]:
        if q["kind"] == "marginal":
            assert q["options"] == next(v for v in out["variables"] if v["name"] == q["var"])["options"]
    assert llm.check_options(llm.options_by_var(MED), tr, "de") == []
    bad = dict(tr)
    k = MED["variables"][0]["name"]
    bad[k] = [bad[k][0]] * len(bad[k])
    assert "options_not_distinct" in llm.check_options(llm.options_by_var(MED), bad, "de")


# ----------------------------------------------------------------------------------------------------------------------
# generated data (if present)
# ----------------------------------------------------------------------------------------------------------------------

DATA = ROOT / "data" / "v2" / "stress"


@pytest.mark.skipif(not any(DATA.glob("*/*.jsonl")), reason="data/v2/stress not generated")
def test_generated_files_invariant():
    src_cache = {}

    def source(inst):
        sid = inst["meta"]["source_id"]
        path = inst["meta"].get("source_file")
        if path is None:
            fam, split, _ = re.match(r"^([a-z_]+)-([a-z_]+)-(\d+)$", sid).groups()
            path = f"data/{split}/{fam}.jsonl"
        path = ROOT / path
        if path not in src_cache:
            src_cache[path] = {d["id"]: d for d in iter_jsonl(path)}
        return src_cache[path][sid]

    n = 0
    for f in sorted(DATA.glob("*/*.jsonl")):
        for j, inst in enumerate(iter_jsonl(f)):
            if j >= 5:
                break
            if inst["transform"] == "scaling":
                scaling.validate_composite(inst)
                continue
            validate(inst)
            assert_invariant(source(inst), inst)
            n += 1
    assert n > 0


def test_generic_clauses_for_unknown_family():
    """Instances that are not in the bookiebench.sims registry (e.g. other packs) get generic but exact clauses."""
    import copy
    src = copy.deepcopy(SOURCES[0])
    src["id"] = "otherpack-test-000001"
    for name in ("negation", "disjunction", "nested", "conjunction", "base_rate"):
        out = TRANSFORMS[name](src)
        validate(out)
        assert out["meta"]["stress"]["native_clauses"] is False
        assert_invariant(src, out)
        for qid in out["meta"]["stress"].get("added_qids", []):
            q = next(x for x in out["queries"] if x["id"] == qid)
            assert 'the answer to "' in q["text"]


def test_quoted_terms_ignore_possessives():
    assert llm.quoted_terms("chosen with 75 percent for Jin's box and 25 percent for Pavel's box") == []
    assert llm.quoted_terms("The word 'invoice' appears; Amara's word 'slow' too.") == ["invoice", "slow"]


# ----------------------------------------------------------------------------------------------------------------------
# release (tv=2, options shuffled per instance by core.finalize_release)
# ----------------------------------------------------------------------------------------------------------------------

from bookiebench.sims.core import finalize_release  # noqa: E402
from bookiebench.sims.packs.stress.common import Clauses  # noqa: E402

RELEASED = [finalize_release(s) for s in SOURCES]


def test_release_clauses_follow_the_shuffle():
    """On a shuffled instance, the clause for new option index i is the clause of the canonical option perm[i]."""
    n_native = 0
    for src, rel in zip(SOURCES, RELEASED):
        c0, c1 = Clauses(src), Clauses(rel)
        assert c1.native == c0.native
        n_native += c1.native
        perm = rel["meta"]["option_perm"]
        for v in rel["variables"]:
            for i in range(len(v["options"])):
                assert c1.clause(v["name"], [i]) == c0.clause(v["name"], [perm[v["name"]][i]])
    assert n_native == len(SOURCES)


@pytest.mark.parametrize("name", FAST)
def test_transforms_on_released_instances(name):
    for rel in RELEASED:
        out = TRANSFORMS[name](rel)
        validate(out)
        assert_invariant(rel, out)
        st = out["meta"]["stress"]
        for qid in st.get("added_qids", []):
            q = next(x for x in out["queries"] if x["id"] == qid)
            assert np.allclose(exact_answer(out, q), st["added_exact"][qid], atol=1e-9)
            if qid in st.get("forms", {}):
                assert abs(form_prob(out, st["forms"][qid]) - exact_answer(out, q)[0]) < 1e-9


def test_released_added_queries_mean_what_they_say():
    """Semantic check across the shuffle: an added query on the released instance, mapped back to canonical option
    indices, has the same exact answer as the same text's event on the canonical instance."""
    from bookiebench.sims.core import unshuffle_options
    for rel in RELEASED[:12]:
        for name in ("negation", "disjunction", "nested", "conjunction"):
            out = TRANSFORMS[name](rel)
            back = unshuffle_options(out)
            c0 = Clauses(unshuffle_options(rel))
            for qid in out["meta"]["stress"]["added_qids"]:
                q = next(x for x in back["queries"] if x["id"] == qid)
                # the clause texts for the canonical event appear verbatim in the query text
                for var, idxs in q["event"].items():
                    if len(idxs) == 1:
                        assert c0.clause(var, idxs) in q["text"] or c0.not_clause(var, idxs) in q["text"] or \
                            len(back["variables"][var_names(back).index(var)]["options"]) > 2, (q["text"], var, idxs)
                assert np.allclose(exact_answer(back, q), exact_answer(out, next(x for x in out["queries"]
                                                                                 if x["id"] == qid)))


RELEASE_DIR = ROOT / "data" / "release" / "test"


@pytest.mark.skipif(not (RELEASE_DIR / "urn.jsonl").exists(), reason="data/release not built")
def test_real_release_instances():
    from bookiebench.sims.core import read_jsonl
    for fam in ("urn", "medical", "witness"):
        for rel in read_jsonl(RELEASE_DIR / f"{fam}.jsonl")[:5]:
            assert Clauses(rel).native, rel["id"]
            for name in ("negation", "conjunction", "format_csv", "base_rate"):
                out = TRANSFORMS[name](rel)
                validate(out)
                assert_invariant(rel, out)


MANIFEST = ROOT / "data" / "release" / "stress" / "manifest.json"


@pytest.mark.skipif(not MANIFEST.exists(), reason="release stress manifest not built")
def test_paired_manifest():
    man = json.loads(MANIFEST.read_text())
    for t, rows in man["transforms"].items():
        for fam, r in rows.items():
            assert r["n_kept"] == len(r["kept_source_ids"])
            assert set(r["kept_source_ids"]) | set(r["rejected_source_ids"]) == set(
                json.loads(json.dumps(r["kept_source_ids"] + r["rejected_source_ids"])))
            assert r["n_kept"] + len(r["rejected_source_ids"]) == r["n_source"]
            f = MANIFEST.parent / t / f"{fam}.jsonl"
            if f.exists():
                assert [o["meta"]["source_id"] for o in iter_jsonl(f)][:20] == r["kept_source_ids"][:20]
    for fam, r in man["families"].items():
        for rows in man["transforms"].values():
            assert set(r["kept_by_all"]) <= set(rows[fam]["kept_source_ids"])


def test_zh_double_negation():
    from bookiebench.sims.packs.stress.llm import zh_double_negation
    q = "Is it untrue that the ticket will be escalated?"
    assert zh_double_negation(q, "工单不会被升级，这是否不属实?")
    assert not zh_double_negation(q, "工单会被升级，这是否不属实?")
    assert not zh_double_negation(q, "是否并非工单会被升级?")
    assert not zh_double_negation("Is it untrue that no intruder entered?", "没有入侵者进入，这是否不属实?")
    assert not zh_double_negation("Is it false that the next toss will come up heads?",
                                  "“未来的下一次抛掷会显示正面”这一说法是否不成立?")
