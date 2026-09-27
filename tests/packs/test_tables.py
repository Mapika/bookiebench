"""Tests for the v2 `tables` pack: SPEC validity, recount of every joint from the rendered text, martingale identity,
perms, determinism and the size knob."""
from __future__ import annotations

import numpy as np
import pytest

from bookiebench.sims.core import dumps, exact_answer, joint_array, mart_queries, marginal, validate, var_index
from bookiebench.sims.packs.tables import FAMILIES, PACK, generate
from bookiebench.sims.packs.tables.model import FORMATS, parse_rows

FAMS = list(FAMILIES)
SEED = 3


def _code(v: str, spec):
    if spec["type"] == "cat":
        return spec["options"].index(v)
    return int(np.searchsorted(spec["edges"], float(v), side="right"))


def _holds(f, row, tcol):
    x = row[f["col"]] if f["col"] in row else row["__ts__"]
    op, v = f["op"], f["value"]
    if op == "eq":
        return x == str(v)
    if op == "ne":
        return x != str(v)
    if op == "ge":
        return float(x) >= v
    if op == "lt":
        return float(x) < v
    if op == "hour_lt":
        return int(x[11:13]) < v
    if op == "hour_ge":
        return int(x[11:13]) >= v
    if op == "date_lt":
        return x[:10] < v
    if op == "date_ge":
        return x[:10] >= v
    raise ValueError(op)


def _count_joint(inst, rows):
    names = [v["name"] for v in inst["variables"]]
    specs = inst["meta"]["columns"]
    C = np.zeros([len(v["options"]) for v in inst["variables"]])
    for r in rows:
        C[tuple(_code(r[n], specs[n]) for n in names)] += 1
    return C


def _parse(inst, text):
    rows = parse_rows(text, inst["meta"]["style"])
    tcol = inst["meta"]["table_columns"][1]
    for r in rows:
        if "__ts__" in r:
            r[tcol] = r.pop("__ts__")
    return rows


@pytest.mark.parametrize("fam", FAMS)
def test_validate_and_fields(fam):
    styles, modes = set(), set()
    for i in range(40 if fam != "tab_big" else 12):
        inst = generate(fam, "dev", i, seed=SEED)
        validate(inst)
        assert inst["pack"] == PACK and inst["family"] == fam
        assert inst["level"] == ("L4" if fam == "tab_big" else "L3")
        m = inst["meta"]
        styles.add(m["style"]); modes.add(m["mode"])
        if fam != "tab_big":
            assert 10 <= m["n_rows"] <= 200
        assert "perms" in inst
        if m["mode"] == "pick":
            assert all("next_evidence" in s for s in inst["steps"][:-1])
        else:
            assert all("next_evidence" not in s for s in inst["steps"])
    if fam != "tab_big":
        assert styles == set(FORMATS)


@pytest.mark.parametrize("fam", FAMS)
def test_deterministic(fam):
    a, b = generate(fam, "dev", 4, seed=SEED), generate(fam, "dev", 4, seed=SEED)
    assert dumps(a) == dumps(b)
    assert dumps(a) != dumps(generate(fam, "train", 4, seed=SEED))


@pytest.mark.parametrize("fam", FAMS)
def test_recount_from_text(fam):
    """Every step's joint equals the count table recomputed from the rendered text (parsed back)."""
    for i in range(30 if fam != "tab_big" else 6):
        inst = generate(fam, "dev", i, seed=SEED)
        m = inst["meta"]
        tcol = m["table_columns"][1]
        if m["mode"] == "pick":
            rows = _parse(inst, inst["prelude"])
            assert len(rows) == m["n_rows"]
            assert len(m["facts"]) == len(inst["steps"])
            live = rows
            for k, f in enumerate(m["facts"]):
                live = [r for r in live if _holds(f, r, tcol)]
                C = _count_joint(inst, live)
                assert np.allclose(C / C.sum(), joint_array(inst, k), atol=1e-7), (inst["id"], k)
        else:
            seen = []
            for k, st in enumerate(inst["steps"]):
                batch = _parse(inst, st["evidence"])
                assert len(batch) == m["batches"][k]
                seen += batch
                C = _count_joint(inst, seen)
                assert np.allclose(C / C.sum(), joint_array(inst, k), atol=1e-7), (inst["id"], k)
            assert len(seen) == m["n_rows"]
            live = seen
        # conditional queries are exact count ratios
        names = [v["name"] for v in inst["variables"]]
        specs = m["columns"]
        for q in inst["queries"]:
            if q["kind"] != "cond":
                continue
            g = [r for r in live if all(_code(r[n], specs[n]) in a for n, a in q["given"].items())]
            a = [r for r in g if all(_code(r[n], specs[n]) in al for n, al in q["event"].items())]
            assert abs(exact_answer(inst, q)[0] - len(a) / len(g)) < 1e-6  # 8-decimal storage / small P(given)


@pytest.mark.parametrize("fam", ["tab_pick", "tab_big"])
def test_martingale_pick(fam):
    n = 0
    for i in range(25 if fam == "tab_pick" else 8):
        inst = generate(fam, "dev", i, seed=SEED)
        if inst["meta"]["mode"] != "pick":
            continue
        mq = mart_queries(inst)
        ax = var_index(inst, inst["mart_var"])
        for k, st in enumerate(inst["steps"][:-1]):
            pk = np.array(exact_answer(inst, mq[k]))
            mix = sum(a["prob"] * np.array(a["mart_post"]) for a in st["next_evidence"])
            assert np.allclose(mix, pk, atol=1e-6)
            nxt = inst["steps"][k + 1]["evidence"]
            alt = next(a for a in st["next_evidence"] if a["evidence"] == nxt)
            assert np.allclose(alt["mart_post"], marginal(joint_array(inst, k + 1), ax), atol=1e-6)
            n += 1
    assert n > 0


def test_perms_pick_and_gold():
    for i in range(20):
        inst, trace, w = generate("tab_pick", "dev", i, seed=SEED, return_world=True)
        J = w.project(w.posterior(trace["obs"]))
        for p in inst["perms"]:
            assert np.allclose(w.project(w.posterior([trace["obs"][j] for j in p])), J, atol=1e-12)
        # gold = the selected row's cell, which has positive final probability
        cell = tuple(inst["gold"][v["name"]] for v in inst["variables"])
        assert joint_array(inst)[cell] > 0
        assert list(w.proj[trace["z"]]) == list(cell)


def test_stream_perms_and_gold():
    for i in range(20):
        inst = generate("tab_stream", "dev", i, seed=SEED)
        cell = tuple(inst["gold"][v["name"]] for v in inst["variables"])
        assert joint_array(inst)[cell] > 0
        # final joint is the full table's counts regardless of batch order
        rows = [r for st in inst["steps"] for r in _parse(inst, st["evidence"])]
        for p in inst["perms"]:
            rows_p = [r for j in p for r in _parse(inst, inst["steps"][j]["evidence"])]
            assert np.array_equal(_count_joint(inst, rows), _count_joint(inst, rows_p))


def test_size_knob():
    inst = generate("tab_pick", "dev", 0, seed=SEED, n_rows=600)
    validate(inst)
    assert inst["meta"]["n_rows"] == 600
    a = generate("tab_big", "dev", 1, seed=SEED, target_tokens=3000)
    b = generate("tab_big", "dev", 1, seed=SEED, target_tokens=12000)
    assert b["meta"]["n_rows"] > 2.5 * a["meta"]["n_rows"]


# ---------------------------------------------------------------------------------------------------------------------
# oracle: exact answers score KL = 0, dutch = 0, mart = 0 through bookiebench.metrics
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
def test_oracle_scores_zero(fam):
    from bookiebench.metrics.core import evaluate_instance
    for i in range(8):
        inst = generate(fam, "dev", i, seed=SEED)
        ans = {q["id"]: exact_answer(inst, q) for q in inst["queries"]}
        variants = {"base": {"answers": ans}}
        mq = mart_queries(inst)
        for k, st in enumerate(inst["steps"]):
            for j, alt in enumerate(st.get("next_evidence") or []):
                variants[f"next:{k}:{j}"] = {"answers": {mq[k]["id"]: alt["mart_post"]}}
        for pi, _ in enumerate(inst.get("perms", [])):
            variants[f"evperm:{pi}"] = {"answers": ans}
        rec = evaluate_instance(inst, variants)
        assert max(rec["kl_marg"] + rec["kl_bin"]) < 1e-4  # metric floors probabilities at ~1e-6
        assert rec["dutch"] < 1e-6
        assert all(m < 1e-6 for m in rec["mart"])
        assert all(e < 1e-9 for e in rec["evperm"])


@pytest.mark.parametrize("fam", FAMS)
def test_few_trivially_certain_queries(fam):
    from bookiebench.sims.packs.programs.queryfix import is_trivial
    n = t = 0
    for i in range(60 if fam != "tab_big" else 20):
        inst = generate(fam, "dev", i, seed=SEED)
        for q in inst["queries"]:
            n += 1
            t += is_trivial(inst, q)
    assert t / n < 0.05, t / n


def test_batch_headers_position_neutral_and_log_time_named():
    import re as _re
    bad = _re.compile(r"\b(initial|another|first|so far|more|starts|next|last)\b", _re.I)
    logs = 0
    for fam in ("tab_stream", "tab_big"):
        for i in range(30):
            inst = generate(fam, "dev", i, seed=SEED)
            if inst["meta"]["mode"] == "stream":
                for st in inst["steps"]:
                    assert not bad.search(st["evidence"].split("\n", 1)[0]), st["evidence"][:80]
    for i in range(40):
        inst = generate("tab_pick", "dev", i, seed=SEED)
        if inst["meta"]["style"] == "log":
            tcol = inst["meta"]["table_columns"][1]
            body = inst["prelude"].split("```")[1]
            assert all(l.startswith(tcol + "=") for l in body.strip().split("\n")[0:] if l.strip())
            logs += 1
    assert logs > 0


@pytest.mark.parametrize("fam", FAMS)
def test_release_compatible(fam):
    """tv=2 output passes the shared release finalisation (option shuffle) and its inverse; tv=1 still validates."""
    from bookiebench.sims.common import template_version
    from bookiebench.sims.core import finalize_release, unshuffle_options
    from bookiebench.sims import world as W
    for i in range(6):
        with template_version(2):
            inst = generate(fam, "dev", i, seed=SEED)
        fin = finalize_release(inst, seed=1)
        validate(fin)
        back = unshuffle_options(fin)
        for q in inst["queries"]:
            q2 = next(x for x in back["queries"] if x["id"] == q["id"])
            assert np.allclose(exact_answer(inst, q), exact_answer(back, q2), atol=1e-7)
            if q["kind"] != "marginal":
                assert q["text"].endswith("?") and not W.names_absent(q["text"], inst["prelude"] + "\n".join(
                    s["evidence"] for s in inst["steps"]))
        with template_version(1):
            validate(generate(fam, "dev", i, seed=SEED))
