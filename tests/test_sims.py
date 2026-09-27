"""Tests for bookiebench/sims: SPEC validity, exactness (vs brute-force Monte Carlo), coherence, perms, martingale."""
from __future__ import annotations

import numpy as np
import pytest

import json
from pathlib import Path

from bookiebench.sims import (FAMILIES, HELDOUT_FAMILIES, PRIOR_FAMILIES, SPLITS, TRAIN_FAMILIES, VAL_FAMILIES, generate,
                         make_world)
from bookiebench.sims.lexicon import DOMAINS
from bookiebench.sims.nuisance import augment
from bookiebench.sims.core import (cell_assignment, dumps, exact_answer, joint_array, joint_shape, loads, mart_queries,
                              marginal, state_text, validate, var_index, InvalidInstance)
from bookiebench.sims.world import build_instance

ALL = list(FAMILIES)


def _split(fam):
    if fam in TRAIN_FAMILIES or fam in PRIOR_FAMILIES:
        return "train"
    return "val" if fam in VAL_FAMILIES else "heldout"


def _gen(fam, i, trace=False, split=None):
    return generate(fam, split or _split(fam), i, seed=123, return_trace=trace)


def _world_and_trace(fam, i):
    split = _split(fam)
    world, rng = make_world(fam, split, i, seed=123)
    inst, tr = build_instance(world, rng, family=fam, split=split, idx=i, return_trace=True,
                              decimals=getattr(FAMILIES[fam], "DECIMALS", 10))
    return world, inst, tr


MART_TOL = 1e-6


# ---------------------------------------------------------------------------------------------------------------------
# registry / determinism
# ---------------------------------------------------------------------------------------------------------------------

def test_registry():
    assert len(ALL) >= 10
    assert len(TRAIN_FAMILIES) == 7 and len(HELDOUT_FAMILIES) >= 3 and len(VAL_FAMILIES) == 2
    groups = [TRAIN_FAMILIES, PRIOR_FAMILIES, VAL_FAMILIES, HELDOUT_FAMILIES]
    assert sum(len(g) for g in groups) == len(set().union(*map(set, groups))) == len(ALL)
    assert SPLITS["test"] == SPLITS["robust"] == TRAIN_FAMILIES
    assert SPLITS["train"] == TRAIN_FAMILIES + PRIOR_FAMILIES
    assert not set(SPLITS["train"]) & (set(VAL_FAMILIES) | set(HELDOUT_FAMILIES))
    assert len(DOMAINS) >= 40


@pytest.mark.parametrize("fam", ALL)
def test_deterministic_and_roundtrip(fam):
    a, b = _gen(fam, 5), _gen(fam, 5)
    assert dumps(a) == dumps(b)
    assert loads(dumps(a)) == a
    assert dumps(_gen(fam, 6)) != dumps(a)
    if fam in TRAIN_FAMILIES:
        assert dumps(generate(fam, "test", 5, seed=123)) != dumps(generate(fam, "train", 5, seed=123))


# (split, directory, generate kwargs): the internal (pre-release) data must stay byte-identical under tv=1
LEGACY = [("test", "test", {}), ("heldout", "heldout", {}), ("val", "val", {}), ("test_prior", "test_prior", {}),
          ("robust", "robust", {}), ("train", "train", {}), ("train", "train_nuisance", {"nuisance": 0.5, "decimals": 8})]


@pytest.mark.parametrize("split,dirname,kw", LEGACY, ids=[d for _, d, _ in LEGACY])
def test_legacy_outputs_unchanged_under_tv1(split, dirname, kw):
    """Regression: the committed internal data files are reproduced exactly by the generator with tv=1 (seed 0)."""
    root = Path(__file__).resolve().parents[1] / "data" / dirname
    fams = SPLITS[split] if dirname != "train_nuisance" else TRAIN_FAMILIES
    checked = 0
    for fam in fams:
        path = root / f"{fam}.jsonl"
        if not path.exists():
            continue
        k = dict(kw)
        if dirname == "train" and fam in PRIOR_FAMILIES:
            k["nuisance"] = 0.3
        with open(path) as f:
            lines = [next(f).rstrip("\n") for _ in range(3)]
        for i, line in enumerate(lines):
            assert dumps(generate(fam, split, i, seed=0, tv=1, **k)) == line, (dirname, fam, i)
            checked += 1
    if not checked:
        pytest.skip("data not generated")


# ---------------------------------------------------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", ALL)
def test_validate_family(fam):
    for i in range(60):
        inst = _gen(fam, i)
        assert validate(inst)
        assert inst["family"] == fam
        assert np.prod(joint_shape(inst)) <= 256
        assert 2 <= len(inst["variables"]) <= 4
        assert len(inst["steps"]) >= 2
        for q in inst["queries"]:
            assert q["text"][0].isupper() and q["text"].endswith("?")


def test_validate_rejects_bad():
    inst = _gen("urn", 0)
    bad = loads(dumps(inst))
    bad["steps"][0]["joint"] = (np.asarray(bad["steps"][0]["joint"]) * 1.1).tolist()
    with pytest.raises(InvalidInstance):
        validate(bad)
    bad = loads(dumps(inst))
    bad["queries"] = [q for q in bad["queries"] if q["kind"] != "cond"]
    with pytest.raises(InvalidInstance):
        validate(bad)
    bad = loads(dumps(inst))
    bad["queries"][0]["options"] = bad["queries"][0]["options"][::-1]
    with pytest.raises(InvalidInstance):
        validate(bad)


# ---------------------------------------------------------------------------------------------------------------------
# exact_answer coherence against explicit cell enumeration
# ---------------------------------------------------------------------------------------------------------------------

def _brute(inst, q, step=None):
    k = q.get("step") if step is None else step
    J = joint_array(inst, k).ravel()
    names = [v["name"] for v in inst["variables"]]

    def holds(cell, ev):
        return all(cell[n] in allowed for n, allowed in ev.items())

    cells = [cell_assignment(inst, f) for f in range(J.size)]
    if q["kind"] == "marginal":
        out = np.zeros(len(q["options"]))
        for f, c in enumerate(cells):
            out[c[q["var"]]] += J[f]
        return out.tolist()
    ev = [holds(c, q["event"]) for c in cells]
    if q.get("neg"):
        ev = [not e for e in ev]
    if q["kind"] == "noul":
        return [sum(J[f] for f in range(J.size) if ev[f])]
    g = [holds(c, q["given"]) for c in cells]
    return [sum(J[f] for f in range(J.size) if ev[f] and g[f]) / sum(J[f] for f in range(J.size) if g[f])]


@pytest.mark.parametrize("fam", ALL)
def test_exact_answer_matches_enumeration(fam):
    for i in range(15):
        inst = _gen(fam, i)
        for q in inst["queries"]:
            np.testing.assert_allclose(exact_answer(inst, q), _brute(inst, q), atol=1e-9)
            np.testing.assert_allclose(exact_answer(inst, q, step=0), _brute(inst, q, step=0), atol=1e-9)
        # marginals are consistent with the joint and neg is the complement
        J = joint_array(inst)
        for q in inst["queries"]:
            if q["kind"] == "marginal" and "step" not in q:
                np.testing.assert_allclose(exact_answer(inst, q), marginal(J, var_index(inst, q["var"])), atol=1e-12)
            if q["kind"] == "noul":
                flipped = dict(q, neg=not q.get("neg", False))
                assert abs(exact_answer(inst, q)[0] + exact_answer(inst, flipped)[0] - 1) < 1e-9
        # step marginals of mart_var are consistent with the per-step joints
        for k, q in mart_queries(inst).items():
            np.testing.assert_allclose(exact_answer(inst, q), marginal(joint_array(inst, k),
                                                                       var_index(inst, inst["mart_var"])), atol=1e-12)


# ---------------------------------------------------------------------------------------------------------------------
# brute-force Monte Carlo (tests only)
# ---------------------------------------------------------------------------------------------------------------------

def _mc_generic(world, obs, n, rng, min_acc=5000, max_batches=30):
    """Rejection sampling from the generative process: z ~ prior, o_k ~ lik(., past)[z]; keep matches."""
    J = np.zeros(world.shape)
    acc = 0
    for _ in range(max_batches):
        z = rng.choice(len(world.prior), size=n, p=world.prior / world.prior.sum())
        for k, o in enumerate(obs):
            past = obs[:k]
            alts = world.alternatives(k, past)
            P = np.stack([world.lik(a, past)[z] for a in alts], 1)
            u = rng.random(len(z))[:, None]
            draw = (np.cumsum(P, 1) < u).sum(1)
            z = z[draw == alts.index(o)]
        np.add.at(J, tuple(world.proj[z].T), 1.0)
        acc += len(z)
        if acc >= min_acc:
            break
    return J / max(acc, 1), acc


@pytest.mark.parametrize("fam", ALL)
def test_monte_carlo_generic(fam):
    rng = np.random.default_rng(0)
    n_inst = 6 if fam in PRIOR_FAMILIES else 2
    for i in range(n_inst):
        world, inst, tr = _world_and_trace(fam, i)
        # full evidence if acceptance allows, else the longest prefix that does
        for k in range(world.T - 1, -1, -1):
            J_mc, acc = _mc_generic(world, tr["obs"][: k + 1], 1_000_000, rng, max_batches=8)
            if acc >= 3000:
                break
        assert acc >= 3000, f"too few accepted samples ({acc})"
        J = joint_array(inst, k)
        tol = 5 * np.sqrt(0.25 / acc) + 1e-3
        assert np.abs(J - J_mc).max() < tol, (fam, i, k, np.abs(J - J_mc).max(), tol)
        if k != world.T - 1:
            continue
        # also every query answer
        for q in inst["queries"]:
            if "step" in q:
                continue
            qi = dict(inst, steps=[dict(inst["steps"][-1], joint=J_mc.tolist())])
            if q["kind"] == "cond" and exact_answer(inst, dict(q, kind="noul", event=q["given"], neg=False))[0] < 0.05:
                continue
            np.testing.assert_allclose(exact_answer(inst, q), exact_answer(qi, q), atol=2.5 * tol)


def _mc_story_urn(p, obs, n, rng):
    """Independent simulation: pick an urn, draw with replacement T+2 times."""
    K, C = p["theta"].shape
    u = rng.choice(K, size=n, p=p["prior"])
    draws = np.stack([(rng.random(n)[:, None] > np.cumsum(p["theta"][u], 1)).sum(1) for _ in range(len(obs) + 2)], 1)
    keep = (draws[:, : len(obs)] == np.array(obs)).all(1)
    cols = [u, draws[:, len(obs)]] + ([draws[:, len(obs) + 1]] if p["three"] else [])
    J = np.zeros([K] + [C] * (len(cols) - 1))
    np.add.at(J, tuple(c[keep] for c in cols), 1.0)
    return J / keep.sum(), keep.sum()


def _mc_story_cards(p, obs, n, rng):
    """Independent simulation: pick a deck, shuffle it, deal the first T cards then the next two."""
    cnt = p["counts"].astype(int)
    K, C = cnt.shape
    T = len(obs)
    J = np.zeros((K, C, C))
    d = rng.choice(K, size=n, p=p["prior"])
    acc = 0
    for k in range(K):
        m = int((d == k).sum())
        deck = np.repeat(np.arange(C), cnt[k])
        keys = rng.random((m, len(deck)))
        order = deck[np.argsort(keys, 1)][:, : T + 2]
        keep = (order[:, :T] == np.array(obs)).all(1)
        acc += keep.sum()
        np.add.at(J, (np.full(keep.sum(), k), order[keep, T], order[keep, T + 1]), 1.0)
    return J / acc, acc


def _mc_story_weather(p, obs, n, rng):
    """Independent simulation: pick a regime, run the Markov chain for T+2 days."""
    T = len(obs)
    r = (rng.random(n) >= p["p_regime0"]).astype(int)
    w = (rng.random(n)[:, None] > np.cumsum(p["init"][r], 1)).sum(1)
    seq = [w]
    for _ in range(T + 1):
        w = (rng.random(n)[:, None] > np.cumsum(p["trans"][r, w], 1)).sum(1)
        seq.append(w)
    seq = np.stack(seq, 1)
    keep = (seq[:, :T] == np.array([o[1] for o in obs])).all(1)
    J = np.zeros((2, 3, 3))
    np.add.at(J, (r[keep], seq[keep, T], seq[keep, T + 1]), 1.0)
    return J / keep.sum(), keep.sum()


@pytest.mark.parametrize("fam,story", [("urn", _mc_story_urn), ("cards", _mc_story_cards),
                                       ("weather", _mc_story_weather)])
def test_monte_carlo_independent_story(fam, story):
    rng = np.random.default_rng(1)
    for i in range(3):
        world, inst, tr = _world_and_trace(fam, i)
        J_mc, acc = story(world.params, tr["obs"], 600_000, rng)
        assert acc >= 2000
        tol = 5 * np.sqrt(0.25 / acc) + 1e-3
        J = joint_array(inst)
        assert np.abs(J - J_mc).max() < tol, (fam, i, np.abs(J - J_mc).max(), tol)


# ---------------------------------------------------------------------------------------------------------------------
# likelihoods, gold, perms, martingale
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", ALL)
def test_likelihoods_normalised(fam):
    for i in range(10):
        world, inst, tr = _world_and_trace(fam, i)
        obs = tr["obs"]
        post = world.prior / world.prior.sum()
        for k in range(world.T):
            alts = world.alternatives(k, obs[:k])
            S = sum(world.lik(a, obs[:k]) for a in alts)
            live = post > 0
            np.testing.assert_allclose(S[live], 1.0, atol=1e-12)
            post = world.posterior(obs[: k + 1])
        assert tuple(world.proj[tr["z"]]) == tuple(inst["gold"][v["name"]] for v in inst["variables"])


@pytest.mark.parametrize("fam", ALL)
def test_perms_leave_final_joint_unchanged(fam):
    n_with = 0
    for i in range(15):
        world, inst, tr = _world_and_trace(fam, i)
        if not world.exchangeable:
            assert "perms" not in inst
            continue
        assert inst["perms"], "exchangeable family must provide perms"
        n_with += 1
        J = joint_array(inst)
        for p in inst["perms"]:
            obs_p = [tr["obs"][j] for j in p]
            Jp = world.project(world.posterior(obs_p))
            np.testing.assert_allclose(Jp, J, atol=MART_TOL)
            # the permuted state text contains exactly the same evidence lines
            assert sorted(state_text(inst, order=p).split("\n")) == sorted(state_text(inst).split("\n"))
    if FAMILIES[fam].__name__.endswith(("sensor", "weather", "randhmm")):
        assert n_with == 0


@pytest.mark.parametrize("fam", ALL)
def test_martingale_identity(fam):
    for i in range(25):
        inst = _gen(fam, i)
        T = len(inst["steps"])
        ax = var_index(inst, inst["mart_var"])
        for k, st in enumerate(inst["steps"]):
            if k == T - 1:
                assert "next_evidence" not in st
                continue
            ne = st["next_evidence"]
            probs = np.array([a["prob"] for a in ne])
            assert abs(probs.sum() - 1) < MART_TOL and (probs >= 0).all()
            cur = marginal(joint_array(inst, k), ax)
            expect = sum(a["prob"] * np.asarray(a["mart_post"]) for a in ne)
            np.testing.assert_allclose(expect, cur, atol=MART_TOL)
            # the realised alternative's posterior is the next step's marginal
            real = [a for a in ne if a["evidence"] == inst["steps"][k + 1]["evidence"]]
            assert len(real) == 1
            np.testing.assert_allclose(real[0]["mart_post"], marginal(joint_array(inst, k + 1), ax), atol=MART_TOL)


@pytest.mark.parametrize("fam", ["urn", "medical", "sensor", "spam", "randbn", "randhmm", "tracking"])
def test_next_evidence_probs_match_monte_carlo(fam):
    """P(next evidence | prefix) from the sim equals the empirical frequency under the generative process."""
    rng = np.random.default_rng(2)
    world, inst, tr = _world_and_trace(fam, 0)
    obs = tr["obs"]
    z = rng.choice(len(world.prior), size=400_000, p=world.prior / world.prior.sum())
    for k in range(1):
        past = obs[: k + 1]
        # condition on prefix k (step 0), then draw the next observation
        alts0 = world.alternatives(0, [])
        P0 = np.stack([world.lik(a, [])[z] for a in alts0], 1)
        d0 = (np.cumsum(P0, 1) < rng.random(len(z))[:, None]).sum(1)
        zz = z[d0 == alts0.index(obs[0])]
        alts = world.alternatives(1, past)
        P1 = np.stack([world.lik(a, past)[zz] for a in alts], 1)
        d1 = (np.cumsum(P1, 1) < rng.random(len(zz))[:, None]).sum(1)
        freq = np.bincount(d1, minlength=len(alts)) / len(zz)
        texts = [world.render(1, a) for a in alts]
        sim = {a["evidence"]: a["prob"] for a in inst["steps"][0]["next_evidence"]}
        for t, f in zip(texts, freq):
            assert abs(sim.get(t, 0.0) - f) < 0.01


# ---------------------------------------------------------------------------------------------------------------------
# nuisance augmentation, robust split, procedural prior coverage
# ---------------------------------------------------------------------------------------------------------------------

def _semantics(inst):
    return (json.dumps([st["joint"] for st in inst["steps"]]), json.dumps(inst["gold"]),
            json.dumps([[a["prob"] for a in st.get("next_evidence", [])] for st in inst["steps"]]),
            json.dumps([(q["id"], q["kind"], q.get("event"), q.get("given"), q.get("neg"), q.get("step"), q["text"])
                        for q in inst["queries"]]), json.dumps(inst["variables"]), inst.get("perms"))


@pytest.mark.parametrize("fam", ALL)
def test_nuisance_preserves_semantics(fam):
    n_changed = 0
    for i in range(20):
        inst = _gen(fam, i)
        aug = augment(inst, np.random.default_rng(i))
        assert validate(aug)
        assert _semantics(aug) == _semantics(inst)
        n_changed += aug["prelude"] != inst["prelude"]
        # every original prelude number-bearing sentence survives (possibly moved)
        assert len(aug["prelude"]) > len(inst["prelude"])
        for st0, st1 in zip(inst["steps"], aug["steps"]):
            assert st1["evidence"].startswith(st0["evidence"])
        assert aug["meta"]["nuisance"]
    assert n_changed == 20


def test_nuisance_keeps_json_parseable():
    n = 0
    for fam in ALL:
        for i in range(30):
            inst = _gen(fam, i)
            if inst["meta"].get("style") != "json":
                continue
            aug = augment(inst, np.random.default_rng(i))
            pre = aug["prelude"]
            obj = json.loads(pre[pre.index("{"):])
            orig = json.loads(inst["prelude"][inst["prelude"].index("{"):])
            assert all(obj[k] == v for k, v in orig.items())
            n += 1
    assert n > 20


def test_robust_split_pairs_with_test():
    for fam in TRAIN_FAMILIES:
        r, t = generate(fam, "robust", 7), generate(fam, "test", 7)
        assert r["split"] == "robust" and r["id"] == f"{fam}-robust-000007" and r["meta"]["paired_with"] == t["id"]
        assert _semantics(r) == _semantics(t)
        assert r["prelude"] != t["prelude"]
        assert validate(r)


def test_train_nuisance_fraction():
    flags = [bool(generate("urn", "train", i, nuisance=0.5)["meta"].get("nuisance")) for i in range(200)]
    assert 60 < sum(flags) < 140
    # the underlying world is the same as without nuisance
    for i in range(10):
        assert _semantics(generate("urn", "train", i, nuisance=1.0)) == _semantics(generate("urn", "train", i))


def test_randbn_prior_coverage():
    modes, structs, styles, themes, nvars, T = set(), set(), set(), set(), set(), set()
    for i in range(400):
        inst = _gen("randbn", i)
        m = inst["meta"]
        modes.add(m["mode"]); structs.add(m["structure"]); styles.add(m["style"]); themes.add(m["theme"])
        nvars.add(len(inst["variables"])); T.add(len(inst["steps"]))
        assert np.prod(joint_shape(inst)) <= 256
    assert modes == {"single", "plate"}
    assert {"chain", "fork", "collider", "naive", "random", "noisyor"} <= structs
    assert {"prose", "bullets", "json"} <= styles
    assert len(themes) >= 35 and nvars == {2, 3, 4} and min(T) == 2 and max(T) >= 5


def test_randhmm_prior_coverage():
    Ks, Hs, sens = set(), set(), set()
    for i in range(200):
        inst = _gen("randhmm", i)
        Ks.add(inst["meta"]["K"]); Hs.add(inst["meta"]["H"]); sens.add(inst["meta"]["sensors"])
        assert "perms" not in inst
    assert Ks == {2, 3, 4} and len(Hs) >= 5 and sens == {1, 2}


# ---------------------------------------------------------------------------------------------------------------------
# template version 2 (release) and release post-processing
# ---------------------------------------------------------------------------------------------------------------------
import re as _re

from bookiebench.sims import template_version
from bookiebench.sims.core import degenerate_fraction, event_mask, finalize_release, mart_moves, shuffle_options

V2_BANNED = [r"^How likely", r"^What is the probability", r"^If .*, is it the case that", r"\ba either\b",
             r"\ban? either\b", r"positive a ", r"positive the ", r"\b1 (coin|die)s? that land\b", r"Headaches occurs",
             r"will be (vowel|consonant|blank)\b", r"Did an? .* happen", r"neither .* nor .* or "]


@pytest.mark.parametrize("fam", ALL)
def test_tv2_query_wording(fam):
    for i in range(25):
        inst = generate(fam, _split(fam), i, tv=2)
        assert inst["meta"]["tv"] == 2 and validate(inst)
        T = len(inst["steps"])
        J = joint_array(inst)
        for q in inst["queries"]:
            t = q["text"]
            for pat in V2_BANNED:
                assert not _re.search(pat, t), (fam, t)
            if q["kind"] == "marginal":
                assert q.get("step", -1) != T - 1  # no duplicate of the final marginal (#9)
                continue
            if q.get("neg") and len(q["event"]) >= 2:
                assert "both" in t
            if q["kind"] == "noul" and not q.get("neg") and len(q["event"]) >= 2:
                assert ("both that" in t or "of the following hold" in t or _re.search(", that .* and that", t)), t
            if q["kind"] == "cond":
                assert J[event_mask(inst, q["given"])].sum() > 1e-3  # (#8)
            assert names_absent_ok(inst, q)
        assert set(mart_queries(inst)) == set(range(T))


def names_absent_ok(inst, q):
    from bookiebench.sims.world import names_absent
    return q["kind"] != "marginal" or not names_absent(q["text"], inst["prelude"])


@pytest.mark.parametrize("fam", ["cards", "urn", "genetics", "randbn", "randhmm"])
def test_tv2_fewer_degenerate_queries(fam):
    def rate(tv):
        n = d = 0
        for i in range(60):
            inst = generate(fam, _split(fam), i, tv=tv)
            for q in inst["queries"]:
                if q["kind"] != "marginal":
                    p = exact_answer(inst, q)[0]
                    n += 1
                    d += p < 1e-9 or p > 1 - 1e-9
        return d / n
    r1, r2 = rate(1), rate(2)
    assert r2 <= 0.5 * r1 + 0.01 and r2 < 0.2, (r1, r2)


def test_template_version_context():
    from bookiebench.sims.common import get_tv
    assert get_tv() == 2
    with template_version(1):
        assert get_tv() == 1
        assert "tv" not in generate("urn", "test", 0)["meta"]
    assert get_tv() == 2


@pytest.mark.parametrize("fam", ["urn", "alarm", "sensor", "randbn", "tracking", "factory"])
def test_shuffle_options_is_a_relabelling(fam):
    for i in range(15):
        a = generate(fam, _split(fam), i)
        b = finalize_release(a, seed=3)
        assert validate(b) and b == finalize_release(a, seed=3)
        perm = b["meta"]["option_perm"]
        for qa, qb in zip(a["queries"], b["queries"]):
            x, y = exact_answer(a, qa), exact_answer(b, qb)
            if qa["kind"] == "marginal":
                x = [x[j] for j in perm[qa["var"]]]
            np.testing.assert_allclose(x, y, atol=1e-12)
        for v in a["variables"]:
            nb = [w for w in b["variables"] if w["name"] == v["name"]][0]
            assert nb["options"][b["gold"][v["name"]]] == v["options"][a["gold"][v["name"]]]
        for sa, sb in zip(a["steps"], b["steps"]):
            for xa, xb in zip(sa.get("next_evidence", []), sb.get("next_evidence", [])):
                mp = [xa["mart_post"][j] for j in perm[a["mart_var"]]]
                np.testing.assert_allclose(mp, xb["mart_post"])
        from bookiebench.sims.core import unshuffle_options
        c = unshuffle_options(b)
        c["meta"].pop("release")
        assert dumps(c) == dumps({**a, "meta": a.get("meta", {})})


def test_release_one_instance():
    from bookiebench.sims.release import _one, keys
    line, k, a, df, mv, ndup = _one("v1", "urn", "test", 3, 0, 0.0, None)
    inst = json.loads(line)
    assert validate(inst) and inst["meta"]["release"] and inst["meta"]["release_attempt"] == a
    assert len(k) == 2 and (not mv or mart_moves(inst)) and abs(degenerate_fraction(inst) - df) < 1e-12
    # a banned key forces a redraw
    line2, k2, a2, *_ = _one("v1", "urn", "test", 3, 0, 0.0, set(k))
    assert a2 > a and k2 != k


def test_release_holdout_plan():
    from bookiebench.sims.release import HOLDOUT_TRAIN, plan
    jobs = plan()
    groups = {}
    for pack, fam, split, n, nuis, rel, group, train_rel in jobs:
        groups.setdefault(fam, set()).add((split, group))
    for fam in HOLDOUT_TRAIN:
        assert ("train", "keys_only") in groups[fam] and ("dev", "new_mechanics") in groups[fam]
    nm = {f for f, g in groups.items() if any(x[1] == "new_mechanics" for x in g)}
    assert nm == HOLDOUT_TRAIN | {"genetics", "tracking"} and len(nm) == 11
    written_train = {f for f, g in groups.items() if ("train", "train") in g}
    assert not (written_train & nm)
    assert len({f for f, g in groups.items() if any(x[1] == "in_family_v2" for x in g)}) == 13
