"""Tests for the BookieBench v2 `mechanics` pack: SPEC validity, determinism, exactness against independent physical
Monte Carlo simulations, likelihood normalisation, perms invariance, martingale identity, levels, scale and CLI."""
from __future__ import annotations

import json

import numpy as np
import pytest

from bookiebench.sims.core import (dumps, exact_answer, joint_array, joint_shape, loads, marginal, state_text, var_index)
from bookiebench.sims.packs.mechanics import FAMILIES, generate, make_world, validate
from bookiebench.sims.world import build_instance

ALL = list(FAMILIES)
TOL = 1e-6


def _wit(fam, i, level=None, split="dev"):
    world, rng, lv = make_world(fam, split, i, seed=7, level=level)
    inst, tr = build_instance(world, rng, family=fam, split=split, idx=i, return_trace=True)
    return world, inst, tr


def test_registry():
    assert len(ALL) >= 15
    for fam, m in FAMILIES.items():
        assert m.NAME == fam and callable(m.make_world) and callable(m.simulate)


@pytest.mark.parametrize("fam", ALL)
def test_validate_and_tags(fam):
    levels = set()
    for i in range(40):
        inst = generate(fam, "dev", i, seed=7)
        assert validate(inst)
        assert inst["pack"] == "mechanics" and inst["family"] == fam and inst["transform"] == "none"
        assert inst["level"] in ("L0", "L1", "L2")
        levels.add(inst["level"])
        assert np.prod(joint_shape(inst)) <= 256
        assert 2 <= len(inst["variables"]) <= 4
        for q in inst["queries"]:
            assert q["text"][0].isupper() and q["text"].endswith("?")
        assert inst["meta"]["style"] in ("prose", "json")
    assert levels == {"L0", "L1", "L2"}


@pytest.mark.parametrize("fam", ALL)
def test_levels(fam):
    for i in range(8):
        for lv in range(3):
            inst = generate(fam, "dev", i, seed=3, level=lv)
            validate(inst)
            assert inst["level"] == f"L{lv}"
            if lv == 0:
                assert len(inst["variables"]) == 2 and len(inst["steps"]) <= 2


@pytest.mark.parametrize("fam", ALL)
def test_deterministic_and_roundtrip(fam):
    a, b = generate(fam, "dev", 5), generate(fam, "dev", 5)
    assert dumps(a) == dumps(b)
    assert loads(dumps(a)) == a
    assert dumps(generate(fam, "dev", 6)) != dumps(a)
    assert dumps(generate(fam, "train", 5)) != dumps(a)
    assert dumps(generate(fam, "dev", 5, seed=1)) != dumps(a)


@pytest.mark.parametrize("fam", ALL)
def test_scale_knob(fam):
    for i in range(4):
        inst = generate(fam, "dev", i, scale=2)
        assert validate(inst)
        assert inst["meta"]["scale"] == 2


# ---------------------------------------------------------------------------------------------------------------------
# exactness
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", ALL)
def test_monte_carlo_physical(fam):
    """Independent simulation of the physical story (shuffle / place / run the process), rejection on the evidence."""
    rng = np.random.default_rng(11)
    m = FAMILIES[fam]
    for lv in range(3):
        world, inst, tr = _wit(fam, lv, level=lv)
        obs = tr["obs"]
        sims = []
        for _ in range(60000):
            s = m.simulate(world, rng)
            if s is not None:
                sims.append(s)
        assert len(sims) > 500, "simulator rejects almost everything"
        for k in range(world.T - 1, -1, -1):
            acc = [a for a, o in sims if list(o[: k + 1]) == list(obs[: k + 1])]
            if len(acc) >= 1500:
                break
        assert len(acc) >= 300, (fam, lv, len(acc))
        J = np.zeros(world.shape)
        for a in acc:
            J[tuple(a)] += 1
        J /= len(acc)
        tol = 5 * np.sqrt(0.25 / len(acc)) + 2e-3
        err = np.abs(J - joint_array(inst, k)).max()
        assert err < tol, (fam, lv, k, err, tol)


@pytest.mark.parametrize("fam", ALL)
def test_monte_carlo_generic(fam):
    """Rejection sampling from the enumerated generative process (z ~ prior, o ~ lik) agrees with the joints."""
    rng = np.random.default_rng(5)
    for i in range(3):
        world, inst, tr = _wit(fam, 10 + i)
        n = 200_000
        for k in range(world.T - 1, -1, -1):
            z = rng.choice(len(world.prior), size=n, p=world.prior / world.prior.sum())
            obs = tr["obs"]
            for j in range(k + 1):
                alts = world.alternatives(j, obs[:j])
                P = np.stack([world.lik(a, obs[:j])[z] for a in alts], 1)
                d = (np.cumsum(P, 1) < rng.random(len(z))[:, None] * P.sum(1, keepdims=True)).sum(1)
                z = z[d == alts.index(obs[j])]
            if len(z) >= 3000:
                break
        J = np.zeros(world.shape)
        np.add.at(J, tuple(world.proj[z].T), 1.0)
        J /= len(z)
        tol = 5 * np.sqrt(0.25 / len(z)) + 1e-3
        assert np.abs(J - joint_array(inst, k)).max() < tol


@pytest.mark.parametrize("fam", ALL)
def test_likelihoods_normalised_and_gold(fam):
    for i in range(10):
        world, inst, tr = _wit(fam, i)
        obs = tr["obs"]
        post = world.prior / world.prior.sum()
        for k in range(world.T):
            alts = world.alternatives(k, obs[:k])
            S = sum(world.lik(a, obs[:k]) for a in alts)
            live = post > 0
            np.testing.assert_allclose(S[live], 1.0, atol=1e-9)
            for a in alts:
                assert (world.lik(a, obs[:k])[live] >= 0).all()
            post = world.posterior(obs[: k + 1])
        assert tuple(world.proj[tr["z"]]) == tuple(inst["gold"][v["name"]] for v in inst["variables"])


@pytest.mark.parametrize("fam", ALL)
def test_exact_answers_consistent(fam):
    for i in range(10):
        inst = generate(fam, "dev", i)
        J = joint_array(inst)
        for q in inst["queries"]:
            a = exact_answer(inst, q)
            if q["kind"] == "marginal" and "step" not in q:
                np.testing.assert_allclose(a, marginal(J, var_index(inst, q["var"])), atol=1e-12)
            if q["kind"] == "noul":
                flipped = dict(q, neg=not q.get("neg", False))
                assert abs(a[0] + exact_answer(inst, flipped)[0] - 1) < 1e-9


# ---------------------------------------------------------------------------------------------------------------------
# perms and martingale
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", ALL)
def test_perms_invariance(fam):
    n_with = 0
    for i in range(15):
        world, inst, tr = _wit(fam, i)
        if not world.exchangeable or world.T < 2:
            assert "perms" not in inst
            continue
        assert inst["perms"]
        n_with += 1
        J = joint_array(inst)
        for p in inst["perms"]:
            Jp = world.project(world.posterior([tr["obs"][j] for j in p]))
            np.testing.assert_allclose(Jp, J, atol=TOL)
            assert sorted(state_text(inst, order=p).split("\n")) == sorted(state_text(inst).split("\n"))
    if FAMILIES[fam].NAME != "montyhall":
        assert n_with > 0


@pytest.mark.parametrize("fam", ALL)
def test_martingale_identity(fam):
    n_checked = 0
    for i in range(25):
        inst = generate(fam, "dev", i)
        T = len(inst["steps"])
        ax = var_index(inst, inst["mart_var"])
        for k, st in enumerate(inst["steps"]):
            if k == T - 1:
                assert "next_evidence" not in st
                continue
            ne = st["next_evidence"]
            probs = np.array([a["prob"] for a in ne])
            assert abs(probs.sum() - 1) < TOL and (probs >= 0).all()
            cur = marginal(joint_array(inst, k), ax)
            np.testing.assert_allclose(sum(a["prob"] * np.asarray(a["mart_post"]) for a in ne), cur, atol=TOL)
            real = [a for a in ne if a["evidence"] == inst["steps"][k + 1]["evidence"]]
            assert len(real) == 1
            np.testing.assert_allclose(real[0]["mart_post"], marginal(joint_array(inst, k + 1), ax), atol=TOL)
            n_checked += 1
    assert n_checked > 0


@pytest.mark.parametrize("fam", ["blackjack", "minesweeper", "montyhall", "epidemic", "recapture"])
def test_next_evidence_matches_simulation(fam):
    """P(next evidence | step-0 evidence) from the sim equals the frequency under the physical simulator."""
    rng = np.random.default_rng(3)
    m = FAMILIES[fam]
    for i in range(20):
        world, inst, tr = _wit(fam, i, level=1 if fam == "minesweeper" else 2)
        if world.T >= 2:
            break
    n = 400_000 if fam == "minesweeper" else 80_000
    sims = [s for s in (m.simulate(world, rng) for _ in range(n)) if s is not None]
    o0 = tr["obs"][0]
    nxt = [o[1] for _, o in sims if o[0] == o0]
    assert len(nxt) > 1000
    alts = world.alternatives(1, [o0])
    texts = {world.render(1, a): a for a in alts}
    sim = {a["evidence"]: a["prob"] for a in inst["steps"][0]["next_evidence"]}
    for t, p in sim.items():
        f = np.mean([x == texts[t] for x in nxt])
        assert abs(f - p) < 5 * np.sqrt(0.25 / len(nxt)) + 1e-3, (t, f, p)


def test_cli(tmp_path):
    from bookiebench.sims.packs.mechanics.generate import main
    main(["--out", str(tmp_path), "--n", "6", "--n-train", "4", "--families", "poker", "gauge", "--workers", "2"])
    for split, n in (("dev", 6), ("train", 4)):
        for fam in ("poker", "gauge"):
            lines = (tmp_path / split / f"{fam}.jsonl").read_text().strip().split("\n")
            assert len(lines) == n
            inst = json.loads(lines[0])
            assert validate(inst) and inst["split"] == split
            assert dumps(inst) == dumps(generate(fam, split, 0))
    rep = json.loads((tmp_path / "dev_sanity.json").read_text())
    assert set(rep) == {"poker", "gauge"}


@pytest.mark.parametrize("fam", ALL)
def test_oracle_scores_zero(fam):
    """Exact answers are coherent: no Dutch book and zero KL under bookiebench.metrics."""
    from bookiebench.metrics.core import evaluate_instance
    from bookiebench.metrics.dutch import dutch_book
    for i in range(4):
        inst = generate(fam, "dev", i)
        answers = {q["id"]: exact_answer(inst, q) for q in inst["queries"]}
        assert dutch_book(inst, answers)["dutch"] < 1e-6
        rec = evaluate_instance(inst, {"base": {"answers": answers}})
        assert max(rec["kl_marg"] + rec["kl_bin"]) < 1e-4  # metric clips probabilities at ~1e-6


# ---------------------------------------------------------------------------------------------------------------------
# the state names everything the queries mention; no silently repeated noisy evidence
# ---------------------------------------------------------------------------------------------------------------------

_STOP = {"Door", "Is", "What", "How", "Which", "Who", "Given", "Suppose", "Assuming", "If", "In", "Among", "Across", "For",
         "Will", "Does", "Did", "Was", "Are", "Do", "At", "When", "After", "Of", "Where", "From", "Counting", "With",
         "Behind", "Would", "The", "Has", "Its", "Were", "On", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
         "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"}


@pytest.mark.parametrize("fam", ALL)
def test_query_names_appear_in_state(fam):
    import re
    for i in range(60):
        inst = generate(fam, "dev", i)
        st = state_text(inst)
        for q in inst["queries"]:
            names = set(re.findall(r"(?<![.?!]\s)(?<!^)\b([A-Z][a-z]{2,})\b", q["text"]))
            unk = {n for n in names if n not in st and n not in _STOP}
            assert not unk, (inst["id"], inst["meta"]["style"], q["text"], unk)
        for v in inst["variables"]:
            for o in v["options"]:
                for n in re.findall(r"\b([A-Z][a-z]{2,})\b", o):
                    assert n in st or n in _STOP, (inst["id"], o)


UNIQUE_EVIDENCE = ["matching", "birthday", "minesweeper", "epidemic", "raters", "reliability", "search", "tournament",
                   "forensic", "montyhall", "channel", "inventory", "poker"]


@pytest.mark.parametrize("fam", UNIQUE_EVIDENCE)
def test_no_verbatim_repeated_evidence(fam):
    """Repeated noisy tests carry distinct labels (sample A/B, run A/B, ...) and deterministic facts are not restated."""
    for i in range(300):
        ev = [s["evidence"] for s in generate(fam, "dev", i)["steps"]]
        assert len(set(ev)) == len(ev), (fam, i, ev)


def test_montyhall_pending_friend_is_specified():
    for i in range(300):
        inst = generate("montyhall", "dev", i)
        if "second_friend" in [v["name"] for v in inst["variables"]]:
            assert "second friend" in inst["prelude"].lower() and ("signals a door" in inst["prelude"] or
                                                                   "friend" in inst["prelude"])
            w, _, _ = make_world("montyhall", "dev", i)
            assert "friend" in w.params["kinds"]


def _answer_keys(args):
    fam, split, lo, hi = args
    out = []
    for i in range(lo, hi):
        J = joint_array(generate(fam, split, i))
        out.append((J.shape, tuple(np.round(J.ravel(), 4))))
    return out


@pytest.mark.parametrize("fam", list(FAMILIES))
def test_answer_vector_repeat_rate(fam):
    """Shortcut guard: < 5% of 300 dev instances may share their (rounded) final joint with any of 2000 train instances."""
    from concurrent.futures import ProcessPoolExecutor
    chunks = [(fam, "train", lo, lo + 250) for lo in range(0, 2000, 250)] + [(fam, "dev", lo, lo + 50) for lo in range(0, 300, 50)]
    with ProcessPoolExecutor(max_workers=14) as ex:
        res = list(ex.map(_answer_keys, chunks))
    train = {k for r in res[:8] for k in r}
    dev = [k for r in res[8:] for k in r]
    rate = float(np.mean([k in train for k in dev]))
    assert rate < 0.05, (fam, rate)
