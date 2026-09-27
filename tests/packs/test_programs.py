"""Tests for the v2 `programs` pack: SPEC validity, exactness against Monte Carlo execution of the rendered code
(Python via exec, JavaScript via node), martingale identity, perms, determinism."""
from __future__ import annotations

import json
import re
import random
import shutil
import subprocess

import numpy as np
import pytest

from bookiebench.sims.core import dumps, exact_answer, joint_array, mart_queries, marginal, validate, var_index
from bookiebench.sims.packs.programs import FAMILIES, PACK, generate
from bookiebench.sims.packs.programs import lang
from bookiebench.sims.packs.programs.build import obs_code
from bookiebench.sims.packs.programs.lang import PY_PRELUDE, negate, render, render_expr

FAMS = list(FAMILIES)
SEED = 7


def _gen(fam, i, split="dev"):
    return generate(fam, split, i, seed=SEED, return_world=True)


# ---------------------------------------------------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fam", FAMS)
def test_validate_and_fields(fam):
    modes, styles, levels = set(), set(), set()
    for i in range(40):
        inst, _, w = _gen(fam, i)
        validate(inst)
        assert inst["pack"] == PACK and inst["family"] == fam and inst["level"] in ("L2", "L3")
        assert inst["transform"] == "none"
        assert int(np.prod([len(v["options"]) for v in inst["variables"]])) <= 256
        m = inst["meta"]
        modes.add(m["mode"]); styles.add(m["style"]); levels.add(inst["level"])
        assert "```" in inst["prelude"]
        for k, st in enumerate(inst["steps"][:-1]):
            assert "next_evidence" in st
        if m["mode"] == "observe":
            assert "perms" in inst
        else:
            assert "perms" not in inst
    assert modes == {"observe", "print"}
    assert styles == {"python", "js", "pseudo"}
    assert levels == {"L2", "L3"}


@pytest.mark.parametrize("fam", FAMS)
def test_deterministic(fam):
    for i in (0, 5):
        a = generate(fam, "dev", i, seed=SEED)
        b = generate(fam, "dev", i, seed=SEED)
        assert dumps(a) == dumps(b)
    assert dumps(generate(fam, "dev", 1, seed=SEED)) != dumps(generate(fam, "train", 1, seed=SEED))


@pytest.mark.parametrize("fam", FAMS)
def test_martingale_and_next_evidence(fam):
    for i in range(25):
        inst, trace, w = _gen(fam, i)
        mq = mart_queries(inst)
        ax = var_index(inst, inst["mart_var"])
        for k, st in enumerate(inst["steps"][:-1]):
            pk = np.array(exact_answer(inst, mq[k]))
            mix = sum(a["prob"] * np.array(a["mart_post"]) for a in st["next_evidence"])
            assert np.allclose(mix, pk, atol=1e-6)
            # the realised alternative's mart_post matches the next step's marginal
            nxt = inst["steps"][k + 1]["evidence"]
            alt = next(a for a in st["next_evidence"] if a["evidence"] == nxt)
            assert np.allclose(alt["mart_post"], marginal(joint_array(inst, k + 1), ax), atol=1e-6)


@pytest.mark.parametrize("fam", FAMS)
def test_perms_final_posterior_invariant(fam):
    n = 0
    for i in range(30):
        inst, trace, w = _gen(fam, i)
        if "perms" not in inst:
            continue
        J = w.project(w.posterior(trace["obs"]))
        for p in inst["perms"]:
            obs = [trace["obs"][j] for j in p]
            assert np.allclose(w.project(w.posterior(obs)), J, atol=1e-9)
            n += 1
    assert n > 0


def test_enumerator_exact_small():
    # x = randint(1, 3); y = flip(0.5); observe(x + (1 if y else 0) >= 3)
    prog = lang.Program((), (("assign", "x", ("randint", 1, 3)), ("assign", "y", ("flip", 50)),
                             ("observe", ("bin", ">=", ("bin", "+", ("var", "x"), ("ifexp", ("var", "y"), ("lit", 1), ("lit", 0))), ("lit", 3)))))
    dist, mass, _ = lang.enumerate_program(prog)
    assert mass == lang.Fraction(1, 2)
    px = {}
    for (fz, outs), p in dist.items():
        x = dict((k, v[1]) for k, v in fz)["x"]
        px[x] = px.get(x, 0) + p / mass
    assert px == {2: lang.Fraction(1, 3), 3: lang.Fraction(2, 3)}


def test_break_and_helpers():
    # geometric with cap via helper loop + early return, and a retry loop with break
    f = ("g", (), (("for", "i", 3, (("if", ((("flip", 50), (("return", ("bin", "+", ("var", "i"), ("lit", 1))),)),), None),)),
                   ("return", ("lit", 0))))
    body = (("assign", "k", ("call", "g", ())), ("assign", "n", ("lit", 0)),
            ("for", "t", 4, (("assign", "n", ("bin", "+", ("var", "n"), ("lit", 1))),
                             ("if", ((("bin", ">=", ("var", "n"), ("var", "k")), (("break",),)),), None))))
    dist, mass, _ = lang.enumerate_program(lang.Program((f,), body))
    pk = {}
    for (fz, outs), p in dist.items():
        env = dict((a, b[1]) for a, b in fz)
        pk[(env["k"], env["n"])] = pk.get((env["k"], env["n"]), 0) + p
    F = lang.Fraction
    assert pk == {(1, 1): F(1, 2), (2, 2): F(1, 4), (3, 3): F(1, 8), (0, 1): F(1, 8)}


# ---------------------------------------------------------------------------------------------------------------------
# Monte Carlo: execute the rendered Python program with sampling + rejection
# ---------------------------------------------------------------------------------------------------------------------

def _py_runner(prog, extra_lines, style):
    code = render(prog, "python") + "\n" + "\n".join(extra_lines)
    compiled = compile(code, "<prog>", "exec")
    base = {}
    exec(PY_PRELUDE, base)

    def fmt(x):
        return lang.fmt_value(x, style)

    def run(rng):
        out = []
        base["_rng"] = rng  # the primitives' globals are `base`
        g = dict(base)
        g["print"] = lambda *a: out.append(" ".join(fmt(x) for x in a))
        try:
            exec(compiled, g)
        except g["_Reject"]:
            return None
        return g, out
    return run


def _cells(w, g):
    return tuple(sp[1][lang._key(g[nm])] for nm, sp in zip(w.params["qvars"], w.params["specs"]))


def _mc_check(inst, trace, w, n_runs, rng):
    prog = w.params["program"]
    style = w.params["style"]
    mode = inst["meta"]["mode"]
    extra = []
    if mode == "observe":
        for ci, b in trace["obs"]:
            c = w.params["pool"][ci][0]
            extra.append(obs_code(c if b else negate(c), "python"))
    run_prior = _py_runner(prog, [], style)
    run_post = _py_runner(prog, extra, style)
    shape = tuple(len(v["options"]) for v in inst["variables"])
    Cp, Cq = np.zeros(shape), np.zeros(shape)
    for _ in range(n_runs):
        r = run_prior(rng)
        if r is not None:
            Cp[_cells(w, r[0])] += 1
        r = run_post(rng)
        if r is None:
            continue
        g, out = r
        if mode == "print" and tuple(out) != tuple(trace["obs"]):
            continue
        Cq[_cells(w, g)] += 1
    # prior over query cells (before any evidence)
    Jp = w.project(w.prior)
    for C, J in ((Cp, Jp), (Cq, joint_array(inst))):
        n = C.sum()
        assert n >= 300, f"too few accepted samples ({n}) for {inst['id']}"
        E = C / n
        se = np.sqrt(J * (1 - J) / n)
        assert np.all(np.abs(E - J) <= 5 * se + 0.01), (inst["id"], np.abs(E - J).max())


@pytest.mark.parametrize("fam", FAMS)
def test_monte_carlo_python(fam):
    rng = random.Random(11)
    checked = 0
    for i in range(40):
        inst, trace, w = _gen(fam, i)
        # rejection on the realised evidence: skip instances whose evidence is too unlikely for cheap MC
        pe = 1.0
        post = w.prior.copy()
        for k, o in enumerate(trace["obs"]):
            l = w.lik(o, trace["obs"][:k])
            pe *= float(post @ l)
            post = post * l
            post /= post.sum()
        if pe < 0.05:
            continue
        _mc_check(inst, trace, w, n_runs=int(min(40000, 3000 / pe)), rng=rng)
        checked += 1
        if checked >= 10:
            break
    assert checked >= 5


# ---------------------------------------------------------------------------------------------------------------------
# JavaScript rendering: run it under node and compare the prior over the query cells
# ---------------------------------------------------------------------------------------------------------------------

JS_EXHAUSTIVE = r"""
// Exhaustive enumeration of every random-choice sequence of each program under real node.js semantics.
const progs = JSON.parse(require("fs").readFileSync(0, "utf8"));
const CAP = 400000;
const out = [];
for (const P of progs) {
  const body = P.code + "\nreturn [" + P.names.join(", ") + "];";
  const f = new Function("flip", "randint", "choice", "observe", "console", body);
  const dist = {}; let rejected = 0, runs = 0, error = null, truncated = false;
  const stack = [[]];
  class Rej {}
  while (stack.length) {
    if (++runs > CAP) { truncated = true; break; }
    const prefix = stack.pop(); const cur = []; let pos = 0; let prob = 1;
    const pickIdx = (probs) => {
      let i;
      if (pos < prefix.length) { i = prefix[pos]; }
      else {
        const nz = probs.map((p, k) => k).filter(k => probs[k] > 0);
        i = nz[0];
        for (const k of nz.slice(1)) stack.push(cur.concat([k]));
      }
      cur.push(i); pos++; prob *= probs[i]; return i;
    };
    const flip = (p) => pickIdx([p, 1 - p]) === 0;
    const randint = (a, b) => a + pickIdx(Array.from({length: b - a + 1}, () => 1 / (b - a + 1)));
    const choice = (xs, w) => { if (!w) w = xs.map(() => 1); const t = w.reduce((a, b) => a + b, 0);
                                return xs[pickIdx(w.map(x => x / t))]; };
    const observe = (c) => { if (!c) throw new Rej(); };
    try {
      const vals = f(flip, randint, choice, observe, {log: () => {}});
      const k = JSON.stringify(vals);
      dist[k] = (dist[k] || 0) + prob;
    } catch (e) {
      if (e instanceof Rej) rejected += prob; else { error = String(e); break; }
    }
  }
  out.push({dist, rejected, error, truncated});
}
process.stdout.write(JSON.stringify(out));
"""


def _node_exhaustive(items):
    out = subprocess.run(["node", "-e", JS_EXHAUSTIVE], input=json.dumps(items), capture_output=True, text=True,
                         timeout=1200)
    assert out.returncode == 0, out.stderr[:2000]
    return json.loads(out.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize("fam", FAMS)
def test_js_rendering_every_program_exact(fam):
    """EVERY program (not only js-styled instances) rendered as JavaScript runs under node without errors, and the
    exhaustive node distribution over the query cells equals the exact prior."""
    items, meta = [], []
    for i in range(80):
        inst, trace, w = _gen(fam, i)
        items.append({"code": render(w.params["program"], "js"), "names": w.params["qvars"]})
        meta.append((inst, w))
    res = _node_exhaustive(items)
    n_exact = 0
    for (inst, w), r in zip(meta, res):
        assert r["error"] is None, (inst["id"], r["error"])
        if r["truncated"]:
            continue
        n_exact += 1
        shape = tuple(len(v["options"]) for v in inst["variables"])
        C = np.zeros(shape)
        for k, p in r["dist"].items():
            vals = json.loads(k)
            C[tuple(sp[1][lang._key(v)] for v, sp in zip(vals, w.params["specs"]))] += p
        assert np.allclose(C / C.sum(), w.project(w.prior), atol=1e-9), inst["id"]
    assert n_exact >= 60


def test_pseudo_rendering_balanced():
    for fam in FAMS:
        for i in range(20):
            inst, trace, w = _gen(fam, i)
            lines = [l.strip() for l in render(w.params["program"], "pseudo").splitlines()]
            assert sum(l.startswith("IF ") for l in lines) == sum(l == "END IF" for l in lines)
            assert sum(l.startswith("FOR ") for l in lines) == sum(l == "END FOR" for l in lines)
            assert sum(l.startswith("FUNCTION ") for l in lines) == sum(l == "END FUNCTION" for l in lines)


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
    for i in range(60):
        inst = generate(fam, "dev", i, seed=SEED)
        for q in inst["queries"]:
            n += 1
            t += is_trivial(inst, q)
            if q["kind"] == "cond":
                g = joint_array(inst)[__import__("bookiebench.sims.core", fromlist=["event_mask"]).event_mask(inst, q["given"])].sum()
                assert g >= 1e-3 - 1e-9
    assert t / n < 0.05, t / n


def test_pseudo_prints_match_literals():
    """pseudo-code writes TRUE/FALSE, so its printed output must spell booleans the same way."""
    seen = 0
    for fam in FAMS:
        for i in range(80):
            inst = generate(fam, "dev", i, seed=SEED)
            if inst["meta"]["style"] != "pseudo" or inst["meta"]["mode"] != "print":
                continue
            texts = [s["evidence"] for s in inst["steps"]] + [a["evidence"] for s in inst["steps"]
                                                               for a in s.get("next_evidence", [])]
            for t in texts:
                assert not re.search(r"\b(true|false|True|False)\b", t), t
            seen += 1
    assert seen > 0


def _replay_python(prog, style, cap=200_000):
    """Independent exact enumeration: execute the rendered Python once per random-choice sequence (odometer DFS)."""
    from fractions import Fraction as F
    code = compile(render(prog, "python"), "<prog>", "exec")

    class Rej(Exception):
        pass

    prefix, dist, n = [], [], 0
    while True:
        rec, prob = [], [F(1)]

        def pick(opts):
            c = prefix[len(rec)] if len(rec) < len(prefix) else 0
            rec.append((len(opts), c))
            prob[0] *= opts[c][1]
            return opts[c][0]

        g = {"flip": lambda p: pick([(True, F(str(p))), (False, 1 - F(str(p)))]),
             "randint": lambda a, b: pick([(v, F(1, b - a + 1)) for v in range(a, b + 1)]),
             "choice": lambda xs, w=None: pick([(x, F(wi, sum(w or [1] * len(xs))))
                                                for x, wi in zip(xs, w or [1] * len(xs)) if wi]),
             "observe": lambda c: None if c else (_ for _ in ()).throw(Rej()),
             "print": lambda *a: None, "__builtins__": {"min": min, "max": max, "abs": abs, "range": range}}
        try:
            exec(code, g)
            dist.append((prob[0], g))
        except Rej:
            pass
        n += 1
        assert n <= cap
        j = len(rec) - 1
        while j >= 0 and rec[j][1] + 1 >= rec[j][0]:
            j -= 1
        if j < 0:
            return dist, n
        prefix = [c for _, c in rec[:j]] + [rec[j][1] + 1]


@pytest.mark.parametrize("fam", FAMS)
def test_trace_cap_and_exact_replay(fam):
    """Every program has <= 1e5 raw traces (meta.n_traces, an upper bound), and an independent trace-replay of the
    rendered Python reproduces the exact prior over the query cells."""
    from bookiebench.sims.packs.programs.build import MAX_TRACES
    for i in range(25):
        inst, trace, w = _gen(fam, i)
        assert inst["meta"]["n_traces"] <= MAX_TRACES
        if inst["meta"]["n_traces"] > 20000:
            continue
        dist, n = _replay_python(w.params["program"], w.params["style"])
        assert n <= inst["meta"]["n_traces"]  # upper bound: replay stops a trace at a failing observe
        C = np.zeros(tuple(len(v["options"]) for v in inst["variables"]))
        tot = sum(p for p, _ in dist)
        for p, g in dist:
            C[_cells(w, g)] += float(p / tot)
        assert np.allclose(C, w.project(w.prior), atol=1e-9), inst["id"]


@pytest.mark.parametrize("fam", FAMS)
def test_release_compatible(fam):
    from bookiebench.sims.common import template_version
    from bookiebench.sims.core import finalize_release, unshuffle_options
    for i in range(6):
        with template_version(2):
            inst = generate(fam, "dev", i, seed=SEED)
        fin = finalize_release(inst, seed=1)
        validate(fin)
        back = unshuffle_options(fin)
        for q in inst["queries"]:
            q2 = next(x for x in back["queries"] if x["id"] == q["id"])
            assert np.allclose(exact_answer(inst, q), exact_answer(back, q2), atol=1e-7)
        with template_version(1):
            validate(generate(fam, "dev", i, seed=SEED))
