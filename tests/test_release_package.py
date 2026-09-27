"""Standalone-package tests: shipped data integrity, oracle exactness on the shipped files, the CLI pipeline
(generate -> run -> score) and the hidden-test generator."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bookiebench.metrics.core import aggregate, evaluate
from bookiebench.runners import baselines, core

ROOT = Path(__file__).resolve().parents[1]
REL = ROOT / "data" / "release"
EVAL_DIRS = ["test", "test_prior", "heldout", "val", "mechanics/dev", "programs/dev", "tables/dev"]
PY = sys.executable


def _files():
    return [f for d in EVAL_DIRS for f in sorted((REL / d).glob("*.jsonl"))]


def test_checksums_match():
    p = subprocess.run([PY, str(ROOT / "tools" / "checksums.py")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_manifest_covers_shipped_files():
    man = json.loads((REL / "sims_manifest.json").read_text())["files"]
    shipped = {str(f.relative_to(REL)) for f in _files()}
    assert shipped, "no shipped eval files"
    for rel in shipped:
        assert rel in man, rel
        n = sum(1 for _ in open(REL / rel))
        assert n == man[rel]["n"], (rel, n)
    assert not (REL / "train").exists(), "train is regenerable and must not be shipped"


@pytest.mark.parametrize("path", _files(), ids=lambda p: str(p.relative_to(REL)))
def test_oracle_exact_on_shipped(path):
    insts = core.load_instances(path)[:25]
    sc = baselines.OracleScorer(insts)
    P = core.plan_instances(insts)
    preds = core.assemble(P, sc.score(P.rows), "oracle")
    by = {}
    for p in preds:
        by.setdefault(p["id"], {})[p["variant"]] = p
    recs, _ = evaluate({i["id"]: i for i in insts}, by)
    m = aggregate(recs)
    assert m["answered_frac"] == 1.0
    assert m["kl"] < 1e-5 and m["dutch"] < 1e-6 and abs(m["skill"] - 1) < 1e-4  # EPS clipping
    if m["mart"] is not None:
        assert m["mart"] < 1e-6
    if m["optperm"] is not None:
        assert m["optperm"] < 1e-9


def test_realcoh_release_modes():
    man = json.loads((REL / "realcoh" / "manifest.json").read_text())
    assert len(man["sources"]) == 25
    for s, info in man["sources"].items():
        rows = [json.loads(l) for l in open(REL / "realcoh" / f"{s}.jsonl")]
        assert len(rows) == info["n"]
        for r in rows:
            ev = r["steps"][0]["evidence"]
            if info["release_mode"] == "keep":
                assert isinstance(ev, str) and ev
            else:
                assert ev is None and len(r["meta"]["evidence_sha256"]) == 64 and r["meta"]["source_ref"]
    assert (REL / "realcoh" / "NOTICE.md").exists()


STRESS = REL / "stress"
STRESS_DATA = ["anchoring", "base_rate", "conjunction", "disjunction", "format_chat", "format_csv", "format_email",
               "format_log", "format_markdown", "negation", "nested", "paraphrase", "lang_de", "lang_es", "lang_pt",
               "lang_zh"]
STRESS_RECIPE = ["long_4k", "long_8k", "long_16k", "long_32k", "scaling"]


def test_stress_layout_and_flags():
    man = json.loads((STRESS / "manifest.json").read_text())
    assert sorted(man["release_format"]["data"]) == sorted(STRESS_DATA)
    assert sorted(man["release_format"]["recipe"]) == sorted(STRESS_RECIPE)
    assert "lang_zh" in man["beta"] and "BETA" in man["beta"]["lang_zh"]
    assert set(man["excluded_v1"]) == {"lang_hu", "framing"}
    for t in ("lang_hu", "framing"):
        assert not (STRESS / t).exists()
    recipe = json.loads((STRESS / "recipe.json").read_text())
    assert sorted(recipe["transforms"]) == sorted(STRESS_RECIPE)
    for t in STRESS_DATA:
        files = sorted((STRESS / t).glob("*.jsonl"))
        assert len(files) == 11, t
        kept = sum(v["n_kept"] for v in man["transforms"][t].values())
        assert kept == sum(1 for f in files for _ in open(f)), t


@pytest.mark.parametrize("t", ["paraphrase", "lang_de", "lang_es", "lang_pt", "lang_zh"])
def test_stress_llm_items_pass_both_judges(t):
    man = json.loads((STRESS / "manifest.json").read_text())["transforms"][t]
    for f in sorted((STRESS / t).glob("*.jsonl")):
        ids = []
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            st = r["meta"]["stress"]
            for j in ("judge", "judge2"):
                assert st[j] and not st[j]["different"] and not st[j]["unparsed"], (t, r["id"], j)
            ids.append(r["meta"]["source_id"])
        assert ids == man[f.stem]["kept_source_ids"], (t, f.stem)


def test_stress_scaling_recipe_sample():
    from bookiebench import stress_rebuild
    rep = stress_rebuild.rebuild(["scaling"], None, families=["urn", "spam"], sample=2, log=lambda *_: None)
    assert all(v["ok"] for v in rep["scaling"].values())


def _long_ready():
    from bookiebench import stress_rebuild
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False
    return not stress_rebuild.check_long_inputs(json.loads(stress_rebuild.RECIPE.read_text()))


@pytest.mark.skipif(not _long_ready(), reason="long-context filler / tokenizer not in the HF cache")
def test_stress_long_recipe_sample():
    from bookiebench import stress_rebuild
    rep = stress_rebuild.rebuild(["long_4k", "long_16k"], None, families=["dice", "hiring"], sample=2,
                                 log=lambda *_: None)
    assert all(v["ok"] for t in rep.values() for v in t.values())


def test_cli_generate_run_score(tmp_path):
    env = dict(os.environ)
    data, res = tmp_path / "data", tmp_path / "results"
    subprocess.run([PY, "-m", "bookiebench.cli", "generate", "--out", str(data), "--packs", "v1", "--n-scale", "0.02",
                    "--eval-only", "--families", "urn", "genetics", "spam", "--workers", "2"],
                   check=True, capture_output=True, env=env)
    inputs = [str(data / d) for d in ("test", "val", "heldout")]
    for m in ("oracle", "uniform"):
        subprocess.run([PY, "-m", "bookiebench.cli", "run", m, *inputs, "--out", str(res / m)], check=True,
                       capture_output=True)
    p = subprocess.run([PY, "-m", "bookiebench.cli", "score", str(res / "oracle"), "--data", str(data)],
                       check=True, capture_output=True, text=True)
    assert "| overall |" in p.stdout
    rep = json.loads((res / "oracle" / "report.json").read_text())["overall"]
    assert rep["kl"] < 1e-5 and rep["dutch"] < 1e-6 and rep["skill"] > 0.999
    p = subprocess.run([PY, "-m", "bookiebench.cli", "compare", str(res / "oracle"), str(res / "uniform"),
                        "--data", str(data), "--recompute"], check=True, capture_output=True, text=True)
    assert "ref:uniform_joint" in p.stdout and "uniform" in p.stdout


SCRIPT = ROOT / "scripts" / "make_hidden_test.py"
DUMMY = "PUBLIC-TEST-VALUE-NOT-A-SECRET"


def test_hidden_requires_secret(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "BOOKIEBENCH_SECRET_SEED"}
    p = subprocess.run([PY, str(SCRIPT), "--edition", "t", "--out", str(tmp_path / "h")], capture_output=True,
                       text=True, env=env)
    assert p.returncode != 0 and "BOOKIEBENCH_SECRET_SEED" in (p.stdout + p.stderr)


def test_hidden_refuses_repo_path():
    env = dict(os.environ, BOOKIEBENCH_SECRET_SEED=DUMMY)
    env.pop("BOOKIEBENCH_ALLOW_IN_REPO", None)
    p = subprocess.run([PY, str(SCRIPT), "--edition", "t", "--out", str(ROOT / "data" / "hidden"),
                        "--no-train-dedupe"], capture_output=True, text=True, env=env)
    assert p.returncode != 0 and "refusing" in (p.stdout + p.stderr)


def test_hidden_deterministic_disjoint(tmp_path):
    from bookiebench.sims import release
    from bookiebench.sims.core import iter_jsonl, validate
    env = dict(os.environ, BOOKIEBENCH_SECRET_SEED=DUMMY)
    args = ["--edition", "t", "--no-train-dedupe", "--n-scale", "0.02", "--packs", "v1",
            "--families", "urn", "genetics", "--workers", "2"]
    for o in ("a", "b"):
        subprocess.run([PY, str(SCRIPT), *args, "--out", str(tmp_path / o)], check=True, capture_output=True, env=env)
    man = json.loads((tmp_path / "a" / "hidden_manifest.json").read_text())
    assert DUMMY not in json.dumps(man) and man["publishable"] is False
    for rel in man["files"]:
        a, b = (tmp_path / "a" / rel).read_bytes(), (tmp_path / "b" / rel).read_bytes()
        assert a == b, rel
        pub = {k for i in iter_jsonl(REL / rel) for k in release.keys(i)}
        for inst in iter_jsonl(tmp_path / "a" / rel):
            assert inst["id"].startswith("hiddent-")
            validate(inst)
            assert not set(release.keys(inst)) & pub
