#!/usr/bin/env python
"""Maintainer tool: copy the shippable stress transforms from the upstream build and write the recipe for the rest.

    python tools/make_stress_recipe.py --src /path/to/upstream/data/release/stress

- Shipped as data: the deterministic transforms except `scaling` and `long_*`, plus the LLM transforms paraphrase,
  lang_de, lang_es, lang_pt and lang_zh. Only items that passed both judges on every segment are shipped (upstream
  finalize with --require judge judge2).
- Shipped as a recipe (`recipe.json`): long_4k/8k/16k/32k, because the filler is third-party text, and scaling,
  because of its size (about 260 MB). The recipe holds per-instance and per-file sha256 values of the upstream
  files, the pinned filler datasets and tokenizer, and fetch commands. `bookiebench rebuild-stress` regenerates and
  verifies them.
- The manifest is copied with `beta` and `release_format` annotations added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "data" / "release" / "stress"
LONG = ["long_4k", "long_8k", "long_16k", "long_32k"]
RECIPE_TRANSFORMS = LONG + ["scaling"]
LLM_SHIP = ["paraphrase", "lang_de", "lang_es", "lang_pt", "lang_zh"]
EXCLUDED = ["lang_hu", "framing"]
BETA = {
    "lang_zh": ("BETA: both LLM judges detect only 0.64 (judge 1) / 0.66 (judge 2) of planted query-negation flips in "
                "Chinese (logs/stress_judge_controls_judge{,2}.json upstream), versus >= 0.96 for de/es/pt and "
                "paraphrase; a rule-based double-negation filter removed 64 further items (logs/stress_finalize.json "
                "upstream, programmatic_dropped). Residual meaning errors in the kept queries are likely. Report "
                "lang_zh separately and do not pool it into headline stress numbers."),
}
FILLER = {
    "news": ("abisee/cnn_dailymail", "3.0.0", "validation", "96df5e686bee6baa90b8bee7c28b81fa3fa6223d",
             "abisee___cnn_dailymail/3.0.0/0.0.0/{rev}/cnn_dailymail-validation.arrow"),
    "wiki": ("microsoft/wiki_qa", "default", "train", "3f104672b5de699878fe7907afc486f0de325eb5",
             "microsoft___wiki_qa/default/0.0.0/{rev}/wiki_qa-train.arrow"),
    "forum": ("SetFit/20_newsgroups", "default", "test", "f1b91292074e7cfb69be58b642d583ec262f30ed",
              "SetFit___20_newsgroups/default/0.0.0/{rev}/20_newsgroups-test.arrow"),
}
TOKENIZER = ("Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
TESTED_WITH = {"transformers": "5.17.0", "tokenizers": "0.23.2", "pyarrow": "25.0.1"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="upstream data/release/stress")
    ap.add_argument("--hf-home", default=None, help="HF_HOME holding the filler cache (for the filler sha256 values)")
    a = ap.parse_args(argv)
    src = Path(a.src)
    man = json.loads((src / "manifest.json").read_text())
    for t in EXCLUDED:
        assert not (src / t).exists() or t not in man["transforms"], f"{t} is excluded from v1"
    # every shipped LLM item must carry two clean judge verdicts
    for t in LLM_SHIP:
        for f in sorted((src / t).glob("*.jsonl")):
            for line in open(f, encoding="utf-8"):
                st = json.loads(line)["meta"]["stress"]
                for j in ("judge", "judge2"):
                    v = st.get(j)
                    assert v and not v.get("different") and not v.get("unparsed"), (t, f.name, j)
    DST.mkdir(parents=True, exist_ok=True)
    shipped = []
    for t in sorted(man["transforms"]):
        if t in RECIPE_TRANSFORMS:
            continue
        (DST / t).mkdir(exist_ok=True)
        for f in sorted((src / t).glob("*.jsonl")):
            shutil.copy2(f, DST / t / f.name)
        shipped.append(t)
    hf = Path(a.hf_home) if a.hf_home else None
    recipe = {
        "version": 1, "seed": 0, "n_per_family": man["n_per_family"],
        "sources": {fam: p for fam, p in man["sources"].items()},
        "rebuild": "bookiebench rebuild-stress --long  (and: bookiebench rebuild-stress --transforms scaling)",
        "why": {"long_*": "filler is third-party text (CNN/DailyMail is ids-only in the licence review; WikiQA, "
                          "20 Newsgroups)", "scaling": "size (~260 MB), fully determined by the shipped sources"},
        "filler": {}, "tokenizer": {"repo": TOKENIZER[0], "revision": TOKENIZER[1],
                                    "fetch": f"huggingface-cli download {TOKENIZER[0]} tokenizer.json "
                                             f"tokenizer_config.json vocab.json merges.txt --revision {TOKENIZER[1]}"},
        "tested_with": TESTED_WITH, "transforms": {},
    }
    for name, (repo, cfg, split, rev, path) in FILLER.items():
        cp = path.format(rev=rev)
        ent = {"dataset": repo, "config": cfg, "split": split, "revision": rev, "cache_path": cp,
               "fetch": f"python -c \"import datasets; datasets.load_dataset('{repo}', '{cfg}', split='{split}', "
                        f"revision='{rev}')\""}
        if hf and (hf / "datasets" / cp).exists():
            ent["arrow_sha256"] = sha((hf / "datasets" / cp).read_bytes())
        recipe["filler"][name] = ent
    for t in RECIPE_TRANSFORMS:
        recipe["transforms"][t] = {}
        for f in sorted((src / t).glob("*.jsonl")):
            raw = f.read_bytes()
            lines = raw.decode("utf-8").split("\n")  # NOT splitlines(): filler text contains U+2028 etc.
            if lines and lines[-1] == "":
                lines.pop()
            recipe["transforms"][t][f.stem] = {"n": len(lines), "bytes": len(raw), "file_sha256": sha(raw),
                                               "line_sha256": [sha(l.encode("utf-8")) for l in lines]}
    (DST / "recipe.json").write_text(json.dumps(recipe, indent=0))
    man["release_format"] = {"data": shipped, "recipe": RECIPE_TRANSFORMS, "excluded_v1": EXCLUDED}
    man["beta"] = BETA
    man["judges"] = {"require": ["judge", "judge2"], "judge": "Qwen/Qwen3.8-27B-FP8 (the rewriting model, temperature 0)",
                     "judge2": "deepseek-v41-flash (second judge)",
                     "rule": "an LLM-transform item is kept only if both judges pass every segment"}
    (DST / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"shipped {len(shipped)} transforms as data; recipe for {RECIPE_TRANSFORMS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
