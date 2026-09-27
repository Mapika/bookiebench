# TODO(stress): final stress manifest pending

The stress pack (`bookiebench.sims.packs.stress`) is **not shipped yet**. The upstream DeepSeek second-judge pass on
the LLM transforms is still running. The upstream `data/release/stress/manifest.json` has no `judge2` field yet, so
the kept sets are not final.

What will go here once that field exists:

- `data/release/stress/<transform>/<family>.jsonl` for the filtered transforms. The upstream manifest currently
  lists 21: anchoring, base_rate, conjunction, disjunction, format_{chat,csv,email,log,markdown},
  lang_{de,es,pt,zh}, long_{4k,8k,16k,32k}, negation, nested, paraphrase, scaling. Sources are
  `data/release/test/` (7 families) and `data/release/heldout/` (4 families), 300 each.
- `data/release/stress/manifest.json`, which gives per transform and family `kept_source_ids` /
  `rejected_source_ids` / `kept_judge_clean_source_ids` and `kept_by_all`, plus `judge2`.
- Excluded, and not to be copied: `lang_hu` and `framing` (see `excluded_v1` in the manifest and the README
  limitations).

Checklist when copying:
1. Confirm that `manifest.json` has `judge2`, and that `lang_hu/` and `framing/` are absent.
2. Copy the directory. It is about 1.7 GB upstream, mostly `long_*`, so consider shipping the long-context
   transforms as a separate download.
3. Run `python tools/checksums.py --write` and `python -m pytest -q`
   (`test_stress_not_shipped_until_final_manifest` checks for the judge2 field).
4. Update README "Known limitations" with the judge-2 kept counts and the DATASHEET.
