# BookieBench leaderboard

> **Placeholder.** Fill this file from the upstream release leaderboard (`results/release/LEADERBOARD.md`,
> produced by `python -m bookiebench.runners.leaderboard`) after the pending metrics and group changes land.
> Do not hand-edit numbers into it.

When filled, it should state:
- the data (public dev subset or hidden edition + commitment hash) and the exact instance ids scored;
- per-group tables with the reference rows (`ref:uniform_joint`, `ref:indep_joint`, `ref:prior`, `ref:oracle`);
- realcoh split by `in_decider_train`;
- paired stress deltas on `kept_source_ids` only;
- per model: runner, variants, temperature, context limit (which `long_*` transforms were run).
