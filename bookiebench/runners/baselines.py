"""Model-free baseline runners (CPU, seconds): useful for smoke tests and as sanity rows.

    bookiebench run uniform data/release/test --out results/uniform
    bookiebench run oracle  data/release/test --out results/oracle

- `uniform`: every choice question gets 1/K per option and every yes/no question 0.5. This is *per-question* uniform,
  which is NOT coherent (a conjunction priced at 0.5 next to marginals at 0.5 is Dutch-bookable). The coherent,
  evidence-blind floor is `ref:uniform_joint`, which `bookiebench score/compare` compute internally.
- `oracle`: answers every question from the instance's exact joint (for `next:<k>:<j>` rows, from the exact
  `mart_post` of that alternative). It should score kl = 0, dutch = 0, skill = 1, sens = 1 on the simulator splits.
  On instances without an exact joint (realcoh) it falls back to the uniform answer.

Both plug into `bookiebench.runners.core` like any model, so they exercise the same row expansion, option permutation
and answer mapping a real model goes through.
"""
from __future__ import annotations

import argparse
import sys

from . import core


class UniformScorer:
    name = "uniform"

    def score(self, rows):
        return [[1.0 / len(r.options)] * len(r.options) if r.kind == "choice" else [0.5] for r in rows]


class OracleScorer:
    """Looks each row's state up among the input instances' rendered prefixes / evidence orders / next alternatives."""
    name = "oracle"

    def __init__(self, instances):
        from bookiebench.sims.core import exact_answer
        self._exact = exact_answer
        self.by_state = {}      # state text -> list of (inst, step, mart_post or None)
        self.qtext = {}         # (inst id, question text) -> query
        for inst in instances:
            if not inst.get("steps") or any(s.get("joint") is None for s in inst["steps"]):
                continue        # realcoh: no exact law
            n = len(inst["steps"])
            for k in range(n):
                self._put(core.render_state(inst, upto=k), (inst, k, None))
            for order in inst.get("perms") or []:
                self._put(core.render_state(inst, order=order), (inst, n - 1, None))
            for k, st in enumerate(inst["steps"]):
                for alt in st.get("next_evidence") or []:
                    self._put(core.render_state(inst, upto=k, extra=alt["evidence"]), (inst, k + 1, alt.get("mart_post")))
            for q in inst["queries"]:
                self.qtext[(inst["id"], q["text"])] = q
                for alt in (inst.get("paraphrases") or {}).get(q["id"], []):
                    self.qtext[(inst["id"], alt)] = q

    def _put(self, state, entry):
        self.by_state.setdefault(state, []).append(entry)

    @staticmethod
    def _uniform(r):
        return [1.0 / len(r.options)] * len(r.options) if r.kind == "choice" else [0.5]

    def score(self, rows):
        out = []
        for r in rows:
            ans = None
            for inst, k, mart_post in self.by_state.get(r.state, []):
                q = self.qtext.get((inst["id"], r.question))
                if q is None:
                    continue
                if mart_post is not None and q["kind"] == "marginal" and q.get("var") == core.mart_var(inst):
                    ex = [float(x) for x in mart_post]
                else:
                    ex = [float(x) for x in self._exact(inst, dict(q, step=k))]
                if r.kind == "choice":
                    opts = core.query_options(inst, q)
                    ans = [ex[opts.index(o)] for o in r.options]
                    s = sum(ans)
                    ans = [x / s for x in ans] if s > 0 else self._uniform(r)
                else:
                    ans = [min(1.0, max(0.0, ex[0]))]
                break
            out.append(ans if ans is not None else self._uniform(r))
        return out


def main(argv=None, which=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    if which is None:
        ap.add_argument("which", choices=["uniform", "oracle"])
    core.add_common_args(ap)
    args = ap.parse_args(argv)
    which = which or args.which
    if which == "uniform":
        sc = UniformScorer()
    else:
        insts = [i for f in core.input_files(args.inputs) for i in core.load_instances(f)]
        sc = OracleScorer(insts)
    if args.name:
        sc.name = args.name
    return core.run_cli(sc, args)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
