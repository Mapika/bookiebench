"""Build the real-world coherence benchmark: data/realcoh/<source>.jsonl in the SPEC §1 instance format.

    python -m bookiebench.realcoh.build            # 6 sources x 200 states, seed 0

Differences from the simulator instances:
- one step, whose "joint" is null (the true law is unknown); the state is prelude + "\n" + steps[0]["evidence"]
- "gold" holds only the variables the dataset labels (it may be empty)
- "paraphrases": {query id: [alternative question text, ...]} for the marginal queries; runners ask these as variants
  "para:<i>" (answers keyed by the original query id), and metrics report `para` = mean TV to the base answer
- no "perms", "next_evidence" or "mart_var": evperm and mart are skipped, and so is kl (no joint)

Queries per state (natural English, templates drawn at random per instance):
all marginals (Choice), 2 conjunctions and 1 negation (Noul), 2 conditionals (Noul, "Given that ..., ...?"), and one
extra negated conjunction. The outcome space is the product of the variables' options, so dutch is well-defined.
"""
import argparse
import json
import os
import random
from pathlib import Path

from .sources import SOURCES

CONJ = [
    "Is it true both that {a} and that {b}?",
    "Would you say that {a}, and also that {b}?",
    "Are both of these correct: {a}; {b}?",
    "Is it the case that {a} and {b}?",
]
NEG = [
    "Is it false that {a}?",
    "Would it be wrong to say that {a}?",
    "Is it not the case that {a}?",
]
NEG_CONJ = [
    "Is it false that {a} and {b} at the same time?",
    "Is it not the case that both {a} and {b}?",
]
COND = [
    "Given that {g}, is it the case that {e}?",
    "Suppose {g}. In that case, would you say {e}?",
    "Assuming {g}, is it true that {e}?",
    "If {g}, would you conclude that {e}?",
]


def _pick_event(rng, variables, nvars):
    vs = rng.sample(range(len(variables)), nvars)
    return [(variables[v], rng.randrange(len(variables[v]["options"]))) for v in vs]


def make_queries(rng, variables):
    qs, para = [], {}
    for v in variables:
        qid = f"q{len(qs)}"
        qs.append({"id": qid, "kind": "marginal", "var": v["name"], "text": v["questions"][0], "options": list(v["options"])})
        para[qid] = list(v["questions"][1:])
    seen = set()

    def fresh(evs):
        key = tuple(sorted((v["name"], k) for v, k in evs))
        if key in seen:
            return False
        seen.add(key); return True

    n = 0
    while n < 2:                                               # conjunctions over two different variables
        (va, ka), (vb, kb) = _pick_event(rng, variables, 2)
        if not fresh([(va, ka), (vb, kb)]):
            continue
        qs.append({"id": f"q{len(qs)}", "kind": "noul", "event": {va["name"]: [ka], vb["name"]: [kb]},
                   "text": rng.choice(CONJ).format(a=va["clauses"][ka], b=vb["clauses"][kb])})
        n += 1
    (va, ka), = _pick_event(rng, variables, 1)                 # negation of a single option
    qs.append({"id": f"q{len(qs)}", "kind": "noul", "event": {va["name"]: [ka]}, "neg": True,
               "text": rng.choice(NEG).format(a=va["clauses"][ka])})
    (va, ka), (vb, kb) = _pick_event(rng, variables, 2)        # negated conjunction
    qs.append({"id": f"q{len(qs)}", "kind": "noul", "event": {va["name"]: [ka], vb["name"]: [kb]}, "neg": True,
               "text": rng.choice(NEG_CONJ).format(a=va["clauses"][ka], b=vb["clauses"][kb])})
    n, tries = 0, 0
    while n < 2 and tries < 100:                               # conditionals P(E | G), E and G on different variables
        tries += 1
        (ve, ke), (vg, kg) = _pick_event(rng, variables, 2)
        key = ("cond", ve["name"], ke, vg["name"], kg)
        if key in seen:
            continue
        seen.add(key)
        qs.append({"id": f"q{len(qs)}", "kind": "cond", "event": {ve["name"]: [ke]}, "given": {vg["name"]: [kg]},
                   "text": rng.choice(COND).format(g=vg["clauses"][kg], e=ve["clauses"][ke])})
        n += 1
    return qs, para


def build_source(name, n=200, seed=0):
    fn, variables, prelude = SOURCES[name]
    rng = random.Random(f"{seed}:{name}")
    states = fn(rng, n)
    out = []
    for i, (text, gold) in enumerate(states):
        qrng = random.Random(f"{seed}:{name}:{i}")
        qs, para = make_queries(qrng, variables)
        out.append({
            "id": f"realcoh-{name}-{i:06d}", "family": name, "split": "realcoh",
            "variables": [{"name": v["name"], "options": list(v["options"])} for v in variables],
            "steps": [{"evidence": text, "joint": None}],
            "prelude": prelude, "gold": gold, "queries": qs, "paraphrases": para,
            "meta": {"source": name},
        })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "data" / "realcoh"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default=",".join(SOURCES))
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    for s in a.sources.split(","):
        insts = build_source(s, a.n, a.seed)
        with open(os.path.join(a.out, f"{s}.jsonl"), "w") as f:
            for inst in insts:
                f.write(json.dumps(inst, ensure_ascii=False) + "\n")
        ng = sum(len(i["gold"]) for i in insts)
        print(f"{s}: {len(insts)} instances, {sum(len(i['queries']) for i in insts)} queries, {ng} gold labels")


if __name__ == "__main__":
    main()
