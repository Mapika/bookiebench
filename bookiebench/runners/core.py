"""Generic runner core: instance JSONL -> prediction JSONL (SPEC §1 -> §2).

A model plugs in as a *scorer*: an object with

    name: str                                  written as "model" in every prediction line
    score(rows: list[Row]) -> list[list[float]]
        kind "choice": a probability vector over row.options, in the order shown
        kind "noul":   [p_yes]

The core expands every instance into typed questions for all variants, de-duplicates identical rows (same state, question,
options), sends them to the scorer in one list (the scorer batches), maps permuted options back to canonical order and
writes one prediction line per (instance, variant).

Variants (SPEC §2 / §3):
    base            every query at its own step (final step unless "step" is given)
    optperm:<seed>  marginal queries only, options shown in a seeded non-identity permutation, answers mapped back
    evperm:<i>      final-step queries, evidence in the order perms[i]  (i < max_evperm)
    next:<k>:<j>    the step-k martingale marginal query, asked on prefix k followed by steps[k].next_evidence[j]
    para:<i>        query q asked with the wording paraphrases[q][i] (realcoh), answer keyed by q's id
"""
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Row:
    state: str
    question: str
    options: tuple          # shown options (choice); ("yes", "no") placeholder for noul
    kind: str               # "choice" | "noul"


@dataclass
class Plan:
    rows: list = field(default_factory=list)            # unique rows
    index: dict = field(default_factory=dict)           # Row -> position in rows
    # (instance id, variant) -> list of (query id, row position, perm or None)
    slots: dict = field(default_factory=dict)
    order: list = field(default_factory=list)           # (instance, variant) in output order

    def add(self, row):
        k = self.index.get(row)
        if k is None:
            k = self.index[row] = len(self.rows); self.rows.append(row)
        return k


def load_instances(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def render_state(inst, order=None, upto=None, extra=None):
    """prelude + "\n" + evidence_0 ... evidence_k (one line each). order: evidence order (perms[i]); upto: last step index
    (inclusive) of the prefix; extra: an evidence string appended after the prefix (next_evidence alternative)."""
    steps = inst["steps"]
    order = list(range(len(steps))) if order is None else list(order)
    if upto is not None:
        order = order[:upto + 1]
    ev = [steps[i]["evidence"] for i in order]
    if extra is not None:
        ev.append(extra)
    return "\n".join([inst.get("prelude", "")] + ev)


def query_options(inst, q):
    if q.get("options"):
        return list(q["options"])
    for v in inst["variables"]:
        if v["name"] == q["var"]:
            return list(v["options"])
    raise KeyError(f"{inst['id']}/{q['id']}: no options for var {q.get('var')!r}")


def query_step(inst, q):
    last = len(inst["steps"]) - 1
    s = q.get("step")
    return last if s is None else (last + s if s < 0 else s)


def mart_var(inst):
    """The martingale variable: "mart_var" if given, else the variable with marginal queries at the most distinct steps."""
    if inst.get("mart_var"):
        return inst["mart_var"]
    steps = {}
    for q in inst["queries"]:
        if q["kind"] == "marginal" and "step" in q:
            steps.setdefault(q["var"], set()).add(query_step(inst, q))
    return max(steps, key=lambda v: len(steps[v])) if steps else None


def mart_queries(inst):
    """{step: query} of the martingale variable's marginal queries (explicit "step" wins over an implicit final step)."""
    mv = mart_var(inst); out = {}
    if mv is None:
        return out
    for q in inst["queries"]:
        if q["kind"] == "marginal" and q["var"] == mv:
            k = query_step(inst, q)
            if k not in out or "step" in q:
                out[k] = q
    return out


def seeded_perm(n, seed, key):
    """A non-identity permutation of range(n) (for n >= 2), reproducible from (seed, key)."""
    rng = random.Random(f"{seed}:{key}")
    p = list(range(n))
    for _ in range(100):
        rng.shuffle(p)
        if n < 2 or p != sorted(p):
            break
    return p


def _row(inst, q, state, perm=None):
    if q["kind"] == "marginal":
        opts = query_options(inst, q)
        shown = tuple(opts[i] for i in perm) if perm is not None else tuple(opts)
        return Row(state, q["text"], shown, "choice")
    if q["kind"] in ("noul", "cond"):
        return Row(state, q["text"], ("yes", "no"), "noul")
    raise ValueError(f"unknown query kind {q['kind']!r}")


def plan_instances(instances, optperm_seeds=(0, 1), max_evperm=2, do_next=True, variants=None):
    """variants: optional set of variant families to emit ({"base", "optperm", "evperm", "next"}); default all."""
    fams = set(variants or ("base", "optperm", "evperm", "next", "para"))
    P = Plan()

    def put(iid, var, entries):
        if entries:
            P.slots[(iid, var)] = entries; P.order.append((iid, var))

    for inst in instances:
        iid = inst["id"]; last = len(inst["steps"]) - 1
        if "base" in fams:
            put(iid, "base", [(q["id"], P.add(_row(inst, q, render_state(inst, upto=query_step(inst, q)))), None)
                              for q in inst["queries"]])
        if "optperm" in fams:
            for seed in optperm_seeds:
                ent = []
                for q in inst["queries"]:
                    if q["kind"] != "marginal":
                        continue
                    perm = seeded_perm(len(query_options(inst, q)), seed, f"{iid}/{q['id']}")
                    ent.append((q["id"], P.add(_row(inst, q, render_state(inst, upto=query_step(inst, q)), perm)), perm))
                put(iid, f"optperm:{seed}", ent)
        if "para" in fams:                                   # paraphrased wording of a query (realcoh)
            for qid, alts in (inst.get("paraphrases") or {}).items():
                q = next((x for x in inst["queries"] if x["id"] == qid), None)
                if q is None:
                    continue
                for i, text in enumerate(alts):
                    st = render_state(inst, upto=query_step(inst, q))
                    key = (iid, f"para:{i}")
                    ent = P.slots.get(key)
                    if ent is None:
                        ent = P.slots[key] = []; P.order.append(key)
                    ent.append((qid, P.add(_row(inst, dict(q, text=text), st)), None))
        if "evperm" in fams:
            for i, order in enumerate((inst.get("perms") or [])[:max_evperm]):
                st = render_state(inst, order=order)
                put(iid, f"evperm:{i}", [(q["id"], P.add(_row(inst, q, st)), None)
                                         for q in inst["queries"] if query_step(inst, q) == last])
        if "next" in fams and do_next:
            mq = mart_queries(inst)
            for k, step in enumerate(inst["steps"]):
                alts = step.get("next_evidence") or []
                if k not in mq or not alts:
                    continue
                q = mq[k]
                for j, alt in enumerate(alts):
                    st = render_state(inst, upto=k, extra=alt["evidence"])
                    put(iid, f"next:{k}:{j}", [(q["id"], P.add(_row(inst, q, st)), None)])
    return P


def assemble(P, probs, model):
    """probs: scorer output per unique row -> prediction dicts in P.order."""
    out = []
    for iid, var in P.order:
        ans = {}
        for qid, k, perm in P.slots[(iid, var)]:
            if probs[k] is None:                               # a scorer may leave an answer missing (API parse failure)
                continue
            p = [float(x) for x in probs[k]]
            if perm is not None:                               # shown position j holds canonical option perm[j]
                back = [0.0] * len(p)
                for j, ci in enumerate(perm):
                    back[ci] = p[j]
                p = back
            ans[qid] = p
        out.append({"id": iid, "model": model, "variant": var, "answers": ans})
    return out


def check_probs(row, p):
    n = 1 if row.kind == "noul" else len(row.options)
    assert len(p) == n, f"scorer returned {len(p)} values for a {row.kind} row with {n} options"
    if row.kind == "choice":
        s = sum(p); assert abs(s - 1) < 1e-3, f"choice probabilities sum to {s}"
    assert all(-1e-6 <= x <= 1 + 1e-6 for x in p)


def run_file(scorer, in_path, out_dir, log=print, **plan_kw):
    """Predict one instance file. Output: <out_dir>/<split>__<family>.jsonl. Returns a stats dict."""
    insts = load_instances(in_path)
    if not insts:
        return None
    split = insts[0].get("split") or Path(in_path).parent.name
    family = insts[0].get("family") or Path(in_path).stem
    P = plan_instances(insts, **plan_kw)
    t = time.time()
    if getattr(scorer, "wants_groups", False):
        probs = scorer.score(P.rows, groups=[[k for _, k, _ in P.slots[key]] for key in P.order])
    else:
        probs = scorer.score(P.rows)
    dt = time.time() - t
    for r, p in zip(P.rows, probs):
        if p is not None:
            check_probs(r, p)
    preds = assemble(P, probs, scorer.name)
    os.makedirs(out_dir, exist_ok=True)
    out = Path(out_dir) / f"{split}__{family}.jsonl"
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    os.replace(tmp, out)
    n_ref = sum(len(v) for v in P.slots.values())
    st = dict(file=str(in_path), out=str(out), instances=len(insts), lines=len(preds), answers=n_ref, rows=len(P.rows),
              seconds=round(dt, 2), rows_per_s=round(len(P.rows) / max(dt, 1e-9), 1))
    log(json.dumps(st))
    return st


def input_files(paths):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    return files


def add_common_args(ap):
    ap.add_argument("inputs", nargs="+", help="instance .jsonl files or directories (data/<split>)")
    ap.add_argument("--out", default=None, help="output dir (default results/<model>)")
    ap.add_argument("--name", default=None, help="model name in predictions and results/<name>")
    ap.add_argument("--optperm-seeds", default="0,1")
    ap.add_argument("--max-evperm", type=int, default=2)
    ap.add_argument("--no-next", action="store_true")
    ap.add_argument("--variants", default="base,optperm,evperm,next,para")
    ap.add_argument("--subdir-by-parent", action="store_true",
                    help="write each input file's predictions under <out>/<its parent dir name>/ (stress/<transform>)")
    ap.add_argument("--skip-existing", action="store_true", help="skip inputs whose output file already exists")
    ap.add_argument("--limit", type=int, default=None, help="first N instances per file (smoke tests)")
    ap.add_argument("--sample", type=int, default=None, help="a seeded random N instances per file (calibration runs)")
    ap.add_argument("--seed", type=int, default=0, help="seed of --sample")
    return ap


def run_cli(scorer, args, root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    out_dir = args.out or str(root / "results" / scorer.name)
    seeds = tuple(int(s) for s in args.optperm_seeds.split(",") if s != "")
    kw = dict(optperm_seeds=seeds, max_evperm=args.max_evperm, do_next=not args.no_next,
              variants=set(args.variants.split(",")))
    stats = []
    base_out = out_dir
    for f in input_files(args.inputs):
        out_dir = str(Path(base_out) / Path(f).parent.name) if getattr(args, "subdir_by_parent", False) else base_out
        if getattr(args, "skip_existing", False):
            first = next((json.loads(l) for l in open(f) if l.strip()), None)
            if first is None or (Path(out_dir) / f"{first.get('split') or Path(f).parent.name}__{first.get('family') or Path(f).stem}.jsonl").exists():
                continue
        if args.limit or getattr(args, "sample", None):
            import tempfile
            insts = load_instances(f)
            if getattr(args, "sample", None) and args.sample < len(insts):
                idx = sorted(random.Random(f"{args.seed}:{Path(f).name}").sample(range(len(insts)), args.sample))
                insts = [insts[i] for i in idx]
            insts = insts[:args.limit] if args.limit else insts
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
                for i in insts:
                    tf.write(json.dumps(i) + "\n")
            st = run_file(scorer, tf.name, out_dir, **kw)
            os.unlink(tf.name)
        else:
            st = run_file(scorer, f, out_dir, **kw)
        if st:
            stats.append(st)
    tot_rows = sum(s["rows"] for s in stats); tot_s = sum(s["seconds"] for s in stats)
    summary = dict(model=scorer.name, files=len(stats), instances=sum(s["instances"] for s in stats), rows=tot_rows,
                   seconds=round(tot_s, 1), rows_per_s=round(tot_rows / max(tot_s, 1e-9), 1),
                   **getattr(scorer, "info", lambda: {})())
    print(json.dumps(summary))
    out_dir = base_out
    os.makedirs(out_dir, exist_ok=True)
    splits = "+".join(sorted({Path(s["out"]).name.split("__")[0] for s in stats})) or "none"
    with open(Path(out_dir) / f"run_info__{splits}.json", "w") as f:
        json.dump(dict(summary=summary, files=stats), f, indent=1)
    return summary
