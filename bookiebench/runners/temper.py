"""Post-hoc temperature for any runner's predictions (softmax(l/T) = normalise(p1 ** (1/T)) for p1 = softmax(l)).

    # fit one scalar on a calibration split, against the exact answers (mean KL over base marginal + noul/cond answers)
    python -m bookiebench.runners.temper fit results/Qwen3.5-4B-Base --data data/val
    # write results/<name>-T<t>/ with every answer rescaled (all variants)
    python -m bookiebench.runners.temper apply results/Qwen3.5-4B-Base --t 1.7

t is relative to the temperature the predictions were made at (the logit runner defaults to 1, so t is then the
temperature itself).

Joint-emitting records (a "joint" field, the heads models): the temperature is applied to the joint,
p ∝ joint^(1/t), and every final-step answer is recomputed from the tempered joint with the scorer's own query
semantics (bookiebench.metrics.refs.answers_from_joint), so a coherent model stays coherent. Answers the stored joint does
not determine (earlier-step queries) and records without a joint (LLMs, decider, Julia, the heads' non-base variants)
keep per-answer tempering. fit/apply need the instances (--data) for this; without them every record is per-answer.
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path

EPS = 1e-12


def rescale(p, t):
    if len(p) == 1:
        q = [max(min(p[0], 1 - EPS), EPS), 1 - max(min(p[0], 1 - EPS), EPS)]
        return [rescale(q, t)[0]]
    lg = [math.log(max(x, EPS)) / t for x in p]; m = max(lg)
    e = [math.exp(x - m) for x in lg]; s = sum(e)
    return [x / s for x in e]


def _kl(p, q):
    if len(p) == 1:
        p, q = [p[0], 1 - p[0]], [q[0], 1 - q[0]]
    return sum(a * (math.log(max(a, EPS)) - math.log(max(b, EPS))) for a, b in zip(p, q) if a > 0)


def _exact_fn():
    try:
        from bookiebench.sims.core import exact_answer
        return exact_answer
    except Exception:
        from tests.test_runners import exact                  # fixture-only fallback
        return exact


def load_insts(*roots):
    """id -> instance for every *.jsonl under the given data dirs (recursive)."""
    insts = {}
    for r in roots:
        for f in sorted(Path(r).rglob("*.jsonl")):
            if f.name == "manifest.json":
                continue
            for line in open(f):
                if line.strip():
                    i = json.loads(line)
                    if "queries" in i:
                        insts[i["id"]] = i
    return insts


def temper_record(p, t, inst=None):
    """Answers of one prediction record at relative temperature t (see the module doc for the joint rule).
    Returns (answers, tempered joint or None)."""
    if p.get("joint") is None or inst is None:
        return {k: rescale(v, t) for k, v in p["answers"].items()}, None
    import numpy as np
    from bookiebench.metrics.exact import is_final
    from bookiebench.metrics.refs import answers_from_joint
    j = np.asarray(p["joint"], dtype=float)
    jt = np.asarray(rescale(j.ravel().tolist(), t)).reshape(j.shape)
    a = answers_from_joint(inst, jt)
    qs = {q["id"]: q for q in inst["queries"]}
    out = {}
    for k, v in p["answers"].items():
        q = qs.get(k)
        out[k] = a[k] if (q is not None and k in a and is_final(inst, q)) else rescale(v, t)
    return out, jt.tolist()


def records(pred_dir, data_dir):
    """[(exact answers by query id, prediction record, instance)] for the base records of pred_dir found in data_dir."""
    ex = _exact_fn(); out = []
    insts = load_insts(data_dir)
    for f in glob.glob(os.path.join(pred_dir, "*.jsonl")):
        for line in open(f):
            p = json.loads(line)
            if p["variant"] != "base" or p["id"] not in insts:
                continue
            inst = insts[p["id"]]; e = {}
            for q in inst["queries"]:
                if q["id"] in p["answers"]:
                    x = ex(inst, q); e[q["id"]] = list(x) if isinstance(x, (list, tuple)) else [float(x)]
            out.append((e, p, inst))
    return out


def pairs(pred_dir, data_dir):
    return [(e[k], p["answers"][k]) for e, p, _ in records(pred_dir, data_dir) for k in e]


def fit(pred_dir, data_dir):
    from scipy.optimize import minimize_scalar
    rec = records(pred_dir, data_dir)
    npairs = sum(len(e) for e, _, _ in rec)
    if not npairs:
        raise SystemExit("no (exact, predicted) pairs: check --data matches the predictions")

    def loss(logt):
        t = math.exp(logt); tot = 0.0
        for e, p, inst in rec:
            a, _ = temper_record(p, t, inst)
            tot += sum(_kl(e[k], a[k]) for k in e)
        return tot / npairs
    r = minimize_scalar(loss, bounds=(math.log(0.05), math.log(50)), method="bounded")
    return dict(t=math.exp(r.x), kl_fitted=r.fun, kl_t1=loss(0.0), n=npairs,
                joint_records=sum(p.get("joint") is not None for _, p, _ in rec))


def apply(pred_dir, t, out_dir=None, name=None, data=None):
    """data: instance dirs (str or list) for the joint rule; records whose instance is not found are per-answer."""
    insts = load_insts(*([data] if isinstance(data, (str, Path)) else (data or [])))
    src = Path(pred_dir); name = name or f"{src.name}-T{t:.3g}"
    out = Path(out_dir or src.parent / name); out.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*.jsonl"):                             # recursive: stress/<transform>/ keeps its layout
        dst = out / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as g:
            for line in open(f):
                p = json.loads(line); p["model"] = name
                p["answers"], jt = temper_record(p, t, insts.get(p["id"]))
                if jt is not None:
                    p["joint"] = jt
                g.write(json.dumps(p) + "\n")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit"); a.add_argument("pred_dir"); a.add_argument("--data", required=True)
    a.add_argument("--apply", action="store_true", help="also write the rescaled copy")
    b = sub.add_parser("apply"); b.add_argument("pred_dir"); b.add_argument("--t", type=float, required=True)
    b.add_argument("--out", default=None); b.add_argument("--name", default=None)
    b.add_argument("--data", nargs="*", default=None, help="instance dirs (needed for joint-emitting records)")
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        r = fit(args.pred_dir, args.data); print(json.dumps(r))
        if args.apply:
            print(apply(args.pred_dir, r["t"], data=args.data))
    else:
        print(apply(args.pred_dir, args.t, args.out, args.name, data=args.data))


if __name__ == "__main__":
    main()
