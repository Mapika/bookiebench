"""Post-hoc temperature for any runner's predictions (softmax(l/T) = normalise(p1 ** (1/T)) for p1 = softmax(l)).

    # fit one scalar on a calibration split, against the exact answers (mean KL over base marginal + noul/cond answers)
    python -m bookiebench.runners.temper fit results/Qwen3.5-4B-Base --data data/val
    # write results/<name>-T<t>/ with every answer rescaled (all variants)
    python -m bookiebench.runners.temper apply results/Qwen3.5-4B-Base --t 1.7

t is relative to the temperature the predictions were made at (the logit runner defaults to 1, so t is then the
temperature itself).
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


def pairs(pred_dir, data_dir):
    ex = _exact_fn(); out = []
    insts = {}
    for f in glob.glob(os.path.join(data_dir, "*.jsonl")):
        for line in open(f):
            if line.strip():
                i = json.loads(line); insts[i["id"]] = i
    for f in glob.glob(os.path.join(pred_dir, "*.jsonl")):
        for line in open(f):
            p = json.loads(line)
            if p["variant"] != "base" or p["id"] not in insts:
                continue
            inst = insts[p["id"]]
            for q in inst["queries"]:
                if q["id"] in p["answers"]:
                    e = ex(inst, q); e = list(e) if isinstance(e, (list, tuple)) else [float(e)]
                    out.append((e, p["answers"][q["id"]]))
    return out


def fit(pred_dir, data_dir):
    from scipy.optimize import minimize_scalar
    pr = pairs(pred_dir, data_dir)
    if not pr:
        raise SystemExit("no (exact, predicted) pairs: check --data matches the predictions")

    def loss(logt):
        t = math.exp(logt)
        return sum(_kl(e, rescale(p, t)) for e, p in pr) / len(pr)
    r = minimize_scalar(loss, bounds=(math.log(0.05), math.log(50)), method="bounded")
    return dict(t=math.exp(r.x), kl_fitted=r.fun, kl_t1=loss(0.0), n=len(pr))


def apply(pred_dir, t, out_dir=None, name=None):
    src = Path(pred_dir); name = name or f"{src.name}-T{t:.3g}"
    out = Path(out_dir or src.parent / name); out.mkdir(parents=True, exist_ok=True)
    for f in src.rglob("*.jsonl"):                             # recursive: stress/<transform>/ keeps its layout
        dst = out / f.relative_to(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as g:
            for line in open(f):
                p = json.loads(line); p["model"] = name
                p["answers"] = {k: rescale(v, t) for k, v in p["answers"].items()}
                g.write(json.dumps(p) + "\n")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fit"); a.add_argument("pred_dir"); a.add_argument("--data", required=True)
    a.add_argument("--apply", action="store_true", help="also write the rescaled copy")
    b = sub.add_parser("apply"); b.add_argument("pred_dir"); b.add_argument("--t", type=float, required=True)
    b.add_argument("--out", default=None); b.add_argument("--name", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        r = fit(args.pred_dir, args.data); print(json.dumps(r))
        if args.apply:
            print(apply(args.pred_dir, r["t"]))
    else:
        print(apply(args.pred_dir, args.t, args.out, args.name))


if __name__ == "__main__":
    main()
