"""Random latent model -> a concrete table of N rows, and the text formats it is rendered in.

The latent model is a small random Bayesian network over the chosen columns, optionally with a hidden class
variable that is a parent of several columns (so columns are correlated in varied ways). Categorical CPT rows are
Dirichlet draws with a random concentration; numeric columns are (log)normal with a parent-dependent shift. The
generated table itself is the object of every question (empirical distribution), so the latent model only has
to make tables look varied and realistic; it is never needed to compute an answer.
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import json

import numpy as np

from bookiebench.sims.common import pick


# ----------------------------------------------------------------------------------------------------------------------
# table generation
# ----------------------------------------------------------------------------------------------------------------------

def _num_draw(rng, spec, shift, n):
    if spec["log"]:
        x = np.exp(rng.normal(np.log(spec["mu"]) + shift, spec["sigma"], size=n))
    else:
        x = rng.normal(spec["mu"] + shift * spec["sigma"], spec["sigma"], size=n)
    x = np.clip(x, spec["lo"], spec["hi"])
    x = np.round(x, spec["dec"])
    if spec["dec"] == 0:
        return [int(v) for v in x]
    return [float(v) for v in x]


def choose_columns(rng, dom, n_cols):
    cats = [c for c in dom["cols"] if c[0] == "cat"]
    nums = [c for c in dom["cols"] if c[0] == "num"]
    n_cat = min(len(cats), max(2, n_cols - int(rng.integers(0, min(len(nums), 2) + 1))))
    n_num = min(len(nums), max(0, n_cols - n_cat))
    ci = sorted(int(i) for i in rng.choice(len(cats), size=n_cat, replace=False))
    ni = sorted(int(i) for i in rng.choice(len(nums), size=n_num, replace=False)) if n_num else []
    return [cats[i] for i in ci] + [nums[i] for i in ni]


def generate_table(rng, dom, N, cols):
    """rows: list of dicts with id, time and every column in `cols`."""
    K = int(pick(rng, [1, 2, 2, 3]))
    z = rng.choice(K, size=N, p=rng.dirichlet(np.full(K, 2.0)))
    order = [cols[int(i)] for i in rng.permutation(len(cols))]
    vals: dict = {}
    codes: dict = {}  # column -> small integer code per row (for use as a parent)
    for j, c in enumerate(order):
        kind, name = c[0], c[1]
        parents = []
        if K > 1 and rng.random() < 0.65:
            parents.append(z)
        prev = order[:j]
        if prev and rng.random() < 0.6:
            p = prev[int(rng.integers(len(prev)))]
            parents.append(codes[p[1]])
        if parents:
            sizes = [int(p.max()) + 1 for p in parents]
            cfg = np.zeros(N, dtype=int)
            for p, s in zip(parents, sizes):
                cfg = cfg * s + p
            n_cfg = int(np.prod(sizes))
        else:
            cfg = np.zeros(N, dtype=int)
            n_cfg = 1
        if kind == "cat":
            V = c[2]
            alpha = float(pick(rng, [0.3, 0.6, 1.0, 2.0, 5.0]))
            base = rng.dirichlet(np.full(len(V), 1.5))
            out = np.zeros(N, dtype=int)
            for g in range(n_cfg):
                m = cfg == g
                row = rng.dirichlet(alpha * len(V) * base + 0.05)
                out[m] = rng.choice(len(V), size=int(m.sum()), p=row)
            vals[name] = [V[i] for i in out]
            codes[name] = out
        else:
            spec = c[2]
            out = [None] * N
            shifts = rng.normal(0, 0.6, size=n_cfg)
            for g in range(n_cfg):
                idx = np.nonzero(cfg == g)[0]
                draws = _num_draw(rng, spec, float(shifts[g]), len(idx))
                for i, v in zip(idx, draws):
                    out[int(i)] = v
            vals[name] = out
            arr = np.array(out, dtype=float)
            codes[name] = (arr > np.median(arr)).astype(int)
    # ids and timestamps
    start_id = int(rng.integers(100, 90000))
    ids, cur = [], start_id
    for _ in range(N):
        ids.append(dom["id_fmt"].format(n=cur))
        cur += int(rng.integers(1, 4))
    t0 = dt.datetime(int(rng.integers(2021, 2025)), int(rng.integers(1, 13)), int(rng.integers(1, 28)),
                     int(rng.integers(0, 24)), int(rng.integers(0, 60)), int(rng.integers(0, 60)))
    date_only = dom["time_col"] in ("order_date", "last_review", "shipped_at", "admitted_at", "opened")
    if date_only:
        span_days = float(pick(rng, [30, 90, 365, 730]))
        gaps = rng.exponential(span_days / N, size=N)
    else:
        span_min = float(pick(rng, [60, 600, 1440, 4320]))
        gaps = rng.exponential(span_min / N, size=N)
    times, t = [], t0
    for g in gaps:
        t = t + (dt.timedelta(days=float(g)) if date_only else dt.timedelta(minutes=float(g)))
        times.append(t.strftime("%Y-%m-%d") if date_only else t.strftime("%Y-%m-%d %H:%M:%S"))
    rows = []
    for i in range(N):
        r = {dom["id_col"]: ids[i], dom["time_col"]: times[i]}
        for c in cols:
            r[c[1]] = vals[c[1]][i]
        rows.append(r)
    return rows, {"latent_classes": K, "date_only": date_only}


# ----------------------------------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------------------------------

FORMATS = ["csv", "json", "jsonl", "markdown", "log"]


def fmt_cell(v, spec=None):
    if isinstance(v, float):
        dec = spec["dec"] if spec else 2
        return f"{v:.{dec}f}"
    return str(v)


def render_rows(rows, columns, fmt, specs, rng_state=None, header=True):
    """Render rows (list of dicts) with the given column order. specs: {col: numeric spec} for float formatting."""
    def cell(r, c):
        return fmt_cell(r[c], specs.get(c))

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        if header:
            w.writerow(columns)
        for r in rows:
            w.writerow([cell(r, c) for c in columns])
        return "```csv\n" + buf.getvalue().rstrip("\n") + "\n```"
    if fmt == "markdown":
        lines = []
        if header:
            lines.append("| " + " | ".join(columns) + " |")
            lines.append("|" + "|".join("---" for _ in columns) + "|")
        for r in rows:
            lines.append("| " + " | ".join(cell(r, c) for c in columns) + " |")
        return "\n".join(lines)
    if fmt in ("json", "jsonl"):
        objs = []
        for r in rows:
            o = {}
            for c in columns:
                v = r[c]
                o[c] = round(v, specs[c]["dec"]) if isinstance(v, float) and c in specs else v
            objs.append(json.dumps(o, ensure_ascii=False))
        if fmt == "jsonl":
            return "```jsonl\n" + "\n".join(objs) + "\n```"
        return "```json\n[\n" + ",\n".join("  " + o for o in objs) + "\n]\n```"
    if fmt == "log":
        lines = []
        tcol = columns[1]
        for r in rows:
            kv = " ".join(f"{c}={cell(r, c)}" for c in columns if c != tcol)
            lines.append(f"{tcol}={str(r[tcol]).replace(' ', 'T')} {kv}")  # the time column is named, ISO-8601
        return "```\n" + "\n".join(lines) + "\n```"
    raise ValueError(fmt)


def parse_rows(text, fmt, columns=None):
    """Parse rendered rows back into a list of {col: str} dicts (used by the recount tests)."""
    body = text
    if "```" in body:
        parts = body.split("```")
        chunks = [parts[i] for i in range(1, len(parts), 2)]
    else:
        chunks = [body]
    rows = []
    for ch in chunks:
        lines = ch.split("\n")
        if lines and lines[0].strip() in ("csv", "json", "jsonl", ""):
            lines = lines[1:]
        lines = [l for l in lines if l.strip()]
        if fmt == "csv":
            rd = csv.reader(lines)
            hdr = next(rd)
            rows += [dict(zip(hdr, r)) for r in rd]
        elif fmt in ("json", "jsonl"):
            txt = "\n".join(lines).strip()
            objs = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in lines]
            rows += [{k: fmt_cell(v) if not isinstance(v, float) else repr(v) for k, v in o.items()} for o in objs]
        elif fmt == "log":
            for l in lines:
                r = {}
                for i, kv in enumerate(l.split(" ")):
                    k, v = kv.split("=", 1)
                    r[k] = v.replace("T", " ") if i == 0 else v
                rows.append(r)
    if fmt == "markdown":
        rows = []
        hdr = None
        for l in body.split("\n"):
            l = l.strip()
            if not (l.startswith("|") and l.endswith("|")):
                hdr = None if not l else hdr
                continue
            cells = [c.strip() for c in l.strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue
            if hdr is None or cells == hdr:
                hdr = cells
                continue
            rows.append(dict(zip(hdr, cells)))
    return rows
