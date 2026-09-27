"""Instances for the `tables` pack.

Modes
  pick    The whole table (N rows) is in the prelude. One row is selected uniformly at random; the evidence steps are
          facts about the selected row ("its channel is chat", "its amount is at least 50"). The posterior after
          k facts is the empirical distribution of the rows consistent with them, so every answer is a ratio of
          counts. The latent is the row index, so this is an bookiebench.sims World: exact martingale, next_evidence
          (all possible values of the next fact) and perms (facts about one row are exchangeable).
  stream  Rows arrive in batches (the evidence steps). Questions are about the empirical distribution of all
          rows received so far ("If we pick a row uniformly at random from the records so far ..."), so the joint at
          step k is the count table of rows 0..n_k divided by n_k. The quantity is re-defined by each batch, so there
          is no next_evidence and the martingale metric is skipped (meta.mart_semantics = "empirical-so-far");
          perms (batch orders) are provided because the final table does not depend on the order.
"""
from __future__ import annotations

import numpy as np

from bookiebench.sims.common import get_tv, pick
from bookiebench.sims.world import Var, World, build_instance, _r

from . import model
from .queries_v2 import PICK_V2, STREAM_V2, make_queries_v2
from .schema import DOMAINS

DECIMALS = 8


class Reject(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# query columns
# ----------------------------------------------------------------------------------------------------------------------

def bucket_numeric(rng, spec, values):
    arr = np.array(values, dtype=float)
    lo, hi = np.quantile(arr, 0.1), np.quantile(arr, 0.9)
    cuts = [c for c in spec["cuts"] if lo < c <= hi]
    if not cuts:
        med = float(np.median(arr))
        cuts = [min(spec["cuts"], key=lambda c: abs(c - med))]
        if not (arr.min() < cuts[0] <= arr.max()):
            return None
    k = int(rng.integers(1, min(3, len(cuts)) + 1))
    edges = sorted(int(i) for i in rng.choice(len(cuts), size=k, replace=False))
    edges = [cuts[i] for i in edges]
    names = [f"under {edges[0]}"]
    for a, b in zip(edges[:-1], edges[1:]):
        names.append(f"at least {a} and under {b}")
    names.append(f"{edges[-1]} or more")
    return edges, names


def code_of(v, qc):
    if qc["type"] == "cat":
        return qc["options"].index(v)
    return int(np.searchsorted(qc["edges"], float(v), side="right"))


def choose_query_cols(rng, rows, cols, k):
    cands = []
    for c in cols:
        if c[0] == "cat" and len(c[2]) <= 5:
            vals = [r[c[1]] for r in rows]
            cnt = {v: vals.count(v) for v in c[2]}
            if max(cnt.values()) <= 0.95 * len(rows) and sum(x > 0 for x in cnt.values()) >= 2:
                cands.append(c)
        elif c[0] == "num":
            cands.append(c)
    rng.shuffle(cands)
    chosen, qspecs, cells = [], {}, 1
    for c in cands:
        if len(chosen) >= k:
            break
        if c[0] == "cat":
            qc = {"type": "cat", "options": list(c[2])}
        else:
            b = bucket_numeric(rng, c[2], [r[c[1]] for r in rows])
            if b is None:
                continue
            qc = {"type": "num", "edges": b[0], "options": b[1], "dec": c[2]["dec"]}
            codes = [code_of(r[c[1]], qc) for r in rows]
            if len(set(codes)) < 2:
                continue
        if cells * len(qc["options"]) > 256:
            continue
        cells *= len(qc["options"])
        chosen.append(c)
        qspecs[c[1]] = qc
    if len(chosen) < 2:
        raise Reject("not enough query columns")
    return chosen, qspecs


# ----------------------------------------------------------------------------------------------------------------------
# queries (table-flavoured wording; same SPEC query set as bookiebench.sims.world.make_queries)
# ----------------------------------------------------------------------------------------------------------------------

PICK_TPL = dict(
    conj2=["Is it the case that {a} and {b}?", "What is the probability that {a} and {b}?",
           "How likely is it that {a} and, at the same time, {b}?", "Is it true both that {a} and that {b}?"],
    conj3=["Is it true that {a}, that {b}, and that {c}?", "What is the probability that {a}, {b}, and {c}?"],
    neg1=["Is it false that {a}?", "Is it not the case that {a}?", "How likely is it that it is NOT true that {a}?"],
    neg2=["Is it false that both {a} and {b}?", "Is it not the case that both {a} and {b}?"],
    cond=["If {g}, is it the case that {a}?", "Suppose {g}. How likely is it then that {a}?",
          "Assuming {g}, what is the probability that {a}?", "Given that {g}, is it true that {a}?"],
)
STREAM_TPL = dict(
    conj2=["In what fraction of the {ents} so far is it true that {a} and {b}?",
           "If we pick one of the {ents} so far uniformly at random, how likely is it that {a} and {b}?",
           "What share of all {ents} received so far are ones where {a} and {b}?",
           "Among all {ents} so far, what fraction have both of these: {a}; {b}?"],
    conj3=["In what fraction of the {ents} so far is it true that {a}, {b}, and {c}?",
           "If we pick one of the {ents} so far uniformly at random, how likely is it that {a}, {b}, and {c}?"],
    neg1=["In what fraction of the {ents} so far is it NOT the case that {a}?",
          "If we pick one of the {ents} so far uniformly at random, how likely is it that it is not true that {a}?",
          "What share of the {ents} so far do not satisfy: {a}?"],
    neg2=["In what fraction of the {ents} so far is it false that both {a} and {b}?",
          "If we pick one of the {ents} so far at random, how likely is it that it is not the case that both {a} and {b}?"],
    cond=["Among the {ents} so far for which {g}, what fraction are ones where {a}?",
          "Restricting to {ents} so far where {g}, what share have: {a}?",
          "If we pick uniformly at random among the {ents} so far where {g}, how likely is it that {a}?"],
)


def _event_text(var: Var, idxs):
    return var.says(idxs)


def make_queries(V: list[Var], J: np.ndarray, rng, T: int, mart_var: str, tpl: dict, ents: str) -> list[dict]:
    nv = len(V)
    qs, seen = [], set()

    def key(q):
        return (q["kind"], q.get("neg", False), tuple(sorted((k, tuple(v)) for k, v in q.get("event", {}).items())),
                tuple(sorted((k, tuple(v)) for k, v in q.get("given", {}).items())))

    def add(q):
        if q["kind"] != "marginal":
            kk = key(q)
            if kk in seen:
                return False
            seen.add(kk)
        q["id"] = f"q{len(qs)}"
        qs.append(q)
        return True

    def ev(vi, multi=True):
        n = len(V[vi].options)
        if multi and n >= 3 and rng.random() < 0.25:
            return sorted(int(i) for i in rng.choice(n, size=2, replace=False))
        return [int(rng.integers(n))]

    F = lambda s, **kw: s.format(ents=ents, **kw)  # noqa: E731
    for v in V:
        add({"kind": "marginal", "var": v.name, "text": pick(rng, v.questions), "options": list(v.options)})
    n_conj, made = int(rng.integers(2, 4)), 0
    for _ in range(60):
        if made >= n_conj:
            break
        k = 3 if (nv >= 3 and rng.random() < 0.3) else 2
        vids = sorted(int(i) for i in rng.choice(nv, size=k, replace=False))
        e = {vi: ev(vi) for vi in vids}
        cl = [V[vi].says(e[vi]) for vi in vids]
        text = F(pick(rng, tpl["conj2"]), a=cl[0], b=cl[1]) if k == 2 else F(pick(rng, tpl["conj3"]), a=cl[0], b=cl[1], c=cl[2])
        made += add({"kind": "noul", "event": {V[vi].name: e[vi] for vi in vids}, "text": text})
    n_neg, made = int(rng.integers(1, 3)), 0
    for _ in range(60):
        if made >= n_neg:
            break
        if rng.random() < 0.6:
            vi = int(rng.integers(nv))
            e = {vi: ev(vi)}
            text = F(pick(rng, tpl["neg1"]), a=V[vi].says(e[vi]))
        else:
            vids = sorted(int(i) for i in rng.choice(nv, size=2, replace=False))
            e = {vi: ev(vi, False) for vi in vids}
            text = F(pick(rng, tpl["neg2"]), a=V[vids[0]].says(e[vids[0]]), b=V[vids[1]].says(e[vids[1]]))
        made += add({"kind": "noul", "event": {V[vi].name: e[vi] for vi in e}, "neg": True, "text": text})
    n_cond, made = int(rng.integers(1, 3)), 0
    for _ in range(60):
        if made >= n_cond:
            break
        a, g = (int(i) for i in rng.choice(nv, size=2, replace=False))
        e = ev(a, False)
        mg = J.sum(axis=tuple(i for i in range(nv) if i != g))
        ok = [i for i in range(len(V[g].options)) if mg[i] > 1e-6]
        if not ok:
            continue
        gi = [int(pick(rng, ok))]
        text = F(pick(rng, tpl["cond"]), g=V[g].says(gi), a=V[a].says(e))
        made += add({"kind": "cond", "event": {V[a].name: e}, "given": {V[g].name: gi}, "text": text})
    mv = next(v for v in V if v.name == mart_var)
    for k in range(T):
        add({"kind": "marginal", "var": mv.name, "step": k, "text": pick(rng, mv.questions), "options": list(mv.options)})
    return qs


# ----------------------------------------------------------------------------------------------------------------------
# facts about the selected row (pick mode)
# ----------------------------------------------------------------------------------------------------------------------

def fact_phrase(f):
    c, op, v = f["col"], f["op"], f["value"]
    if op in ("hour_lt", "hour_ge"):
        return f"{c} has a time of day {'before' if op == 'hour_lt' else 'at or after'} {int(v):02d}:00"
    return {"eq": f"{c} is {v}", "ne": f"{c} is not {v}", "ge": f"{c} is at least {v}", "lt": f"{c} is under {v}",
            "date_lt": f"{c} date is before {v}", "date_ge": f"{c} date is on or after {v}"}[op]


def fact_sql(f):
    c, op, v = f["col"], f["op"], f["value"]
    q = (lambda x: f"'{x}'")
    return {"eq": f"{c} = {q(v)}", "ne": f"{c} <> {q(v)}", "ge": f"{c} >= {v}", "lt": f"{c} < {v}",
            "hour_lt": f"HOUR({c}) < {v}", "hour_ge": f"HOUR({c}) >= {v}", "date_lt": f"{c} < '{v}'",
            "date_ge": f"{c} >= '{v}'"}[op]


def fact_holds(f, row):
    c, op, v = f["col"], f["op"], f["value"]
    x = row[c]
    if op == "eq":
        return str(x) == str(v)
    if op == "ne":
        return str(x) != str(v)
    if op == "ge":
        return float(x) >= float(v)
    if op == "lt":
        return float(x) < float(v)
    if op in ("hour_lt", "hour_ge"):
        h = int(str(x)[11:13])
        return h < v if op == "hour_lt" else h >= v
    if op in ("date_lt", "date_ge"):
        return (str(x) < v) if op == "date_lt" else (str(x) >= v)
    raise ValueError(op)


FACT_TPL = [
    "We learn that the selected {ent}'s {p}.",
    "Additional detail about the selected {ent}: its {p}.",
    "An auditor notes that for the selected {ent}, {p}.",
    "The selected {ent} matches the filter `{sql}`.",
]


def fact_groups(rows, cols, qspecs, meta_time, dom, rng):
    """Candidate fact groups: each is a list of mutually exclusive, exhaustive facts (the alternatives)."""
    groups = []
    for c in cols:
        name = c[1]
        vals = [r[name] for r in rows]
        if c[0] == "cat":
            if name not in qspecs:
                groups.append([{"col": name, "op": "eq", "value": v} for v in c[2]])
            if len(c[2]) > 2:
                for v in c[2]:
                    groups.append([{"col": name, "op": "ne", "value": v}, {"col": name, "op": "eq", "value": v}])
        else:
            arr = np.array(vals, dtype=float)
            cuts = [t for t in c[2]["cuts"] if np.quantile(arr, 0.1) < t <= np.quantile(arr, 0.9)]
            for t in cuts:
                groups.append([{"col": name, "op": "ge", "value": t}, {"col": name, "op": "lt", "value": t}])
    tcol = dom["time_col"]
    if meta_time["date_only"]:
        ds = sorted(r[tcol] for r in rows)
        for q in (0.3, 0.5, 0.7):
            d = ds[int(q * (len(ds) - 1))]
            groups.append([{"col": tcol, "op": "date_lt", "value": d}, {"col": tcol, "op": "date_ge", "value": d}])
    else:
        for h in (9, 12, 15, 18):
            groups.append([{"col": tcol, "op": "hour_lt", "value": h}, {"col": tcol, "op": "hour_ge", "value": h}])
    return groups


# ----------------------------------------------------------------------------------------------------------------------
# instances
# ----------------------------------------------------------------------------------------------------------------------

def _vars(qcols, qspecs, rng, ent, mode, ents=None):
    ents = ents or ent + "s"
    out = []
    for c in qcols:
        name = c[1]
        qc = qspecs[name]
        num = qc["type"] == "num"
        if mode == "pick":
            if num:
                qs = [f"Which range contains the selected {ent}'s {name}?",
                      f"For the randomly selected {ent}, which bucket does {name} fall in?",
                      f"What is the selected {ent}'s {name} (by range)?"]
                clause = [f"the selected {ent}'s {name} is {o}" for o in qc["options"]]
            else:
                qs = [f"What is the selected {ent}'s {name}?",
                      f"For the randomly selected {ent}, what is the value of {name}?",
                      f"What does the {name} field of the selected {ent} say?"]
                clause = [f"the selected {ent} has {name} = {o}" for o in qc["options"]]
        else:
            if num:
                qs = [f"If we pick one of the {ents} so far uniformly at random, which range contains its {name}?",
                      f"What fraction of the {ents} so far fall in each {name} range?",
                      f"Across all {ents} received so far, how are they distributed over these {name} ranges?"]
                clause = [f"{name} is {o}" for o in qc["options"]]
            else:
                qs = [f"If we pick one of the {ents} so far uniformly at random, what is its {name}?",
                      f"What fraction of the {ents} so far have each value of {name}?",
                      f"Across all {ents} received so far, how are they distributed over {name}?"]
                clause = [f"{name} = {o}" for o in qc["options"]]
        out.append(Var(name, list(qc["options"]), qs, clause))
    return out


def _table_setup(rng, n_rows, n_cols=None, domain=None, fmt=None):
    dom = domain or DOMAINS[int(rng.integers(len(DOMAINS)))]
    if isinstance(n_cols, list):
        cols = n_cols  # explicit column specs
    else:
        n_cols = n_cols or int(rng.integers(4, 8))
        cols = model.choose_columns(rng, dom, n_cols)
    rows, tmeta = model.generate_table(rng, dom, n_rows, cols)
    k = int(rng.integers(2, 5))
    qcols, qspecs = choose_query_cols(rng, rows, cols, k)
    fmt = fmt or pick(rng, model.FORMATS)
    order = [cols[int(i)][1] for i in rng.permutation(len(cols))]
    columns = [dom["id_col"], dom["time_col"]] + order
    specs = {c[1]: c[2] for c in cols if c[0] == "num"}
    return dom, cols, rows, tmeta, qcols, qspecs, fmt, columns, specs


FMT_NAME = {"csv": "CSV", "json": "a JSON array of records", "jsonl": "JSON Lines (one record per line)",
            "markdown": "a markdown table", "log": "log lines of key=value fields (the first field is the timestamp)"}


def pick_world(rng, n_rows, *, domain=None, n_cols=None, fmt=None) -> World:
    dom, cols, rows, tmeta, qcols, qspecs, fmt, columns, specs = _table_setup(rng, n_rows, n_cols, domain, fmt)
    ent, ents = dom["ent"], dom["ents"]
    N = len(rows)
    proj = np.array([[code_of(r[c[1]], qspecs[c[1]]) for c in qcols] for r in rows], dtype=int)
    prior = np.full(N, 1.0 / N)
    qnames = [c[1] for c in qcols]
    cells = [tuple(p) for p in proj]
    groups = fact_groups(rows, cols, qspecs, tmeta, dom, rng)
    gvecs = []
    for g in groups:
        M = np.array([[fact_holds(f, r) for r in rows] for f in g], dtype=bool)
        assert np.all(M.sum(0) == 1)
        gvecs.append(M)

    def info(M, post):
        tot = post.sum()
        pq = {}
        for c, p in zip(cells, post):
            pq[c] = pq.get(c, 0.0) + p
        tv = 0.0
        for m in M:
            pa = float(post[m].sum())
            if pa <= 0:
                continue
            d = {}
            for c, p, x in zip(cells, post, m):
                if x:
                    d[c] = d.get(c, 0.0) + p
            tv += 0.5 * sum(abs(d.get(c, 0.0) - pa * pc / tot) for c, pc in pq.items())
        return tv / tot

    T = int(rng.integers(2, 5))
    base_seed = int(rng.integers(2**31))
    cache = {}

    def alternatives(k, past):
        key = (k, tuple(past))
        if key in cache:
            return cache[key]
        post = prior.copy()
        used = set()
        for gi, ai in past:
            post = post * gvecs[gi][ai]
            used.add(gi)
        r = np.random.default_rng([base_seed, k] + [gi * 16 + ai for gi, ai in past])
        live = post > 0
        n_live = int(live.sum())
        cands = []
        for gi, M in enumerate(gvecs):
            if gi in used:
                continue
            sizes = (M & live).sum(1)
            nz = sizes[sizes > 0]
            if len(nz) < 2 or nz.max() > 0.9 * n_live or (n_live >= 6 and nz.max() > n_live - 2 and len(nz) == 2):
                continue
            cands.append(gi)

        def keeps_uncertain(M):
            # every outcome of the fact leaves each query column with >= 2 distinct values among the live rows
            for m in M:
                rows_m = live & m
                if rows_m.any() and any(len(set(proj[rows_m, j])) < 2 for j in range(proj.shape[1])):
                    return False
            return True

        good = [gi for gi in cands if keeps_uncertain(gvecs[gi])]
        if not good:
            good = [gi for gi in range(len(gvecs)) if gi not in used and keeps_uncertain(gvecs[gi])
                    and sum(((gvecs[gi][a] & live).sum() > 0) for a in range(len(gvecs[gi]))) >= 2]
        cands = good or cands
        if not cands:
            cands = [gi for gi in range(len(gvecs)) if gi not in used]
        scores = np.array([info(gvecs[gi], post) for gi in cands]) + 1e-3
        gi = int(cands[int(r.choice(len(cands), p=scores / scores.sum()))])
        cache[key] = [(gi, ai) for ai in range(len(groups[gi]))]
        return cache[key]

    def lik(o, past):
        gi, ai = o
        return gvecs[gi][ai].astype(float)

    tpl_i = int(rng.integers(len(FACT_TPL)))

    def render(k, o):
        f = groups[o[0]][o[1]]
        return FACT_TPL[tpl_i].format(ent=ent, p=fact_phrase(f), sql=fact_sql(f))

    table = model.render_rows(rows, columns, fmt, specs)
    intro = pick(rng, dom["framing"])
    fmt_s = pick(rng, [f"It has {N} rows, given as {FMT_NAME[fmt]}.", f"The {N} records are shown as {FMT_NAME[fmt]}.",
                       f"All {N} {ents} are listed below ({FMT_NAME[fmt]})."])
    sel = pick(rng, [
        f"One {ent} is selected uniformly at random from these {N} rows (every row equally likely); call it the selected {ent}. Facts about the selected {ent} are then revealed one at a time.",
        f"A reviewer picks one of the {N} {ents} at random, each with equal probability. Below, details about that selected {ent} come in one by one.",
        f"Suppose we draw a single row uniformly at random from this table: the selected {ent}. We then learn some of its fields.",
    ])
    q = pick(rng, ["Questions are about the selected " + ent + ".",
                   f"Answer every question about the selected {ent}, using the table.",
                   "All probabilities follow from counting rows in the table."])
    prelude = f"{intro} {fmt_s}\n{table}\n{sel} {q}"
    variables = _vars(qcols, qspecs, rng, ent, "pick", ents)
    mart_var = qnames[int(rng.integers(len(qnames)))]
    meta = {"style": fmt, "theme": dom["name"], "mode": "pick", "n_rows": N, "entity": ent,
            "columns": {c: {k: v for k, v in qspecs[c].items()} for c in qnames}, "table_columns": columns,
            "latent_classes": tmeta["latent_classes"]}
    params = {"rows": rows, "groups": groups, "qspecs": qspecs, "qnames": qnames, "fmt": fmt, "columns": columns}
    return World(variables=variables, prior=prior, proj=proj, T=T, alternatives=alternatives, lik=lik, render=render,
                 prelude=prelude, mart_var=mart_var, exchangeable=True, meta=meta, params=params)


def pick_instance(world: World, rng, *, family, split, idx):
    inst, trace = build_instance(world, rng, family=family, split=split, idx=idx, return_trace=True, decimals=DECIMALS)
    # make_queries with table wording instead of the generic one
    J = np.asarray(inst["steps"][-1]["joint"], dtype=float)
    ent = world.meta["entity"]
    if get_tv() >= 2:
        ctx = "\n".join([inst["prelude"]] + [s["evidence"] for s in inst["steps"]])
        inst["queries"] = make_queries_v2(world.variables, J, rng, world.T, world.mart_var, PICK_V2,
                                          {"ent": ent, "ents": ent + "s"}, ctx)
    else:
        inst["queries"] = make_queries(world.variables, J, rng, world.T, world.mart_var, PICK_TPL, ent + "s")
    inst["meta"]["facts"] = [world.params["groups"][gi][ai] for gi, ai in trace["obs"]]
    return inst, trace


def stream_instance(rng, n_rows, *, family, split, idx, domain=None, n_cols=None, fmt=None):
    dom, cols, rows, tmeta, qcols, qspecs, fmt, columns, specs = _table_setup(rng, n_rows, n_cols, domain, fmt)
    ent, ents = dom["ent"], dom["ents"]
    N = len(rows)
    T = int(rng.integers(2, 6))
    first = max(5, int(N * rng.uniform(0.2, 0.5)))
    rest = N - first
    if rest < T - 1:
        T = max(2, rest + 1)
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, rest), size=T - 2, replace=False)) if T > 2 else []
    sizes = [first] + [b - a for a, b in zip([0] + cuts, cuts + [rest])]
    assert sum(sizes) == N and all(s > 0 for s in sizes)
    qnames = [c[1] for c in qcols]
    shape = tuple(len(qspecs[n]["options"]) for n in qnames)
    codes = np.array([[code_of(r[n], qspecs[n]) for n in qnames] for r in rows], dtype=int)
    variables = _vars(qcols, qspecs, rng, ent, "stream", ents)
    steps, lo = [], 0
    # position-neutral batch headers (no "initial", "another", "so far"), so permuted batch orders read naturally
    tpl = pick(rng, ["A batch of {n} {ents} arrives:", "Batch received ({n} rows):", "{n} {ents} come in:",
                     "Incoming batch of {n} {ents}:"])
    Js = []
    for k, s in enumerate(sizes):
        chunk = rows[lo:lo + s]
        C = np.zeros(shape)
        for r in codes[: lo + s]:
            C[tuple(r)] += 1
        J = C / C.sum()
        Js.append(J)
        head = tpl.format(n=s, ents=ents)
        steps.append({"evidence": head + "\n" + model.render_rows(chunk, columns, fmt, specs), "joint": _r(J, DECIMALS)})
        lo += s
    mart_var = qnames[int(rng.integers(len(qnames)))]
    g = int(rng.integers(N))
    intro = pick(rng, dom["framing"])
    prelude = (f"{intro} Records arrive in batches, each given as {FMT_NAME[fmt]}. "
               + pick(rng, [f"All questions are about the empirical distribution of the {ents} received so far: picking one of them uniformly at random.",
                            f"Answer by counting: probabilities and fractions refer to all {ents} shown so far, each equally weighted.",
                            f"Treat the {ents} received so far as the whole population; a random {ent} means one of them chosen uniformly."]))
    inst = {
        "id": f"{family}-{split}-{idx:06d}", "family": family, "split": split,
        "variables": [{"name": v.name, "options": list(v.options)} for v in variables],
        "prelude": prelude, "steps": steps,
        "gold": {n: int(codes[g, j]) for j, n in enumerate(qnames)},
        "mart_var": mart_var,
    }
    perms, seen = [], {tuple(range(T))}
    for _ in range(20):
        p = tuple(int(i) for i in rng.permutation(T))
        if p not in seen:
            seen.add(p)
            perms.append(list(p))
        if len(perms) >= 3:
            break
    inst["perms"] = perms
    if get_tv() >= 2:
        ctx = "\n".join([prelude] + [s["evidence"] for s in steps])
        inst["queries"] = make_queries_v2(variables, Js[-1], rng, T, mart_var, STREAM_V2, {"ent": ent, "ents": ents}, ctx)
    else:
        inst["queries"] = make_queries(variables, Js[-1], rng, T, mart_var, STREAM_TPL, ents)
    inst["meta"] = {"style": fmt, "theme": dom["name"], "mode": "stream", "n_rows": N, "entity": ent,
                    "batches": sizes, "columns": {c: dict(qspecs[c]) for c in qnames}, "table_columns": columns,
                    "mart_semantics": "empirical-so-far", "latent_classes": tmeta["latent_classes"]}
    return inst, {"rows": rows, "codes": codes, "gold_row": g, "variables": variables}
