"""Procedural random Bayesian networks over a large theme lexicon (broad PFN-style prior over worlds).

Two modes:
  * single: one DAG (chain / fork / collider / naive-Bayes / noisy-OR / random), nodes split into query variables,
    summed-out hidden nodes, and evidence nodes that are observed one at a time (exchangeable -> perms).
  * plate:  instance-level "global" nodes plus a per-case sub-network repeated i.i.d. over cases (matches, batches,
    ...); each evidence step records the observed per-case nodes of a new case (exchangeable -> perms). Variables:
    some globals and per-case nodes of the NEXT case.
CPT rows are exact rationals (integer counts over a denominator), drawn from Dirichlet rows with random
concentration (near-deterministic to flat), and rendered as prose / bullet tables / JSON with %, decimals,
"x out of y", ratios or odds.
"""
from __future__ import annotations

import itertools
import re

import numpy as np

from .common import a_an, get_tv, cap, join_list, json_block, json_line, pick
from .lexicon import DOMAINS, instantiate, pseudo
from .procedural import ALPHAS, DENOMS, NumStyle, Row, is_whether, questions, rand_row, statement
from .world import Var, World

DECIMALS = 8
MAX_Z = 8192
MAX_PARAMS = 64


def _slug(s, used):
    base = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    base = re.sub(r"^(the|whether)_", "", base)[:28].strip("_") or "var"
    name, k = base, 2
    while name in used:
        name, k = f"{base}_{k}", k + 1
    used.add(name)
    return name


class Node:
    def __init__(self, tmpl, states, kind):
        self.tmpl = tmpl  # noun phrase, may contain {c}
        self.states = states
        self.kind = kind  # "global" | "case"
        self.parents: list[int] = []
        self.table = None  # ndarray (*parent_states, n_states)
        self.rows: dict = {}  # parent config -> Row
        self.noisy = None  # dict for noisy-OR nodes

    def np(self, c=""):
        return self.tmpl.replace("{c}", c)


def _structure(rng, N, struct):
    parents = [[] for _ in range(N)]
    for i in range(1, N):
        if struct == "chain":
            parents[i] = [i - 1]
        elif struct in ("fork", "naive"):
            parents[i] = [0]
        elif struct == "collider":
            if i == 1:
                parents[i] = []
            elif i == 2:
                parents[i] = [0, 1]
            else:
                parents[i] = [int(rng.integers(i))]
        else:  # random, noisyor
            k = min(i, int(rng.integers(1, 3)) if struct == "random" else int(rng.integers(1, 4)))
            parents[i] = sorted(int(x) for x in rng.choice(i, size=k, replace=False))
    if struct == "noisyor":
        n_roots = int(rng.integers(1, min(3, N - 1) + 1))
        for i in range(n_roots):
            parents[i] = []
        for i in range(n_roots, N):
            if not parents[i] or max(parents[i]) >= i:
                parents[i] = [int(rng.integers(i))]
    return parents


def _fill_cpt(rng, node: Node, nodes, noisy: bool):
    pcs = [len(nodes[p].states) for p in node.parents]
    n = len(node.states)
    table = np.zeros(pcs + [n])
    if noisy and node.parents and n == 2 and all(c == 2 for c in pcs):
        on = int(rng.integers(2))
        act = [int(rng.integers(2)) for _ in node.parents]
        ps = [int(pick(rng, list(range(30, 100, 5)))) for _ in node.parents]
        leak = int(pick(rng, [0, 1, 2, 5, 5, 10, 15, 20]))
        for cfg in itertools.product(*[range(c) for c in pcs]):
            off = 1 - leak / 100
            for x, a, p in zip(cfg, act, ps):
                if x == a:
                    off *= 1 - p / 100
            table[cfg + (1 - on,)] = off
            table[cfg + (on,)] = 1 - off
        node.noisy = {"on": on, "act": act, "ps": ps, "leak": leak}
    else:
        D = int(pick(rng, DENOMS))
        alpha = float(pick(rng, ALPHAS))
        for cfg in itertools.product(*[range(c) for c in pcs]):
            row = Row(rand_row(rng, n, D, alpha), D)
            node.rows[cfg] = row
            table[cfg] = row.probs
    node.table = table


def _eval(nodes, cols, idxs):
    """Product of CPT entries of nodes `idxs` for assignment matrix `cols` (n_rows x n_nodes)."""
    p = np.ones(cols.shape[0])
    for i in idxs:
        nd = nodes[i]
        key = tuple(cols[:, q] for q in nd.parents) + (cols[:, i],)
        p = p * nd.table[key]
    return p


def _cond_text(nodes, cfg_parents, cfg, c):
    return join_list([statement(nodes[p].np(c(nodes[p])), nodes[p].states[v]) for p, v in zip(cfg_parents, cfg)])


def _render_cpts(rng, nodes, order, ns: NumStyle, layout, c):
    """Return (units, json_dict) describing every node's CPT; c(node) gives the case reference for per-case nodes."""
    units, J = [], {}
    for i in order:
        nd = nodes[i]
        np_ = nd.np(c(nd))
        if nd.noisy is not None:
            z = nd.noisy
            on, off = nd.states[z["on"]], nd.states[1 - z["on"]]
            causes = [(statement(nodes[p].np(c(nodes[p])), nodes[p].states[a]), ns.scalar(q))
                      for p, a, q in zip(nd.parents, z["act"], z["ps"])]
            if layout == "json":
                J[f"rule for {np_}"] = {"type": "noisy-OR", "effect": f"{np_} = {on}",
                                        "causes": {k: v for k, v in causes},
                                        "background probability": ns.scalar(z["leak"]),
                                        "otherwise": off}
            else:
                txt = (f"{cap(np_)} follows a noisy-OR rule: " +
                       "; ".join(f"if {k}, that alone makes it {on} with probability {v}" for k, v in causes) +
                       (f". Independently of these, there is a background probability of {ns.scalar(z['leak'])} "
                        f"that it is {on}." if z["leak"] else ". There is no other way for it to be " + on + ".") +
                       f" It is {on} if any of these independent triggers fires, and {off} otherwise.")
                units.append(txt)
            continue
        if not nd.parents:
            row = nd.rows[()]
            if layout == "json":
                J[f"P({np_})"] = ns.cells(row, nd.states)
            elif layout == "bullets":
                cells = ns.cells(row, nd.states)
                units.append(f"{cap(np_)}: " + ", ".join(f"{k} {v}" for k, v in cells.items()) + ".")
            else:
                units.append(cap(ns.row_text(row, nd.states, np_)) + ".")
            continue
        pn = [nodes[p].np(c(nodes[p])) for p in nd.parents]
        if layout == "json":
            J[f"P({np_} | {', '.join(pn)})"] = {
                ", ".join(nodes[p].states[v] for p, v in zip(nd.parents, cfg)): ns.cells(row, nd.states)
                for cfg, row in nd.rows.items()}
        elif layout == "bullets":
            lines = [f"{cap(np_)}, depending on {join_list(pn)}:"]
            for cfg, row in nd.rows.items():
                cells = ns.cells(row, nd.states)
                lines.append(f"- if {_cond_text(nodes, nd.parents, cfg, c)}: " +
                             ", ".join(f"{k} {v}" for k, v in cells.items()))
            units.append("\n".join(lines))
        else:
            for cfg, row in nd.rows.items():
                cond = _cond_text(nodes, nd.parents, cfg, c)
                form = pick(rng, ["When {cond}, {row}.", "If {cond}, then {row}.", "Given that {cond}, {row}."])
                units.append(form.format(cond=cond, row=ns.row_text(row, nd.states, np_)))
    return units, J


def _n_params(nodes):
    return sum(int(np.prod(nd.table.shape)) if nd.noisy is None else len(nd.parents) + 1 for nd in nodes)


def _pick_concepts(rng, pool, k, binary=False):
    cands = [x for x in pool if (not binary or len(x.split("|")[1].split(",")) == 2)]
    if len(cands) < k:
        return None
    return [cands[int(i)] for i in rng.choice(len(cands), size=k, replace=False)]


def make_world(rng) -> World:
    for _ in range(200):
        w = _try_world(rng)
        if w is not None:
            return w
    raise RuntimeError("randbn: could not build a world")


def _try_world(rng):
    dom = pick(rng, DOMAINS)
    case = dom["case"]
    setting = dom["setting"].replace("{X}", pseudo(rng))
    mode = "plate" if rng.random() < 0.5 else "single"
    ns = NumStyle(rng)
    layout = pick(rng, ["prose", "prose", "bullets", "json"])
    if mode == "single":
        return _single(rng, dom, case, setting, ns, layout)
    return _plate(rng, dom, case, setting, ns, layout)


def _finish_vars(rng, nodes, var_idx, c_var, future_flags):
    used = set()
    variables = []
    for i, fut in zip(var_idx, future_flags):
        nd = nodes[i]
        np_ = nd.np(c_var(nd))
        variables.append(Var(_slug(np_, used), list(nd.states), questions(np_, future=fut),
                             [statement(np_, s, future=fut) for s in nd.states]))
    return variables


def _single(rng, dom, case, setting, ns, layout):
    struct = pick(rng, ["chain", "fork", "collider", "naive", "random", "random", "noisyor"])
    n_var = int(rng.integers(2, 5))
    n_hid = int(rng.integers(0, 2))
    n_ev = int(rng.integers(2, 6))
    N = n_var + n_hid + n_ev
    if N > 8:
        return None
    concepts = _pick_concepts(rng, dom["g"] + dom["p"], N, binary=(struct == "noisyor"))
    if concepts is None:
        return None
    this = f"this {case}"
    nodes = []
    for cc in concepts:
        np_, states = instantiate(rng, cc)
        nodes.append(Node(np_, states, "global"))
    if len({nd.np(this) for nd in nodes}) < N:
        return None
    parents = _structure(rng, N, struct)
    for i, nd in enumerate(nodes):
        nd.parents = parents[i]
        if np.prod([len(nodes[p].states) for p in nd.parents]) > 9:
            return None
    for nd in nodes:
        _fill_cpt(rng, nd, nodes, noisy=(struct == "noisyor"))
    if _n_params(nodes) > MAX_PARAMS:
        return None
    # roles
    idx = list(range(N))
    if struct == "naive":
        rest = [int(i) for i in rng.permutation(np.arange(1, N))]
        var_idx = [0] + rest[: n_var - 1]
        ev_idx = rest[n_var - 1: n_var - 1 + n_ev]
    elif struct == "collider":
        rest = [int(i) for i in rng.permutation(np.arange(2, N))]
        var_idx = [0, 1] + rest[: n_var - 2]
        others = rest[max(0, n_var - 2):]
        ev_idx = others[:n_ev]
    else:
        perm = [int(i) for i in rng.permutation(N)]
        var_idx, ev_idx = perm[:n_var], perm[n_var:n_var + n_ev]
    if len(ev_idx) < 2 or len(var_idx) < 2:
        return None
    var_idx = sorted(var_idx)
    shape = [len(nodes[i].states) for i in var_idx]
    if np.prod(shape) > 256:
        return None
    sizes = [len(nd.states) for nd in nodes]
    if np.prod(sizes) > MAX_Z:
        return None
    Z = np.array(list(itertools.product(*[range(s) for s in sizes])))
    prior = _eval(nodes, Z, idx)
    ev_idx = [ev_idx[int(i)] for i in rng.permutation(len(ev_idx))]
    T = len(ev_idx)
    c = lambda nd: this  # noqa: E731

    units, J = _render_cpts(rng, nodes, idx, ns, layout, c)
    intro = pick(rng, [f"This concerns {this} at {setting}.", f"Consider {this} at {setting}.",
                       f"At {setting}, analysts model {this} with the following probabilities."])
    tail = pick(rng, ["These probabilities describe everything that is relevant.",
                      "Each quantity depends only on the ones it is stated to depend on.", ""])
    tv2 = get_tv() >= 2
    if tv2:  # the listed dependencies are complete (#12)
        tail = ("Each quantity depends directly only on the quantities it is stated to depend on; there are no other "
                "dependencies.")
    json_ev = layout == "json"
    if layout == "json":
        J = {"setting": setting, "about": this,
             "quantities": {nodes[i].np(this): nodes[i].states for i in idx}, **J}
        if tv2:
            J["dependencies"] = "complete as listed: each quantity depends only on the quantities it is conditioned on"
        prelude = json_block(rng, J)
    else:
        prelude = "\n".join([intro] + units + ([tail] if tail else [])) if layout == "bullets" else \
            " ".join([intro] + units + ([tail] if tail else []))
    ev_forms = ["Observation: {s}.", "It is found that {s}.", "A reliable report confirms that {s}.", "We learn that {s}."]
    pos = [pick(rng, ev_forms) for _ in range(T)]

    def render(k, o):
        j, v = o
        nd = nodes[ev_idx[j]]
        if json_ev:
            return json_line({"observed": nd.np(this), "value": nd.states[v]})
        return cap(pos[k].format(s=statement(nd.np(this), nd.states[v])))

    variables = _finish_vars(rng, nodes, var_idx, c, [False] * len(var_idx))
    proj = Z[:, var_idx]
    ev_cols = [Z[:, i] for i in ev_idx]
    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: [(k, v) for v in range(len(nodes[ev_idx[k]].states))],
                 lik=lambda o, past: (ev_cols[o[0]] == o[1]).astype(float),
                 render=render, prelude=prelude, mart_var=variables[int(rng.integers(len(variables)))].name,
                 exchangeable=True,
                 meta={"style": "json" if layout == "json" else layout, "theme": dom["name"], "mode": "single",
                       "structure": struct})


def _plate(rng, dom, case, setting, ns, layout):
    nG = int(rng.integers(1, 4))
    nP = int(rng.integers(1, 4))
    noisy = rng.random() < 0.2
    gc = _pick_concepts(rng, dom["g"], nG)
    pc = _pick_concepts(rng, dom["p"], nP, binary=noisy)
    if gc is None or pc is None:
        return None
    nodes = []
    for cc in gc:
        np_, st = instantiate(rng, cc)
        nodes.append(Node(np_, st, "global"))
    for cc in pc:
        np_, st = instantiate(rng, cc)
        nodes.append(Node(np_, st, "case"))
    G = list(range(nG))
    P = list(range(nG, nG + nP))
    gstruct = pick(rng, ["independent", "chain", "fork"])
    for i in G[1:]:
        if gstruct == "chain":
            nodes[i].parents = [i - 1]
        elif gstruct == "fork":
            nodes[i].parents = [0]
    for j, i in enumerate(P):
        cands = G + P[:j]
        k = min(len(cands), int(rng.integers(1, 3)))
        ps = sorted(int(x) for x in rng.choice(cands, size=k, replace=False))
        if j == 0 and not any(p in G for p in ps):
            ps = sorted(set(ps) | {int(pick(rng, G))})
        nodes[i].parents = ps
    for nd in nodes:
        if np.prod([len(nodes[p].states) for p in nd.parents]) > 9:
            return None
    for nd in nodes:
        _fill_cpt(rng, nd, nodes, noisy=noisy and nd.kind == "case" and len(nd.states) == 2 and
                  all(len(nodes[p].states) == 2 for p in nd.parents))
    if _n_params(nodes) > MAX_PARAMS:
        return None
    # observed per-case nodes (1-2), hidden per-case nodes summed out
    n_obs = int(rng.integers(1, min(2, nP) + 1))
    O = sorted(int(x) for x in rng.choice(P, size=n_obs, replace=False))
    n_alt = int(np.prod([len(nodes[i].states) for i in O]))
    if n_alt > 16:
        return None
    # variables: >=1 global (mart var) + 1-2 next-case nodes
    n_gv = int(rng.integers(1, nG + 1))
    gv = sorted(int(x) for x in rng.choice(G, size=n_gv, replace=False))
    n_pv = int(rng.integers(1, min(2, nP) + 1))
    n_pv = min(n_pv, 4 - len(gv))
    if n_pv < 1:
        return None
    pv = sorted(int(x) for x in rng.choice(P, size=n_pv, replace=False))
    var_idx = gv + pv
    if not 2 <= len(var_idx) <= 4 or np.prod([len(nodes[i].states) for i in var_idx]) > 256:
        return None
    sizes = [len(nd.states) for nd in nodes]
    if np.prod(sizes) > MAX_Z:
        return None
    Z = np.array(list(itertools.product(*[range(s) for s in sizes])))
    prior = _eval(nodes, Z, list(range(len(nodes))))
    case_p = _eval(nodes, Z, P)
    g_idx = np.ravel_multi_index(tuple(Z[:, i] for i in G), [sizes[i] for i in G])
    o_shape = [sizes[i] for i in O]
    o_idx = np.ravel_multi_index(tuple(Z[:, i] for i in O), o_shape)
    M = np.zeros((int(np.prod([sizes[i] for i in G])), n_alt))
    np.add.at(M, (g_idx, o_idx), case_p)
    # each global config appears once per per-case assignment, so M rows already sum to 1
    L = M[g_idx]  # n_Z x n_alt
    T = int(rng.integers(3, 7))

    tv2 = get_tv() >= 2
    a_case, the_case, next_case = (a_an(case) if tv2 else f"a {case}"), f"the {case}", f"the next {case}"
    c_desc = lambda nd: a_case if nd.kind == "case" else ""  # noqa: E731
    units, J = _render_cpts(rng, nodes, G + P, ns, layout, c_desc)
    gnames = join_list([nodes[i].np("") for i in G])
    s_ind = pick(rng, [
        f"Different {case}s are independent of each other once {gnames} {'is' if nG == 1 else 'are'} fixed, and every "
        f"{case} follows the same rules; statements about {a_case} refer to one and the same {case}.",
        f"Each {case} is generated independently by the same rules, given {gnames}.",
    ])
    if tv2:
        s_ind = (f"Different {case}s are independent of each other once {gnames} {'is' if nG == 1 else 'are'} fixed, "
                 f"and every {case} follows the same rules; within one {case}, all statements about {a_case} refer to "
                 f"one and the same {case}. Each quantity depends directly only on the quantities it is stated to "
                 f"depend on; there are no other dependencies.")
    intro = pick(rng, [f"At {setting}, {case}s are recorded one after another.",
                       f"We follow a series of {case}s at {setting}.",
                       f"This concerns repeated {case}s at {setting}."])
    json_ev = layout == "json"
    if layout == "json":
        J = {"setting": setting, "unit": case,
             "instance-level quantities": {nodes[i].np(''): nodes[i].states for i in G},
             f"per-{case} quantities": {nodes[i].np(a_case): nodes[i].states for i in P},
             **J, "independence": f"{case}s are i.i.d. given the instance-level quantities"}
        if tv2:
            J["same_case"] = f"within one {case}, all per-{case} statements refer to one and the same {case}"
            J["dependencies"] = "complete as listed: each quantity depends only on the quantities it is conditioned on"
        prelude = json_block(rng, J)
    else:
        sep = "\n" if layout == "bullets" else " "
        prelude = sep.join([intro] + units + [s_ind])
    o_assign = list(itertools.product(*[range(sizes[i]) for i in O]))
    forms = ["A {case} is recorded: {s}.", "New {case}: {s}.", "Another {case} is logged, and {s}.",
             "Record for one more {case}: {s}."]
    pos = [pick(rng, forms[:2] + forms[3:]) for _ in range(T)]

    def render(k, o):
        vals = o_assign[o]
        if json_ev:
            return json_line({case: {nodes[i].np(the_case): nodes[i].states[v] for i, v in zip(O, vals)}})
        s = join_list([statement(nodes[i].np(the_case), nodes[i].states[v]) for i, v in zip(O, vals)])
        f = pos[k]
        if tv2:  # article agreement and no order words under perms
            f = f.replace("A {case}", cap(a_case)).replace("Record for one more {case}", "Record for " + a_case)
        return cap(f.format(case=case, s=s))

    c_var = lambda nd: next_case if nd.kind == "case" else ""  # noqa: E731
    variables = _finish_vars(rng, nodes, var_idx, c_var, [nodes[i].kind == "case" for i in var_idx])
    proj = Z[:, var_idx]
    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: list(range(n_alt)),
                 lik=lambda o, past: L[:, o],
                 render=render, prelude=prelude, mart_var=variables[int(rng.integers(len(gv)))].name,
                 exchangeable=True,
                 meta={"style": "json" if layout == "json" else layout, "theme": dom["name"], "mode": "plate",
                       "structure": gstruct + ("+noisyor" if noisy else "")})
