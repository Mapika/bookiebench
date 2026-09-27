"""Turn a generated Program into an bookiebench.sims World (exact posterior over the program's accepted executions).

Latent space Z = the distinct accepted final states (final environment, printed lines), with the exact
probabilities from `lang.enumerate_program` (normalised by the accepted mass, i.e. conditioned on every in-program
observe()). Query variables are final values of program variables (numbers bucketed into named ranges).

Two evidence modes:
  * observe: extra `observe(cond)` lines appended to the end of the program are revealed one at a time. Step k's
    alternatives are `observe(c)` / `observe(not c)` for a reveal condition c chosen from a pool as a deterministic
    function of the realised past (never of z). Facts about one run are exchangeable -> perms.
  * print:   the program contains print statements on every path (the same number on every accepted run); step k
    reveals printed line k. The alternatives are all possible texts of line k. Sequential -> no perms.
"""
from __future__ import annotations

import zlib
from fractions import Fraction

import numpy as np

from bookiebench.sims.common import pick
from bookiebench.sims.world import Var, World

from . import lang
from .lang import Budget, Program, negate, render, render_expr

STYLES = ["python", "python", "js", "pseudo"]
FENCE = {"python": "python", "js": "javascript", "pseudo": "text"}
LANG_NAME = {"python": "Python", "js": "JavaScript", "pseudo": "pseudo-code"}


MAX_TRACES = 100_000


class Reject(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------
# small analysis helpers
# ----------------------------------------------------------------------------------------------------------------------

def final_table(dist: dict, mass):
    """(keys, prior ndarray, envs list[dict], outs list[tuple]) sorted deterministically."""
    keys = sorted(dist, key=repr)
    prior = np.array([float(dist[k] / mass) for k in keys])
    envs = [lang._thaw(k[0]) for k in keys]
    outs = [k[1] for k in keys]
    return keys, prior, envs, outs


def var_marginal(envs, prior, name):
    m = {}
    for e, p in zip(envs, prior):
        k = lang._key(e[name])
        m[k] = m.get(k, 0.0) + p
    return m


def uses(prog: Program) -> set:
    """primitive names used anywhere in the program."""
    found = set()

    def ex(e):
        t = e[0]
        if t in ("flip", "randint", "choice"):
            found.add(t)
        if t == "bin":
            ex(e[2]); ex(e[3])
        elif t == "not":
            ex(e[1])
        elif t == "ifexp":
            ex(e[1]); ex(e[2]); ex(e[3])
        elif t == "call":
            for a in e[2]:
                ex(a)

    def st(ss):
        for s in ss:
            t = s[0]
            if t == "assign":
                ex(s[2])
            elif t == "aug":
                ex(s[3])
            elif t in ("observe", "return"):
                if t == "observe":
                    found.add("observe")
                ex(s[1])
            elif t == "print":
                found.add("print")
                for x in s[2]:
                    ex(x)
            elif t == "if":
                for c, b in s[1]:
                    ex(c); st(b)
                if s[2]:
                    st(s[2])
            elif t == "for":
                st(s[3])

    for f in prog.funcs:
        st(f[2])
    st(prog.body)
    return found


# ----------------------------------------------------------------------------------------------------------------------
# query variables (bucketing)
# ----------------------------------------------------------------------------------------------------------------------

def bucketize(marg: dict, rng, max_opts=5):
    """marg: {(type, value): prob}. Returns (options, value->option index, kind)."""
    items = sorted(marg.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    tname = items[0][0][0]
    vals = [k[1] for k, _ in items]
    probs = [p for _, p in items]
    if tname == "bool":
        return None  # style-specific; handled by caller
    if tname == "str":
        if len(vals) > max_opts:
            return None
        return [f'"{v}"' for v in vals], {lang._key(v): i for i, v in enumerate(vals)}, "str"
    if len(vals) <= max_opts and rng.random() < 0.8:
        return [str(v) for v in vals], {lang._key(v): i for i, v in enumerate(vals)}, "int"
    # contiguous buckets balancing mass
    nb = int(rng.integers(3, min(max_opts, len(vals)) + 1)) if len(vals) >= 3 else 2
    cum = np.cumsum(probs)
    cuts = []
    for j in range(1, nb):
        target = j / nb
        c = int(np.searchsorted(cum, target))
        c = min(max(c + 1, (cuts[-1] + 1) if cuts else 1), len(vals) - (nb - j))
        cuts.append(c)
    groups, lo = [], 0
    for c in cuts + [len(vals)]:
        groups.append(vals[lo:c])
        lo = c
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return None
    opts, mp = [], {}
    for gi, g in enumerate(groups):
        if len(g) == 1:
            name = str(g[0])
        elif gi == 0:
            name = f"at most {g[-1]}"
        elif gi == len(groups) - 1:
            name = f"{g[0]} or more"
        else:
            name = f"between {g[0]} and {g[-1]}"
        opts.append(name)
        for v in g:
            mp[lang._key(v)] = gi
    return opts, mp, "bucket"


BOOL_OPTS = {"python": ["True", "False"], "js": ["true", "false"], "pseudo": ["TRUE", "FALSE"]}

QUESTIONS = [
    "What is the final value of `{v}`?",
    "When the program finishes, what is `{v}`?",
    "Which value does `{v}` hold at the end of the run?",
    "After the run completes, what does `{v}` equal?",
]
QUESTIONS_BUCKET = [
    "Which range does the final value of `{v}` fall in?",
    "When the program finishes, which of these describes `{v}`?",
    "At the end of the run, where does `{v}` land?",
]
CLAUSES = [
    "the final value of `{v}` is {{opt}}",
    "`{v}` is {{opt}} when the program finishes",
    "`{v}` ends the run as {{opt}}",
]


def make_var(name, opts, kind, rng) -> Var:
    qs = QUESTIONS_BUCKET if kind == "bucket" else QUESTIONS
    return Var(name, list(opts), [q.format(v=name) for q in qs], pick(rng, CLAUSES).format(v=name))


def _entropy(keys, prior):
    m = {}
    for k, p in zip(keys, prior):
        m[k] = m.get(k, 0.0) + p
    v = np.array([x for x in m.values() if x > 0])
    return float(-(v * np.log(v)).sum())


def mutual_info(envs, prior, name):
    """I(name ; all other variables) under the final distribution (nats)."""
    own = [lang._key(e[name]) for e in envs]
    rest = [tuple((k, lang._key(v)) for k, v in sorted(e.items()) if k != name) for e in envs]
    both = list(zip(own, rest))
    return max(0.0, _entropy(own, prior) + _entropy(rest, prior) - _entropy(both, prior))


def info_tv(vec, cells, prior):
    """TV distance between the query-joint posterior given a binary fact and given its negation."""
    pa = float(prior[vec].sum())
    if pa <= 1e-9 or pa >= 1 - 1e-9:
        return 0.0
    cats = {}
    for c, v, p in zip(cells, vec, prior):
        d = cats.setdefault(c, [0.0, 0.0])
        d[0 if v else 1] += p
    return 0.5 * sum(abs(a / pa - b / (1 - pa)) for a, b in cats.values())


def choose_query_vars(envs, prior, rng, style, prefer=(), exclude=(), max_vars=None):
    names = sorted(set.intersection(*[set(e) for e in envs])) if envs else []
    cands = []
    for n in names:
        if n in exclude:
            continue
        m = var_marginal(envs, prior, n)
        ps = sorted(m.values(), reverse=True)
        if len(ps) < 2 or ps[0] > 0.93 or ps[1] < 0.03:
            continue
        cands.append(n)
    if len(cands) < 2:
        raise Reject("too few uncertain variables")
    k = int(rng.integers(2, min(4, len(cands)) + 1)) if max_vars is None else min(max_vars, len(cands))
    pref = [c for c in prefer if c in cands]
    rest = [c for c in cands if c not in pref]
    rng.shuffle(pref)
    # other variables: prefer ones entangled with the rest of the state (mutual information), not isolated coins
    if rest:
        mi = np.array([mutual_info(envs, prior, n) for n in rest]) + 0.01
        order_r = rng.choice(len(rest), size=len(rest), replace=False, p=mi / mi.sum())
        rest = [rest[int(i)] for i in order_r]
    order = pref + rest
    # mix: mostly preferred, sometimes other variables
    if pref and rest and rng.random() < 0.35:
        order = pref[: max(1, k - 1)] + rest + pref[max(1, k - 1):]
    chosen, specs, cells = [], [], 1
    for n in order:
        if len(chosen) >= k:
            break
        m = var_marginal(envs, prior, n)
        tname = next(iter(m))[0]
        if tname == "bool":
            opts = BOOL_OPTS[style]
            mp = {("bool", True): 0, ("bool", False): 1}
            kind = "bool"
        else:
            b = bucketize(m, rng)
            if b is None:
                continue
            opts, mp, kind = b
        # merge options that end up with ~zero mass are fine (they stay in the option list)
        if cells * len(opts) > 256:
            continue
        cells *= len(opts)
        chosen.append(n)
        specs.append((opts, mp, kind))
    if len(chosen) < 2:
        raise Reject("could not pick 2 query variables")
    return chosen, specs


# ----------------------------------------------------------------------------------------------------------------------
# reveal-condition pool (observe mode)
# ----------------------------------------------------------------------------------------------------------------------

def cond_pool(envs, prior, rng, var_types: dict, max_pool=60, cells=None, min_info=0.1):
    names = sorted(set.intersection(*[set(e) for e in envs]))
    atoms = []
    for n in names:
        m = var_marginal(envs, prior, n)
        vals = sorted(k[1] for k in m)
        t = next(iter(m))[0]
        V = ("var", n)
        if t == "bool":
            atoms += [V, ("not", V)]
        elif t == "str":
            for v in vals:
                atoms.append(("bin", "==", V, ("lit", v)))
                atoms.append(("bin", "!=", V, ("lit", v)))
        else:
            for v in vals[1:]:
                atoms.append(("bin", ">=", V, ("lit", v)))
                atoms.append(("bin", "<", V, ("lit", v)))
            for v in vals:
                atoms.append(("bin", "==", V, ("lit", v)))
                if len(vals) > 2:
                    atoms.append(("bin", "!=", V, ("lit", v)))
            if len(vals) > 2:
                for v in vals[:-1]:
                    atoms.append(("bin", "<=", V, ("lit", v)))
                    atoms.append(("bin", ">", V, ("lit", v)))
    ints = [n for n in names if next(iter(var_marginal(envs, prior, n)))[0] == "int"]
    for i, a in enumerate(ints):
        for b in ints[i + 1:]:
            for op in (">", "<", "==", ">=", "!="):
                atoms.append(("bin", op, ("var", a), ("var", b)))

    def vec(e):
        return np.array([bool(lang.eval_in(e, lang._freeze(env))) for env in envs])

    pool, seen = [], set()
    atom_vecs = []
    for a in atoms:
        v = vec(a)
        p = float(prior[v].sum())
        atom_vecs.append((a, v, p))
    order = rng.permutation(len(atom_vecs))
    for i in order:
        a, v, p = atom_vecs[int(i)]
        if 0.06 <= p <= 0.94 and v.tobytes() not in seen:
            seen.add(v.tobytes())
            pool.append((a, v))
    # compounds
    good = [x for x in atom_vecs if 0.15 <= x[2] <= 0.85]
    for _ in range(40):
        if len(good) < 2:
            break
        i, j = rng.choice(len(good), size=2, replace=False)
        (a, va, _), (b, vb, _) = good[int(i)], good[int(j)]
        if a[0] == "bin" and b[0] == "bin" and a[2] == b[2]:
            continue
        op = "and" if rng.random() < 0.5 else "or"
        v = (va & vb) if op == "and" else (va | vb)
        p = float(prior[v].sum())
        if 0.08 <= p <= 0.92 and v.tobytes() not in seen:
            seen.add(v.tobytes())
            pool.append((("bin", op, a, b), v))
    if cells is not None:
        pool = [(c, v) for c, v in pool if info_tv(v, cells, prior) >= min_info]
    rng.shuffle(pool)
    return pool[:max_pool]


# ----------------------------------------------------------------------------------------------------------------------
# prose
# ----------------------------------------------------------------------------------------------------------------------

def primitives_text(used: set, style: str, rng) -> str:
    S = style
    T = "TRUE" if S == "pseudo" else ("true" if S == "js" else "True")
    fn = (lambda n: n.upper()) if S == "pseudo" else (lambda n: n)
    parts = []
    if "flip" in used:
        parts.append(pick(rng, [f"`{fn('flip')}(p)` returns {T} with probability p",
                                f"`{fn('flip')}(p)` is a biased coin that comes up {T} with probability p",
                                f"`{fn('flip')}(p)` yields {T} with chance p (and the opposite otherwise)"]))
    if "randint" in used:
        parts.append(pick(rng, [f"`{fn('randint')}(a, b)` returns a uniformly random integer from a to b inclusive",
                                f"`{fn('randint')}(a, b)` picks each integer in a..b (both ends included) with equal probability",
                                f"`{fn('randint')}(a, b)` draws an integer between a and b, inclusive, uniformly"]))
    if "choice" in used:
        parts.append(pick(rng, [f"`{fn('choice')}(xs, w)` returns xs[i] with probability w[i] / sum(w) (uniform when no weights are given)",
                                f"`{fn('choice')}(xs, w)` picks one element of xs, element i with weight w[i] (equal weights if w is omitted)",
                                f"`{fn('choice')}(xs, w)` samples from xs in proportion to the weights w (or uniformly without w)"]))
    obs_kw = "OBSERVE c" if S == "pseudo" else "observe(c)"
    parts.append(pick(rng, [f"`{obs_kw}` discards the run unless c is true, so everything is conditioned on every observe statement that is reached holding",
                            f"`{obs_kw}` conditions on c: runs in which c is false are rejected and do not count",
                            f"`{obs_kw}` is a hard constraint; a run that reaches it with c false is thrown away (rejection sampling)"]))
    lead = pick(rng, ["Semantics: ", "Here ", "In this code, ", "Primitives: "])
    return lead + "; ".join(parts) + ". Each random call is independent of all others."


def assemble_prelude(rng, framing: str, code: str, style: str, used: set, mode: str) -> str:
    sem = primitives_text(used, style, rng)
    run = pick(rng, ["The program is run once.", "Consider a single run of this program.",
                     "We execute the program one time."])
    block = f"```{FENCE[style]}\n{code}\n```"
    if mode == "observe":
        ev = pick(rng, [
            "After the program, additional observe lines are appended one at a time; each one further conditions the same run.",
            "Extra conditions about the same run are then added as observe statements at the very end of the program.",
            "Below, more observe statements get appended to the end of the program; they all refer to this one run.",
        ])
    else:
        ev = pick(rng, [
            "Its printed output is revealed line by line (only runs that pass every observe are kept).",
            "Below you see what the (accepted) run prints, one output line at a time.",
            "The console output of the run is shown incrementally, one line per update; rejected runs print nothing.",
        ])
    q = pick(rng, ["Questions refer to the values of variables when the program finishes.",
                   "All questions are about variable values at the end of the run.",
                   "Answer about the final values of the variables."])
    order = pick(rng, [0, 1])
    if order == 0:
        return "\n".join([framing + " " + sem, block, run + " " + ev + " " + q])
    return "\n".join([framing, block, sem, run + " " + ev + " " + q])


OBS_TPL = {
    "python": ["Appended line: `{c}`", "The program also ends with `{c}`.", "+ {c}", "Added at the end: {c}",
               "Condition added to the program: `{c}`"],
    "js": ["Appended line: `{c};`", "The script also ends with `{c};`.", "+ {c};", "Added at the end: {c};",
           "Condition added to the script: `{c};`"],
    "pseudo": ["Appended line: {c}", "The procedure also ends with: {c}", "+ {c}", "Added at the end: {c}",
               "Condition added: {c}"],
}
PRINT_TPL = {
    "python": ["Output: `{o}`", "The program prints: {o}", "stdout> {o}", "Printed line: {o}"],
    "js": ["Console: `{o}`", "The script logs: {o}", "> {o}", "Logged line: {o}"],
    "pseudo": ["Output: {o}", "The procedure prints: {o}", "PRINTED: {o}", "Printed line: {o}"],
}


def obs_code(c, style):
    s = render_expr(c, style)
    return f"OBSERVE {s}" if style == "pseudo" else f"observe({s})"


# ----------------------------------------------------------------------------------------------------------------------
# world
# ----------------------------------------------------------------------------------------------------------------------

def prefix_states(prog: Program, max_states=20000):
    """[(top-level index, states)] distributions before each top-level statement (and at the end)."""
    en = lang.Enumerator(prog, max_states=max_states)
    states = {((), (), None): Fraction(1)}
    out = [states]
    for s in prog.body:
        states = en.stmt(s, states)
        if len(states) > max_states:
            raise Budget("states")
        out.append(states)
    return out


def _live_env_marg(states, name):
    m, tot = {}, Fraction(0)
    for (fz, outs, ctrl), p in states.items():
        env = lang._thaw(fz)
        if name not in env:
            return None
        k = lang._key(env[name])
        m[k] = m.get(k, Fraction(0)) + p
        tot += p
    return {k: float(v / tot) for k, v in m.items()} if tot > 0 else None


def instrument_prints(prog: Program, rng, query_vars, n_prints, labels: dict | None = None):
    """Insert print statements printing informative expressions, for about n_prints output lines in total.

    Sites: a top-level position (one output line), or the end of the body of a top-level for-loop without `break`
    (one line per iteration, so every run prints the same number of lines)."""
    body = list(prog.body)
    pre = prefix_states(prog)
    last_assign = {}
    for i, s in enumerate(body):
        for n in lang.assigned_names([s]):
            last_assign[n] = i
    cands = []  # (pos, kind, stmt, n_lines)
    for pos in range(1, len(body) + 1):
        states = pre[pos]
        env0 = lang._thaw(next(iter(states))[0]) if states else {}
        for n in sorted(env0):
            if n in query_vars and last_assign.get(n, -1) < pos:
                continue  # would print a query variable's final value
            m = _live_env_marg(states, n)
            if not m or len(m) < 2 or max(m.values()) > 0.92:
                continue
            t = next(iter(m))[0]
            vals = sorted(k[1] for k in m)
            if t == "int" and len(vals) > 8:
                thr = vals[len(vals) // 2]
                e = ("bin", ">=", ("var", n), ("lit", thr))
                lab = f"{n}>={thr}:"
            else:
                e = ("var", n)
                # always labelled, so two prints of different variables can never produce the same line
                lab = (labels or {}).get(n, pick(rng, [f"{n}:", f"{n} =", n]))
            cands.append((pos, "top", ("print", lab, (e,)), 1))
        # loop sites
        s = body[pos - 1]
        if s[0] == "for" and not lang.has(s[3], "break") and s[2] <= 4:
            inner = sorted(lang.assigned_names(s[3]))
            after = pre[pos]
            for n in inner:
                m = _live_env_marg(after, n)
                if not m or len(m) < 2:
                    continue
                t = next(iter(m))[0]
                vals = sorted(k[1] for k in m)
                final_q = n in query_vars and last_assign.get(n, -1) <= pos - 1
                if final_q and t != "int":
                    continue
                if (t == "int" and len(vals) > 8) or final_q:
                    e = ("bin", ">=", ("var", n), ("lit", vals[len(vals) // 2]))
                else:
                    e = ("var", n)
                # always print the loop index too, so no two output lines of a run can be identical restatements
                st = ("print", pick(rng, [s[1], f"{s[1]}:", "iter"]), (("var", s[1]), e))
                cands.append((pos - 1, "loop", st, s[2]))
    if not cands:
        raise Reject("not enough print candidates")
    order = [int(i) for i in rng.permutation(len(cands))]
    chosen, lines, loops_used, vars_used = [], 0, set(), set()
    for i in order:
        c = cands[i]
        pv = tuple(sorted(str(x) for x in c[2][2]))
        if pv in vars_used:
            continue
        if lines + c[3] > max(n_prints, 2) + (1 if c[1] == "loop" else 0) or lines + c[3] > 6:
            continue
        if c[1] == "loop" and c[0] in loops_used:
            continue
        chosen.append(c)
        vars_used.add(pv)
        lines += c[3]
        if c[1] == "loop":
            loops_used.add(c[0])
        if lines >= n_prints:
            break
    if lines < 2:
        raise Reject("not enough print candidates")
    # insert back to front so positions stay valid; loop prints go at the end of that loop's body
    for pos, kind, st, _ in sorted(chosen, key=lambda c: (-c[0], c[1] == "top")):
        if kind == "top":
            body.insert(pos, st)
        else:
            f = body[pos]
            body[pos] = ("for", f[1], f[2], tuple(f[3]) + (st,))
    return Program(prog.funcs, tuple(body))


def make_program_world(rng, prog: Program, *, style: str, mode: str, framing: str, theme: str, family: str,
                       prefer=(), var_types=None, exclude=(), extra_meta=None) -> World:
    """Build the World (raises Reject if the program is unsuitable)."""
    try:
        dist, mass, st = lang.enumerate_program(prog, style=style)
    except Budget as e:
        raise Reject(f"budget: {e}")
    if mass < Fraction(1, 20):
        raise Reject("observe acceptance too low")
    # raw trace space (what a trace-replay checker has to visit) must stay <= MAX_TRACES, per the pack spec
    try:
        n_traces = lang.count_traces(prog)
    except Budget:
        raise Reject("trace space too large")
    if n_traces > MAX_TRACES:
        raise Reject("trace space too large")
    keys, prior, envs, outs = final_table(dist, mass)
    qvars, specs = choose_query_vars(envs, prior, rng, style, prefer=prefer, exclude=exclude)

    if mode == "print":
        if not lang.has(prog.body, "print"):
            T = int(rng.integers(2, 6))
            prog = instrument_prints(prog, rng, set(qvars), T)
        try:
            dist, mass, st = lang.enumerate_program(prog, style=style)
        except Budget as e:
            raise Reject(f"budget: {e}")
        keys, prior, envs, outs = final_table(dist, mass)
        lens = {len(o) for o in outs}
        if len(lens) != 1:
            raise Reject("variable number of printed lines")
        T = lens.pop()
        if not 2 <= T <= 6:
            raise Reject("bad number of prints")
        alt_vals = []
        for k in range(T):
            vs = sorted({o[k] for o in outs})
            if len(vs) > 16:
                raise Reject("too many print alternatives")
            alt_vals.append(vs)
        if sum(len(v) > 1 for v in alt_vals) < 2:
            raise Reject("prints not informative")
        # the full output must not (usually) pin down every query variable
        cellof = [tuple(sp[1][lang._key(e[nm])] for nm, sp in zip(qvars, specs)) for e in envs]
        by_out: dict = {}
        for i, o in enumerate(outs):
            d = by_out.setdefault(o, {})
            d[cellof[i]] = d.get(cellof[i], 0.0) + prior[i]
        pmax = sum(max(d.values()) for d in by_out.values())
        if pmax > 0.9:
            raise Reject("prints reveal the answer")
        nq = len(qvars)
        pinned = 0.0
        for d in by_out.values():
            if any(len({c[j] for c, p in d.items() if p > 1e-12}) < 2 for j in range(nq)):
                pinned += sum(d.values())
        if pinned > 0.15:
            raise Reject("prints often pin a query variable")
        pq = {}
        for i, c in enumerate(cellof):
            pq[c] = pq.get(c, 0.0) + prior[i]
        exp_tv = 0.0
        for d in by_out.values():
            po = sum(d.values())
            exp_tv += 0.5 * sum(abs(d.get(c, 0.0) - po * pc) for c, pc in pq.items())
        if exp_tv < 0.1:
            raise Reject("prints uninformative about the queries")
        if any(len(v) < 2 for v in alt_vals):
            raise Reject("a print line is deterministic")
        for k in range(1, T):
            groups = {}
            for o in outs:
                groups.setdefault(o[:k], set()).add(o[k])
            if all(len(g) == 1 for g in groups.values()):
                raise Reject("a print line is determined by earlier lines")
        out_arr = np.empty((len(outs), T), dtype=object)
        for i, o in enumerate(outs):
            out_arr[i, :] = o
    else:
        T = int(rng.integers(2, 6))
        cellof = [tuple(sp[1][lang._key(e[nm])] for nm, sp in zip(qvars, specs)) for e in envs]
        pool = cond_pool(envs, prior, rng, var_types or {}, cells=cellof, max_pool=100)
        if len(pool) < T + 2:
            raise Reject("reveal pool too small")
        # secondary pool: conditions informative about the run but not (a priori) about the queries; used when the
        # primary pool is exhausted under the realised past
        seen_v = {v.tobytes() for _, v in pool}
        for c, v in cond_pool(envs, prior, rng, var_types or {}, max_pool=40):
            if v.tobytes() not in seen_v:
                seen_v.add(v.tobytes())
                pool.append((c, v))
        shape_q = tuple(len(sp[0]) for sp in specs)
        cell_flat = np.array([np.ravel_multi_index(c, shape_q) for c in cellof])
        n_cells = int(np.prod(shape_q))

        qcodes = np.array(cellof, dtype=int).reshape(len(cellof), -1)

        def keeps_uncertain(v, post):
            """True if, whichever way the fact turns out (with positive mass), every query variable keeps >= 2
            values with mass: facts that could pin a query variable are only used as a last resort."""
            for m in (v, ~v):
                w = post * m
                tot = float(w.sum())
                if tot <= 1e-12:
                    continue
                for j in range(qcodes.shape[1]):
                    if int((np.bincount(qcodes[:, j], weights=w) > 1e-9 * tot).sum()) < 2:
                        return False
            return True

        def info_now(v, post):
            pa = float(post[v].sum())
            if pa <= 1e-12 or pa >= 1 - 1e-12:
                return 0.0
            da = np.bincount(cell_flat, weights=post * v, minlength=n_cells) / pa
            db = np.bincount(cell_flat, weights=post * ~v, minlength=n_cells) / (1 - pa)
            return 0.5 * float(np.abs(da - db).sum())

    n = len(keys)
    proj = np.zeros((n, len(qvars)), dtype=int)
    for j, (name, (opts, mp, kind)) in enumerate(zip(qvars, specs)):
        for i, e in enumerate(envs):
            proj[i, j] = mp[lang._key(e[name])]
    variables = [make_var(nm, sp[0], sp[2], rng) for nm, sp in zip(qvars, specs)]

    # mart var: the most uncertain query variable (ties random)
    ent = []
    for j in range(len(qvars)):
        m = np.bincount(proj[:, j], weights=prior, minlength=len(specs[j][0]))
        m = m[m > 0]
        ent.append(float(-(m * np.log(m)).sum()) + 1e-3 * rng.random())
    mart_var = qvars[int(np.argmax(ent))] if rng.random() < 0.6 else qvars[int(rng.integers(len(qvars)))]

    base_seed = int(rng.integers(2**31))
    code_style_tpl = [int(rng.integers(len(OBS_TPL[style]))) for _ in range(T)]
    uniform_tpl = rng.random() < 0.6
    if uniform_tpl:
        code_style_tpl = [code_style_tpl[0]] * T

    if mode == "observe":
        cache: dict = {}

        def alternatives(k, past):
            key = (k, tuple(past))
            if key in cache:
                return cache[key]
            post = prior.copy()
            used = set()
            for ci, b in past:
                post = post * (pool[ci][1] == b)
                used.add(ci)
            post = post / post.sum()
            r = np.random.default_rng([base_seed, k] + [ci * 2 + int(b) for ci, b in past])
            ps = [(ci, float(post[v].sum()), info_now(v, post), keeps_uncertain(v, post))
                  for ci, (c, v) in enumerate(pool) if ci not in used]
            tiers = [[ci for ci, p, inf, ok in ps if ok and 0.1 <= p <= 0.9 and inf >= 0.1],
                     [ci for ci, p, inf, ok in ps if ok and 0.04 <= p <= 0.96 and inf >= 0.02],
                     [ci for ci, p, inf, ok in ps if ok and 0.04 <= p <= 0.96],
                     [ci for ci, p, inf, ok in ps if 0.04 <= p <= 0.96],
                     [ci for ci, p, inf, ok in ps]]
            ti = next(i for i, t in enumerate(tiers) if t)
            tier = tiers[ti]
            ci = int(tier[int(r.integers(len(tier)))])
            cache[key] = [(ci, True), (ci, False)]
            tier_of[key] = ti
            return cache[key]

        tier_of: dict = {}

        exhaust = np.zeros(T + 1)

        def explore(past, k, mass):
            """Accumulate, per depth, the prior-predictive probability that the reveals run out of non-trivial
            facts at that step (nothing left with 0.04 <= P <= 0.96)."""
            if k == T or mass < 1e-4:
                return
            alternatives(k, past)
            if tier_of[(k, tuple(past))] >= 3:
                exhaust[k] += mass
                return
            post = prior.copy()
            for ci, b in past:
                post = post * (pool[ci][1] == b)
            tot = float(post.sum())
            ci = cache[(k, tuple(past))][0][0]
            for b in (True, False):
                pb = float(post[pool[ci][1] == b].sum()) / tot
                if pb > 1e-12:
                    explore(past + [(ci, b)], k + 1, mass * pb)

        explore([], 0, 1.0)
        T_inf = T
        for k in range(T):
            if exhaust[: k + 1].sum() > 0.1:
                T_inf = k
                break
        if T_inf < T and rng.random() < 0.5:
            raise Reject("reveals exhaust the state before T steps")  # favours richer programs
        T = T_inf
        if T < 2:
            raise Reject("reveals exhaust the state too quickly")

        def lik(o, past):
            ci, b = o
            return (pool[ci][1] == b).astype(float)

        def render_ev(k, o):
            ci, b = o
            c = pool[ci][0] if b else negate(pool[ci][0])
            code = obs_code(c, style)
            return OBS_TPL[style][code_style_tpl[k]].format(c=code)
    else:
        def alternatives(k, past):
            return alt_vals[k]

        tpl = PRINT_TPL[style][int(rng.integers(len(PRINT_TPL[style])))]

        def lik(o, past):
            return (out_arr[:, len(past)] == o).astype(float)

        def render_ev(k, o):
            return tpl.format(o=o)

    code = render(prog, style, indent=int(pick(rng, [4, 4, 2])))
    prelude = assemble_prelude(rng, framing, code, style, uses(prog), mode)
    meta = {"style": style, "theme": theme, "mode": mode, "n_states": n, "peak_states": st["peak_states"],
            "n_traces": n_traces,
            "accept_mass": round(float(mass), 8), "size": lang.count_stmts(prog.body) + sum(
                lang.count_stmts(f[2]) for f in prog.funcs), "depth": lang.depth(prog.body),
            "helpers": len(prog.funcs)}
    meta.update(extra_meta or {})
    params = {"program": prog, "qvars": qvars, "specs": specs, "envs": envs, "outs": outs, "prior": prior,
              "pool": pool if mode == "observe" else None, "style": style}
    return World(variables=variables, prior=prior, proj=proj, T=T, alternatives=alternatives, lik=lik,
                 render=render_ev, prelude=prelude, mart_var=mart_var, exchangeable=(mode == "observe"),
                 meta=meta, params=params)


def _count_prints(stmts):
    n = 0
    for s in stmts:
        if s[0] == "print":
            n += 1
        elif s[0] == "for":
            n += _count_prints(s[3])
    return n


def level_of(prog: Program) -> str:
    size = lang.count_stmts(prog.body) - _count_prints(prog.body) + sum(lang.count_stmts(f[2]) for f in prog.funcs)
    d = lang.depth(prog.body)
    if not prog.funcs and size <= 12 and d <= 1:
        return "L2"
    return "L3"


def seed_of(*parts) -> list[int]:
    return [zlib.crc32(str(p).encode()) if not isinstance(p, int) else p for p in parts]
