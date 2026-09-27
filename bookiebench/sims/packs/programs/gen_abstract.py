"""Random abstract probabilistic programs (grammar-based, abstract variable names).

Statements are generated one at a time; after each, the exact state distribution of the program prefix is
recomputed, so thresholds, compared values and branch conditions are chosen from the variables' actual supports
(conditions are neither always true nor always false). New variables are only introduced at the top level (inside
blocks only existing variables are reassigned), so every variable is defined on every path.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from bookiebench.sims.common import pick, sample

from . import lang
from .build import Reject, STYLES, level_of, make_program_world
from .lang import Budget, Program

INT_NAMES = ["x", "y", "z", "n", "k", "m", "a", "b", "c", "s", "t", "u", "v", "w", "total", "score", "count", "acc",
             "val", "pos", "steps", "level", "tally", "q", "r", "d"]
BOOL_NAMES = ["flag", "ok", "hit", "done", "p", "b1", "b2", "on", "found", "win", "alive", "seen"]
STR_NAMES = ["color", "mode", "label", "side", "kind", "tag", "state", "shape", "lane"]
LABEL_SETS = [["red", "green", "blue"], ["A", "B", "C"], ["left", "right"], ["low", "mid", "high"],
              ["cat", "dog", "fox"], ["up", "down"], ["north", "south", "east"], ["alpha", "beta", "gamma"],
              ["x1", "x2", "x3", "x4"], ["on", "off"], ["hot", "cold"], ["spade", "heart", "club"]]
LOOP_VARS = ["i", "j", "r", "step", "it", "rep", "trial", "round_"]
FUNC_NAMES = ["f", "g", "h", "advance", "roll", "noisy", "bump", "gate", "draw", "sample_one", "mix", "trial_fn"]
PCTS = [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 15, 35, 45, 55, 65, 85, 5, 95]


def V(n):
    return ("var", n)


def L(v):
    return ("lit", v)


def B(op, a, b):
    return ("bin", op, a, b)


class Ctx:
    def __init__(self, rng, level):
        self.rng = rng
        self.level = level
        self.body: list = []
        self.funcs: list = []
        self.types: dict = {}
        self.en = None
        self.states = {((), (), None): Fraction(1)}
        self.labels = pick(rng, LABEL_SETS)
        self.used_names = set()
        self.loopvars = set()

    # ------------------------------------------------------------------------------------------------------------
    def prog(self):
        return Program(tuple(self.funcs), tuple(self.body))

    def fresh(self, pool):
        free = [n for n in pool if n not in self.used_names and n not in self.loopvars]
        if not free:
            return None
        n = pick(self.rng, free)
        return n

    def marg(self, name):
        m, tot = {}, Fraction(0)
        for (fz, outs, ctrl), p in self.states.items():
            env = lang._thaw(fz)
            k = lang._key(env[name])
            m[k] = m.get(k, Fraction(0)) + p
            tot += p
        return {k[1]: float(v / tot) for k, v in m.items()}

    def vars_of(self, t):
        return [n for n, tt in self.types.items() if tt == t]

    def try_add(self, stmts, new_types=None):
        """Append statements if the enumeration stays within budget and values stay small."""
        en = lang.Enumerator(Program(tuple(self.funcs), ()), max_states=4000)
        try:
            st = self.states
            for s in stmts:
                st = en.stmt(s, st)
                if len(st) > 4000:
                    raise Budget("states")
        except (Budget, KeyError):
            return False
        if not st:
            return False
        mass = sum(st.values(), Fraction(0))
        if mass < Fraction(1, 8):
            return False
        # value-range guard
        types = dict(self.types)
        types.update(new_types or {})
        vals = {}
        for (fz, outs, ctrl), p in st.items():
            for k, (tn, v) in fz:
                vals.setdefault(k, set()).add(v)
        if any(len(v) > 14 for v in vals.values()):
            return False
        if any(isinstance(x, int) and not isinstance(x, bool) and abs(x) > 60 for v in vals.values() for x in v):
            return False
        self.states = st
        self.body += stmts
        self.types = types
        for s in stmts:
            self.used_names |= lang.assigned_names([s])
        return True

    # conditions -------------------------------------------------------------------------------------------------
    def cond(self, allow_compound=True, extra_ints=(), avoid=(), min_flip=5):
        """A random condition over existing variables that is uncertain under the current distribution.
        `avoid`: variable names not to test (e.g. those tested by an enclosing branch)."""
        rng = self.rng
        avoid = set(avoid)
        pcts = [p for p in PCTS if min_flip <= p <= 100 - min_flip]
        for _ in range(30):
            kind = rng.random()
            ints = [n for n in self.vars_of("int") + list(extra_ints) if n not in avoid]
            bools = [n for n in self.vars_of("bool") if n not in avoid]
            strs = [n for n in self.vars_of("str") if n not in avoid]
            if kind < 0.45 and ints:
                n = pick(rng, ints)
                if n in extra_ints:
                    c = B(pick(rng, ["==", ">=", "<"]), V(n), L(int(rng.integers(0, 3))))
                else:
                    m = self.marg(n)
                    vals = sorted(m)
                    if len(vals) < 2:
                        continue
                    cum = np.cumsum([m[v] for v in vals])
                    j = int(np.searchsorted(cum, rng.uniform(0.25, 0.75)))
                    j = min(max(j, 0), len(vals) - 2)
                    op = pick(rng, [">", "<=", ">=", "<", "=="])
                    t = vals[j] if op in (">", "<=") else vals[j + 1]
                    if op == "==":
                        t = pick(rng, vals)
                    c = B(op, V(n), L(int(t)))
            elif kind < 0.65 and bools:
                n = pick(rng, bools)
                c = V(n) if rng.random() < 0.6 else ("not", V(n))
            elif kind < 0.8 and strs:
                n = pick(rng, strs)
                vals = sorted(self.marg(n))
                c = B("==" if rng.random() < 0.8 else "!=", V(n), L(pick(rng, vals)))
            elif kind < 0.9 and len(ints) >= 2:
                a, b = sample(rng, ints, 2)
                c = B(pick(rng, [">", "<", ">=", "=="]), V(a), V(b))
            elif kind >= 0.9:
                c = ("flip", pick(rng, pcts))
            else:
                continue
            if allow_compound and rng.random() < 0.2:
                c2 = self.cond(False, extra_ints, avoid=avoid | cond_vars(c), min_flip=min_flip)
                if c2[0] != "flip" or c[0] != "flip":
                    c = B(pick(rng, ["and", "or"]), c, c2)
            return c
        return ("flip", pick(rng, pcts))

    # expressions ------------------------------------------------------------------------------------------------
    def rand_int_expr(self):
        rng = self.rng
        r = rng.random()
        if r < 0.55:
            lo = int(rng.integers(0, 4))
            return ("randint", lo, lo + int(rng.integers(1, 5)))
        if r < 0.8:
            k = int(rng.integers(2, 4))
            vals = tuple(sorted(int(v) for v in rng.choice(np.arange(0, 9), size=k, replace=False)))
            w = tuple(int(x) for x in rng.integers(1, 6, size=k)) if rng.random() < 0.7 else None
            return ("choice", vals, w)
        return B("+", ("randint", 0, int(rng.integers(1, 3))), ("randint", 0, int(rng.integers(1, 3))))

    def derived_int_expr(self):
        rng = self.rng
        ints = self.vars_of("int")
        bools = self.vars_of("bool")
        if not ints:
            return None
        x = pick(rng, ints)
        r = rng.random()
        if r < 0.3:
            return B("+", V(x), ("randint", 0, int(rng.integers(1, 3))))
        if r < 0.45 and len(ints) >= 2:
            y = pick(rng, [i for i in ints if i != x])
            return B(pick(rng, ["+", "-"]), V(x), V(y))
        if r < 0.55 and len(ints) >= 2:
            y = pick(rng, [i for i in ints if i != x])
            return ("call", pick(rng, ["max", "min"]), (V(x), V(y)))
        if r < 0.65:
            return B("*", V(x), L(int(rng.integers(2, 4))))
        if r < 0.8 and bools:
            b = pick(rng, bools)
            return ("ifexp", V(b), B("+", V(x), L(int(rng.integers(1, 3)))), V(x))
        if r < 0.9 and len(ints) >= 2:
            y = pick(rng, [i for i in ints if i != x])
            return ("call", "abs", (B("-", V(x), V(y)),))
        return B("+", V(x), ("choice", (0, 1, 2), tuple(int(v) for v in rng.integers(1, 5, size=3))))

    # statements -------------------------------------------------------------------------------------------------
    def reassign(self, allow_loopvar=None):
        """A block-level reassignment of an existing variable."""
        rng = self.rng
        opts = []
        ints, bools, strs = self.vars_of("int"), self.vars_of("bool"), self.vars_of("str")
        if ints:
            n = pick(rng, ints)
            r = rng.random()
            if r < 0.35:
                opts.append(("aug", n, pick(rng, ["+", "+", "-"]), L(int(rng.integers(1, 3)))))
            elif r < 0.6:
                opts.append(("assign", n, self.rand_int_expr()))
            elif r < 0.8:
                opts.append(("aug", n, "+", ("randint", 0, int(rng.integers(1, 3)))))
            elif len(ints) >= 2:
                m = pick(rng, [i for i in ints if i != n])
                opts.append(("assign", n, B("+", V(n), V(m)) if rng.random() < 0.5 else V(m)))
            else:
                opts.append(("assign", n, L(int(rng.integers(0, 5)))))
        if bools:
            n = pick(rng, bools)
            r = rng.random()
            if r < 0.5:
                opts.append(("assign", n, ("flip", pick(rng, PCTS))))
            elif r < 0.75:
                opts.append(("assign", n, ("not", V(n))))
            else:
                opts.append(("assign", n, L(bool(rng.random() < 0.5))))
        if strs:
            n = pick(rng, strs)
            vals = self.str_vals(n)
            if rng.random() < 0.5:
                opts.append(("assign", n, L(pick(rng, vals))))
            else:
                opts.append(("assign", n, ("choice", tuple(vals), tuple(int(v) for v in rng.integers(1, 6, size=len(vals))))))
        if not opts:
            return None
        return pick(rng, opts)

    def str_vals(self, n):
        return sorted(set(self.labels) | set(self.marg(n)))

    def g_new_rand(self):
        rng = self.rng
        r = rng.random()
        if r < 0.45:
            n = self.fresh(INT_NAMES)
            return n and ([("assign", n, self.rand_int_expr())], {n: "int"})
        if r < 0.75:
            n = self.fresh(BOOL_NAMES)
            return n and ([("assign", n, ("flip", pick(rng, PCTS)))], {n: "bool"})
        n = self.fresh(STR_NAMES)
        labs = self.labels
        w = tuple(int(v) for v in rng.integers(1, 6, size=len(labs))) if rng.random() < 0.7 else None
        return n and ([("assign", n, ("choice", tuple(labs), w))], {n: "str"})

    def g_new_derived(self):
        rng = self.rng
        if rng.random() < 0.6:
            e = self.derived_int_expr()
            n = self.fresh(INT_NAMES)
            return e and n and ([("assign", n, e)], {n: "int"})
        n = self.fresh(BOOL_NAMES)
        c = self.cond()
        if rng.random() < 0.3:
            c = B(pick(rng, ["and", "or"]), c, ("flip", pick(rng, PCTS)))
        return n and ([("assign", n, c)], {n: "bool"})

    def block_body(self, depth_left, avoid=()):
        rng = self.rng
        k = int(rng.integers(1, 3))
        out, targets = [], set()
        for _ in range(k):
            if depth_left > 0 and rng.random() < 0.3:
                out.append(self.if_stmt(depth_left - 1, avoid))
            elif rng.random() < 0.08 and self.level == "L3":
                out.append(("observe", self.cond(False, avoid=avoid, min_flip=30)))
            else:
                for _ in range(4):
                    s = self.reassign()
                    if s and s[1] not in targets:  # no dead stores
                        targets.add(s[1])
                        out.append(s)
                        break
        return [s for s in out if s] or None

    def if_stmt(self, depth_left, avoid=()):
        rng = self.rng
        n_arms = 1 + (rng.random() < 0.35) + (rng.random() < 0.15)
        arms = []
        tested = set(avoid)
        prev_bools = set()
        for _ in range(n_arms):
            for _ in range(10):  # elif arms: no repeated condition, no re-test of a boolean already tested
                c = self.cond(avoid=set(avoid) | prev_bools)
                if all(c != a[0] for a in arms):
                    break
            else:
                break
            tested |= cond_vars(c)
            prev_bools |= {n for n in cond_vars(c) if self.types.get(n) == "bool"}
            b = self.block_body(depth_left, tested)
            if not b:
                return None
            arms.append((c, tuple(b)))
        if not arms:
            return None
        els = None
        if rng.random() < 0.55:
            b = self.block_body(depth_left, tested)
            els = tuple(b) if b else None
        return ("if", tuple(arms), els)

    def g_if(self):
        s = self.if_stmt(1 if self.level == "L3" else 0)
        return s and ([s], {})

    def g_loop(self):
        rng = self.rng
        lv = pick(rng, [v for v in LOOP_VARS if v not in self.used_names])
        n = int(rng.integers(2, 5))
        pre, new_types = [], {}
        self.loopvars.add(lv)  # before choosing fresh names, so the counter can never be the loop variable
        cnt = self.fresh(INT_NAMES)
        if cnt is None:
            return None
        pre.append(("assign", cnt, L(0)))
        new_types[cnt] = "int"
        p = pick(rng, PCTS)
        r = rng.random()
        body = []
        if r < 0.35:
            body.append(("if", ((("flip", p), (("aug", cnt, "+", L(1)),)),), None))
        elif r < 0.55:
            body.append(("aug", cnt, "+", ("randint", 0, 1 + int(rng.random() < 0.3))))
        elif r < 0.75:
            # data-dependent increment
            c = self.cond(extra_ints=(lv,))
            body.append(("if", ((c, (("aug", cnt, "+", L(1)),)),), (("aug", cnt, "+", ("randint", 0, 1)),)
                         if rng.random() < 0.3 else None))
        else:
            hit = self.fresh(BOOL_NAMES)
            if hit is None:
                return None
            pre.append(("assign", hit, L(False)))
            new_types[hit] = "bool"
            body.append(("aug", cnt, "+", L(1)))
            body.append(("if", ((("flip", p), (("assign", hit, L(True)), ("break",))),), None))
        if self.level == "L3" and rng.random() < 0.4:
            self.types.update(new_types)
            s = self.reassign()
            for k in new_types:
                self.types.pop(k)
            if s and s[1] != lv:
                body.append(("if", ((("flip", pick(rng, PCTS)), (s,)),), None))
        if rng.random() < 0.2 and r < 0.75:
            thr = int(rng.integers(1, n))
            body.append(("if", ((B(">=", V(cnt), L(thr)), (("break",),)),), None))
        self.loopvars.add(lv)
        return pre + [("for", lv, n, tuple(body))], new_types

    def g_observe(self):
        c = self.cond(False, min_flip=30)
        return [("observe", c)], {}

    def g_helper(self):
        rng = self.rng
        name = pick(rng, [f for f in FUNC_NAMES if f not in {g[0] for g in self.funcs}])
        ints = self.vars_of("int")
        bools = self.vars_of("bool")
        shape = int(rng.integers(0, 6))
        out = self.fresh(INT_NAMES)
        if out is None:
            return None
        if shape == 0 and ints:  # noisy increment
            k = int(rng.integers(1, 3))
            f = (name, ("n",), (("if", ((("flip", pick(rng, PCTS)), (("return", B("+", V("n"), L(k))),)),), None),
                                ("return", V("n"))))
            call = ("call", name, (V(pick(rng, ints)),))
            t = "int"
        elif shape == 1:  # sum of two dice
            a, b = int(rng.integers(0, 2)), int(rng.integers(2, 4))
            f = (name, (), (("return", B("+", ("randint", a, b), ("randint", a, b))),))
            call = ("call", name, ())
            t = "int"
        elif shape == 2 and ints:  # state-dependent coin
            x = pick(rng, ints)
            thr = int(pick(rng, sorted(self.marg(x))))
            f = (name, ("a",), (("if", ((B(">", V("a"), L(thr)), (("return", ("flip", pick(rng, PCTS))),)),), None),
                                ("return", ("flip", pick(rng, PCTS)))))
            call = ("call", name, (V(x),))
            t = "bool"
            out = self.fresh(BOOL_NAMES)
        elif shape == 3:  # capped geometric: loop with early return
            m = int(rng.integers(2, 5))
            f = (name, (), (("for", "i", m, (("if", ((("flip", pick(rng, PCTS)), (("return", B("+", V("i"), L(1))),)),),
                                                   None),)),
                            ("return", L(0))))
            call = ("call", name, ())
            t = "int"
        elif shape == 4 and bools:  # bool -> label
            labs = tuple(self.labels)
            w1 = tuple(int(v) for v in rng.integers(1, 6, size=len(labs)))
            w2 = tuple(int(v) for v in rng.integers(1, 6, size=len(labs)))
            f = (name, ("a",), (("if", ((V("a"), (("return", ("choice", labs, w1)),)),), None),
                                ("return", ("choice", labs, w2))))
            call = ("call", name, (V(pick(rng, bools)),))
            t = "str"
            out = self.fresh(STR_NAMES)
        elif ints:  # clamp with noise
            lo, hi = int(rng.integers(0, 2)), int(rng.integers(2, 5))
            f = (name, ("v",), (("return", ("call", "min", (("call", "max", (B("+", V("v"), ("randint", -1, 1)),
                                                                              L(lo))), L(hi)))),))
            call = ("call", name, (V(pick(rng, ints)),))
            t = "int"
        else:
            return None
        if out is None:
            return None
        self.funcs.append(f)
        ok = self.try_add([("assign", out, call)], {out: t})
        if not ok:
            self.funcs.pop()
            return None
        # sometimes call it a second time
        if rng.random() < 0.3 and f[1] == ():
            n2 = self.fresh(INT_NAMES if t == "int" else (BOOL_NAMES if t == "bool" else STR_NAMES))
            if n2:
                self.try_add([("assign", n2, call)], {n2: t})
        return "done"


def cond_vars(e) -> set:
    t = e[0]
    if t == "var":
        return {e[1]}
    if t == "bin":
        return cond_vars(e[2]) | cond_vars(e[3])
    if t == "not":
        return cond_vars(e[1])
    return set()


def make_program(rng, level):
    ctx = Ctx(rng, level)
    n_top = int(rng.integers(3, 6)) if level == "L2" else int(rng.integers(5, 9))
    # start with 1-2 random variables
    for _ in range(int(rng.integers(1, 3))):
        r = ctx.g_new_rand()
        if r:
            ctx.try_add(*r)
    gens = [ctx.g_new_rand, ctx.g_new_derived, ctx.g_if, ctx.g_loop, ctx.g_observe]
    w = np.array([2.0, 2.0, 2.5, 1.5, 0.5])
    if level == "L3":
        gens.append(ctx.g_helper)
        w = np.append(w, 1.2)
    n_obs = 0
    n_helpers = 0
    tries = 0
    while len(ctx.body) < n_top and tries < 60:
        tries += 1
        gi = int(rng.choice(len(gens), p=w / w.sum()))
        g = gens[gi]
        if g == ctx.g_observe and (n_obs >= (1 if level == "L2" else 2) or len(ctx.body) < 2):
            continue
        if g == ctx.g_helper:
            if n_helpers >= 2:
                continue
            if g():
                n_helpers += 1
            continue
        r = g()
        if not r:
            continue
        if ctx.try_add(*r) and g == ctx.g_observe:
            n_obs += 1
    if level == "L3" and n_helpers == 0 and rng.random() < 0.6:
        ctx.g_helper()
    return ctx.prog(), ctx.types


FRAMINGS = [
    "Consider the following probabilistic program.",
    "Here is a short randomized program.",
    "The code below uses random primitives.",
    "A probabilistic program is given below.",
    "Look at this small program with random choices.",
]


def make_world(rng):
    mode = "observe" if rng.random() < 0.55 else "print"
    for attempt in range(300):
        level = "L2" if rng.random() < 0.45 else "L3"
        style = pick(rng, STYLES)
        prog, types = make_program(rng, level)
        if len(prog.body) < 3:
            continue
        try:
            w = make_program_world(rng, prog, style=style, mode=mode, framing=pick(rng, FRAMINGS), theme="abstract",
                                   family="prog_abstract", var_types=types)
        except Reject:
            continue
        w.meta["level"] = level_of(w.params["program"])
        return w
    raise RuntimeError("could not generate an abstract program")
