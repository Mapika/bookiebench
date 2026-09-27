"""A tiny probabilistic language: IR, exact enumeration and renderers (Python / JavaScript / pseudo-code).

IR (plain tuples, so programs are hashable and cheap to copy):

  expressions
    ('lit', v)                       int / bool / str literal
    ('var', name)
    ('bin', op, a, b)                op in + - * < <= > >= == != and or
    ('not', a)
    ('ifexp', c, a, b)               a if c else b
    ('call', fname, (args...))       user helper, or builtin min / max / abs
    ('flip', pct)                    True with probability pct/100
    ('randint', lo, hi)              uniform integer in [lo, hi]
    ('choice', (values...), (weights...) | None)

  statements
    ('assign', name, expr)
    ('aug', name, op, expr)          name op= expr, op in + -
    ('if', ((cond, body), ...), else_body | None)
    ('for', var, n, body)            for var in range(n)
    ('break',)
    ('observe', cond)                condition the run on cond (rejection)
    ('print', label | None, (exprs...))
    ('return', expr)                 helpers only

  program = Program(funcs=((name, params, body), ...), body=(stmts...))

Enumeration is exact: it propagates a distribution over program states (environment, printed lines), merging equal
states after every statement, so the cost is the number of distinct states rather than the number of traces.
Mass lost to failing observe() statements is dropped; `run` returns the unnormalised accepted distribution and the
accepted mass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

CMP = ("<", "<=", ">", ">=", "==", "!=")
ARITH = ("+", "-", "*")


class Budget(Exception):
    """Raised when enumeration exceeds its state budget (the generator then rejects the program)."""


@dataclass(frozen=True)
class Program:
    funcs: tuple = ()
    body: tuple = ()


# ----------------------------------------------------------------------------------------------------------------------
# exact enumeration
# ----------------------------------------------------------------------------------------------------------------------

def _key(v):
    # bool and int hash equal in Python (True == 1); tag the type so states never merge across types
    return (type(v).__name__, v)


def _freeze(env: dict):
    return tuple(sorted((k, _key(v)) for k, v in env.items()))


def _thaw(fz):
    return {k: v for k, (_, v) in fz}


def fmt_value(v, style="python") -> str:
    """How print() shows a value in the given style."""
    if isinstance(v, bool):
        if style == "python":
            return "True" if v else "False"
        if style == "pseudo":
            return "TRUE" if v else "FALSE"  # same spelling as the pseudo-code literals
        return "true" if v else "false"
    return str(v)


class Enumerator:
    def __init__(self, program: Program, max_states: int = 20000, style: str = "python", count: bool = False):
        """count=True: every random option gets weight 1 and observe() never rejects, so the total weight is an
        upper bound on the number of raw execution traces (a trace-replay enumerator stops a trace at its first
        failing observe, so it visits at most this many)."""
        self.count = count
        self.funcs = {f[0]: f for f in program.funcs}
        self.program = program
        self.max_states = max_states
        self.style = style
        self.work = 0
        self.peak = 0

    # expressions ------------------------------------------------------------------------------------------------------
    def ev(self, e, env) -> list:
        """list of (value, prob) (probabilities are Fractions; may sum to < 1 if a helper hits a failing observe)."""
        self.work += 1
        if self.work > 50 * self.max_states + 200000:
            raise Budget("work")
        t = e[0]
        if t == "lit":
            return [(e[1], Fraction(1))]
        if t == "var":
            return [(env[e[1]], Fraction(1))]
        if t == "flip":
            if self.count:
                return [(True, Fraction(1)), (False, Fraction(1))]
            p = Fraction(e[1], 100)
            return [(True, p), (False, 1 - p)]
        if t == "randint":
            lo, hi = e[1], e[2]
            n = hi - lo + 1
            return [(v, Fraction(1) if self.count else Fraction(1, n)) for v in range(lo, hi + 1)]
        if t == "choice":
            vals, w = e[1], e[2]
            w = w or (1,) * len(vals)
            if self.count:
                return [(v, Fraction(1)) for v, wi in zip(vals, w) if wi]  # one trace per option
            tot = sum(w)
            out = {}
            for v, wi in zip(vals, w):
                if wi:
                    out[_key(v)] = out.get(_key(v), Fraction(0)) + Fraction(wi, tot)
            return [(k[1], p) for k, p in out.items()]
        if t == "not":
            return [(not v, p) for v, p in self.ev(e[1], env)]
        if t == "ifexp":
            out = []
            for c, p in self.ev(e[1], env):
                for v, q in self.ev(e[2] if c else e[3], env):
                    out.append((v, p * q))
            return _merge_vals(out)
        if t == "bin":
            op = e[1]
            out = []
            for a, p in self.ev(e[2], env):
                if op == "and" and not a:
                    out.append((a, p))
                    continue
                if op == "or" and a:
                    out.append((a, p))
                    continue
                for b, q in self.ev(e[3], env):
                    out.append((_binop(op, a, b), p * q))
            return _merge_vals(out)
        if t == "call":
            name, args = e[1], e[2]
            argd = [[]]
            argp = [Fraction(1)]
            for a in args:
                vals = self.ev(a, env)
                argd = [d + [v] for d in argd for v, _ in vals]
                argp = [pd * q for pd in argp for _, q in vals]
            out = []
            for vals, p in zip(argd, argp):
                if name == "min":
                    out.append((min(vals), p))
                elif name == "max":
                    out.append((max(vals), p))
                elif name == "abs":
                    out.append((abs(vals[0]), p))
                else:
                    for v, q in self.call(name, vals):
                        out.append((v, p * q))
            return _merge_vals(out)
        raise ValueError(f"bad expr {e!r}")

    def call(self, name, vals) -> list:
        _, params, body = self.funcs[name]
        env = dict(zip(params, vals))
        states = {(_freeze(env), (), None): Fraction(1)}
        states = self.block(body, states)
        out = []
        for (fz, outs, ctrl), p in states.items():
            if ctrl is None or ctrl[0] != "ret":
                raise ValueError(f"helper {name} fell off without return")
            out.append((ctrl[1], p))
        return _merge_vals(out)

    # statements -------------------------------------------------------------------------------------------------------
    def block(self, stmts, states: dict) -> dict:
        for s in stmts:
            states = self.stmt(s, states)
            if len(states) > self.max_states:
                raise Budget("states")
            self.peak = max(self.peak, len(states))
        return states

    def stmt(self, s, states: dict) -> dict:
        t = s[0]
        out: dict = {}

        def put(fz, outs, ctrl, p):
            if p == 0:
                return
            k = (fz, outs, ctrl)
            out[k] = out.get(k, Fraction(0)) + p

        if t == "for":
            _, var, n, body = s
            live = {}
            for k, p in states.items():
                if k[2] is None:
                    live[k] = p
                else:
                    put(*k, p)
            for i in range(n):
                cur = {}
                for (fz, outs, ctrl), p in live.items():
                    env = _thaw(fz)
                    env[var] = i
                    kk = (_freeze(env), outs, None)
                    cur[kk] = cur.get(kk, Fraction(0)) + p
                cur = self.block(body, cur)
                live = {}
                for (fz, outs, ctrl), p in cur.items():
                    if ctrl == ("break",):
                        env = _thaw(fz)
                        env.pop(var, None)
                        put(_freeze(env), outs, None, p)
                    elif ctrl is None:
                        live[(fz, outs, ctrl)] = p
                    else:
                        put(fz, outs, ctrl, p)  # return from inside a helper loop
            for (fz, outs, ctrl), p in live.items():
                env = _thaw(fz)
                env.pop(var, None)
                put(_freeze(env), outs, None, p)
            return out

        if t == "if":
            _, arms, els = s
            for k, p in states.items():
                fz, outs, ctrl = k
                if ctrl is not None:
                    put(fz, outs, ctrl, p)
                    continue
                env = _thaw(fz)
                self._if_arms(arms, els, 0, env, fz, outs, p, put)
            return out

        for k, p in states.items():
            fz, outs, ctrl = k
            if ctrl is not None:
                put(fz, outs, ctrl, p)
                continue
            env = _thaw(fz)
            if t == "assign" or t == "aug":
                name = s[1]
                expr = s[2] if t == "assign" else s[3]
                for v, q in self.ev(expr, env):
                    e2 = dict(env)
                    e2[name] = v if t == "assign" else _binop(s[2], env[name], v)
                    put(_freeze(e2), outs, None, p * q)
            elif t == "observe":
                for v, q in self.ev(s[1], env):
                    if v or self.count:
                        put(fz, outs, None, p * q)
            elif t == "print":
                label, exprs = s[1], s[2]
                combos = [([], Fraction(1))]
                for ex in exprs:
                    vals = self.ev(ex, env)
                    combos = [(c + [v], cp * q) for c, cp in combos for v, q in vals]
                for c, q in combos:
                    parts = ([label] if label is not None else []) + [fmt_value(v, self.style) for v in c]
                    put(fz, outs + (" ".join(parts),), None, p * q)
            elif t == "break":
                put(fz, outs, ("break",), p)
            elif t == "return":
                for v, q in self.ev(s[1], env):
                    put(fz, outs, ("ret", v), p * q)
            else:
                raise ValueError(f"bad stmt {s!r}")
        return out

    def _if_arms(self, arms, els, i, env, fz, outs, p, put):
        if i == len(arms):
            if els is None:
                put(fz, outs, None, p)
            else:
                sub = self.block(els, {(fz, outs, None): p})
                for k, q in sub.items():
                    put(*k, q)
            return
        cond, body = arms[i]
        for v, q in self.ev(cond, env):
            if v:
                sub = self.block(body, {(fz, outs, None): p * q})
                for k, r in sub.items():
                    put(*k, r)
            else:
                self._if_arms(arms, els, i + 1, env, fz, outs, p * q, put)

    def run(self, body=None):
        body = self.program.body if body is None else body
        states = {((), (), None): Fraction(1)}
        states = self.block(body, states)
        dist = {}
        for (fz, outs, ctrl), p in states.items():
            k = (fz, outs)
            dist[k] = dist.get(k, Fraction(0)) + p
        mass = sum(dist.values(), Fraction(0))
        return dist, mass


def _merge_vals(pairs):
    out = {}
    for v, p in pairs:
        k = _key(v)
        out[k] = out.get(k, Fraction(0)) + p
    return [(k[1], p) for k, p in out.items() if p != 0]


def _binop(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "and":
        return a and b
    if op == "or":
        return a or b
    raise ValueError(op)


def enumerate_program(program: Program, max_states=20000, style="python"):
    """Exact accepted distribution {(frozen_env, outputs): prob (Fraction, unnormalised)}, accepted mass, stats."""
    en = Enumerator(program, max_states=max_states, style=style)
    dist, mass = en.run()
    return dist, mass, {"peak_states": en.peak, "work": en.work}


def count_traces(program: Program, max_states=200000) -> int:
    """Upper bound on the number of raw execution traces (random-choice sequences) of the program."""
    en = Enumerator(program, max_states=max_states, count=True)
    dist, mass = en.run()
    return int(mass)


def eval_in(expr, fz_env, program: Program | None = None):
    """Evaluate a deterministic expression on a frozen final environment (used for reveal conditions)."""
    en = Enumerator(program or Program())
    vals = en.ev(expr, _thaw(fz_env))
    assert len(vals) == 1, "reveal expression must be deterministic"
    return vals[0][0]


# ----------------------------------------------------------------------------------------------------------------------
# static helpers
# ----------------------------------------------------------------------------------------------------------------------

def assigned_names(stmts) -> set:
    out = set()
    for s in stmts:
        t = s[0]
        if t in ("assign", "aug"):
            out.add(s[1])
        elif t == "if":
            for _, b in s[1]:
                out |= assigned_names(b)
            if s[2]:
                out |= assigned_names(s[2])
        elif t == "for":
            out |= assigned_names(s[3])
    return out


def count_stmts(stmts) -> int:
    n = 0
    for s in stmts:
        n += 1
        if s[0] == "if":
            n += sum(count_stmts(b) for _, b in s[1]) + (count_stmts(s[2]) if s[2] else 0)
        elif s[0] == "for":
            n += count_stmts(s[3])
    return n


def depth(stmts) -> int:
    d = 0
    for s in stmts:
        if s[0] == "if":
            d = max(d, 1 + max([depth(b) for _, b in s[1]] + [depth(s[2]) if s[2] else 0]))
        elif s[0] == "for":
            d = max(d, 1 + depth(s[3]))
    return d


def has(stmts, kind) -> bool:
    for s in stmts:
        if s[0] == kind:
            return True
        if s[0] == "if" and (any(has(b, kind) for _, b in s[1]) or (s[2] and has(s[2], kind))):
            return True
        if s[0] == "for" and has(s[3], kind):
            return True
    return False


# ----------------------------------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------------------------------

_PREC = {"or": 1, "and": 2, "not": 3, "<": 4, "<=": 4, ">": 4, ">=": 4, "==": 4, "!=": 4, "+": 5, "-": 5, "*": 6}


class Renderer:
    """style in {python, js, pseudo}."""

    def __init__(self, style="python", indent=4):
        self.style = style
        self.ind = " " * indent

    # literals / expressions -----------------------------------------------------------------------------------------
    def lit(self, v):
        if isinstance(v, bool):
            return {"python": "True" if v else "False", "js": "true" if v else "false",
                    "pseudo": "TRUE" if v else "FALSE"}[self.style]
        if isinstance(v, str):
            return '"' + v + '"'
        return str(v)

    def op(self, op):
        if self.style == "js":
            return {"and": "&&", "or": "||", "==": "===", "!=": "!=="}.get(op, op)
        if self.style == "pseudo":
            return {"and": "AND", "or": "OR", "==": "=", "!=": "<>"}.get(op, op)
        return op

    def prob(self, pct):
        s = f"{pct / 100:.2f}".rstrip("0")
        return s if not s.endswith(".") else s + "0"

    def expr(self, e, parent=0) -> str:
        t = e[0]
        S = self.style
        if t == "lit":
            return self.lit(e[1])
        if t == "var":
            return e[1]
        if t == "flip":
            return f"{'FLIP' if S == 'pseudo' else 'flip'}({self.prob(e[1])})"
        if t == "randint":
            return f"{'RANDINT' if S == 'pseudo' else 'randint'}({e[1]}, {e[2]})"
        if t == "choice":
            vals = ", ".join(self.lit(v) for v in e[1])
            fn = "CHOICE" if S == "pseudo" else "choice"
            if e[2] is None:
                return f"{fn}([{vals}])"
            return f"{fn}([{vals}], [{', '.join(str(w) for w in e[2])}])"
        if t == "not":
            inner = e[1]
            if S == "js":
                s = "!" + self.expr(inner, 99)
                return s
            kw = "not " if S == "python" else "NOT "
            s = kw + self.expr(inner, _PREC["not"])
            return f"({s})" if parent > _PREC["not"] else s
        if t == "ifexp":
            if S == "python":
                s = f"{self.expr(e[2], 1)} if {self.expr(e[1], 1)} else {self.expr(e[3], 1)}"
            elif S == "js":
                s = f"{self.expr(e[1], 1)} ? {self.expr(e[2], 1)} : {self.expr(e[3], 1)}"
            else:
                s = f"IF {self.expr(e[1], 1)} THEN {self.expr(e[2], 1)} ELSE {self.expr(e[3], 1)}"
            return f"({s})"
        if t == "bin":
            op = e[1]
            pr = _PREC[op]
            if S == "js" and op in ("and", "or"):
                pr = {"or": 1, "and": 2}[op]
            # comparisons are non-associative; arithmetic is left-assoc; right operand of '-' needs parens at equal prec
            lhs = self.expr(e[2], pr + (1 if pr == 4 else 0))
            rhs = self.expr(e[3], pr + 1)
            s = f"{lhs} {self.op(op)} {rhs}"
            return f"({s})" if parent > pr else s
        if t == "call":
            name = e[1]
            args = ", ".join(self.expr(a) for a in e[2])
            if S == "js" and name in ("min", "max", "abs"):
                name = "Math." + name
            return f"{name}({args})"
        raise ValueError(e)

    # statements -----------------------------------------------------------------------------------------------------
    def block(self, stmts, lvl, declared: set, in_func=False) -> list[str]:
        out = []
        for s in stmts:
            out += self.stmt(s, lvl, declared, in_func)
        return out

    def scope_block(self, stmts, lvl, declared: set, in_func=False) -> list[str]:
        """A function body / the program top level. In JS, a variable whose first assignment is nested inside a
        block (if/for) is declared with `let x;` right before the top-level statement containing it, so it is
        scoped to the whole function/program rather than to the inner block."""
        out = []
        for s in stmts:
            if self.style == "js":
                direct = {s[1]} if s[0] in ("assign", "aug") else set()
                nested = sorted(assigned_names([s]) - direct - declared)
                if nested:
                    out.append(f"{self.ind * lvl}let {', '.join(nested)};")
                    declared |= set(nested)
            out += self.stmt(s, lvl, declared, in_func)
        return out

    def stmt(self, s, lvl, declared, in_func) -> list[str]:
        I = self.ind * lvl
        S = self.style
        t = s[0]
        semi = ";" if S == "js" else ""
        if t == "assign":
            name, e = s[1], s[2]
            if S == "pseudo":
                return [f"{I}{name} <- {self.expr(e)}"]
            if S == "js" and name not in declared:
                declared.add(name)
                if lvl > (1 if in_func else 0):  # never block-scope a variable (scope_block pre-declares these)
                    raise ValueError(f"undeclared nested assignment to {name!r}")
                return [f"{I}let {name} = {self.expr(e)};"]
            return [f"{I}{name} = {self.expr(e)}{semi}"]
        if t == "aug":
            name, op, e = s[1], s[2], s[3]
            if S == "pseudo":
                return [f"{I}{name} <- {name} {op} {self.expr(e, _PREC[op] + 1)}"]
            return [f"{I}{name} {op}= {self.expr(e)}{semi}"]
        if t == "observe":
            if S == "pseudo":
                return [f"{I}OBSERVE {self.expr(s[1])}"]
            return [f"{I}observe({self.expr(s[1])}){semi}"]
        if t == "print":
            parts = ([f'"{s[1]}"'] if s[1] is not None else []) + [self.expr(x) for x in s[2]]
            if S == "python":
                return [f"{I}print({', '.join(parts)})"]
            if S == "js":
                return [f"{I}console.log({', '.join(parts)});"]
            return [f"{I}PRINT {', '.join(parts)}"]
        if t == "break":
            return [f"{I}{'BREAK' if S == 'pseudo' else 'break'}{semi}"]
        if t == "return":
            return [f"{I}{'RETURN' if S == 'pseudo' else 'return'} {self.expr(s[1])}{semi}"]
        if t == "for":
            _, var, n, body = s
            if S == "python":
                head = [f"{I}for {var} in range({n}):"]
                return head + self.block(body, lvl + 1, declared, in_func)
            if S == "js":
                return ([f"{I}for (let {var} = 0; {var} < {n}; {var}++) {{"]
                        + self.block(body, lvl + 1, declared, in_func) + [f"{I}}}"])
            return ([f"{I}FOR {var} FROM 0 TO {n - 1}:"] + self.block(body, lvl + 1, declared, in_func)
                    + [f"{I}END FOR"])
        if t == "if":
            _, arms, els = s
            out = []
            for i, (c, b) in enumerate(arms):
                cs = self.expr(c)
                if S == "python":
                    out.append(f"{I}{'if' if i == 0 else 'elif'} {cs}:")
                elif S == "js":
                    out.append(f"{I}if ({cs}) {{" if i == 0 else f"{I}}} else if ({cs}) {{")
                else:
                    out.append(f"{I}IF {cs} THEN" if i == 0 else f"{I}ELSE IF {cs} THEN")
                out += self.block(b, lvl + 1, declared, in_func)
            if els:
                out.append({"python": f"{I}else:", "js": f"{I}}} else {{", "pseudo": f"{I}ELSE"}[S])
                out += self.block(els, lvl + 1, declared, in_func)
            if S == "js":
                out.append(f"{I}}}")
            elif S == "pseudo":
                out.append(f"{I}END IF")
            return out
        raise ValueError(s)

    def func(self, f) -> list[str]:
        name, params, body = f
        S = self.style
        declared = set(params)
        if S == "python":
            return [f"def {name}({', '.join(params)}):"] + self.scope_block(body, 1, declared, True)
        if S == "js":
            return [f"function {name}({', '.join(params)}) {{"] + self.scope_block(body, 1, declared, True) + ["}"]
        return [f"FUNCTION {name}({', '.join(params)}):"] + self.scope_block(body, 1, declared, True) + ["END FUNCTION"]

    def program(self, prog: Program) -> str:
        lines = []
        for f in prog.funcs:
            lines += self.func(f)
            lines.append("")
        lines += self.scope_block(prog.body, 0, set())
        return "\n".join(lines)


def render(prog: Program, style="python", indent=4) -> str:
    return Renderer(style, indent).program(prog)


def render_expr(e, style="python") -> str:
    return Renderer(style).expr(e)


def render_stmt(s, style="python") -> str:
    return "\n".join(Renderer(style).stmt(s, 0, set(), False))


# ----------------------------------------------------------------------------------------------------------------------
# negation (for rendering the complementary observe alternative naturally)
# ----------------------------------------------------------------------------------------------------------------------

_NEG_CMP = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


def negate(e):
    t = e[0]
    if t == "bin" and e[1] in _NEG_CMP:
        return ("bin", _NEG_CMP[e[1]], e[2], e[3])
    if t == "not":
        return e[1]
    if t == "var":
        return ("not", e)
    if t == "bin" and e[1] == "and":
        return ("bin", "or", negate(e[2]), negate(e[3]))
    if t == "bin" and e[1] == "or":
        return ("bin", "and", negate(e[2]), negate(e[3]))
    return ("not", e)


# ----------------------------------------------------------------------------------------------------------------------
# sampling execution of the Python rendering (tests / Monte Carlo only)
# ----------------------------------------------------------------------------------------------------------------------

PY_PRELUDE = '''
class _Reject(Exception):
    pass

def flip(p):
    return _rng.random() < p

def randint(a, b):
    return _rng.randint(a, b)

def choice(xs, w=None):
    return _rng.choices(xs, weights=w)[0]

def observe(c):
    if not c:
        raise _Reject()

def print(*a):
    _out.append(" ".join(str(x) for x in a))
'''
