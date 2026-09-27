"""Domain-flavoured probabilistic programs: retry logic, A/B routing, sampling load balancers, loot drops, caches,
alert debouncing, canary rollouts, board-game moves, queue workers.

Each template draws its numbers, names and optional structural pieces at random, then goes through the same
generic pipeline as the abstract programs (exact enumeration, query-variable bucketing, observe/print evidence).
"""
from __future__ import annotations

import numpy as np

from bookiebench.sims.common import pick

from .build import Reject, STYLES, level_of, make_program_world
from .lang import Program


def V(n):
    return ("var", n)


def L(v):
    return ("lit", v)


def B(op, a, b):
    return ("bin", op, a, b)


def IF(c, body, els=None, elifs=()):
    arms = ((c, tuple(body)),) + tuple((ec, tuple(eb)) for ec, eb in elifs)
    return ("if", arms, tuple(els) if els else None)


def A(n, e):
    return ("assign", n, e)


def INC(n, e=1):
    return ("aug", n, "+", L(e) if isinstance(e, int) else e)


def FLIP(p):
    return ("flip", int(p))


def CH(vals, w=None):
    return ("choice", tuple(vals), tuple(int(x) for x in w) if w is not None else None)


def pct(rng, lo=10, hi=90, step=5):
    return int(pick(rng, list(range(lo, hi + 1, step))))


def weights(rng, k, lo=1, hi=9):
    return [int(x) for x in rng.integers(lo, hi + 1, size=k)]


def nm(rng, *opts):
    return pick(rng, list(opts))


# ----------------------------------------------------------------------------------------------------------------------
# templates: each returns (Program, theme, framing, prefer)
# ----------------------------------------------------------------------------------------------------------------------

def t_retry(rng):
    fn = nm(rng, "call_backend", "send_request", "try_upload", "fetch_quote")
    ok = nm(rng, "succeeded", "ok", "delivered")
    att = nm(rng, "attempts", "tries", "calls_made")
    maxr = int(rng.integers(2, 5))
    funcs, body = [], []
    variant = int(rng.integers(0, 3))
    if variant == 0:
        funcs.append((fn, (), (("return", FLIP(pct(rng, 30, 80))),)))
        call = ("call", fn, ())
    elif variant == 1:
        lat = sorted(int(x) for x in rng.choice([30, 60, 120, 250, 400, 800, 1500], size=3, replace=False))
        timeout = int(pick(rng, [x for x in [100, 200, 300, 500, 1000] if lat[0] < x <= lat[-1]] or [lat[1] + 1]))
        funcs.append((fn, (), (A("latency_ms", CH(lat, weights(rng, 3))), ("return", B("<", V("latency_ms"), L(timeout))))))
        call = ("call", fn, ())
    else:
        reg = nm(rng, "region", "zone")
        regs = pick(rng, [("eu", "us"), ("east", "west"), ("primary", "backup")])
        body.append(A(reg, CH(regs, weights(rng, 2))))
        funcs.append((fn, ("where",), (IF(B("==", V("where"), L(regs[0])), [("return", FLIP(pct(rng, 20, 60)))]),
                                       ("return", FLIP(pct(rng, 50, 90))))))
        call = ("call", fn, (V(reg),))
    body += [A(att, L(0)), A(ok, L(False))]
    loop = [INC(att), IF(call, [A(ok, L(True)), ("break",)])]
    wait = None
    if rng.random() < 0.5:
        wait = nm(rng, "backoff_ms", "waited_ms")
        body.append(A(wait, L(0)))
        step = int(pick(rng, [50, 100, 200]))
        loop.append(INC(wait, B("*", L(step), B("+", V("attempt"), L(1)))) if rng.random() < 0.5 else INC(wait, step))
    body.append(("for", "attempt", maxr, tuple(loop)))
    prefer = [ok, att]
    if rng.random() < 0.6:
        src = nm(rng, "served_from", "response_source")
        body.append(IF(("not", V(ok)), [A(src, CH(["cache", "stale", "error"], weights(rng, 3)))],
                       els=[A(src, L("backend"))]))
        body.insert(0, A(src, L("backend")))
        prefer.append(src)
    if rng.random() < 0.35:
        body.append(("observe", B(">", V(att), L(1))))
    if wait:
        prefer.append(wait)
    framing = pick(rng, [
        "Below is the retry wrapper a client uses when calling a flaky upstream service.",
        "A service calls an unreliable API with the following retry policy.",
        "This snippet models how a mobile app retries a failing request.",
    ])
    return Program(tuple(funcs), tuple(body)), "retry", framing, prefer


def t_ab(rng):
    arm = nm(rng, "variant", "arm", "bucket_name")
    body = []
    if rng.random() < 0.5:
        body.append(A("user_bucket", ("randint", 0, 9)))
        k = int(rng.integers(2, 8))
        body.append(IF(B("<", V("user_bucket"), L(k)), [A(arm, L("B"))], els=[A(arm, L("A"))]))
    else:
        body.append(A(arm, CH(["A", "B"], weights(rng, 2))))
    dev = nm(rng, "is_mobile", "on_mobile", "from_app")
    body.append(A(dev, FLIP(pct(rng, 30, 70))))
    clk = nm(rng, "clicked", "engaged", "opened")
    body.append(IF(B("==", V(arm), L("A")), [A(clk, FLIP(pct(rng, 10, 50)))],
                   elifs=[(V(dev), [A(clk, FLIP(pct(rng, 20, 70)))])], els=[A(clk, FLIP(pct(rng, 10, 60)))]))
    conv = nm(rng, "converted", "purchased", "signed_up")
    body.append(A(conv, L(False)))
    body.append(IF(V(clk), [A(conv, FLIP(pct(rng, 20, 70)))]))
    prefer = [arm, clk, conv]
    if rng.random() < 0.6:
        rev = nm(rng, "revenue", "order_value")
        body.append(A(rev, L(0)))
        body.append(IF(V(conv), [A(rev, CH(sorted(int(x) for x in rng.choice([5, 10, 20, 25, 40, 50, 99], 3, replace=False)),
                                           weights(rng, 3)))]))
        prefer.append(rev)
    if rng.random() < 0.3:
        body.append(("observe", B("or", V(dev), V(clk))))
    framing = pick(rng, [
        "An A/B test routes each visitor with the code below.",
        "The following snippet simulates one visitor in a two-arm experiment.",
        "Traffic splitting and conversion for a single user are modelled as follows.",
    ])
    return Program((), tuple(body)), "ab_routing", framing, prefer


def t_lb(rng):
    nreq = int(rng.integers(2, 4))
    body = []
    variant = int(rng.integers(0, 2))
    if variant == 0:
        servers = pick(rng, [("a", "b", "c"), ("s1", "s2", "s3"), ("east", "west", "north")])
        names = [f"load_{s}" for s in servers]
        for n in names:
            body.append(A(n, ("randint", 0, int(rng.integers(1, 3)))))
        w = weights(rng, 3)
        loop = [A("target", CH(servers, w)),
                IF(B("==", V("target"), L(servers[0])), [INC(names[0])],
                   elifs=[(B("==", V("target"), L(servers[1])), [INC(names[1])])], els=[INC(names[2])])]
        body.append(A("target", L(servers[0])))
        body.append(("for", "req", nreq, tuple(loop)))
        cap = int(rng.integers(2, 5))
        body.append(A("overloaded", B(">=", ("call", "max", (V(names[0]), ("call", "max", (V(names[1]), V(names[2]))))),
                                      L(cap))))
        prefer = ["overloaded", "target"] + names
    else:
        # sampling-based: two backends, noisy load probes, route to the one that looks less busy
        body += [A("load_a", ("randint", 0, 2)), A("load_b", ("randint", 0, 2))]
        loop = [A("probe_a", B("+", V("load_a"), ("randint", 0, 1))),
                A("probe_b", B("+", V("load_b"), ("randint", 0, 1))),
                IF(B("<=", V("probe_a"), V("probe_b")), [INC("load_a")], els=[INC("load_b")])]
        body += [A("probe_a", L(0)), A("probe_b", L(0))]
        body.append(("for", "req", nreq, tuple(loop)))
        body.append(A("imbalance", ("call", "abs", (B("-", V("load_a"), V("load_b")),))))
        prefer = ["load_a", "load_b", "imbalance"]
    if variant == 0 and rng.random() < 0.25:
        body.append(("observe", ("not", V("overloaded"))))
    elif variant == 1 and rng.random() < 0.25:
        body.append(("observe", B("<=", V("imbalance"), L(1))))
    framing = pick(rng, [
        "A sampling-based load balancer assigns incoming requests like this.",
        "The following code routes a burst of requests across backends.",
        "Consider this toy model of a randomized load balancer.",
    ])
    return Program((), tuple(body)), "load_balancer", framing, prefer


def t_loot(rng):
    n = int(rng.integers(2, 4))
    fn = nm(rng, "roll_rarity", "open_chest", "drop_table")
    rar = pick(rng, [("common", "rare", "epic"), ("common", "rare", "legendary"), ("junk", "gear", "relic")])
    funcs = [(fn, (), (("return", CH(rar, [int(pct(rng, 50, 80)), int(pct(rng, 10, 30)), int(pick(rng, [3, 5, 8, 10]))])),))]
    body = [A("gold", L(0)), A("best", L(rar[0])), A("drop", L(rar[0]))]
    top = nm(rng, "epics", "big_drops")
    body.append(A(top, L(0)))
    loop = [A("drop", ("call", fn, ())),
            IF(B("==", V("drop"), L(rar[2])), [INC(top), A("best", L(rar[2])), INC("gold", int(pick(rng, [30, 50])))],
               elifs=[(B("==", V("drop"), L(rar[1])), [INC("gold", int(pick(rng, [10, 20]))),
                                                     IF(B("!=", V("best"), L(rar[2])), [A("best", L(rar[1]))])])],
               els=[INC("gold", ("randint", 1, 2))])]
    body.append(("for", "chest", n, tuple(loop)))
    prefer = ["best", top, "gold"]
    if rng.random() < 0.5:
        body.append(IF(B("==", V(top), L(0)), [IF(FLIP(pct(rng, 10, 40)), [INC(top), A("best", L(rar[2]))])]))
    if rng.random() < 0.25:
        body.append(("observe", B("!=", V("best"), L(rar[0]))))
    framing = pick(rng, [
        "A game opens several loot chests with the following drop logic.",
        "Here is the loot-drop code of a dungeon crawler (with a pity rule in some versions).",
        "The server-side loot roll for one player looks like this.",
    ])
    return Program(tuple(funcs), tuple(body)), "loot", framing, prefer


def t_cache(rng):
    body = [A("hit", FLIP(pct(rng, 40, 90)))]
    fast = sorted(int(x) for x in rng.choice([1, 2, 3, 5], 2, replace=False))
    slow = sorted(int(x) for x in rng.choice([20, 40, 80, 150, 300], 3, replace=False))
    body.append(IF(V("hit"), [A("latency_ms", CH(fast, weights(rng, 2)))], els=[A("latency_ms", CH(slow, weights(rng, 3)))]))
    body.insert(0, A("latency_ms", L(0)))
    body.append(A("stale", L(False)))
    body.append(IF(V("hit"), [A("stale", FLIP(pct(rng, 5, 30)))]))
    prefer = ["hit", "latency_ms", "stale"]
    if rng.random() < 0.6:
        body.append(A("refetched", L(False)))
        body.append(IF(V("stale"), [A("refetched", FLIP(pct(rng, 30, 80))),
                                    IF(V("refetched"), [INC("latency_ms", int(pick(rng, [20, 40, 80])))])]))
        prefer.append("refetched")
    slo = int(pick(rng, [s for s in [10, 25, 50, 100, 200] if fast[-1] < s <= slow[-1]] or [slow[0]]))
    body.append(A("slo_met", B("<=", V("latency_ms"), L(slo))))
    prefer.insert(0, "slo_met")
    framing = pick(rng, [
        "A read-through cache serves one request as follows.",
        "This code models the latency of a single cached lookup.",
        "Consider the request path of a caching proxy, written as a probabilistic program.",
    ])
    return Program((), tuple(body)), "cache", framing, prefer


def t_alert(rng):
    n = int(rng.integers(3, 6))
    k = int(rng.integers(2, 4))
    p = pct(rng, 20, 70)
    hi = nm(rng, "high", "breach", "over_limit")
    body = [A("streak", L(0)), A("alerted", L(False)), A(hi, L(False)), A("breaches", L(0))]
    loop = [A(hi, FLIP(p)),
            IF(V(hi), [INC("streak"), INC("breaches")], els=[A("streak", L(0))]),
            IF(B(">=", V("streak"), L(k)), [A("alerted", L(True))])]
    if rng.random() < 0.3:
        loop.append(IF(V("alerted"), [("break",)]))
    body.append(("for", "minute", n, tuple(loop)))
    prefer = ["alerted", "breaches", "streak"]
    if rng.random() < 0.3:
        body.append(("observe", B(">=", V("breaches"), L(1))))
    framing = pick(rng, [
        f"A monitoring rule pages the on-call engineer after {k} consecutive bad readings.",
        "The alert debouncer below checks a noisy metric once per minute.",
        "This code decides whether a flapping health check triggers an alert.",
    ])
    return Program((), tuple(body)), "alerting", framing, prefer


def t_canary(rng):
    body = [A("bad_build", FLIP(pct(rng, 10, 50)))]
    body.append(IF(V("bad_build"), [A("error_rate", CH(["low", "elevated", "high"], weights(rng, 3)))],
                   els=[A("error_rate", CH(["low", "elevated", "high"], [int(pct(rng, 60, 90)), int(pct(rng, 5, 30)), int(pick(rng, [1, 2, 5]))]))]))
    body.insert(0, A("error_rate", L("low")))
    body.append(A("checks_failed", L(0)))
    n = int(rng.integers(2, 4))
    loop = [IF(B("!=", V("error_rate"), L("low")), [IF(FLIP(pct(rng, 40, 90)), [INC("checks_failed")])],
               els=[IF(FLIP(pct(rng, 5, 20)), [INC("checks_failed")])])]
    body.append(("for", "check", n, tuple(loop)))
    body.append(A("rolled_back", B(">=", V("checks_failed"), L(int(rng.integers(1, n + 1))))))
    prefer = ["bad_build", "rolled_back", "error_rate", "checks_failed"]
    framing = pick(rng, [
        "A canary deployment runs automated health checks before promoting a build.",
        "The rollout controller below decides whether to roll back a new release.",
        "Consider this simplified canary analysis for one deploy.",
    ])
    return Program((), tuple(body)), "canary", framing, prefer


def t_board(rng):
    n = int(rng.integers(2, 4))
    size = int(rng.integers(5, 9))
    snake = int(rng.integers(2, size))
    body = [A("pos", L(0)), A("bumped", L(False))]
    loop = [INC("pos", ("randint", 1, int(rng.integers(2, 4)))),
            IF(B("==", V("pos"), L(snake)), [A("pos", L(max(0, snake - int(rng.integers(1, 3))))), A("bumped", L(True))]),
            A("pos", ("call", "min", (V("pos"), L(size))))]
    body.append(("for", "turn", n, tuple(loop)))
    body.append(A("won", B(">=", V("pos"), L(size - int(rng.integers(0, 3))))))
    prefer = ["pos", "won", "bumped"]
    framing = pick(rng, [
        "A board-game token moves according to the code below.",
        f"In a small race game, a trap sits on square {snake}.",
        "This program plays a few turns of a dice-driven board game.",
    ])
    return Program((), tuple(body)), "board_game", framing, prefer


def t_queue(rng):
    n = int(rng.integers(2, 5))
    body = [A("queue", ("randint", 0, 2)), A("dropped", L(0))]
    cap = int(rng.integers(2, 5))
    loop = [IF(FLIP(pct(rng, 30, 80)), [IF(B("<", V("queue"), L(cap)), [INC("queue")], els=[INC("dropped")])]),
            IF(B("and", B(">", V("queue"), L(0)), FLIP(pct(rng, 30, 80))), [("aug", "queue", "-", L(1))])]
    body.append(("for", "tick", n, tuple(loop)))
    body.append(A("backlogged", B(">=", V("queue"), L(max(1, cap - 1)))))
    prefer = ["queue", "dropped", "backlogged"]
    framing = pick(rng, [
        "A worker drains a bounded job queue; arrivals and completions are random.",
        "The following code simulates a few ticks of a bounded queue.",
        "Consider this model of a message broker with a capacity limit.",
    ])
    return Program((), tuple(body)), "queue", framing, prefer


TEMPLATES = [t_retry, t_ab, t_lb, t_loot, t_cache, t_alert, t_canary, t_board, t_queue]


def make_world(rng):
    mode = "observe" if rng.random() < 0.55 else "print"
    for attempt in range(300):
        tpl = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
        prog, theme, framing, prefer = tpl(rng)
        style = pick(rng, STYLES)
        try:
            w = make_program_world(rng, prog, style=style, mode=mode, framing=framing, theme=theme,
                                   family="prog_domain", prefer=prefer)
        except Reject:
            continue
        w.meta["level"] = level_of(w.params["program"])
        return w
    raise RuntimeError("could not generate a domain program")
