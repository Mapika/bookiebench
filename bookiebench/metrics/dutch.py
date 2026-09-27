"""Maximum Dutch-book profit against a model's prices for one instance (SPEC §3 `dutch`).

Each answered probability is a price for a bet on an event over the outcome space = joint cells:

- marginal vector entry k of variable V  -> indicator event {V = k}
- noul                                   -> event E (or its complement if `neg`)
- cond P(E | G) = p                      -> conditional bet paying 1_{E and G} - p * 1_G (called off when G is false)

The bookie picks stakes s_i in [-1, 1] (s_i > 0: bookie sells the bet, receiving p_i and paying the indicator; s_i < 0:
bookie buys). Profit in outcome w is  sum_i s_i * (p_i * G_i(w) - E_i(w) * G_i(w)).  We maximise the guaranteed
(minimum over outcomes) profit with an LP. By de Finetti the optimum is 0 iff the prices are coherent (extend to a
probability over the cells), and positive otherwise.

Tolerance (`dutch@delta`). Each price is read as the interval [p - delta, p + delta] clipped to [0, 1]: the model buys
at its lower price and sells at its upper price, so the bookie receives lo_i when selling and pays hi_i when buying.
With s = s+ - s- this stays an LP; the value is 0 iff some coherent probability lies within delta of every price.
delta = 0.005 absorbs 2-decimal rounding.

Missing prices (no answer, NaN, or a marginal vector of the wrong length) are **adversarial**: the bookie may pick any
price in [0, 1] for them. For a sell the best price is 1 and for a buy it is 0, so a missing bet is the interval
[lo, hi] = [1, 0]. Because lo > hi, holding both s+ and s- would be an arbitrage no single price allows, so each missing
bet gets a binary sign variable and the problem becomes a small MILP (scipy.optimize.milp, HiGHS). Justification: the
resulting value is >= the value for *every* possible answer to the missing queries, so omitting an answer can never
lower `dutch` -- it is weakly dominated by answering. A fixed penalty would need an arbitrary scale and could still be
cheaper than a bad answer.

Query selection: final-step queries only (prices elicited at other time points are not simultaneous), minus queries
tagged `rel == "paraphrase-dup"` and structurally identical duplicates of an earlier final-step query (e.g. the
final-step martingale marginal repeating the final marginal), which would otherwise double the stake budget.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .exact import event_mask, is_final, marginal_mask, shape_of

DELTAS = (0.005, 0.01)
MILP_TIME_LIMIT = 30.0  # seconds; real instances (<= ~30 bets) solve in < 0.2 s


@dataclass
class Bet:
    event: np.ndarray  # flat bool mask over cells: E (for cond: E, the G restriction is applied separately)
    given: np.ndarray  # flat bool mask: G (all True for unconditional bets)
    price: float  # NaN = missing (adversarial price)
    qid: str = ""

    @property
    def missing(self) -> bool:
        return not math.isfinite(self.price)


def _vi(inst: dict, name: str) -> int:
    for i, v in enumerate(inst["variables"]):
        if v["name"] == name:
            return i
    raise KeyError(name)


def _query_key(q: dict) -> str:
    return json.dumps({k: q.get(k) for k in ("kind", "var", "event", "given", "neg")}, sort_keys=True)


def dutch_queries(inst: dict, final_only: bool = True) -> list[dict]:
    """Queries that enter the book: final step, no paraphrase duplicates, no structural duplicates."""
    out, seen = [], set()
    for q in inst["queries"]:
        if final_only and not is_final(inst, q):
            continue
        if q.get("rel") == "paraphrase-dup":
            continue
        key = _query_key(q)
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def bets_for_instance(inst: dict, answers: dict, final_only: bool = True,
                      include_missing: bool = False) -> tuple[list[Bet], int]:
    """Translate queries into bets. With `include_missing` (what `dutch_book` uses), unanswered / NaN / wrong-length
    answers become missing bets (price NaN, adversarial); by default only answered bets are returned (the original
    API). Returns (bets, number of queries used)."""
    answers = answers or {}
    bets: list[Bet] = []
    nq = 0
    ncell = int(np.prod(shape_of(inst)))
    full = np.ones(ncell, dtype=bool)
    for q in dutch_queries(inst, final_only):
        a = answers.get(q["id"])
        try:
            a = None if a is None else np.atleast_1d(np.asarray(a, dtype=float)).ravel()
        except (TypeError, ValueError):
            a = None
        kind = q["kind"]
        qbets: list[Bet] = []
        if kind == "marginal":
            K = len(q.get("options") or inst["variables"][_vi(inst, q["var"])]["options"])
            if a is None or a.size != K:
                a = np.full(K, np.nan)
            for k, p in enumerate(a):
                qbets.append(Bet(marginal_mask(inst, q["var"], k).ravel(), full, float(p), f"{q['id']}[{k}]"))
        elif kind in ("noul", "cond"):
            p = float(a[0]) if a is not None and a.size >= 1 else float("nan")
            e = event_mask(inst, q["event"], bool(q.get("neg"))).ravel()
            g = event_mask(inst, q.get("given")).ravel() if kind == "cond" else full
            if not g.any():
                continue  # always called off: no bet
            qbets.append(Bet(e, g, p, q["id"]))
        else:
            continue
        if not include_missing:
            qbets = [b for b in qbets if not b.missing]
            if not qbets:
                continue
        bets.extend(qbets)
        nq += 1
    return bets, nq


def max_dutch_book(bets: list[Bet], stake_bound: float = 1.0, delta: float = 0.0) -> tuple[float, np.ndarray]:
    """Max guaranteed bookie profit (>= 0) and the optimal net stakes. `delta`: price tolerance (see module doc)."""
    n = len(bets)
    if n == 0:
        return 0.0, np.zeros(0)
    G = np.stack([b.given for b in bets]).astype(float)  # n x W
    EG = np.stack([b.event for b in bets]).astype(float) * G
    price = np.array([b.price for b in bets])
    miss = ~np.isfinite(price)
    lo = np.where(miss, 1.0, np.clip(price - delta, 0.0, 1.0))
    hi = np.where(miss, 0.0, np.clip(price + delta, 0.0, 1.0))
    a_sell = lo[:, None] * G - EG  # bookie profit per unit sold
    a_buy = EG - hi[:, None] * G  # bookie profit per unit bought
    W = G.shape[1]
    if not miss.any() and delta == 0.0:
        # plain LP over net stakes (the original formulation)
        c = np.zeros(n + 1)
        c[-1] = -1.0
        A = np.hstack([-a_sell.T, np.ones((W, 1))])
        res = linprog(c, A_ub=A, b_ub=np.zeros(W), bounds=[(-stake_bound, stake_bound)] * n + [(None, None)],
                      method="highs")
        if res.status != 0:
            raise RuntimeError(f"Dutch-book LP failed: {res.message}")
        return max(0.0, float(-res.fun)), res.x[:n]
    # split stakes: x = [s+ (n), s- (n), z (m binaries for missing bets), t]
    mi = np.flatnonzero(miss)
    m = mi.size
    nv = 2 * n + m + 1
    c = np.zeros(nv)
    c[-1] = -1.0
    rows, lb, ub = [], [], []
    # t - sum s+ a_sell - sum s- a_buy <= 0 for each outcome
    rows.append(np.hstack([-a_sell.T, -a_buy.T, np.zeros((W, m)), np.ones((W, 1))]))
    lb.append(np.full(W, -np.inf))
    ub.append(np.zeros(W))
    # s+ + s- <= bound
    rows.append(np.hstack([np.eye(n), np.eye(n), np.zeros((n, m + 1))]))
    lb.append(np.full(n, -np.inf))
    ub.append(np.full(n, stake_bound))
    if m:
        # missing bet j: s+ <= bound * z, s- <= bound * (1 - z)
        Ez = np.zeros((m, m))
        Ez[np.arange(m), np.arange(m)] = 1.0
        sp = np.zeros((m, n))
        sp[np.arange(m), mi] = 1.0
        rows.append(np.hstack([sp, np.zeros((m, n)), -stake_bound * Ez, np.zeros((m, 1))]))
        lb.append(np.full(m, -np.inf))
        ub.append(np.zeros(m))
        rows.append(np.hstack([np.zeros((m, n)), sp, stake_bound * Ez, np.zeros((m, 1))]))
        lb.append(np.full(m, -np.inf))
        ub.append(np.full(m, stake_bound))
    cons = LinearConstraint(np.vstack(rows), np.concatenate(lb), np.concatenate(ub))
    lower = np.concatenate([np.zeros(2 * n + m), [-np.inf]])
    upper = np.concatenate([np.full(2 * n, stake_bound), np.ones(m), [np.inf]])
    integrality = np.concatenate([np.zeros(2 * n), np.ones(m), [0]])
    res = milp(c, constraints=cons, bounds=Bounds(lower, upper), integrality=integrality,
               options={"mip_rel_gap": 1e-9, "time_limit": MILP_TIME_LIMIT})
    # on a time-out keep the incumbent: a valid (lower-bound) adversarial profit
    if res.x is None or res.status not in (0, 1):
        raise RuntimeError(f"Dutch-book MILP failed: {res.message}")
    return max(0.0, float(-res.fun)), res.x[:n] - res.x[n:2 * n]


def dutch_book(inst: dict, answers: dict, deltas=DELTAS) -> dict:
    """Per-instance Dutch-book result.

    dutch          profit with missing prices adversarial (headline)
    dutch_norm     dutch / #queries in the book;  dutch_per_bet  dutch / #bets
    dutch@<delta>  tolerance-aware profit (prices are +-delta intervals), missing still adversarial
    dutch_ans      profit over answered prices only (the pre-fix behaviour, a diagnostic that omission can game)
    """
    bets, nq = bets_for_instance(inst, answers, include_missing=True)
    profit, _ = max_dutch_book(bets)
    n_miss = sum(b.missing for b in bets)
    out = {"dutch": profit, "dutch_norm": profit / nq if nq else 0.0,
           "dutch_per_bet": profit / len(bets) if bets else 0.0,
           "n_queries": nq, "n_bets": len(bets), "n_missing_bets": n_miss}
    for d in deltas:
        out[f"dutch@{d}"] = max_dutch_book(bets, delta=d)[0] if profit > 0 else 0.0
    out["dutch_ans"] = max_dutch_book([b for b in bets if not b.missing])[0] if n_miss else profit
    return out
