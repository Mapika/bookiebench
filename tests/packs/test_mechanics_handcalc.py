"""Hand-computed posteriors for every mechanics family, derived from the stated parameters and the observed evidence
with independent code (closed forms or brute force written from scratch; no World prior/lik/proj). Results are
compared with the stored exact marginals *by option label*, so a latent-to-label inversion (like the forensic
contamination bug) fails here even though the enumeration itself is internally consistent."""
from __future__ import annotations

import itertools
import math
import re

import numpy as np
import pytest

from bookiebench.sims.core import finalize_release, joint_array, marginal, var_index
from bookiebench.sims.packs.mechanics import FAMILIES, make_world
from bookiebench.sims.world import build_instance


def _case(fam, level, pred=lambda w, inst: True, start=0):
    for i in range(start, start + 400):
        world, rng, _ = make_world(fam, "dev", i, seed=13, level=level)
        inst, tr = build_instance(world, rng, family=fam, split="dev", idx=i, return_trace=True)
        if pred(world, inst):
            # check the release form: options are shuffled per variable, so every comparison below is by label
            return world, finalize_release(inst, seed=7), tr["obs"]
    raise RuntimeError("no suitable instance")


def _final(inst, name):
    for v in inst["variables"]:
        if v["name"] == name:
            return dict(zip(v["options"], marginal(joint_array(inst), var_index(inst, name))))
    raise KeyError(name)


def _check(inst, name, expected: dict, tol=1e-7):
    got = _final(inst, name)
    exp = {k: v for k, v in expected.items()}
    s = sum(exp.values())
    exp = {k: v / s for k, v in exp.items()}
    for lab in set(got) | set(exp):
        assert abs(got.get(lab, 0.0) - exp.get(lab, 0.0)) < tol, (inst["id"], name, lab, got.get(lab), exp.get(lab))


def _vname(inst, prefix):
    return next(v["name"] for v in inst["variables"] if v["name"].startswith(prefix))


# ---------------------------------------------------------------------------------------------------------------------

def test_blackjack():
    w, inst, obs = _case("blackjack", 0)
    cnt, hand = w.params["counts"], w.params["hand"]
    N, k = sum(cnt.values()), len(obs)
    nm = {1: "an ace", 2: "a two", 3: "a three", 4: "a four", 5: "a five", 6: "a six", 7: "a seven", 8: "an eight",
          9: "a nine", 10: "a ten-valued card"}
    # the hole card is exchangeable with every unseen card: P(v) = (c_v - #v revealed) / (N - k)
    p = {v: (c - obs.count(v)) / (N - k) for v, c in cnt.items()}
    _check(inst, "hole_card", {nm[v]: q for v, q in p.items()})

    def tot(cards):
        t = sum(cards)
        return t + 10 if 1 in cards and t + 10 <= 21 else t

    hit = {}
    for v, q in p.items():  # the player's next card has the same marginal
        t = tot(hand + [v])
        lab = "bust (over 21)" if t > 21 else ("17 to 21" if t >= 17 else "16 or less")
        hit[lab] = hit.get(lab, 0) + q
    _check(inst, "after_hit", hit)


def test_poker():
    w, inst, obs = _case("poker", 0)
    ts, U = w.params["ts"], w.params["U"]
    suit = ["hearts", "spades", "diamonds", "clubs"][ts]
    s1 = suit[:-1]
    rp = w.params["raise_p"]  # raise rates: pocket pair, >= 1 target-suit card, other
    burned = [c for kd, c in obs if kd == "burn"]
    exposed = [c for kd, c in obs if kd == "expose"]
    bets = [c for kd, c in obs if kd == "bet"]
    opp_suit, nxt = {}, {}
    for a, b in itertools.combinations(U, 2):  # brute force: opponent hand, then the next board card
        if a in burned or b in burned or (exposed and exposed[0] not in (a, b)):
            continue
        pr = 0.5 if exposed else 1.0
        cat = 0 if a // 4 == b // 4 else (1 if ts in (a % 4, b % 4) else 2)
        for bt in bets:
            pr *= rp[cat] / 100 if bt == 0 else 1 - rp[cat] / 100
        rest = [c for c in U if c not in (a, b) and c not in burned]
        k = (a % 4 == ts) + (b % 4 == ts)
        lab = [f"no {suit}", f"one {s1}", f"two {suit}"][k]
        opp_suit[lab] = opp_suit.get(lab, 0) + pr
        for c in rest:
            l2 = f"a {s1}" if c % 4 == ts else f"not a {s1}"
            nxt[l2] = nxt.get(l2, 0) + pr / len(rest)
    _check(inst, "opp_suit", opp_suit)
    _check(inst, "next_card", nxt)

def test_minesweeper():
    w, inst, obs = _case("minesweeper", 0)
    p = w.params
    H, W, M = p["H"], p["W"], p["M"]
    cells = [(r, c) for r in range(H) for c in range(W)]

    def nb(x):
        return [(r, c) for r in range(x[0] - 1, x[0] + 2) for c in range(x[1] - 1, x[1] + 2)
                if (r, c) != x and 0 <= r < H and 0 <= c < W]

    tally = {}
    for mines in itertools.combinations(cells, M):  # brute force over the whole board
        ms = set(mines)
        if any(x in ms or sum(n in ms for n in nb(x)) != p["num"][x] for x in p["opened"]):
            continue
        pr = 1.0
        for (kd, i), (_, v) in zip(p["sched"], obs):
            x = p["unk"][i]
            if kd == "reveal":
                pr *= (-1 if x in ms else sum(n in ms for n in nb(x))) == v
            else:  # detector: beep with prob sens if mine, fp otherwise
                pb = (p["sens"] if x in ms else p["fp"]) / 100
                pr *= pb if v else 1 - pb
        for i in p["var_cells"]:
            key = (i, "a mine" if p["unk"][i] in ms else "safe")
            tally[key] = tally.get(key, 0) + pr
    for i in p["var_cells"]:
        x = p["unk"][i]
        name = f"cell_{'ABCDEFGH'[x[1]]}{x[0] + 1}"
        _check(inst, name, {lab: tally.get((i, lab), 0) for lab in ("a mine", "safe")})

def test_montyhall():
    for start in (0, 50, 100):
        w, inst, obs = _case("montyhall", 0, start=start)
        p = w.params
        Nd, pick_, host, q, fa = p["Nd"], p["pick"], p["host"], p["lazy_q"] / 100, p["fa"] / 100
        post = {}
        for car in range(Nd):
            pr = p["w"][car]
            opened = []
            for o in obs:
                if o[0] == "friend":
                    pr *= fa if o[1] == car else (1 - fa) / (Nd - 1)
                    continue
                _, d, c = o
                closed = [x for x in range(Nd) if x != pick_ and x not in opened]
                if host == "ignorant":
                    pr *= ((0 if d == car else 2) == c) / len(closed)
                else:
                    elig = [x for x in closed if x != car]
                    if d not in elig:
                        pr = 0.0
                    elif host == "knows" or len(elig) == 1:
                        pr *= 1 / len(elig)
                    else:  # lazy: lowest-numbered goat door with probability q
                        pr *= q if d == min(elig) else (1 - q) / (len(elig) - 1)
                opened.append(d)
            post[car] = pr
        _check(inst, "car_door", {f"Door {j + 1}": v for j, v in post.items()})
        _check(inst, "stay_wins", {"yes": post[pick_], "no": sum(post.values()) - post[pick_]})

def test_birthday():
    w, inst, obs = _case("birthday", 0)
    p = w.params
    n, pw = p["n"], p["pw"]
    shared = {"yes": 0.0, "no": 0.0}
    for a in itertools.product(range(len(pw)), repeat=n):
        ok = True
        for (k, v), (kd, i, j) in zip(obs, p["ev"]):
            val = a[i] if kd == "reveal" else (int(a[i] == a[j]) if kd == "pair" else int(a[i] in j))
            ok &= val == v
        if ok:
            shared["yes" if len(set(a)) < n else "no"] += float(np.prod([pw[x] for x in a]))
    _check(inst, "any_shared", shared)


def test_matching():
    w, inst, obs = _case("matching", 0)
    p = w.params
    n, q, err = p["n"], p["q"] / 100, p["err"] / 100
    words = {0: "nobody", 1: "exactly one person", 3: "all three people"} if n == 3 else \
        {0: "nobody", 1: "exactly one person", 2: "exactly two people", 3: "three or more people"}
    perms = list(itertools.permutations(range(n)))
    ov = next(v["name"] for v in inst["variables"] if v["name"].endswith("_own") and v["name"] != "n_own")
    X = 0  # matching.make_world asks the own-item question about the first-named person (X, Y = 0, 1)
    cnt, own = {}, {"yes": 0.0, "no": 0.0}
    for s in perms:
        # prior: everything returned correctly with probability q, else a uniform matching
        pr = (1 - q) / len(perms) + (q if all(s[i] == i for i in range(n)) else 0.0)
        for (_, v), (kd, i, j) in zip(obs, p["ev"]):
            if kd == "own":  # noisy own-item report
                truth = s[i] == i
                pr *= (1 - err) if bool(v) == truth else err
            elif kd == "got":  # names the owner; a wrong answer is uniform over the other n-1 people
                pr *= (1 - err) if s[i] == v else err / (n - 1)
            else:  # noisy yes/no "do you hold j's item?"
                pr *= (1 - err) if (s[i] == j) == bool(v) else err
        f = min(sum(s[i] == i for i in range(n)), 3)
        cnt[words[f]] = cnt.get(words[f], 0) + pr
        own["yes" if s[X] == X else "no"] += pr
    _check(inst, "n_own", cnt)
    _check(inst, ov, own)

def test_reliability():
    w, inst, obs = _case("reliability", 0, lambda w, i: not w.params["has_batch"])
    p = w.params
    n = p["n"]
    res = {"working": 0.0, "failed": 0.0}
    for up in itertools.product((0, 1), repeat=n):
        pr = float(np.prod([(1 - p["fail"][c] / 100) if up[c] else p["fail"][c] / 100 for c in range(n)]))
        for (_, v), (kd, x) in zip(obs, p["ev"]):
            if kd == "test":
                pf = p["sens"] / 100 if not up[x] else p["fa"] / 100
                pr *= pf if v else 1 - pf
            else:
                b, comps = p["blocks"][x]
                pr *= (sum(up[c] for c in comps) >= (2 if b == "2of3" else 1)) == bool(v)
        ok = all(sum(up[c] for c in comps) >= (2 if b == "2of3" else 1) for b, comps in p["blocks"])
        res["working" if ok else "failed"] += pr
    _check(inst, "system", res)


def test_tournament():
    w, inst, obs = _case("tournament", 0)
    R = [int(w.params["rating"][t, 0]) for t in range(4)]
    teams = list(w.variables[0].options)  # canonical (unshuffled) order from the world
    known = {}
    for (_, win), e in zip(obs, w.params["ev"]):
        known[e[1]] = win
    pw = lambda i, j: R[i] / (R[i] + R[j])  # noqa: E731
    champ = {t: 0.0 for t in teams}
    for w1 in (0, 1):
        for w2 in (2, 3):
            p1 = (w1 == known[0]) if 0 in known else pw(w1, 1 - w1)
            p2 = (w2 == known[1]) if 1 in known else pw(w2, 5 - w2)
            champ[teams[w1]] += p1 * p2 * pw(w1, w2)
            champ[teams[w2]] += p1 * p2 * pw(w2, w1)
    _check(inst, "champion", champ)


def test_channel():
    w, inst, obs = _case("channel", 0)
    p = w.params
    f = p["flips"][0] / 100
    post = {}
    for b in (0, 1):
        pr = p["pm"][b]
        for (_, r) in obs:
            pr *= (1 - f) if r == b else f
        post[str(b)] = pr
    _check(inst, "message", post)
    s = sum(post.values())
    p1 = sum(q / s * ((1 - f) if int(b) == 1 else f) for b, q in post.items())
    _check(inst, "retransmission", {"1": p1, "0": 1 - p1})


def test_search():
    w, inst, obs = _case("search", 0)
    p = w.params
    names = list(w.variables[0].options)  # canonical order from the world; the math below is by index
    fa = p["fa"] / 100
    post = {}
    for loc in range(len(p["w"])):
        pr = p["w"][loc]
        for (k, r), c in zip(obs, p["plan"]):
            pc = p["dg"][c] / 100 if c == loc else fa
            pr *= pc if r == 0 else 1 - pc
        post[loc] = pr
    _check(inst, "location", {names[i]: q for i, q in post.items()})
    s = sum(post.values())
    pc = sum(q / s * (p["dg"][p["nxt"]] / 100 if i == p["nxt"] else fa) for i, q in post.items())
    _check(inst, "next_search", {"contact": pc, "nothing": 1 - pc})


def test_epidemic():
    w, inst, obs = _case("epidemic", 0)
    p = w.params
    n, adj, days, beta = p["n"], p["adj"], p["days"], p["beta"] / 100
    pairs = [(i, j) for i in range(n) for j in adj[i]]  # directed contacts
    names = list(w.variables[0].options)  # canonical (unshuffled) order from the world
    src_post, status = {nm: 0.0 for nm in names}, {"never infected": 0.0, "infected": 0.0}
    se, sp = p["ab"][0] / 100, p["ab"][1] / 100
    for src in range(n):
        for flips in itertools.product((0, 1), repeat=len(pairs) * days):  # every transmission coin, every day
            inf = {src}
            pr = 1.0 / n
            for d in range(days):
                new = set(inf)
                for e, (i, j) in enumerate(pairs):
                    x = flips[d * len(pairs) + e]
                    pr *= beta if x else 1 - beta
                    if x and i in inf and j not in inf:
                        new.add(j)
                inf = new
            for (_, r), (kind, i) in zip(obs, p["tests"]):
                pos = se if i in inf else 1 - sp
                pr *= pos if r else 1 - pos
            src_post[names[src]] += pr
            status["infected" if p["target"] in inf else "never infected"] += pr
    _check(inst, "patient_zero", src_post)
    _check(inst, _vname(inst, "status_"), status)


def test_gauge():
    w, inst, obs = _case("gauge", 0)
    p = w.params
    post = {}
    for g, wt in zip(p["grid"], p["pw"]):
        pr = wt
        for (_, r) in obs:
            e = r - g
            pr *= p["eA"][e + p["KA"]] if abs(e) <= p["KA"] else 0
        post[g] = pr
    got = _final(inst, "true_value")
    s = sum(post.values())
    for lab, q in got.items():
        g = int(re.match(r"(\d+)", lab).group(1))
        assert abs(q - post[g] / s) < 1e-7, (lab, q, post[g] / s)


def test_queue():
    w, inst, obs = _case("queue", 0)
    p = w.params
    c = p["c"]

    def pmf(lam, k):
        if k < c:
            return math.exp(-lam) * lam ** k / math.factorial(k)
        return 1 - sum(math.exp(-lam) * lam ** j / math.factorial(j) for j in range(c))

    post = {nm: rp * np.prod([pmf(lam, k) for k in obs]) for nm, rp, lam in zip(["quiet", "busy"], p["rp"], p["lams"])}
    _check(inst, "regime", post)
    s = sum(post.values())
    nxt = {(str(k) if k < c else f"{c} or more"): sum(post[nm] / s * pmf(lam, k) for nm, lam in zip(["quiet", "busy"], p["lams"]))
           for k in range(c + 1)}
    _check(inst, "next_count", nxt)


def test_inventory():
    w, inst, obs = _case("inventory", 0)
    p = w.params
    post = {}
    for nm, lp, tab in zip(["slow", "strong"], p["lp"], p["tabs"]):
        pr = lp
        for (k, sold) in obs:
            S = p["stock"][k]
            pr *= tab[sold] / 100 if sold < S else sum(tab[S:]) / 100  # a sell-out hides the demand
        post[nm] = pr
    _check(inst, "demand_level", post)


def test_forensic_contamination():
    w, inst, obs = _case("forensic", 2, lambda w, i: w.params["use_dna"])
    p = w.params
    sus = w.variables[0].options[0].replace(" did it", "")
    g_post, c_post = {f"{sus} did it": 0.0, f"{sus} did not do it": 0.0}, {"clean": 0.0, "contaminated": 0.0}
    for guilty in (True, False):
        for cont in (False, True):
            pr = (p["pG"] if guilty else 1 - p["pG"]) * (p["pcont"] / 100 if cont else 1 - p["pcont"] / 100)
            for (_, r), (kind, i) in zip(obs, p["ev"]):
                if kind == "dna":
                    pm = p["pcm"] / 100 if cont else (p["dna"] / 100 if guilty else p["rmp"] / 100)
                else:
                    pm = p["specs"][i]["pgv"] / 100 if guilty else p["specs"][i]["piv"] / 100
                pr *= pm if r == 0 else 1 - pm
            g_post[f"{sus} did it" if guilty else f"{sus} did not do it"] += pr
            c_post["contaminated" if cont else "clean"] += pr
    _check(inst, "culprit", g_post)
    _check(inst, "contamination", c_post)


def test_forensic_contamination_prior_direction():
    """With no DNA result observed yet the contamination marginal equals the stated prior P(contaminated)."""
    for i in range(200):
        world, rng, _ = make_world("forensic", "dev", i, seed=21, level=2)
        if not world.params["use_dna"]:
            continue
        inst = build_instance(world, rng, family="forensic", split="dev", idx=i)
        k = next(j for j, (kind, _) in enumerate(world.params["ev"]) if kind == "dna")
        if k == 0:
            continue
        v = inst["variables"]
        ax = var_index(inst, "contamination")
        m = dict(zip(v[ax]["options"], marginal(joint_array(inst, k - 1), ax)))
        assert abs(m["contaminated"] - world.params["pcont"] / 100) < 1e-9
        return
    pytest.skip("no instance with a late DNA result")


def test_raters():
    w, inst, obs = _case("raters", 1, lambda w, i: w.params["anon"])
    p = w.params
    K = p["K"]
    classes = list(w.variables[0].options)  # canonical (unshuffled) order from the world
    truth, wtype = {c: 0.0 for c in classes}, {"diligent": 0.0, "spammer": 0.0}
    for c in range(K):
        for spam in (0, 1):
            pr = p["prev"][c] / 100 * (p["p_spam"] / 100 if spam else 1 - p["p_spam"] / 100)
            for (_, lab), (kind, i) in zip(obs, p["ev"]):
                if kind == "rater":
                    pr *= p["conf"][i][c][lab] / 100
                else:
                    t = c if kind == "anon" else p["ctrl"][i]
                    pr *= (p["spam"][lab] if spam else p["dil"][t][lab]) / 100
            truth[classes[c]] += pr
            wtype["spammer" if spam else "diligent"] += pr
    _check(inst, "true_label", truth)
    if "worker_type" in [x["name"] for x in inst["variables"]]:
        _check(inst, "worker_type", wtype)


def test_recapture():
    w, inst, obs = _case("recapture", 0)
    p = w.params
    m = p["m"]
    post = {}
    for N, wt in zip(p["grid"], p["pw"]):
        pr, mk = float(wt), 0
        for k, x in enumerate(obs):  # sequential draws without replacement
            pm = (m - mk) / (N - k)
            pr *= pm if x else 1 - pm
            mk += x
        post[N] = pr
    s = sum(post.values())
    for lab, q in _final(inst, "population").items():
        N = int(lab.split()[0])
        assert abs(q - post[N] / s) < 1e-7
    mk, k = sum(obs), len(obs)
    pn = sum(post[N] / s * (m - mk) / (N - k) for N in post)
    _check(inst, "next_marked", {"marked": pn, "unmarked": 1 - pn})


def test_every_family_has_a_handcalc():
    names = {n[5:] for n in globals() if n.startswith("test_")}
    for fam in FAMILIES:
        assert any(n == fam or n.startswith(fam + "_") for n in names), fam
