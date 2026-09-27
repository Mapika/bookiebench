"""Minesweeper local boards: exact counting over consistent mine configurations.

The prelude shows a partly opened board (numbers on opened cells, total mine count) and states that every
arrangement of the mines consistent with what is shown is equally likely. Latent = the set of consistent
configurations of the unopened cells (uniform). Evidence: a referee reveals the content of chosen unopened cells
(a mine, or the number it would show) -- deterministic given the configuration, hence exchangeable.
Variables: mine/safe status of 1-3 specific cells and (L2) the number of mines in a named group of cells.
"""
from __future__ import annotations

import itertools

import numpy as np

from ._base import Var, cap, is_json, join_list, json_block, json_line, mk_world, n_steps, per_step, pick

NAME = "minesweeper"
COLS = "ABCDEFGH"
MINE = -1


def cell_name(r, c):
    return f"{COLS[c]}{r + 1}"


def neighbors(r, c, H, W):
    return [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
            if (rr, cc) != (r, c) and 0 <= rr < H and 0 <= cc < W]


def _layout(rng, level, scale):
    if level == 0:
        H, W = pick(rng, [(3, 3), (3, 4), (4, 3)])
        M, umax = int(rng.integers(2, 4)), 7
    elif level == 1:
        H, W = pick(rng, [(4, 4), (4, 5), (5, 4)])
        M, umax = int(rng.integers(3, 5)), 11
    else:
        H, W = pick(rng, [(5, 5), (4, 6), (5, 6)])
        M, umax = int(rng.integers(4, 7)), 15
    if scale > 1:
        H, W, M, umax = min(8, H + scale - 1), min(8, W + scale - 1), M + 2 * (scale - 1), umax + 3 * (scale - 1)
    return H, W, M, umax


def make_world(rng, level, scale=1):
    nvar = min({0: 2, 1: int(rng.integers(2, 4)), 2: 2}[level] + (scale - 1), 4 if level < 2 else 3)
    T = n_steps(rng, level, 2, 5)
    for _ in range(500):
        H, W, M, umax = _layout(rng, level, scale)
        cells = [(r, c) for r in range(H) for c in range(W)]
        mines = {cells[int(i)] for i in rng.choice(len(cells), size=M, replace=False)}
        num = {(r, c): sum((n in mines) for n in neighbors(r, c, H, W)) for r, c in cells}
        safe = [x for x in cells if x not in mines]
        opened = set()

        def click(x0):  # open a safe cell, flood-filling through zeros as the real game does
            stack = [x0]
            while stack:
                x = stack.pop()
                if x in opened:
                    continue
                opened.add(x)
                if num[x] == 0:
                    stack.extend(n for n in neighbors(x[0], x[1], H, W) if n not in opened)

        click(safe[int(rng.integers(len(safe)))])
        while len(cells) - len(opened) > umax:
            cand = [x for x in cells if x not in mines and x not in opened]
            click(cand[int(rng.integers(len(cand)))])
        unk = [x for x in cells if x not in opened]
        U = len(unk)
        if U < M + 2:
            continue
        cfgs = np.array([[i in comb for i in range(U)] for comb in itertools.combinations(range(U), M)], bool)
        uidx = {x: i for i, x in enumerate(unk)}
        ok = np.ones(len(cfgs), bool)
        for x in opened:
            nb = [uidx[n] for n in neighbors(*x, H, W) if n in uidx]
            ok &= cfgs[:, nb].sum(1) == num[x]
        cfgs = cfgs[ok]
        if len(cfgs) < 4:
            continue
        pm = cfgs.mean(0)
        und = [i for i in range(U) if 0.02 < pm[i] < 0.98]
        need = nvar + T + (3 if level == 2 else 0)
        if len(und) < nvar + 1 or U < need:
            continue
        order = [int(i) for i in rng.permutation(und)]
        var_cells = order[:nvar]
        rest = [i for i in rng.permutation(U) if int(i) not in var_cells]
        rest = [int(i) for i in rest]
        grp = []
        if level == 2:
            grp = rest[:3]
            rest = rest[3:]
        # evidence: noisy detector scans of uncertain cells (the variable cells included), plus at most one exact
        # referee reveal on L2; scans keep the posterior moving instead of pinning cells at 0/1
        sched, used_reveal = [], False
        # scan only cells that bear on a queried cell: the variable cells and cells correlated with them
        Cf = cfgs.astype(float)
        sd = Cf.std(0) + 1e-12
        corr = np.abs(((Cf - Cf.mean(0)).T @ (Cf - Cf.mean(0)) / len(Cf)) / np.outer(sd, sd))
        scan_pool = [i for i in und if i not in grp and (i in var_cells or corr[i, var_cells].max() > 0.15)]
        for _k in range(T):
            if level == 2 and not used_reveal and rest and rng.random() < 0.3:
                sched.append(("reveal", rest.pop(0)))
                used_reveal = True
            else:
                sched.append(("scan", int(pick(rng, scan_pool))))
        break
    else:
        raise RuntimeError("could not build a minesweeper board")

    # cell values under each configuration: MINE or the number it shows
    def value(i):
        x = unk[i]
        nb = [uidx[n] for n in neighbors(*x, H, W) if n in uidx]
        return np.where(cfgs[:, i], MINE, cfgs[:, nb].sum(1))

    VAL = {i: value(i) for kd, i in sched if kd == "reveal"}
    MINEAT = {i: cfgs[:, i] for kd, i in sched if kd == "scan"}
    sens = int(rng.integers(60, 96))
    fp = int(rng.integers(3, 31))
    from ._base import Fmt, repeat_tags, tag_text
    fmt = Fmt(rng)
    stag = [f"scan {t}" if t else "" for t in repeat_tags(sched)]
    cols = [np.where(cfgs[:, i], 0, 1) for i in var_cells]
    variables = []
    for i in var_cells:
        nm = cell_name(*unk[i])
        variables.append(Var(f"cell_{nm}", ["a mine", "safe"],
                             [f"Is cell {nm} a mine?", f"Does {nm} hide a mine?", f"What is under cell {nm}?"],
                             [f"{nm} hides a mine", f"{nm} is safe"]))
    glo = 0
    if grp:
        cnt = cfgs[:, grp].sum(1)
        lo, hi = int(cnt.min()), int(cnt.max())
        if hi == lo:
            lo, hi = (lo - 1, hi) if lo > 0 else (lo, hi + 1)
        opts = list(range(lo, hi + 1))
        gname = join_list([cell_name(*unk[i]) for i in grp])
        words = {0: "no mines", 1: "exactly one mine", 2: "exactly two mines", 3: "three mines"}
        assert hi <= 3
        variables.append(Var("group_count", [words[o] for o in opts],
                             [f"How many mines are among {gname}?", f"How many of the cells {gname} hold mines?",
                              f"Counting the cells {gname}, how many mines are there?"],
                             [f"there {'is' if o == 1 else 'are'} {words[o]} among {gname}" for o in opts]))
        cols.append(cnt - lo)
        glo = lo
    proj = np.stack(cols, 1)
    prior = np.ones(len(cfgs))

    grid = []
    for r in range(H):
        grid.append("".join(str(num[(r, c)]) if (r, c) in opened else "?" for c in range(W)))
    js = is_json(rng)
    if js:
        prelude = json_block(rng, {
            "board": {f"row {r + 1}": grid[r] for r in range(H)}, "columns": COLS[:W],
            "cell_names": f"column letter then row number, e.g. {COLS[1]}{2} is column {COLS[1]}, row 2 (rows counted from the top)",
            "legend": "digit = opened cell showing how many of its up to 8 neighbours hold mines; ? = unopened",
            "total_mines": M,
            "assumption": "every arrangement of the mines consistent with the board is equally likely",
            "detector": {"P(beep | mine)": fmt.p(sens), "P(beep | no mine)": fmt.p(fp),
                         "repeats": "repeated scans of one cell (scan A, scan B, ...) are independent given its content"},
            "reveals": "a referee may reveal what is under an unopened cell (a mine, or the number it would show); "
                       "nothing explodes"})
        tpls = [lambda nm, v: json_line({"reveal": nm, "content": "mine" if v == MINE else v}),
                lambda nm, v: json_line({"cell": nm, "is": "mine" if v == MINE else f"safe, shows {v}"})]
        stpls = [lambda nm, b: json_line({"detector_scan": nm, "beep": bool(b)}),
                 lambda nm, b: json_line({"scan": nm, "result": "beep" if b else "silent"})]
    else:
        rows_txt = "\n".join(f"Row {r + 1}: " + " ".join(grid[r]) for r in range(H))
        head = pick(rng, [
            f"A Minesweeper board with {H} rows and {W} columns (columns {COLS[0]}-{COLS[W - 1]}, rows 1-{H}) "
            f"contains exactly {M} mines. Opened cells show how many of their neighbouring cells (including "
            f"diagonals) hold mines; '?' marks an unopened cell.",
            f"Here is a {H}x{W} Minesweeper position with {M} mines in total. Digits are opened cells counting the "
            f"mines among their up to eight neighbours, and '?' cells are still covered. Columns are lettered "
            f"{COLS[0]} to {COLS[W - 1]}, rows numbered from the top.",
            f"Puzzle: {M} mines are hidden in this {W}-column, {H}-row grid. A number on an opened square tells how "
            f"many mines touch it (sideways or diagonally); covered squares are shown as '?'.",
        ])
        tail = pick(rng, [
            "Assume every arrangement of the mines that is consistent with the board is equally likely.",
            "Treat all mine layouts that agree with what is shown as equally likely.",
            "All consistent placements of the mines are equally probable.",
        ])
        tail += (f" A mine detector can scan a covered cell: it beeps with probability {fmt.p(sens)} if the cell hides a "
                 f"mine and with probability {fmt.p(fp)} if it does not; repeated scans of the same cell (labelled "
                 f"scan A, scan B, ...) are independent given what is under it.")
        if any(kd == "reveal" for kd, _ in sched):
            tail += (" A referee may also reveal the contents of a covered cell (a mine, or the number the cell would "
                     "show); nothing explodes.")
        names_txt = (f"Cells are named by column letter and row number: {COLS[1]}2 is column {COLS[1]}, "
                     f"row 2 (rows counted from the top).")
        prelude = f"{head} {names_txt}\n{rows_txt}\n{tail}"
        tpls = [lambda nm, v: f"The referee reveals {nm}: " + ("a mine." if v == MINE else f"safe, showing {v}."),
                lambda nm, v: f"Cell {nm} is uncovered and " + ("holds a mine." if v == MINE else f"shows the number {v}."),
                lambda nm, v: f"{nm}: " + ("mine." if v == MINE else f"no mine, number {v}."),
                lambda nm, v: f"Inspecting {nm} shows " + ("a mine." if v == MINE else f"it is safe with a {v}.")]
        stpls = [lambda nm, b: f"The detector {'beeps' if b else 'stays silent'} over {nm}.",
                 lambda nm, b: f"Scan of {nm}: {'beep' if b else 'no beep'}.",
                 lambda nm, b: f"Scanning {nm}, the detector {'beeps' if b else 'does not beep'}."]
    ptpl, pstp = per_step(rng, T, tpls), per_step(rng, T, stpls)
    alts = [MINE] + list(range(9))

    def alternatives(k, past):
        return [(k, v) for v in alts] if sched[k][0] == "reveal" else [(k, 1), (k, 0)]

    def lik(o, past):
        k, v = o
        kd, i = sched[k]
        if kd == "reveal":
            return (VAL[i] == v).astype(float)
        pb = np.where(MINEAT[i], sens / 100, fp / 100)
        return pb if v else 1 - pb

    def render(k, o):
        kd, i = sched[k]
        nm = cell_name(*unk[i])
        return ptpl[k](nm, o[1]) if kd == "reveal" else tag_text(pstp[k](nm, o[1]), stag[k])

    return mk_world(variables, prior, proj, T, alternatives, lik, render, prelude, variables[0].name, True,
                    "json" if js else "prose", "minesweeper",
                    {"H": H, "W": W, "M": M, "unk": unk, "opened": sorted(opened), "num": num, "var_cells": var_cells,
                     "grp": grp, "sched": sched, "grp_lo": glo, "sens": sens, "fp": fp})


def simulate(world, rng):
    """Place M mines uniformly on the covered cells; None if the board's numbers are contradicted."""
    p = world.params
    H, W, unk = p["H"], p["W"], p["unk"]
    mines = {unk[int(i)] for i in rng.choice(len(unk), size=p["M"], replace=False)}
    for x in p["opened"]:
        if sum(n in mines for n in neighbors(*x, H, W)) != p["num"][x]:
            return None
    a = [0 if unk[i] in mines else 1 for i in p["var_cells"]]
    if p["grp"]:
        a.append(sum(unk[i] in mines for i in p["grp"]) - p["grp_lo"])
    obs = []
    for k, (kd, i) in enumerate(p["sched"]):
        if kd == "reveal":
            obs.append((k, MINE if unk[i] in mines else sum(n in mines for n in neighbors(*unk[i], H, W))))
        else:
            pb = (p["sens"] if unk[i] in mines else p["fp"]) / 100
            obs.append((k, int(rng.random() < pb)))
    return tuple(a), obs
