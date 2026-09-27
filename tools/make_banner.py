"""Render the README banner (docs/assets/bookiebench.gif): BookieBench's story told at a racecourse bookmaker's pitch at dusk.

    python tools/make_banner.py [--out docs/assets/bookiebench.gif] [--png DIR] [--frames DIR]

Pillow + numpy only. Drawn in a 32-colour palette on a 240x135 canvas, scaled 4x (nearest-neighbour) to 960x540.
Frames store only the pixels that changed (transparent elsewhere), which keeps the GIF small.
"""
import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H, SCALE, MS = 240, 135, 4, 90

PALETTE = [
    ('black', (0, 0, 0)), ('ink', (20, 12, 28)), ('night', (38, 26, 56)), ('purple', (68, 36, 82)),
    ('plum', (120, 44, 88)), ('magenta', (184, 64, 98)), ('coral', (232, 106, 84)), ('orange', (246, 160, 72)),
    ('gold', (255, 214, 92)), ('cream', (255, 241, 214)), ('white', (255, 255, 255)), ('dgreen', (22, 70, 52)),
    ('green', (46, 122, 66)), ('lgreen', (112, 184, 72)), ('dbrown', (60, 38, 30)), ('brown', (110, 68, 44)),
    ('tan', (178, 120, 78)), ('skin', (240, 188, 150)), ('skinsh', (196, 136, 108)), ('dgray', (60, 62, 74)),
    ('gray', (110, 114, 128)), ('lgray', (170, 176, 188)), ('silver', (218, 224, 232)), ('red', (220, 40, 56)),
    ('dred', (140, 24, 40)), ('blue', (48, 132, 232)), ('dblue', (30, 72, 150)), ('lblue', (140, 200, 255)),
    ('teal', (40, 180, 160)), ('led', (255, 150, 40)), ('ledoff', (58, 30, 30)), ('lav', (150, 130, 190)),
]
C = {n: i for i, (n, _) in enumerate(PALETTE)}
TRANSP = len(PALETTE)
FLAT = [v for _, c in PALETTE for v in c] + [0, 0, 0] * (256 - len(PALETTE))

# ---------------------------------------------------------------- fonts
F3 = {
    'A': "010101111101101", 'B': "110101110101110", 'C': "011100100100011", 'D': "110101101101110",
    'E': "111100110100111", 'F': "111100110100100", 'G': "011100101101011", 'H': "101101111101101",
    'I': "111010010010111", 'J': "001001001101010", 'K': "101101110101101", 'L': "100100100100111",
    'M': "101111111101101", 'N': "110101101101101", 'O': "010101101101010", 'P': "110101110100100",
    'Q': "010101101110011", 'R': "110101110101101", 'S': "011100010001110", 'T': "111010010010010",
    'U': "101101101101111", 'V': "101101101101010", 'W': "101101111111101", 'X': "101101010101101",
    'Y': "101101010010010", 'Z': "111001010100111", '0': "111101101101111", '1': "010110010010111",
    '2': "110001010100111", '3': "110001010001110", '4': "101101111001001", '5': "111100110001110",
    '6': "011100111101111", '7': "111001010010010", '8': "111101111101111", '9': "111101111001110",
    '%': "101001010100101", '.': "000000000000010", ':': "000010000010000", '!': "010010010000010",
    '?': "110001010000010", '+': "000010111010000", '-': "000000111000000", "'": "010010000000000",
    '/': "001001010100100", '(': "010100100100010", ')': "010001001001010", '=': "000111000111000",
    ',': "000000000010100", ' ': "000000000000000",
}
F5 = {
    'A': "01110 10001 10001 11111 10001 10001 10001", 'B': "11110 10001 10001 11110 10001 10001 11110",
    'C': "01110 10001 10000 10000 10000 10001 01110", 'D': "11110 10001 10001 10001 10001 10001 11110",
    'E': "11111 10000 10000 11110 10000 10000 11111", 'F': "11111 10000 10000 11110 10000 10000 10000",
    'G': "01110 10001 10000 10111 10001 10001 01111", 'H': "10001 10001 10001 11111 10001 10001 10001",
    'I': "01110 00100 00100 00100 00100 00100 01110", 'J': "00111 00010 00010 00010 00010 10010 01100",
    'K': "10001 10010 10100 11000 10100 10010 10001", 'L': "10000 10000 10000 10000 10000 10000 11111",
    'M': "10001 11011 10101 10101 10001 10001 10001", 'N': "10001 10001 11001 10101 10011 10001 10001",
    'O': "01110 10001 10001 10001 10001 10001 01110", 'P': "11110 10001 10001 11110 10000 10000 10000",
    'Q': "01110 10001 10001 10001 10101 10010 01101", 'R': "11110 10001 10001 11110 10100 10010 10001",
    'S': "01111 10000 10000 01110 00001 00001 11110", 'T': "11111 00100 00100 00100 00100 00100 00100",
    'U': "10001 10001 10001 10001 10001 10001 01110", 'V': "10001 10001 10001 10001 10001 01010 00100",
    'W': "10001 10001 10001 10101 10101 10101 01010", 'X': "10001 10001 01010 00100 01010 10001 10001",
    'Y': "10001 10001 01010 00100 00100 00100 00100", 'Z': "11111 00001 00010 00100 01000 10000 11111",
    '0': "01110 10001 10011 10101 11001 10001 01110", '1': "00100 01100 00100 00100 00100 00100 01110",
    '2': "01110 10001 00001 00010 00100 01000 11111", '3': "11111 00010 00100 00010 00001 10001 01110",
    '4': "00010 00110 01010 10010 11111 00010 00010", '5': "11111 10000 11110 00001 00001 10001 01110",
    '6': "00110 01000 10000 11110 10001 10001 01110", '7': "11111 00001 00010 00100 01000 01000 01000",
    '8': "01110 10001 10001 01110 10001 10001 01110", '9': "01110 10001 10001 01111 00001 00010 01100",
    '%': "11000 11001 00010 00100 01000 10011 00011", '!': "00100 00100 00100 00100 00100 00000 00100",
    '.': "00000 00000 00000 00000 00000 01100 01100", ',': "00000 00000 00000 00000 01100 00100 01000",
    '-': "00000 00000 00000 11111 00000 00000 00000", '+': "00000 00100 00100 11111 00100 00100 00000",
    '?': "01110 10001 00001 00010 00100 00000 00100", "'": "01100 00100 01000 00000 00000 00000 00000",
    ':': "00000 01100 01100 00000 01100 01100 00000", '/': "00001 00001 00010 00100 01000 10000 10000",
    '(': "00010 00100 01000 01000 01000 00100 00010", ')': "01000 00100 00010 00010 00010 00100 01000",
    '=': "00000 00000 11111 00000 11111 00000 00000", ' ': "00000 00000 00000 00000 00000 00000 00000",
}
_G3 = {k: [(i % 3, i // 3) for i, b in enumerate(v) if b == '1'] for k, v in F3.items()}
_G5 = {k: [(q, r) for r, row in enumerate(v.split()) for q, b in enumerate(row) if b == '1'] for k, v in F5.items()}


def tw(s, font=5, k=1):
    return ((4 if font == 3 else 6) * len(s) - 1) * k


def text(d, x, y, s, c, font=5, k=1, outline=None, lower=None):
    """Draw s; x='c' centres it. outline: 1-px border colour; lower: colour for the glyph's lower rows."""
    s = s.upper()
    if x == 'c':
        x = (W - tw(s, font, k)) // 2
    adv, g, split = (4, _G3, 3) if font == 3 else (6, _G5, 4)
    cells = [(x + i * adv * k + q * k, y + r * k, r) for i, ch in enumerate(s) for q, r in g.get(ch, g['?'])]
    if outline:
        for X, Y, _ in cells:
            d.rectangle([X - 1, Y - 1, X + k, Y + k], fill=C[outline])
    for X, Y, r in cells:
        d.rectangle([X, Y, X + k - 1, Y + k - 1], fill=C[lower if lower and r >= split else c])
    return x + tw(s, font, k)


# ---------------------------------------------------------------- helpers
def rect(d, x0, y0, x1, y1, c, edge=None):
    d.rectangle([x0, y0, x1, y1], fill=C[c], outline=C[edge] if edge else None)


def px(d, x, y, c):
    d.point((x, y), fill=C[c])


def hsh(*a):
    h = 2166136261
    for v in a:
        h = ((h ^ (int(v) & 0xffffffff)) * 16777619) & 0xffffffff
    return h >> 8


def ease(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def arc(p0, p1, u, h):
    u = max(0.0, min(1.0, u))
    return p0[0] + (p1[0] - p0[0]) * u, p0[1] + (p1[1] - p0[1]) * u - h * math.sin(math.pi * u)


BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


# ---------------------------------------------------------------- static backdrop
def make_base():
    a = np.zeros((H, W), np.uint8)
    sky = ['night', 'purple', 'plum', 'magenta', 'coral', 'orange', 'gold']
    for y in range(66):
        v = (y / 65) ** 1.15 * (len(sky) - 1)
        for x in range(W):
            i = int(v) + (1 if v - int(v) > BAYER[y % 4, x % 4] else 0)
            a[y, x] = C[sky[min(i, len(sky) - 1)]]
    sx, sy, r = 214, 50, 11                                   # setting sun, striped
    for y in range(sy - r - 3, sy + r + 3):
        for x in range(sx - r - 3, sx + r + 3):
            dd = math.hypot(x - sx, y - sy)
            if dd < r and not (y > sy and (y - sy) % 4 == 1):
                a[y, x] = C['cream'] if dd < r * 0.45 else C['gold']
            elif r <= dd < r + 2.5 and BAYER[y % 4, x % 4] < 0.5:
                a[y, x] = C['orange']
    return a


BASE = make_base()


def ground(a, shift):
    ys, xs = np.mgrid[65:114, 0:W]
    stripe = ((xs + shift + (ys - 65)) // 10) % 2
    dark = (xs + ys) % 2 == 0
    g = np.where(stripe == 0, C['green'], np.where(dark, C['dgreen'], C['green']))
    g[:2] = C['dgreen']
    a[65:114] = g
    for i in range(40):                                        # grass tufts
        x = int((i * 37 + 5 + shift) % W)
        y = 72 + (i * 23) % 40
        a[y, x] = C['lgreen']
        if x + 1 < W:
            a[y - 1, x + 1] = C['lgreen']
    ys, xs = np.mgrid[114:119, 0:W]
    speck = ((xs * 7 + ys * 13 + (xs * ys) % 5) % 9) == 0
    a[114:119] = np.where(speck, C['brown'], C['tan'])
    a[114] = C['brown']


CLOUDS = [(10, 20, 30), (80, 16, 22), (150, 25, 36), (220, 18, 26), (270, 28, 20)]
TREES = [(172, 42), (182, 46), (246, 43), (256, 40)]
HORSE = ["..........##.", ".........####", ".#######.##..", "##########...", "#.#######....",
         "..#.....#....", ".#.......#...", "#.........#.."]
HORSE2 = ["..........##.", ".........####", ".#######.##..", "##########...", "#.#######....",
          "...#...#.....", "...#..#......", "..#...#......"]


def horse(d, x, y, t, silk):
    spr = HORSE if (t // 2) % 2 else HORSE2
    for r, row in enumerate(spr):
        for q, b in enumerate(row):
            if b == '#':
                px(d, x + q, y + r, 'dbrown')
    px(d, x + 11, y + 1, 'ink')
    rect(d, x + 5, y - 2, x + 7, y, silk)                     # jockey silks
    rect(d, x + 6, y - 4, x + 7, y - 3, 'ink')                # cap


def grandstand(d, ox, g, jump):
    X = lambda xw: int(round(xw + ox))
    L, R = X(-40), X(172)
    rect(d, L, 35, R, 57, 'night')
    for yy in (42, 47, 52):
        d.line([L, yy, R, yy], fill=C['dgray'])
    for r in range(4):                                         # crowd
        hy = 38 + 5 * r
        for xw in range(-40 + (r % 2) * 2, 170, 4):
            h = hsh(xw, r)
            if h % 11 == 0:
                continue
            x = X(xw)
            if x < -4 or x > W + 4:
                continue
            bob = (2 if (g + h) % 4 < 2 else 0) if jump else (1 if (g // 3 + h) % 9 == 0 else 0)
            skin = ('skin', 'skinsh', 'tan', 'brown')[h % 4]
            shirt = ('red', 'blue', 'cream', 'gold', 'teal', 'magenta', 'lgray', 'orange')[(h // 4) % 8]
            rect(d, x - 1, hy + 2 - bob, x + 2, hy + 3 - bob, shirt)
            rect(d, x, hy - bob, x + 1, hy + 1 - bob, skin)
            if (h // 32) % 5 == 0:
                d.line([x - 1, hy - 1 - bob, x + 2, hy - 1 - bob], fill=C[('ink', 'dred', 'cream')[h % 3]])
            if jump and h % 3 == 0:
                px(d, x + 3, hy - 1 - bob, skin)
    for xw in range(-36, 172, 26):                             # pillars
        rect(d, X(xw), 34, X(xw) + 1, 57, 'gray')
        d.line([X(xw), 34, X(xw), 57], fill=C['lgray'])
    rect(d, L, 29, R, 33, 'dgray')                             # roof
    d.line([L, 29, R, 29], fill=C['lgray'])
    for i, xw in enumerate(range(-40, 172, 6)):                # scalloped awning
        c = 'red' if i % 2 else 'cream'
        rect(d, X(xw), 34, X(xw) + 5, 35, c)
        d.line([X(xw) + 1, 36, X(xw) + 4, 36], fill=C[c])
    rect(d, L, 57, R, 60, 'cream')
    d.line([L, 58, R, 58], fill=C['red'])
    for i, xw in enumerate((-20, 20, 100, 140)):               # pennants
        x = X(xw)
        d.line([x, 22, x, 28], fill=C['lgray'])
        c = ('red', 'blue', 'gold', 'teal')[i]
        if (g // 4 + i) % 2:
            rect(d, x + 1, 22, x + 4, 23, c); px(d, x + 1, 24, c)
        else:
            rect(d, x + 1, 23, x + 4, 24, c); px(d, x + 1, 22, c)
    tx = X(58)                                                 # clock tower
    rect(d, tx, 14, tx + 14, 28, 'cream', 'dbrown')
    d.polygon([(tx - 2, 14), (tx + 7, 6), (tx + 16, 14)], fill=C['red'], outline=C['dred'])
    d.ellipse([tx + 3, 16, tx + 11, 24], fill=C['white'], outline=C['ink'])
    d.line([tx + 7, 20, tx + 7, 17], fill=C['ink'])
    d.line([tx + 7, 20, tx + 9, 20], fill=C['ink'])
    rect(d, tx + 5, 25, tx + 9, 28, 'night')
    d.line([tx + 7, 6, tx + 7, 1], fill=C['lgray'])
    rect(d, tx + 8, 1 + (g // 4) % 2, tx + 11, 2 + (g // 4) % 2, 'gold')


def world(im, d, g, cam=0.0, horses=None, jump=False):
    for i in range(18):                                        # stars
        x, y = (i * 47 + 11) % W, (i * 13 + 3) % 16
        if (g // 5 + i) % 6:
            px(d, x, y, 'cream' if i % 3 == 0 else 'lav')
    for cx, cy, L in CLOUDS:
        x = int((cx - g * 0.25 - cam * 0.1) % (W + 80)) - 40
        d.line([x + 4, cy, x + L - 4, cy], fill=C['purple'])
        d.line([x, cy + 1, x + L, cy + 1], fill=C['purple'])
        d.line([x + 2, cy + 2, x + L - 1, cy + 2], fill=C['coral'])
    o = -cam * 0.15
    d.polygon([(x, 52 - 3 * math.sin((x - o) / 17) - 2 * math.sin((x - o) / 7)) for x in range(-1, W + 2)]
              + [(W + 1, 62), (-1, 62)], fill=C['plum'])
    o = -cam * 0.25
    d.polygon([(x, 56 - 3 * math.sin((x - o) / 29 + 1) - 1.5 * math.sin((x - o) / 11)) for x in range(-1, W + 2)]
              + [(W + 1, 62), (-1, 62)], fill=C['purple'])
    ox = -cam * 0.4
    for xw, top in TREES:
        x = int(xw + ox)
        rect(d, x, top + 9, x + 1, 60, 'dbrown')
        d.ellipse([x - 6, top, x + 7, top + 12], fill=C['dgreen'])
        d.arc([x - 5, top + 1, x + 5, top + 9], 190, 260, fill=C['green'])
    grandstand(d, ox, g, jump)
    rect(d, 0, 61, W - 1, 64, 'tan')                          # track
    so = int(-cam * 0.6)
    for x in range(W):
        h = hsh(x - so)
        if h % 5 == 0:
            px(d, x, 61 + h % 4, 'brown')
    if horses is not None:
        for i, (sp, off, silk) in enumerate(((7, 0, 'red'), (8, -30, 'blue'), (6.5, -55, 'gold'))):
            horse(d, int(-20 + off + sp * horses) , 56, horses + i, silk)
    so = -cam * 0.8                                            # fence
    for yy in (65, 68):
        d.line([0, yy, W - 1, yy], fill=C['cream'])
    for x in range(int(so) % 12 - 12, W, 12):
        rect(d, x, 64, x + 1, 71, 'cream')
        d.line([x + 1, 64, x + 1, 71], fill=C['lgray'])


def new_frame(g, cam=0.0, horses=None, jump=False):
    a = BASE.copy()
    ground(a, int(-cam))
    im = Image.fromarray(a, 'P')
    im.putpalette(FLAT)
    d = ImageDraw.Draw(im)
    world(im, d, g, cam, horses, jump)
    return im, d


# ---------------------------------------------------------------- props and characters
def shadow(d, x0, x1, y):
    d.ellipse([x0, y - 1, x1, y + 2], fill=C['brown'])


def crate(d, ox=0):
    x0 = 182 + ox
    shadow(d, x0 - 2, x0 + 40, 116)
    rect(d, x0, 104, x0 + 38, 116, 'tan', 'dbrown')
    for yy in (108, 112):
        d.line([x0 + 1, yy, x0 + 37, yy], fill=C['brown'])
    for xx in (x0 + 3, x0 + 35):
        d.line([xx, 105, xx, 115], fill=C['brown'])
    rect(d, x0 + 10, 105, x0 + 28, 115, 'tan')
    d.rectangle([x0 + 10, 105, x0 + 28, 115], outline=C['brown'])
    text(d, x0 + 12, 108, 'BETS', 'dbrown', font=3)


# 17x9 LED visor faces ('#' = LED in the mood colour, 'o' = blush)
FACES = {
    'ok': ["................." ,
           "....##.....##....",
           "....##.....##....",
           "....##.....##....",
           ".................",
           ".................",
           "....#.......#....",
           ".....#######.....",
           "................."],
    'happy': [".................",
              "....##.....##....",
              "...#..#...#..#...",
              ".................",
              ".oo...........oo.",
              "...#.........#...",
              "....##.....##....",
              ".....#######.....",
              "......#####......"],
    'sad': ["..##.........##..",
            "....##.....##....",
            "...###.....###...",
            "...###.....###...",
            ".................",
            "......#####......",
            ".....##...##.....",
            "....##.....##....",
            "................."],
}
FACE_BLINK = [".................", ".................", ".................", "....##.....##....",
              ".................", ".................", "....#.......#....", ".....#######.....", "................."]


def robot(d, x, y, t, mood='ok', hold=False):
    hi, mid, lo = ('lgreen', 'green', 'dgreen') if mood == 'happy' else ('silver', 'lgray', 'gray')
    shadow(d, x - 2, x + 25, 116)
    rect(d, x + 6, y + 32, x + 9, y + 35, lo)                  # legs and feet
    rect(d, x + 13, y + 32, x + 16, y + 35, lo)
    rect(d, x + 4, y + 36, x + 10, y + 38, 'dgray', 'ink')
    rect(d, x + 12, y + 36, x + 18, y + 38, 'dgray', 'ink')
    y += (t // 6) % 2                                          # upper body bobs
    rect(d, x + 3, y + 17, x + 19, y + 31, mid, 'ink')        # body
    d.line([x + 4, y + 18, x + 4, y + 30], fill=C[hi])
    d.line([x + 18, y + 18, x + 18, y + 30], fill=C[lo])
    d.line([x + 4, y + 30, x + 18, y + 30], fill=C[lo])
    rect(d, x + 6, y + 20, x + 16, y + 27, 'ink')             # chest panel
    for i, c in enumerate(('red', 'gold', 'lgreen')):
        px(d, x + 8 + 3 * i, y + 23, c if (t // 3) % 3 == i else 'dgray')
    d.line([x + 8, y + 25, x + 14, y + 25], fill=C['dgray'])
    rect(d, x - 1, y + 18, x + 2, y + 28, lo, 'ink')          # left arm
    rect(d, x - 1, y + 29, x + 2, y + 30, 'dgray')
    if hold:                                                   # right arm up, holding the two tickets
        rect(d, x + 20, y + 18, x + 23, y + 21, lo, 'ink')
        rect(d, x + 23, y + 11, x + 26, y + 21, lo, 'ink')
    else:
        rect(d, x + 20, y + 18, x + 23, y + 28, lo, 'ink')
        rect(d, x + 20, y + 29, x + 23, y + 30, 'dgray')
    rect(d, x + 9, y + 15, x + 13, y + 16, lo)                 # neck
    rect(d, x - 2, y + 4, x, y + 9, lo, 'ink')                 # ears
    rect(d, x + 22, y + 4, x + 24, y + 9, lo, 'ink')
    rect(d, x + 1, y + 1, x + 21, y + 13, mid)                 # head (rounded)
    d.line([x + 2, y, x + 20, y], fill=C['ink'])
    d.line([x + 2, y + 14, x + 20, y + 14], fill=C['ink'])
    d.line([x, y + 2, x, y + 12], fill=C['ink'])
    d.line([x + 22, y + 2, x + 22, y + 12], fill=C['ink'])
    for cx, cy in ((x + 1, y + 1), (x + 21, y + 1), (x + 1, y + 13), (x + 21, y + 13)):
        px(d, cx, cy, 'ink')
    d.line([x + 2, y + 1, x + 19, y + 1], fill=C[hi])
    d.line([x + 1, y + 2, x + 1, y + 11], fill=C[hi])
    d.line([x + 21, y + 2, x + 21, y + 12], fill=C[lo])
    d.line([x + 2, y + 13, x + 20, y + 13], fill=C[lo])
    d.rectangle([x + 2, y + 2, x + 20, y + 12], outline=C['dgray'])   # visor bezel
    rect(d, x + 3, y + 3, x + 19, y + 11, 'ink')              # dark visor, bright LED features
    face, col = FACES[mood], {'ok': 'lblue', 'happy': 'lgreen', 'sad': 'red'}[mood]
    if mood == 'ok' and t % 30 in (0, 1):
        face = FACE_BLINK
    for r, row in enumerate(face):
        for q, ch in enumerate(row):
            if ch != '.':
                px(d, x + 3 + q, y + 3 + r, col if ch == '#' else 'coral')
    d.line([x + 11, y - 1, x + 11, y - 4], fill=C['gray'])     # antenna
    bulb = {'ok': ('red', 'gold'), 'happy': ('lgreen', 'green'), 'sad': ('red', 'dred')}[mood][(t // 4) % 2]
    rect(d, x + 10, y - 6, x + 12, y - 4, bulb)
    if hold:
        ticket(d, x + 21, y + 3, 'red')
        ticket(d, x + 26, y + 6, 'blue')


def bookie(d, x, y, t, mood='grin'):
    """y = top of the hat; feet at y + 47. Faces left."""
    rect(d, x + 6, y + 45, x + 13, y + 46, 'ink')              # shoes
    rect(d, x + 15, y + 45, x + 21, y + 46, 'ink')
    rect(d, x + 9, y + 37, x + 19, y + 44, 'dgray')            # trousers
    d.line([x + 14, y + 39, x + 14, y + 44], fill=C['ink'])
    d.line([x + 18, y + 37, x + 18, y + 44], fill=C['ink'])
    for yy in range(y + 21, y + 37):                           # checked jacket, shaded on the right
        for xx in range(x + 6, x + 23):
            chk = (xx - x) % 4 == 0 or (yy - y) % 4 == 0
            c = ('dbrown' if chk else 'brown') if xx >= x + 19 else ('brown' if chk else 'tan')
            px(d, xx, yy, c)
    d.rectangle([x + 5, y + 20, x + 23, y + 37], outline=C['ink'])
    d.polygon([(x + 12, y + 20), (x + 16, y + 20), (x + 14, y + 26)], fill=C['cream'])
    d.line([x + 11, y + 21, x + 13, y + 27], fill=C['dbrown'])
    d.line([x + 17, y + 21, x + 15, y + 27], fill=C['dbrown'])
    for p in ((x + 12, y + 21), (x + 13, y + 22), (x + 16, y + 21), (x + 15, y + 22)):
        px(d, *p, 'red')
    px(d, x + 14, y + 21, 'dred')
    for yy in (y + 29, y + 33):
        px(d, x + 14, yy, 'gold')
    d.line([x + 7, y + 21, x + 20, y + 29], fill=C['dbrown'])   # satchel strap
    rect(d, x + 19, y + 29, x + 27, y + 36, 'brown', 'dbrown')  # satchel with cash
    rect(d, x + 20, y + 26, x + 25, y + 28, 'lgreen', 'dgreen')
    px(d, x + 22, y + 27, 'dgreen')
    rect(d, x + 19, y + 29, x + 27, y + 31, 'tan', 'dbrown')
    px(d, x + 23, y + 32, 'gold')
    if mood == 'greedy':                                       # arm up, waving a fan of notes
        rect(d, x + 2, y + 16, x + 6, y + 27, 'brown', 'ink')
        rect(d, x + 1, y + 13, x + 5, y + 16, 'skin', 'skinsh')
        for i in range(3):
            rect(d, x - 3 + i * 3, y + 7 - (i % 2), x + 1 + i * 3, y + 12 - (i % 2), 'lgreen', 'dgreen')
    else:
        rect(d, x + 2, y + 22, x + 6, y + 31, 'brown', 'ink')
        rect(d, x + 1, y + 31, x + 4, y + 33, 'skin', 'skinsh')
    lift = 4 if mood == 'tip' else 0
    hy = y - lift                                              # bowler dome (face drawn over its lower half)
    d.ellipse([x + 8, hy, x + 20, hy + 16], fill=C['ink'])
    d.line([x + 11, hy + 2, x + 13, hy + 2], fill=C['dgray'])
    px(d, x + 10, hy + 3, 'dgray')
    if lift:
        rect(d, x + 8, y + 8, x + 20, y + 9, 'brown')           # hair under the lifted hat
    rect(d, x + 8, y + 10, x + 20, y + 19, 'skin')             # head
    d.line([x + 19, y + 10, x + 19, y + 19], fill=C['skinsh'])
    d.line([x + 20, y + 11, x + 20, y + 18], fill=C['skinsh'])
    d.line([x + 9, y + 20, x + 18, y + 20], fill=C['skinsh'])
    rect(d, x + 20, y + 12, x + 21, y + 15, 'skinsh')          # ear
    rect(d, x + 6, y + 13, x + 7, y + 15, 'skin')              # nose
    px(d, x + 6, y + 15, 'skinsh')
    px(d, x + 15, y + 15, 'coral')                             # cheek
    brow = y + (10 if mood == 'sweat' else 11)
    d.line([x + 9, brow, x + 11, brow], fill=C['dbrown'])
    d.line([x + 13, brow, x + 15, brow], fill=C['dbrown'])
    eye = 'gold' if mood == 'greedy' else 'ink'
    rect(d, x + 10, y + 12, x + 10, y + 13, eye)
    rect(d, x + 14, y + 12, x + 14, y + 13, eye)
    d.line([x + 8, y + 16, x + 15, y + 16], fill=C['brown'])   # handlebar moustache
    px(d, x + 7, y + 15, 'brown'); px(d, x + 16, y + 15, 'brown')
    if mood == 'sweat':
        for p in ((x + 9, y + 18), (x + 10, y + 17), (x + 11, y + 18), (x + 12, y + 17), (x + 13, y + 18)):
            px(d, *p, 'ink')
        for k, (sx, sy) in enumerate(((x + 21, y + 9), (x + 7, y + 10))):
            dy = (t // 2 + k * 3) % 6
            px(d, sx, sy + dy, 'lblue'); px(d, sx, sy + dy + 1, 'blue')
    else:
        d.line([x + 9, y + 18, x + 13, y + 18], fill=C['ink'])
        d.line([x + 10, y + 17, x + 12, y + 17], fill=C['white'])
        d.line([x + 3, y + 18, x + 8, y + 18], fill=C['tan'])    # cigar
        px(d, x + 2, y + 18, 'orange' if (t // 3) % 2 else 'red')
        for k in range(3):                                      # smoke
            u = ((t + k * 5) % 15) / 15
            px(d, int(x + 2 - 3 * u + math.sin(u * 6 + k)), int(y + 16 - 12 * u), 'lgray' if u < 0.5 else 'gray')
    rect(d, x + 8, hy + 6, x + 20, hy + 7, 'dred')             # hat band and curled brim
    rect(d, x + 5, hy + 8, x + 23, hy + 9, 'ink')
    px(d, x + 5, hy + 7, 'ink'); px(d, x + 23, hy + 7, 'ink')


def ticket(d, x, y, col):
    rect(d, x, y, x + 6, y + 4, col, 'ink')
    d.line([x + 2, y + 2, x + 4, y + 2], fill=C['cream'])


def note(d, x, y, t):
    x, y = int(x), int(y)
    if t % 2:
        rect(d, x, y, x + 6, y + 3, 'lgreen', 'dgreen'); px(d, x + 3, y + 1, 'dgreen')
    else:
        rect(d, x + 1, y - 1, x + 5, y + 4, 'lgreen', 'dgreen'); px(d, x + 3, y + 1, 'dgreen')


# ---------------------------------------------------------------- the RED vs BLUE draw machine
DC, DR = (84, 88), 13
REST = [(78, 96, 'red'), (82, 96, 'blue'), (86, 96, 'red'), (90, 96, 'blue'), (84, 92, 'red')]
CHUTE0, TRAY = (95, 97), (107, 97)


def ball(d, cx, cy, col):
    base, sh, hi = {'red': ('red', 'dred', 'cream'), 'blue': ('blue', 'dblue', 'lblue')}[col]
    x, y = int(round(cx)) - 2, int(round(cy)) - 2
    rect(d, x + 1, y, x + 2, y + 3, base)
    rect(d, x, y + 1, x + 3, y + 2, base)
    px(d, x + 1, y + 1, hi); px(d, x + 2, y + 3, sh); px(d, x + 3, y + 2, sh)


def drum(d, t, draws=(), g=0):
    """draws: (spin_start, draw_frame, colour, clear_frame)."""
    spun, s, out = 0, 0.0, None
    for t0, t1, col, tc in draws:
        spun += max(0, min(t, t1) - t0)
        if t0 <= t < t1:
            s = min(1.0, (t - t0 + 1) / 3)
        if t1 <= t < tc:
            out = (col, t - t1)
    ang = spun * 0.7
    shadow(d, 54, 114, 116)
    rect(d, 56, 102, 112, 103, 'tan', 'dbrown')                # table
    rect(d, 57, 104, 111, 114, 'brown', 'dbrown')
    rect(d, 58, 115, 60, 116, 'dbrown'); rect(d, 108, 115, 110, 116, 'dbrown')
    rect(d, 60, 104, 108, 114, 'gold', 'dbrown')               # name plate (2 px padding round the text)
    x = text(d, 63, 107, 'RED', 'red', font=3)
    x = text(d, x + 5, 107, 'VS', 'dbrown', font=3)
    text(d, x + 5, 107, 'BLUE', 'blue', font=3)
    d.line([80, 92, 74, 101], fill=C['gray']); d.line([81, 92, 75, 101], fill=C['dgray'])
    d.line([88, 92, 94, 101], fill=C['gray']); d.line([87, 92, 93, 101], fill=C['dgray'])
    cx, cy = DC
    for yy in range(cy - DR, cy + DR + 1):                     # glass
        for xx in range(cx - DR, cx + DR + 1):
            if math.hypot(xx - cx, yy - cy) <= DR - 0.5:
                px(d, xx, yy, 'dblue' if (xx + yy) % 2 else 'night')
    for k in range(3):
        a = ang + k * 2 * math.pi / 3
        d.line([cx, cy, cx + 11 * math.cos(a), cy + 11 * math.sin(a)], fill=C['dgray'])
    rest = list(REST)
    if out:
        for i in range(len(rest) - 1, -1, -1):
            if rest[i][2] == out[0]:
                del rest[i]
                break
    for i, (rx, ry, col) in enumerate(rest):
        th = i * 1.3 + ang * (1 + 0.21 * i)
        rho = 3 + (i * 3) % 6 + 1.5 * math.sin(ang * 1.7 + i)
        bx, by = cx + rho * math.cos(th), cy + rho * math.sin(th)
        ball(d, rx * (1 - s) + bx * s, ry * (1 - s) + by * s, col)
    d.arc([cx - 11, cy - 11, cx + 11, cy + 11], 200, 250, fill=C['white'])
    d.arc([cx - 9, cy - 9, cx + 9, cy + 9], 205, 235, fill=C['lblue'])
    d.ellipse([cx - DR - 1, cy - DR - 1, cx + DR + 1, cy + DR + 1], outline=C['ink'])
    d.ellipse([cx - DR, cy - DR, cx + DR, cy + DR], outline=C['silver'])
    rect(d, cx - 1, cy - 1, cx + 1, cy + 1, 'gold')
    hx, hy = cx - DR - 2, cy                                   # crank
    kx, ky = hx + 4 * math.cos(ang * 1.5), hy + 4 * math.sin(ang * 1.5)
    d.line([hx, hy, kx, ky], fill=C['lgray'])
    rect(d, int(kx) - 1, int(ky) - 1, int(kx), int(ky), 'red')
    d.line([94, 96, 104, 98], fill=C['silver'])                 # chute and tray
    d.line([94, 99, 103, 100], fill=C['lgray'])
    rect(d, 103, 99, 111, 101, 'gray', 'ink')
    if out:
        col, k = out
        u = min(1.0, k / 4)
        ball(d, CHUTE0[0] + (TRAY[0] - CHUTE0[0]) * u, CHUTE0[1] + (TRAY[1] - CHUTE0[1]) * u, col)
        if k >= 4:
            sp = ((0, -6), (-5, -3), (5, -3)) if g % 2 else ((-4, -5), (4, -5), (0, -8))
            for dx, dy in sp:
                px(d, TRAY[0] + dx, TRAY[1] + dy, 'gold')


# ---------------------------------------------------------------- odds board, bubbles, captions
def led(d, x, y, s, col, t, flip):
    s = s.rjust(4)
    for j, ch in enumerate(s):
        cx = x + j * 6
        rect(d, cx, y, cx + 4, y + 6, 'ledoff')
        if flip is None or t < flip:
            continue
        if t < flip + 6:
            ch = '0123456789'[hsh(t, j, x, y) % 10] if ch != ' ' else ' '
            c = 'led'
        else:
            c = col
        for q, r in _G5.get(ch, []):
            px(d, cx + q, y + r, c)


def board(d, rows, t, drop=0):
    x0, y0, x1, y1 = 116, 14 + drop, 180, 63 + drop
    for p in (122, 172):
        rect(d, p, y1, p + 2, 115, 'gray')
        d.line([p + 2, y1, p + 2, 115], fill=C['dgray'])
    rect(d, x0, y0, x1, y1, 'brown', 'dbrown')
    rect(d, x0 + 2, y0 + 2, x1 - 2, y1 - 2, 'ink')
    for bx, by in ((x0 + 1, y0 + 1), (x1 - 1, y0 + 1), (x0 + 1, y1 - 1), (x1 - 1, y1 - 1)):
        px(d, bx, by, 'gold')
    rect(d, x0 + 2, y0 + 2, x1 - 2, y0 + 10, 'dred')          # header: 2 px above and below the text
    s = "TODAY'S PRICES"
    text(d, x0 + (x1 - x0 + 1 - tw(s, 3)) // 2, y0 + 4, s, 'cream', font=3)
    for i, (label, val, col, flip) in enumerate(rows):         # labels flush left, LEDs flush right, 3 px in
        ry = y0 + 14 + i * 11 + (2 if i == 2 else 0)
        text(d, x0 + 5, ry + 1, label, 'cream' if i < 2 else 'gold', font=3)
        led(d, x1 - 5 - 22, ry, val, col, t, flip)
    d.line([x0 + 5, y0 + 34, x1 - 5, y0 + 34], fill=C['gray'])


def bubble(d, x, y, lines, tip, font=5, fill='cream'):
    lh, gap = (7, 3) if font == 5 else (5, 2)
    w = max(tw(s, font) for s, _ in lines) + 8
    h = len(lines) * lh + (len(lines) - 1) * gap + 7
    d.line([x + 2, y + h + 1, x + w, y + h + 1], fill=C['ink'])
    d.line([x + w + 1, y + 2, x + w + 1, y + h], fill=C['ink'])
    rect(d, x + 1, y + 1, x + w - 1, y + h - 1, fill)
    for a, b in (((x + 1, y), (x + w - 1, y)), ((x + 1, y + h), (x + w - 1, y + h)),
                 ((x, y + 1), (x, y + h - 1)), ((x + w, y + 1), (x + w, y + h - 1))):
        d.line([a, b], fill=C['ink'])
    bx = min(max(tip[0] - 3, x + 3), x + w - 9)
    d.polygon([(bx, y + h), (bx + 6, y + h), tip], fill=C[fill], outline=C['ink'])
    d.line([bx + 1, y + h, bx + 5, y + h], fill=C[fill])
    for i, (s, c) in enumerate(lines):
        text(d, x + 4, y + 4 + i * (lh + gap), s, c, font=font)


def caption(d, s, c='cream'):
    rect(d, 0, 0, W - 1, 10, 'ink')                           # 2 px above and below the text
    d.line([0, 11, W - 1, 11], fill=C['gold'])
    text(d, 'c', 2, s, c)


def ledger(d, s, c='cream'):
    rect(d, 0, 119, W - 1, H - 1, 'ink')                       # 4 px above and below the text
    d.line([0, 119, W - 1, 119], fill=C['gold'])
    for x in (3, W - 4):                                        # ticket-stub punch holes
        for y in (124, 130):
            px(d, x, y, 'dgray')
    if s:
        text(d, 'c', 124, s, c)


def timed(t, items, default=('', 'cream')):
    cur = default
    for t0, s, c in items:
        if t >= t0:
            cur = (s, c)
    return cur


def big_title(d, y, s='BOOKIEBENCH'):
    x0, y0, x1, y1 = 30, y - 7, 209, y + 44                    # hanging race-day sign
    for rx in (x0 + 14, x1 - 14):
        d.line([rx, 0, rx, y0], fill=C['lgray'])
    rect(d, x0, y0, x1, y1, 'dbrown', 'ink')
    rect(d, x0 + 2, y0 + 2, x1 - 2, y1 - 2, 'ink')
    d.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=C['plum'])
    for bx, by in ((x0 + 1, y0 + 1), (x1 - 1, y0 + 1), (x0 + 1, y1 - 1), (x1 - 1, y1 - 1)):
        px(d, bx, by, 'gold')
    text(d, 'c', y, s, 'gold', k=2, outline='ink', lower='orange')


ROBOT_X, ROBOT_Y, BOOKIE_X, BOOKIE_Y = 10, 78, 188, 57


# ---------------------------------------------------------------- scenes
def scene_title(t, g):
    cam = -60 * (1 - ease(t / 18))
    im, d = new_frame(g, cam, horses=g % 56)
    ox = int(-cam)
    crate(d, ox)
    drum_off(d, ox, t)
    bookie(d, BOOKIE_X + ox, BOOKIE_Y, g, 'grin')
    robot(d, ROBOT_X + ox, ROBOT_Y, g)
    drop = max(0, 24 - t * 3)
    big_title(d, 13 - drop)
    if t > 8:
        text(d, 'c', 35, "CAN A MODEL'S PROBABILITIES", 'cream', outline='ink')
        text(d, 'c', 44, 'BE TRUSTED?', 'cream', outline='ink')
    ledger(d, 'A BENCHMARK THAT BETS AGAINST MODELS' if t > 12 else '', 'gold')
    return im


def drum_off(d, ox, t):
    if ox == 0:
        drum(d, t)
        return
    tmp = Image.new('P', (W, H), TRANSP)
    tmp.putpalette(FLAT)
    drum(ImageDraw.Draw(tmp), t)
    a = np.array(tmp)
    b = np.full_like(a, TRANSP)
    if ox > 0:
        b[:, ox:] = a[:, :W - ox]
    else:
        b = a
    d._image.paste(Image.fromarray(b, 'P'), (0, 0), Image.fromarray(np.where(b != TRANSP, 255, 0).astype(np.uint8), 'L'))


def scene_question(t, g):
    im, d = new_frame(g)
    crate(d)
    drop = -int(60 * (1 - ease(t / 8))) if t < 8 else 0
    blink = 'red' if (g // 3) % 2 else 'led'
    board(d, [('RED', '70%', 'led', 10), ('NOT RED', '50%', 'led', 18),
              ('SUM', '120%', blink if t >= 30 else 'red', 24)], t, drop)
    drum(d, t, g=g)
    bookie(d, BOOKIE_X, BOOKIE_Y, g, 'grin')
    robot(d, ROBOT_X, ROBOT_Y, g)
    if t >= 5:
        lines = [('P(RED)     = 70%', 'red')]
        if t >= 15:
            lines.append(('P(NOT RED) = 50%', 'blue'))
        bubble(d, 4, 38, lines, (24, 70))
    caption(d, 'ASK FOR LINKED PROBABILITIES')
    ledger(d, *timed(t, [(0, 'THE DRUM HOLDS 3 RED AND 2 BLUE BALLS', 'cream'),
                         (26, 'BUT RED + NOT RED = 120%!', 'red')]))
    return im


BOOK_DRAWS = [(20, 30, 'red', 40), (40, 48, 'blue', 70)]


def scene_book(t, g):
    dutch = t >= 54
    im, d = new_frame(g, jump=dutch)
    crate(d)
    board(d, [('RED', '70%', 'led', -99), ('NOT RED', '50%', 'led', -99), ('SUM', '120%', 'red', -99)], t)
    drum(d, t, BOOK_DRAWS, g=g)
    bookie(d, BOOKIE_X, BOOKIE_Y, g, 'greedy' if t >= 30 else 'grin')
    hold = t >= 14
    robot(d, ROBOT_X, ROBOT_Y, g, 'sad' if t >= 30 else 'ok', hold=hold)
    hand, rob = (BOOKIE_X + 1, BOOKIE_Y + 30), (ROBOT_X + 22, ROBOT_Y + 8)
    for i, col in enumerate(('red', 'blue')):                  # tickets out ...
        u = (t - 2 - 4 * i) / 10
        if 0 <= u < 1:
            x, y = arc(hand, rob, u, 18)
            ticket(d, int(x), int(y), col)
    for i in range(5):                                         # ... money in
        u = (t - 6 - 2 * i) / 12
        if 0 <= u < 1:
            x, y = arc((ROBOT_X + 20, ROBOT_Y + 20), (BOOKIE_X + 20, BOOKIE_Y + 26), u, 10 + 3 * i)
            note(d, x, y, t + i)
    if dutch:                                                  # sign drops in left of the board
        oy = -int(50 * (1 - ease((t - 54) / 4)))
        x0, y0, x1, y1 = 14, 16 + oy, 104, 61 + oy
        for rx in (x0 + 12, x1 - 12):
            d.line([rx, 12, rx, y0], fill=C['lgray'])
        rect(d, x0, y0, x1, y1, 'gold', 'ink')
        rect(d, x0 + 2, y0 + 2, x1 - 2, y1 - 2, 'dred')
        d.rectangle([x0 + 3, y0 + 3, x1 - 3, y1 - 3], outline=C['red'])
        for bx, by in ((x0 + 1, y0 + 1), (x1 - 1, y0 + 1), (x0 + 1, y1 - 1), (x1 - 1, y1 - 1)):
            px(d, bx, by, 'ink')
        c = 'gold' if (g // 3) % 2 else 'cream'
        cx = (x0 + x1 + 1) // 2
        for s_, yy in (('DUTCH', y0 + 7), ('BOOK!', y0 + 25)):
            text(d, cx - tw(s_, 5, 2) // 2, yy, s_, c, k=2, outline='ink', lower='orange')
    caption(d, 'THE BOOKIE SELLS YOU BOTH TICKETS', 'orange')
    ledger(d, *timed(t, [(0, 'PAY 0.70 + 0.50 = 1.20', 'cream'),
                         (12, 'ONE TICKET WINS AND PAYS 1.00', 'cream'),
                         (31, 'BALL IS RED:  BOOKIE +0.20', 'red'),
                         (49, 'BALL IS BLUE: BOOKIE +0.20', 'blue'),
                         (58, 'EITHER WAY THE BOOKIE WINS 0.20', 'gold')]))
    return im


COH_DRAWS = [(26, 34, 'red', 40), (40, 47, 'blue', 58)]


def scene_coherent(t, g):
    im, d = new_frame(g)
    crate(d)
    board(d, [('RED', '60%', 'led', 4), ('NOT RED', '40%', 'led', 10), ('SUM', '100%', 'lgreen', 16)], t)
    drum(d, t, COH_DRAWS, g=g)
    bookie(d, BOOKIE_X, BOOKIE_Y, g, 'sweat' if t >= 35 else 'grin')
    robot(d, ROBOT_X, ROBOT_Y, g, 'happy')
    lines = [('P(RED)     = 60%', 'red')]
    if t >= 8:
        lines.append(('P(NOT RED) = 40%', 'blue'))
    bubble(d, 4, 38, lines, (24, 70))
    if t >= 35:
        bubble(d, 184, 22, [('NO SURE BET', 'dgray'), ('...DAMN.', 'dgray')], (199, 53), font=3)
    caption(d, 'A COHERENT, CALIBRATED MODEL', 'lgreen')
    ledger(d, *timed(t, [(0, '3 OF 5 BALLS ARE RED: EXACTLY 60%', 'cream'),
                         (24, 'PAY 0.60 + 0.40 = 1.00, WIN 1.00', 'cream'),
                         (35, 'BALL IS RED:  BOOKIE +0.00', 'red'),
                         (48, 'BALL IS BLUE: BOOKIE +0.00', 'blue'),
                         (53, 'NO DUTCH BOOK TO BE HAD', 'lgreen')]))
    return im


PILLARS = [('COHERENCE', 'NO DUTCH BOOKS, NO ORDER EFFECTS'), ('CALIBRATION', '70% MEANS 70%'),
           ('SKILL', 'VS THE EXACT POSTERIOR'), ('SENSITIVITY', 'BELIEFS MOVE WITH EVIDENCE')]
ROW0, ROWSTEP, WIN = 4, 18, 18          # row i is checked during story frames [ROW0 + i*WIN, ROW0 + (i+1)*WIN)
ALL_DONE = ROW0 + 4 * WIN + 2


def icon_coherence(d, y, u, g):
    x, full = 176, 30                                          # 30 px = 100%
    if u < 9:                                                  # grows to 70% + 50% = 120%, flashes
        e = ease((u - 2) / 4)
        wr, wb, pct = 21 * e, 15 * e, int(round(120 * e))
    else:                                                      # snaps back to 60% + 40% = 100%
        e = ease((u - 9) / 2)
        wr, wb, pct = 21 - 3 * e, 15 - 3 * e, int(round(120 - 20 * e))
    over = u >= 6 and pct > 100
    d.rectangle([x - 1, y + 1, x + 37, y + 7], fill=C['night'])
    if wr >= 1:
        rect(d, x, y + 2, x + int(wr) - 1, y + 6, 'red')
    if wb >= 1:
        rect(d, x + int(wr), y + 2, x + int(wr + wb) - 1, y + 6, 'blue')
    if over and (g // 2) % 2:
        rect(d, x + full, y + 2, x + int(wr + wb) - 1, y + 6, 'white')
    ok = pct == 100 and u >= 11
    d.line([x + full, y, x + full, y + 8], fill=C['lgreen' if ok else ('red' if over else 'gold')])
    if u >= 2:
        s = f'{pct}%'
        text(d, x + 36 - tw(s, 3), y + 10, s, 'lgreen' if ok else ('red' if over else 'cream'), font=3)


CAL_DOTS = [(2, 3), (6, -3), (10, 2), (14, -2), (18, 3)]       # (x along the plot, starting offset)


def icon_calibration(d, y, u, g):
    x = 196                                                    # 20x14 reliability plot
    d.line([x, y, x, y + 14], fill=C['lgray'])
    d.line([x, y + 14, x + 20, y + 14], fill=C['lgray'])
    n = int(20 * ease((u - 2) / 3))
    for k in range(0, n, 2):                                   # dashed diagonal draws itself
        px(d, x + 1 + k, y + 13 - int(k * 13 / 20), 'gray')
    for k, (dx, off) in enumerate(CAL_DOTS):
        a = 5 + k
        if u < a:
            continue
        o = off * (1 - ease((u - a) / 2))                      # each dot settles onto the diagonal
        cy = y + 13 - int(dx * 13 / 20) + int(round(o))
        rect(d, x + 1 + dx - 1, cy - 1, x + 1 + dx, cy, 'gold')


def icon_skill(d, y, u, g):
    cx, cy = 198, y + 7
    for r, c in ((7, 'red'), (5, 'cream'), (3, 'red'), (1, 'cream')):
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=C[c])
    kl = 0.412 * min(1.0, max(0.0, 1 - (u - 2) / 9))
    s = f'KL {kl:.3f}'
    text(d, 152, y + 5, s, 'lgreen' if kl == 0 else 'cream', font=3)
    if u < 2:
        return
    sx, sy = 224, y - 2                                        # arrow flies in from the right
    e = min(1.0, (u - 2) / 4)
    tx, ty = sx + (cx - sx) * e, sy + (cy - sy) * e
    L = math.hypot(sx - cx, sy - cy)
    ux, uy = (sx - cx) / L, (sy - cy) / L
    wob = (1 if g % 2 else -1) if 6 <= u < 10 else 0          # wobble on impact
    ex, ey = tx + 12 * ux, ty + 12 * uy + wob
    d.line([tx, ty, ex, ey], fill=C['silver'])
    d.line([ex, ey, ex + 2, ey - 2], fill=C['tan'])
    d.line([ex, ey, ex + 2, ey + 2], fill=C['tan'])
    px(d, int(tx), int(ty), 'gold')
    if 6 <= u < 9:
        for dx, dy in ((-3, -3), (3, -3), (-3, 3), (3, 3)):
            px(d, cx + dx, cy + dy, 'white')


SENS_EXACT = [11, 8, 6, 3, 1]                                  # exact posterior (row) after 0..4 pieces of evidence
SENS_X0, SENS_X1, SENS_RISERS = 176, 222, (181, 193, 205, 217)  # 4 evenly spaced risers, 5 px margin at both ends
SENS_AXIS, SENS_PIP = 16, 13                                    # axis row; resting pip rows 13-14 (1 px clear of both)


def icon_sensitivity(d, y, u, g):
    xs = (SENS_X0,) + SENS_RISERS
    d.line([SENS_X0, y + SENS_AXIS, SENS_X1, y + SENS_AXIS], fill=C['dgray'])
    for k in range(4):                                         # exact posterior: dashed steps
        for xx in range(xs[k], xs[k + 1] + 1, 2):
            px(d, xx, y + SENS_EXACT[k], 'lav')
        for yy in range(y + SENS_EXACT[k + 1], y + SENS_EXACT[k] + 1, 2):
            px(d, xs[k + 1], yy, 'lav')
    for xx in range(xs[4], SENS_X1 + 1, 2):
        px(d, xx, y + SENS_EXACT[4], 'lav')
    level, pts = 0, [(SENS_X0, y + SENS_EXACT[0])]
    for k, rx in enumerate(SENS_RISERS):
        a = 3 + 2 * k                                          # the pip drops onto the axis under its riser ...
        if u >= a:
            py = int(y - 2 + min(1.0, (u - a) / 2) * (SENS_PIP + 2))
            landed = py >= y + SENS_PIP
            if landed:                                         # dotted guide from the pip up to the riser
                for yy in range(y + SENS_PIP - 2, y + SENS_EXACT[k] + 1, -2):
                    px(d, rx, yy, 'dgray')
            rect(d, rx - 1, py, rx + 1, py + 1, 'gold')
        if u >= a + 2:                                         # ... lands, and then the belief steps up at that riser
            level = k + 1
            pts += [(rx, pts[-1][1]), (rx, y + SENS_EXACT[k + 1])]
    pts.append((SENS_X1 if level == 4 else xs[level] + 3, pts[-1][1]))
    d.line(pts, fill=C['lgreen'])


ICONS = [icon_coherence, icon_calibration, icon_skill, icon_sensitivity]


def tick(d, bx, by, u, col):
    """Tick strokes draw in over u = 12..15; sparkle after."""
    if u >= 12:
        n = min(3, u - 11)
        for k in range(n):
            rect(d, bx + 2 + k, by + 5 + k, bx + 3 + k, by + 5 + k, col)
    if u >= 14:
        n = min(5, (u - 13) * 3)
        for k in range(n):
            rect(d, bx + 5 + k, by + 7 - k, bx + 5 + k, by + 8 - k, col)


def scene_pillars(t, g):
    im, d = new_frame(g)
    crate(d)
    drum(d, t, g=g)
    bookie(d, BOOKIE_X, BOOKIE_Y, g, 'grin')
    robot(d, ROBOT_X, ROBOT_Y, g, 'happy')
    x0, y0, x1, y1 = 10, 14, 229, 115
    for p in (30, 208):
        rect(d, p, y1, p + 3, 118, 'gray')
    rect(d, x0, y0, x1, y1, 'brown', 'dbrown')
    rect(d, x0 + 2, y0 + 2, x1 - 2, y1 - 2, 'ink')
    for bx, by in ((x0 + 1, y0 + 1), (x1 - 1, y0 + 1), (x0 + 1, y1 - 1), (x1 - 1, y1 - 1)):
        px(d, bx, by, 'gold')
    active = min(3, max(0, (t - ROW0) // WIN)) if t < ALL_DONE else None
    for i, (a, b) in enumerate(PILLARS):
        y = y0 + 7 + i * 24
        s0 = ROW0 + i * WIN
        if t < s0:
            continue
        u = t - s0 if t < ALL_DONE else 99
        done = u >= WIN
        dim = done and t < ALL_DONE
        if i == active:                                        # highlight band on the active row
            rect(d, x0 + 4, y - 3, x1 - 4, y + 17, 'night')
            d.line([x0 + 4, y - 3, x0 + 4, y + 17], fill=C['gold'])
        bx, by = x0 + 15, y + 2
        rect(d, bx, by, bx + 10, by + 10, 'ink', 'gray' if dim else 'lgray')
        tick(d, bx, by, u, 'green' if dim else 'lgreen')
        if 15 <= u < 18:                                       # sparkle
            for dx, dy in (((-2, -2), (12, -2), (5, -3)) if g % 2 else ((-3, 5), (13, 5), (12, 12))):
                px(d, bx + dx, by + dy, 'gold')
        text(d, x0 + 30, y, a, ('ledoff' if u < 1 else 'led') if not dim else 'tan')
        text(d, x0 + 30, y + 10, b, 'lgray' if dim else 'cream', font=3)
        ICONS[i](d, y, min(u, 17), g)
    if active is not None and t >= ROW0 - 2:                   # pointer glides to the active row
        tgt = y0 + 7 + active * 24 + 6
        prev = y0 + 7 + max(0, active - 1) * 24 + 6
        k = t - (ROW0 + active * WIN)
        py = int(prev + (tgt - prev) * ease(k / 3)) if active else tgt
        pxx = x0 + 6 + ((g // 3) % 2)
        d.polygon([(pxx, py - 3), (pxx + 5, py), (pxx, py + 3)], fill=C['gold'], outline=C['orange'])
    caption(d, 'FOUR CHECKS, EXACT ANSWERS', 'gold')
    ledger(d, *timed(t, [(0, '37 SIMULATED WORLDS, 25 REAL SOURCES', 'lav'),
                         (ALL_DONE, 'AND 21 STRESS TRANSFORMS', 'lav')]))
    return im


def scene_end(t, g):
    im, d = new_frame(g, horses=g % 56)
    crate(d)
    drum(d, t, g=g)
    bookie(d, BOOKIE_X, BOOKIE_Y, g, 'tip' if t >= 8 else 'sweat')
    robot(d, ROBOT_X, ROBOT_Y, g, 'happy')
    big_title(d, 13)
    text(d, 'c', 35, 'IF THE BOOKIE CAN PROFIT,', 'cream', outline='ink')
    text(d, 'c', 44, 'THE PROBABILITIES LIE.', 'red', outline='ink')
    ledger(d, 'GITHUB.COM/MAPIKA/BOOKIEBENCH', 'gold')
    return im


# (scene, story frames, representative frame, {story frame: extra hold frames})
SCENES = [(scene_title, 22, 21, {21: 6}),
          (scene_question, 40, 36, {16: 3, 31: 4, 39: 6}),
          (scene_book, 70, 66, {36: 8, 54: 8, 69: 11}),
          (scene_coherent, 58, 55, {23: 7, 38: 4, 51: 4, 57: 6}),
          (scene_pillars, 84, 83, {83: 10}),
          (scene_end, 28, 27, {27: 10})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/assets/bookiebench.gif')
    ap.add_argument('--png', help='write one representative PNG per scene here (racetrack_<scene>.png)')
    ap.add_argument('--frames', help='debug: write every Nth frame as PNG here')
    ap.add_argument('--every', type=int, default=10)
    a = ap.parse_args()
    small, g = [], 0
    for fn, n, rep, holds in SCENES:
        for t in range(n):
            for h in range(1 + holds.get(t, 0)):               # holds freeze the story; ambience keeps moving
                arr = np.array(fn(t, g))
                assert arr.max() < TRANSP
                small.append(arr)
                if a.png and t == rep and h == 0:
                    Path(a.png).mkdir(parents=True, exist_ok=True)
                    up(arr).convert('RGB').save(Path(a.png) / f"racetrack_{fn.__name__.replace('scene_', '')}.png")
                if a.frames and g % a.every == 0:
                    Path(a.frames).mkdir(parents=True, exist_ok=True)
                    up(arr).convert('RGB').save(Path(a.frames) / f'f{g:03d}_{fn.__name__[6:]}_{t}.png')
                g += 1
    frames, prev = [], None
    for arr in small:
        b = arr.copy()
        if prev is not None:
            b[arr == prev] = TRANSP
        prev = arr
        frames.append(up(b))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=MS, loop=0, optimize=False, disposal=1,
                   transparency=TRANSP, background=0)
    print(f'{out}: {len(frames)} frames, {len(frames) * MS / 1000:.1f} s, {out.stat().st_size / 1e6:.2f} MB')


def up(arr):
    im = Image.fromarray(np.repeat(np.repeat(arr, SCALE, 0), SCALE, 1), 'P')
    im.putpalette(FLAT)
    return im


if __name__ == '__main__':
    main()
