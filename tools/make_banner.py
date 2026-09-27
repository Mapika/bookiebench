"""Render the README banner (docs/assets/bookiebench.gif): a pixel-art explainer of what BookieBench checks.

    python tools/make_banner.py [--out docs/assets/bookiebench.gif]

Needs Pillow only. Everything is drawn on a 200x112 canvas and scaled 4x with nearest-neighbour.
"""
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

W, H, SCALE, MS = 200, 112, 4, 80   # canvas, upscale, ms per frame

# PICO-8 palette
PAL = dict(black=(0, 0, 0), navy=(29, 43, 83), plum=(126, 37, 83), dgreen=(0, 135, 81), brown=(171, 82, 54),
           dgray=(95, 87, 79), lgray=(194, 195, 199), white=(255, 241, 232), red=(255, 0, 77), orange=(255, 163, 0),
           yellow=(255, 236, 39), green=(0, 228, 54), blue=(41, 173, 255), lav=(131, 118, 156), pink=(255, 119, 168),
           peach=(255, 204, 170))

GLYPHS = {
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
    '>': "100010001010100", '<': "001010100010001", ',': "000000000010100", ' ': "000000000000000",
}


def text_w(s, k=1):
    return (4 * len(s) - 1) * k


def text(d, x, y, s, c, k=1, shadow=None):
    if x == 'c':
        x = (W - text_w(s, k)) // 2
    passes = ([(max(1, k // 2), shadow)] if shadow else []) + [(0, c)]   # shadow first, so it never covers the letters
    for off, col in passes:
        for i, ch in enumerate(s.upper()):
            g = GLYPHS.get(ch, GLYPHS['?'])
            for r in range(5):
                for q in range(3):
                    if g[r * 3 + q] == '1':
                        px, py = x + (i * 4 + q) * k + off, y + r * k + off
                        d.rectangle([px, py, px + k - 1, py + k - 1], fill=PAL[col])


def box(d, x0, y0, x1, y1, c, edge=None):
    d.rectangle([x0, y0, x1, y1], fill=PAL[c], outline=PAL[edge] if edge else None)


def backdrop(d, t):
    box(d, 0, 0, W, H, 'navy')
    for i in range(14):                                   # twinkling stars
        x, y = (i * 53 + 7) % W, (i * 29 + 5) % 60 + 12
        if (t // 4 + i) % 5:
            d.point((x, y), fill=PAL['lav'])
    box(d, 0, 84, W, H, 'dgray')                          # floor
    for x in range(0, W, 8):
        d.point((x + (4 if (x // 8) % 2 else 0), 86), fill=PAL['black'])


def caption(d, s, c='white'):
    box(d, 0, 0, W, 9, 'black')
    text(d, 'c', 2, s, c)


def urn(d, x, y):
    box(d, x + 3, y, x + 23, y + 3, 'brown', 'black')      # rim
    box(d, x, y + 4, x + 26, y + 30, 'peach', 'black')     # jar
    balls = ['red', 'blue', 'red', 'blue', 'red']
    for i, c in enumerate(balls):
        bx, by = x + 4 + (i % 3) * 7, y + 18 - (i // 3) * 7
        d.ellipse([bx, by, bx + 5, by + 5], fill=PAL[c], outline=PAL['black'])
    text(d, x - 1, y + 34, '3 RED', 'red')
    text(d, x - 1, y + 41, '2 BLUE', 'blue')


def robot(d, x, y, body='lgray', mood='ok', t=0):
    bob = (t // 6) % 2
    y += bob
    d.line([x + 8, y - 5, x + 8, y], fill=PAL['dgray'])
    d.point((x + 8, y - 6), fill=PAL['red' if (t // 5) % 2 else 'orange'])
    box(d, x, y, x + 16, y + 11, body, 'black')            # head
    eye = 'blue' if mood == 'ok' else 'red'
    box(d, x + 3, y + 3, x + 5, y + 5, eye)
    box(d, x + 11, y + 3, x + 13, y + 5, eye)
    if mood == 'ok':
        d.line([x + 5, y + 8, x + 11, y + 8], fill=PAL['black'])
    else:
        d.line([x + 5, y + 9, x + 11, y + 7], fill=PAL['black'])
    box(d, x + 2, y + 12, x + 14, y + 26 - bob, body, 'black')   # body
    box(d, x + 6, y + 15, x + 10, y + 19, 'dgray')
    box(d, x + 3, y + 27 - bob, x + 6, y + 30 - bob, 'dgray')
    box(d, x + 10, y + 27 - bob, x + 13, y + 30 - bob, 'dgray')


def bookie(d, x, y, mood='grin', t=0):
    box(d, x + 1, y - 2, x + 17, y, 'black')               # hat brim
    box(d, x + 4, y - 11, x + 14, y - 2, 'black')          # hat
    box(d, x + 4, y - 4, x + 14, y - 3, 'plum')            # hat band
    box(d, x + 3, y + 1, x + 15, y + 12, 'peach', 'black')  # face
    box(d, x + 6, y + 4, x + 7, y + 5, 'black')
    box(d, x + 11, y + 4, x + 12, y + 5, 'black')
    box(d, x + 5, y + 8, x + 13, y + 9, 'brown')           # moustache
    if mood == 'grin':
        d.line([x + 7, y + 11, x + 11, y + 11], fill=PAL['red'])
        if (t // 3) % 2:                                    # cigar glow
            d.point((x + 17, y + 10), fill=PAL['orange'])
        d.line([x + 13, y + 10, x + 16, y + 10], fill=PAL['white'])
    else:
        d.line([x + 7, y + 11, x + 8, y + 10], fill=PAL['black'])
        d.line([x + 9, y + 10, x + 11, y + 11], fill=PAL['black'])
        if (t // 4) % 2:                                    # sweat drop
            d.point((x + 2, y + 5 + (t // 4) % 3), fill=PAL['blue'])
    box(d, x + 1, y + 13, x + 17, y + 28, 'dgreen', 'black')  # suit
    box(d, x + 7, y + 13, x + 11, y + 15, 'red')           # bow tie
    box(d, x + 3, y + 29, x + 7, y + 31, 'black')
    box(d, x + 11, y + 29, x + 15, y + 31, 'black')


def bubble(d, x, y, lines, c='black'):
    w = max(text_w(s) for s, _ in lines) + 6
    h = 7 * len(lines) + 4
    box(d, x, y, x + w, y + h, 'white', 'black')
    d.polygon([(x + 6, y + h), (x + 10, y + h), (x + 6, y + h + 4)], fill=PAL['white'], outline=PAL['black'])
    for i, (s, col) in enumerate(lines):
        text(d, x + 3, y + 3 + 7 * i, s, col)


def price_bar(d, a, b, y=92, grow=1.0):
    """Stacked bar of two prices (fractions) against a 100% marker."""
    x0, full = 34, 100
    wa, wb = int(full * a * grow), int(full * b * grow)
    box(d, x0 - 1, y - 1, x0 + int(full * 1.3), y + 7, 'black')
    box(d, x0, y, x0 + wa, y + 6, 'red')
    box(d, x0 + wa, y, x0 + wa + wb, y + 6, 'blue')
    d.line([x0 + full, y - 3, x0 + full, y + 9], fill=PAL['yellow'])
    text(d, x0 + full - 5, y + 10 - 20, '100%', 'yellow')
    tot = round((a + b) * 100 * grow)
    text(d, x0 + int(full * 1.3) + 4, y + 1, f'{tot}%', 'red' if tot > 100 else 'green')


def coin(d, x, y):
    d.ellipse([x, y, x + 4, y + 4], fill=PAL['yellow'], outline=PAL['orange'])


def scene_title(t, n):
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    backdrop(d, t)
    drop = max(0, 30 - t * 3)
    text(d, 'c', 22 - drop, 'BOOKIEBENCH', 'yellow', k=3, shadow='plum')
    if t > 8:
        text(d, 'c', 48, "CAN A MODEL'S", 'white')
        text(d, 'c', 56, 'PROBABILITIES BE TRUSTED?', 'white')
    if t > 14:
        bookie(d, 150, 54, 'grin', t)
        robot(d, 30, 54, t=t)
    return im


def scene_question(t, n):
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    backdrop(d, t)
    caption(d, 'ASK FOR LINKED PROBABILITIES')
    urn(d, 14, 22)
    robot(d, 82, 52, t=t)
    bookie(d, 160, 52, 'grin', t)
    if t > 6:
        lines = [('P(RED)     = 70%', 'red')]
        if t > 14:
            lines.append(('P(NOT RED) = 50%', 'blue'))
        bubble(d, 72, 20, lines)
    if t > 22:
        price_bar(d, .7, .5, grow=min(1.0, (t - 22) / 8))
    return im


def scene_book(t, n):
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    backdrop(d, t)
    caption(d, 'THE BOOKIE SELLS YOU BOTH BETS', 'orange')
    robot(d, 40, 52, mood='sad', t=t)
    bookie(d, 150, 52, 'grin', t)
    for i in range(6):                                     # coins arc from robot to bookie
        p = ((t - i * 3) % 18) / 18
        if t >= i * 3:
            cx = 58 + p * 90
            cy = 50 - math.sin(p * math.pi) * 24
            coin(d, int(cx), int(cy))
    text(d, 70, 64, 'PAY  0.70+0.50', 'white')
    text(d, 70, 71, 'WIN  1.00', 'white')
    if t > 14:
        text(d, 'c', 90, 'RED OR BLUE: BOOKIE +0.20', 'yellow')
    if t > 24 and (t // 3) % 2:
        text(d, 'c', 14, 'DUTCH BOOK!', 'red', k=2, shadow='black')
    return im


def scene_coherent(t, n):
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    backdrop(d, t)
    caption(d, 'A COHERENT, CALIBRATED MODEL', 'green')
    urn(d, 14, 22)
    robot(d, 82, 52, body='green', t=t)
    bookie(d, 160, 52, 'sweat', t)
    lines = [('P(RED)     = 60%', 'red')]
    if t > 5:
        lines.append(('P(NOT RED) = 40%', 'blue'))
    bubble(d, 58, 20, lines)
    if t > 10:
        price_bar(d, .6, .4, grow=min(1.0, (t - 10) / 8))
    if t > 20:
        bubble(d, 134, 18, [('NO SURE BET', 'dgray'), ('...DAMN', 'dgray')])
    if t > 28:
        text(d, 'c', 104, 'EXACT ANSWER: 3/5 = 60%', 'green')
    return im


PILLARS = [('COHERENCE', 'NO DUTCH BOOKS'), ('CALIBRATION', '70% MEANS 70%'),
           ('SKILL', 'VS THE EXACT POSTERIOR'), ('SENSITIVITY', 'MOVES WITH EVIDENCE')]


def scene_pillars(t, n):
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    backdrop(d, t)
    caption(d, 'FOUR CHECKS, EXACT ANSWERS', 'yellow')
    for i, (a, b) in enumerate(PILLARS):
        if t > i * 7:
            y = 18 + i * 15
            box(d, 12, y, 20, y + 8, 'black', 'lgray')
            if t > i * 7 + 4:
                d.line([14, y + 4, 15, y + 6], fill=PAL['green'])
                d.line([15, y + 6, 18, y + 2], fill=PAL['green'])
            text(d, 26, y + 2, a, 'yellow')
            text(d, 26 + text_w(a) + 6, y + 2, b, 'white')
    if t > 30:
        text(d, 'c', 88, '37 SIMULATED WORLDS, 25 REAL SOURCES', 'lav')
        text(d, 'c', 96, '21 STRESS TRANSFORMS', 'lav')
    return im


def scene_end(t, n):
    im = Image.new('RGB', (W, H)); d = ImageDraw.Draw(im)
    backdrop(d, t)
    text(d, 'c', 24, 'BOOKIEBENCH', 'yellow', k=3, shadow='plum')
    text(d, 'c', 50, 'IF THE BOOKIE CAN PROFIT,', 'white')
    text(d, 'c', 58, 'THE PROBABILITIES LIE.', 'red')
    robot(d, 30, 54, body='green', t=t)
    bookie(d, 150, 54, 'sweat', t)
    return im


SCENES = [(scene_title, 26), (scene_question, 44), (scene_book, 40), (scene_coherent, 44), (scene_pillars, 50),
          (scene_end, 30)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/assets/bookiebench.gif')
    ap.add_argument('--png', help='also write every Nth frame as PNG into this dir (preview)')
    a = ap.parse_args()
    frames = []
    for fn, n in SCENES:
        for t in range(n):
            frames.append(fn(t, n).resize((W * SCALE, H * SCALE), Image.NEAREST))
    pal = Image.new('P', (1, 1))
    flat = [v for c in PAL.values() for v in c]
    pal.putpalette(flat + [0] * (768 - len(flat)))
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(out, save_all=True, append_images=q[1:], duration=MS, loop=0, optimize=True, disposal=1)
    if a.png:
        Path(a.png).mkdir(parents=True, exist_ok=True)
        start = 0
        for fn, n in SCENES:
            frames[start + n - 1].save(Path(a.png) / f'{fn.__name__}.png')
            start += n
    print(f'{out}: {len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
