"""Print artwork for the bag-in-box syrup cartons: a made-up soft-drink brand ("Velmora"), not a real
trademark. Rendered with Pillow from free fonts (Liberation: SIL OFL 1.1; DejaVu: Bitstream Vera licence)
at 4 px/mm onto white clay-coated board. Writes textures/carton_front.png, carton_side.png.

Run once (with Pillow and the fonts available):  python3 make_textures.py [font_dir]
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/share/fonts/truetype")
PX = 4                                  # px per mm
W, H = 273 * PX, 394 * PX               # one carton face
RED = (176, 18, 28)
DARK_RED = (120, 8, 16)
WHITE = (246, 244, 238)
INK = (28, 24, 24)
GREY = (110, 106, 104)
GOLD = (232, 182, 40)


def font(name, size):
    for p in FONTS.rglob(name):
        return ImageFont.truetype(str(p), size)
    raise FileNotFoundError(name)


SERIF_BI = lambda s: font("LiberationSerif-BoldItalic.ttf", s)
SANS_B = lambda s: font("LiberationSans-Bold.ttf", s)
SANS = lambda s: font("LiberationSans-Regular.ttf", s)
NARROW_B = lambda s: font("LiberationSansNarrow-Bold.ttf", s)
MONO_B = lambda s: font("DejaVuSansMono-Bold.ttf", s)
SYMBOL = lambda s: font("DejaVuSans.ttf", s)


def centred(d, xy, text, f, fill, anchor="mm"):
    d.text(xy, text, font=f, fill=fill, anchor=anchor)


def ribbon(d, y0, amp, thick, fill, x0=0, x1=W):
    """A wave ribbon (swoosh) across the face."""
    top, bot = [], []
    for k in range(0, 101):
        x = x0 + (x1 - x0) * k / 100
        y = y0 + amp * math.sin(math.pi * 1.6 * k / 100 - 0.5)
        t = thick * (0.55 + 0.45 * math.sin(math.pi * k / 100))
        top.append((x, y - t / 2))
        bot.append((x, y + t / 2))
    d.polygon(top + bot[::-1], fill=fill)


def brand_block(img, d, y_top, height):
    """Red brand panel with the wordmark, ribbon and product line."""
    d.rectangle([0, y_top, W, y_top + height], fill=RED)
    ribbon(d, y_top + height * 0.78, 34, 46, WHITE)
    ribbon(d, y_top + height * 0.78 + 30, 30, 14, GOLD)
    centred(d, (W / 2, y_top + height * 0.38), "Velmora", SERIF_BI(int(height * 0.40)), WHITE)
    centred(d, (W / 2 + 250, y_top + height * 0.13), "SINCE 1931", SANS_B(30), (250, 220, 200))


def this_side_up(d, x, y, s=1.0):
    for dx in (-26, 26):
        cx = x + dx * s
        d.polygon([(cx, y - 40 * s), (cx - 18 * s, y - 14 * s), (cx - 7 * s, y - 14 * s), (cx - 7 * s, y + 30 * s),
                   (cx + 7 * s, y + 30 * s), (cx + 7 * s, y - 14 * s), (cx + 18 * s, y - 14 * s)], fill=INK)
    d.rectangle([x - 46 * s, y + 36 * s, x + 46 * s, y + 40 * s], fill=INK)
    centred(d, (x, y + 62 * s), "THIS SIDE UP", SANS_B(int(20 * s)), INK)


def recycle(d, x, y, size=70):
    centred(d, (x, y), "♻", SYMBOL(size), INK)
    centred(d, (x, y + size * 0.72), "CORRUGATED", SANS_B(16), INK)
    centred(d, (x, y + size * 0.72 + 20), "RECYCLE", SANS_B(16), INK)


def mix_badge(d, x, y, r=92):
    d.ellipse([x - r, y - r, x + r, y + r], fill=WHITE, outline=RED, width=10)
    centred(d, (x, y - 8), "5+1", SANS_B(70), RED)
    centred(d, (x, y + 50), "MIX RATIO", SANS_B(20), INK)


def front():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    brand_block(img, d, 0, int(H * 0.40))
    y = int(H * 0.40) + 70
    centred(d, (W / 2, y), "Cola · Syrup Concentrate", SANS_B(56), INK)
    centred(d, (W / 2, y + 66), "FOR FOUNTAIN USE ONLY • DO NOT DRINK UNDILUTED", SANS(24), GREY)
    d.rectangle([60, y + 108, W - 60, y + 112], fill=RED)
    centred(d, (W / 2 - 150, y + 190), "Bag-in-Box", SANS_B(54), INK)
    centred(d, (W / 2 - 150, y + 250), "5 gal (18.9 L)", SANS_B(48), RED)
    mix_badge(d, W / 2 + 270, y + 220)
    # knock-out: dashed circle with directions (centre 65 mm above the base)
    cx, cy, r = W / 2, H - 65 * PX, 50 * PX
    for k in range(24):
        a0, a1 = 360 * k / 24, 360 * k / 24 + 9
        d.arc([cx - r, cy - r, cx + r, cy + r], a0, a1, fill=INK, width=6)
    centred(d, (cx, cy - r - 34), "PUSH IN • PULL FITTING OUT", NARROW_B(30), INK)
    this_side_up(d, 110, H - 190, 1.2)
    recycle(d, W - 110, H - 230)
    centred(d, (W - 110, H - 60), "18.9 L", SANS_B(26), GREY)
    return img


def side():
    img = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(img)
    brand_block(img, d, 0, int(H * 0.26))
    y = int(H * 0.26) + 50
    centred(d, (W / 2, y), "Cola · Syrup Concentrate • Bag-in-Box 5 gal (18.9 L)", SANS_B(30), INK)
    # nutrition / ingredients panel
    x0, y0 = 50, y + 50
    d.rectangle([x0, y0, x0 + 520, y0 + 560], outline=INK, width=4)
    d.text((x0 + 16, y0 + 10), "Nutrition Facts", font=SANS_B(46), fill=INK)
    d.text((x0 + 16, y0 + 64), "Serving size 12 fl oz (prepared 5+1)", font=SANS(22), fill=INK)
    d.rectangle([x0 + 12, y0 + 98, x0 + 508, y0 + 112], fill=INK)
    rows = [("Calories", "150"), ("Total Fat", "0 g"), ("Sodium", "45 mg"), ("Total Carbohydrate", "41 g"),
            ("  Total Sugars", "41 g"), ("Protein", "0 g"), ("Caffeine", "34 mg")]
    for k, (a, b) in enumerate(rows):
        yy = y0 + 124 + k * 46
        d.text((x0 + 16, yy), a, font=SANS_B(26) if k == 0 else SANS(24), fill=INK)
        d.text((x0 + 500, yy), b, font=SANS_B(26) if k == 0 else SANS(24), fill=INK, anchor="ra")
        d.line([x0 + 12, yy + 38, x0 + 508, yy + 38], fill=INK, width=2)
    d.text((x0 + 16, y0 + 470), "INGREDIENTS: CARBONATED WATER (ADDED AT", font=SANS(16), fill=INK)
    d.text((x0 + 16, y0 + 494), "FOUNTAIN), HIGH FRUCTOSE CORN SYRUP, CARAMEL", font=SANS(16), fill=INK)
    d.text((x0 + 16, y0 + 518), "COLOR, PHOSPHORIC ACID, NATURAL FLAVORS, CAFFEINE.", font=SANS(16), fill=INK)
    # barcode and code
    rnd = random.Random(7)
    bx, by = x0 + 590, y0 + 30
    x = bx
    while x < bx + 360:
        w = rnd.choice((3, 3, 6, 9))
        d.rectangle([x, by, x + w - 1, by + 200], fill=INK)
        x += w + rnd.choice((3, 6))
    d.text((bx + 180, by + 220), "0 48210 31187 6", font=MONO_B(30), fill=INK, anchor="ma")
    # lot / expiry stamp (dot-matrix style, slightly askew grey ink)
    stamp = Image.new("RGBA", (420, 120), (0, 0, 0, 0))
    sd = ImageDraw.Draw(stamp)
    sd.text((6, 6), "LOT 26318 B2 04:17", font=MONO_B(30), fill=(40, 40, 60, 230))
    sd.text((6, 52), "BEST BY 05/2027", font=MONO_B(30), fill=(40, 40, 60, 230))
    stamp = stamp.rotate(2.5, expand=True)
    img.paste(stamp, (bx - 10, by + 300), stamp)
    this_side_up(d, bx + 90, H - 230, 1.2)
    recycle(d, bx + 290, H - 250)
    d.text((x0, H - 70), "Keep cool and dry. Store at 40–85 °F. Use within 6 months.", font=SANS(22), fill=GREY)
    return img


def main():
    OUT.mkdir(exist_ok=True)
    front().save(OUT / "carton_front.png", optimize=True)
    side().save(OUT / "carton_side.png", optimize=True)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
