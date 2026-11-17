"""Print artwork for the carded product: a made-up brand ("Brightcell" AA alkaline batteries), not a real
trademark. Rendered with Pillow from free fonts (Liberation: SIL OFL 1.1; DejaVu: Bitstream Vera licence) at
4 px/mm. Writes
  textures/card_front.png   the blister card's printed front, 100 x 180 mm (euro slot drawn by the mesh)
  textures/card_back.png    its back (instructions, barcode), 100 x 180 mm
  textures/label.png        the hook's scan-label price strip, 60 x 22 mm
  textures/shipper.png      the cardboard shipper's printed front, 140 x 60 mm
  textures/slat.png         one slatwall panel strip (white melamine slat with its groove), 200 x 76 mm

Run once (with Pillow and the fonts available):  python3 make_textures.py [font_dir]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OUT = HERE / "textures"
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/share/fonts/truetype")
PX = 4
ORANGE = (240, 120, 20)
DEEP = (18, 40, 92)
WHITE = (250, 250, 247)
INK = (24, 26, 30)
GOLD = (236, 190, 40)
KRAFT = (184, 146, 102)


def font(name, size):
    for p in FONTS.rglob(name):
        return ImageFont.truetype(str(p), size)
    raise FileNotFoundError(name)


def mm(v):
    return int(round(v * PX))


def barcode(d, x0, y0, w, h, seed=7):
    rnd = random.Random(seed)
    x = x0
    d.rectangle([x0 - 4, y0 - 4, x0 + w + 4, y0 + h + 22], fill=WHITE)
    while x < x0 + w:
        bw = rnd.choice((2, 2, 3, 4, 6))
        if rnd.random() < 0.55:
            d.rectangle([x, y0, x + bw - 1, y0 + h], fill=INK)
        x += bw
    d.text((x0, y0 + h + 2), "0 74592 31846 5", fill=INK, font=font("LiberationMono-Regular.ttf", 16))


def card_front():
    W, H = mm(100), mm(180)
    im = Image.new("RGB", (W, H), DEEP)
    d = ImageDraw.Draw(im)
    # orange header band with the hang-tab area, white logo
    d.rectangle([0, 0, W, mm(46)], fill=ORANGE)
    d.polygon([(0, mm(46)), (W, mm(38)), (W, mm(50)), (0, mm(58))], fill=GOLD)
    d.text((mm(8), mm(19)), "Brightcell", fill=WHITE, font=font("LiberationSans-BoldItalic.ttf", mm(13)))
    d.text((mm(9), mm(34)), "ALKALINE  POWER", fill=DEEP, font=font("DejaVuSans-Bold.ttf", mm(4.2)))
    # big AA / 4 PACK
    d.text((mm(5), mm(128)), "AA", fill=WHITE, font=font("LiberationSans-Bold.ttf", mm(30)))
    d.text((mm(52), mm(132)), "4 PACK", fill=GOLD, font=font("LiberationSans-Bold.ttf", mm(10)))
    d.text((mm(54), mm(146)), "1.5 V  LR6", fill=WHITE, font=font("DejaVuSans-Bold.ttf", mm(5)))
    d.rounded_rectangle([mm(6), mm(160), mm(94), mm(173)], radius=mm(3), fill=ORANGE)
    d.text((mm(11), mm(162.5)), "LONG LASTING  •  10 YR STORAGE", fill=WHITE, font=font("DejaVuSans-Bold.ttf", mm(3.6)))
    # window outline behind the blister (the batteries show through the bubble)
    d.rounded_rectangle([mm(13), mm(60), mm(87), mm(128)], radius=mm(5), outline=(70, 96, 150), width=mm(1.2),
                        fill=(30, 58, 120))
    im.save(OUT / "card_front.png")


def card_back():
    W, H = mm(100), mm(180)
    im = Image.new("RGB", (W, H), (248, 248, 244))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, mm(22)], fill=ORANGE)
    d.text((mm(22), mm(10)), "Brightcell", fill=WHITE, font=font("LiberationSans-BoldItalic.ttf", mm(9)))
    d.rectangle([0, mm(22), W, mm(25)], fill=DEEP)
    f = font("DejaVuSans.ttf", mm(2.6))
    d.text((mm(8), mm(26)), "Brightcell AA Alkaline Batteries", fill=INK, font=font("DejaVuSans-Bold.ttf", mm(3.4)))
    y = mm(34)
    for line in ("CAUTION: Do not recharge, disassemble,", "heat or dispose of in fire. Insert", "correctly (+/-). Keep away from children.",
                 "If swallowed, seek medical help at once.", "", "Best before: 12-2036", "Made in Thailand"):
        d.text((mm(8), y), line, fill=(70, 72, 78), font=f)
        y += mm(4.4)
    barcode(d, mm(22), mm(140), mm(56), mm(20))
    im.save(OUT / "card_back.png")


def label():
    W, H = mm(60), mm(22)
    im = Image.new("RGB", (W, H), (250, 236, 60))
    d = ImageDraw.Draw(im)
    d.text((mm(2), mm(1.5)), "BRIGHTCELL AA 4PK", fill=INK, font=font("DejaVuSans-Bold.ttf", mm(3.2)))
    d.text((mm(2), mm(7)), "$6.49", fill=(200, 20, 30), font=font("LiberationSans-Bold.ttf", mm(9)))
    rnd = random.Random(3)
    x = mm(34)
    while x < mm(58):
        bw = rnd.choice((2, 3, 4))
        if rnd.random() < 0.55:
            d.rectangle([x, mm(8), x + bw - 1, mm(19)], fill=INK)
        x += bw
    im.save(OUT / "label.png")


def shipper():
    W, H = mm(140), mm(60)
    im = Image.new("RGB", (W, H), KRAFT)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, mm(14)], fill=ORANGE)
    d.text((mm(4), mm(2.5)), "Brightcell", fill=WHITE, font=font("LiberationSans-BoldItalic.ttf", mm(8)))
    d.text((mm(4), mm(20)), "AA ALKALINE  4 PACK", fill=INK, font=font("DejaVuSans-Bold.ttf", mm(5)))
    d.text((mm(4), mm(30)), "12 CARDS  •  PEG DISPLAY", fill=INK, font=font("DejaVuSans-Bold.ttf", mm(4)))
    d.text((mm(4), mm(42)), "THIS SIDE UP  ↑", fill=(60, 40, 20), font=font("DejaVuSans-Bold.ttf", mm(4)))
    im.save(OUT / "shipper.png")


def cable_front():
    W, H = mm(100), mm(180)
    im = Image.new("RGB", (W, H), (245, 245, 242))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, mm(40)], fill=(20, 20, 24))
    d.text((mm(8), mm(18)), "ZipLink", fill=(80, 200, 255), font=font("LiberationSans-BoldItalic.ttf", mm(12)))
    d.text((mm(8), mm(140)), "USB-C CABLE", fill=INK, font=font("LiberationSans-Bold.ttf", mm(9)))
    d.text((mm(8), mm(153)), "3 ft / 0.9 m  \u2022  FAST CHARGE", fill=(70, 72, 78), font=font("DejaVuSans-Bold.ttf", mm(3.6)))
    im.save(OUT / "cable_front.png")


def slat():
    W, H = mm(200), mm(76)
    im = Image.new("RGB", (W, H), (238, 238, 234))
    d = ImageDraw.Draw(im)
    rnd = random.Random(11)
    for _ in range(1800):                          # faint melamine speckle
        x, y = rnd.randrange(W), rnd.randrange(H)
        c = 228 + rnd.randrange(14)
        d.point((x, y), fill=(c, c, c - 3))
    d.rectangle([0, mm(70), W, mm(76)], fill=(150, 152, 156))   # the groove's aluminium insert lip
    d.rectangle([0, mm(71.5), W, mm(74.5)], fill=(60, 62, 66))  # the dark slot
    im.save(OUT / "slat.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    card_front()
    card_back()
    label()
    shipper()
    cable_front()
    slat()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
