"""Detailed visual meshes (no collision) for the label maker, its cover, the tape cassettes and the carton.
Each returns tuples (name, points, triangles, colour, material, opacity) in the owning Part's frame."""
from __future__ import annotations

import math

import numpy as np

import labeler as L
import meshlib as M

DARK = (0.16, 0.17, 0.19)
SHELL = (0.86, 0.86, 0.84)
KEY = (0.95, 0.95, 0.94)
GLYPH = (0.20, 0.21, 0.24)
BLACK = (0.05, 0.05, 0.055)
WHITE = (0.95, 0.95, 0.94)
YELLOW = (0.96, 0.80, 0.12)
BLUE = (0.10, 0.32, 0.66)
GREEN = (0.20, 0.62, 0.30)
RED = (0.80, 0.14, 0.12)
LCD_BG = (0.60, 0.66, 0.56)
CARD_IN = (0.62, 0.48, 0.32)


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# -- label maker body ------------------------------------------------------------------------------
def labeler_vmeshes():
    cx, cy, cz = L.labeler_pos()
    zb = L.TOP - cz                      # desk
    zt = L.SHELL_TOP - cz                # top deck
    bx, by = L.BODY[0], L.BODY[1]
    V = []
    # lower shell: dark, rounded, on four rubber feet
    V.append(_vm("lower", M.rbox((bx, by, 0.024), 8.0, 64), DARK, "satin", offset=(0, 0, zb + 0.0025 + 0.012)))
    for k, (fx, fy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        V.append(_vm(f"foot{k}", M.revolve([(0.0, 0.0), (0.006, 0.0), (0.006, 0.0025), (0.0, 0.0025)], n=20, cap0=True,
                                           cap1=True), BLACK, "matte",
                     offset=(fx * (bx / 2 - 0.016), fy * (by / 2 - 0.016), zb)))
    # upper shell around the bay pocket (light grey), with a parting line above the lower shell
    zu0 = zb + 0.024
        # keyboard deck (front), back rim, right of bay, LCD side
    def deck(name, x0, x1, y0, y1, z0=zu0, z1=zt, e=10.0):
        return _vm(name, M.rbox((x1 - x0, y1 - y0, z1 - z0), e, 48), SHELL, "satin",
                   offset=((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy, (z0 + z1) / 2))
    rc = L.BAY_C[1]
    ry0, ry1 = rc - L.RECESS_W / 2, rc + L.RECESS_W / 2
    V.append(deck("deck_front", L.X0, L.RECESS_X[0] + 0.004, L.Y0, L.Y1))
    V.append(deck("deck_back", L.BAY_X[1], L.X1, L.Y0, L.Y1))
    V.append(deck("deck_right", L.RECESS_X[0], L.BAY_X[1] + 0.002, L.Y0, min(L.BAY_Y[0], ry0), e=6.0))
    V.append(deck("deck_left", L.RECESS_X[0], L.BAY_X[1] + 0.002, max(L.BAY_Y[1], ry1), L.Y1, e=6.0))
    V.append(deck("deck_bayfront", L.RECESS_X[1] - 0.002, L.BAY_X[0], ry0, ry1, z0=zt - 0.004, e=4.0))
    V.append(_vm("recess", M.rbox((L.RECESS_X[1] - L.RECESS_X[0], L.RECESS_W, 0.001), 6.0, 24), (0.12, 0.12, 0.13), "matte",
                 offset=((L.RECESS_X[0] + L.RECESS_X[1]) / 2 - cx, rc - cy, zt - L.RECESS_D)))
    V.append(_vm("parting", M.rbox((bx - 0.002, by - 0.002, 0.0012), 8.0, 64), (0.30, 0.31, 0.33), "matte",
                 offset=(0, 0, zu0 + 0.0002)))
    # the bay pocket: dark floor and walls, the print head block and the exit slot at its right back corner
    fz = L.FLOOR_Z - cz
    V.append(_vm("bay_floor", M.rbox((L.BAY_X[1] - L.BAY_X[0], L.BAY_Y[1] - L.BAY_Y[0], 0.001), 4.0, 16), (0.12, 0.12, 0.13),
                 "matte", offset=(np.mean(L.BAY_X) - cx, np.mean(L.BAY_Y) - cy, fz - 0.0004)))
    V.append(_vm("head", M.rbox(L.POST, 4.0, 16), (0.55, 0.56, 0.58), "steel",
                 offset=(L.POST_C[0] - cx, L.POST_C[1] - cy, fz + L.POST[2] / 2)))
    V.append(_vm("spool_post", M.revolve([(0.0, 0.0), (0.003, 0.0), (0.003, 0.010), (0.0, 0.010)], n=16, cap1=True),
                 (0.85, 0.85, 0.85), "satin", offset=(np.mean(L.BAY_X) - 0.010 - cx, np.mean(L.BAY_Y) + 0.012 - cy, fz)))
    V.append(_vm("exit_slot", M.rbox((0.004, 0.003, 0.016), 3.0, 12), BLACK, "matte",
                 offset=(L.BAY_X[1] - 0.006 - cx, L.Y0 - cy + 0.001, fz + 0.012)))
    # keyboard: a dark recess and 4 rows of 13 keys, each with a printed glyph; the bottom row a space bar
    kx0, kx1 = L.KEYS_X
    ky0, ky1 = L.KEYS_Y
    V.append(_vm("key_well", M.rbox((kx1 - kx0 + 0.004, ky1 - ky0 + 0.004, 0.0012), 6.0, 32), (0.35, 0.36, 0.38), "matte",
                 offset=((kx0 + kx1) / 2 - cx, (ky0 + ky1) / 2 - cy, zt + 0.0003)))
    cols, rows = 13, 5
    px, py = (kx1 - kx0) / rows, (ky1 - ky0) / cols
    kmesh = M.rbox((px - 0.0022, py - 0.0022, 0.0030), 5.0, 20)
    for r in range(rows):
        x = kx0 + px * (r + 0.5) - cx
        c = 0
        while c < cols:
            span = 5 if (r == 0 and c == 4) else 1          # front row: a space bar
            y = ky0 + py * (c + span / 2) - cy
            colour = KEY
            if r == 0 and c in (0, 12):
                colour = (0.80, 0.82, 0.86)
            mesh = M.rbox((px - 0.0022, py * span - 0.0022, 0.0030), 5.0, 20) if span > 1 else kmesh
            V.append(_vm(f"key{r}_{c}", mesh, colour, "satin", offset=(x, y, zt + 0.0024)))
            if span == 1:
                V.append(_vm(f"glyph{r}_{c}", M.rbox((0.0036, 0.0026, 0.0003), 4.0, 8), GLYPH, "matte",
                             offset=(x + 0.0005, y, zt + 0.0040)))
            c += span
    # LCD with its bezel and two lines of characters, and the round power / print keys beside it
    lx, ly = L.LCD_C[0] - cx, L.LCD_C[1] - cy
    V.append(_vm("lcd_bezel", M.rbox((L.LCD[0], L.LCD[1], 0.0016), 8.0, 32), (0.20, 0.21, 0.23), "satin",
                 offset=(lx, ly, zt + 0.0006)))
    V.append(_vm("lcd", M.rbox((L.LCD[0] - 0.012, L.LCD[1] - 0.008, 0.0004), 10.0, 32), LCD_BG, "gloss",
                 offset=(lx, ly, zt + 0.0016)))
    for line, dxl in enumerate((0.006, -0.004)):
        for k in range(7):
            V.append(_vm(f"char{line}_{k}", M.rbox((0.005, 0.0032, 0.0002), 3.0, 8), (0.10, 0.12, 0.10), "matte",
                         offset=(lx + dxl, ly - 0.015 + k * 0.0048, zt + 0.0020)))
    for k, (dy, col) in enumerate(((-0.012, RED), (0.012, GREEN))):
        V.append(_vm(f"btn{k}", M.revolve([(0.0, 0.0), (0.0055, 0.0), (0.0055, 0.0020), (0.0045, 0.0028), (0.0, 0.0028)],
                                          n=24, cap1=True), col, "satin",
                     offset=(L.LCD_C[0] + L.LCD[0] / 2 + 0.010 - cx, ly + dy, zt)))
    # tape cutter lever on the right back corner, by the exit slot
    V.append(_vm("cutter", M.rbox((0.012, 0.018, 0.006), 4.0, 16), (0.30, 0.55, 0.85), "satin",
                 offset=(L.X1 - 0.009 - cx, L.Y0 + 0.014 - cy, zt + 0.003)))
    return tuple(V)


def cover_vmeshes():
    """The hood cover, in its own frame (origin at the plate's centre): a light plate on 19 mm skirts, grip ribs and
    the push latch tab at the foot of its front skirt, a moulded cassette icon and arrow, hinge knuckles at the back."""
    sx, sy, sz = L.COVER_PLATE
    sk = L.COVER_H - sz
    shell = (0.83, 0.83, 0.81)
    V = [_vm("plate", M.rbox((sx, sy, sz), 10.0, 48), shell, "satin")]
    for k, (size, off) in enumerate((((0.003, sy, sk), (-sx / 2 + 0.0015, 0, -sz / 2 - sk / 2)),
                                     ((0.003, sy, sk), (sx / 2 - 0.0015, 0, -sz / 2 - sk / 2)),
                                     ((sx, 0.003, sk), (0, -sy / 2 + 0.0015, -sz / 2 - sk / 2)),
                                     ((sx, 0.003, sk), (0, sy / 2 - 0.0015, -sz / 2 - sk / 2)))):
        V.append(_vm(f"skirt{k}", M.rbox(size, 4.0, 12), shell, "satin", offset=off))
    for k in range(5):
        V.append(_vm(f"rib{k}", M.rbox((0.0016, 0.044, 0.0010), 4.0, 8), (0.62, 0.63, 0.64), "satin",
                     offset=(-sx / 2 + 0.006 + 0.0035 * k, 0.0, sz / 2 + 0.0004)))
    tab = np.array(L.TAB_C) - np.array(L.COVER_POS)
    V.append(_vm("tab", M.rbox(L.TAB, 4.0, 16), (0.25, 0.26, 0.28), "satin", offset=tuple(tab)))
    V.append(_vm("tab_grip", M.rbox((0.0008, 0.018, 0.003), 3.0, 8), (0.45, 0.46, 0.48), "matte",
                 offset=(tab[0] - L.TAB[0] / 2 - 0.0002, 0.0, tab[2])))
    V.append(_vm("icon", M.rbox((0.030, 0.040, 0.0004), 4.0, 16), (0.55, 0.56, 0.58), "matte",
                 offset=(0.012, 0.010, sz / 2 + 0.0002)))
    V.append(_vm("icon_win", M.rbox((0.014, 0.014, 0.0005), 12.0, 16), shell, "matte",
                 offset=(0.012, 0.002, sz / 2 + 0.0004)))
    arrow = (np.array([[0, 0, 0], [0.008, -0.006, 0], [0.008, 0.006, 0]]), np.array([[0, 1, 2]]))
    V.append(_vm("arrow", arrow, (0.40, 0.41, 0.43), "matte", offset=(-sx / 2 + 0.024, -0.034, sz / 2 + 0.0006)))
    for k, dy in enumerate((-0.036, 0.036)):
        V.append(_vm(f"knuckle{k}", M.revolve([(0.0, -0.007), (0.0025, -0.007), (0.0025, 0.007), (0.0, 0.007)], n=16,
                                              axis="Y", cap0=True, cap1=True), (0.40, 0.41, 0.43), "satin",
                     offset=(sx / 2 - 0.001, dy, L.HINGE_PIVOT[2] - L.COVER_POS[2])))
    return tuple(V)


# -- tape cassette -------------------------------------------------------------------------------------
def cassette_vmeshes(full):
    """88 x 66 x 26 mm laminated-tape cassette: black shell, a printed label, a clear window over the tape
    roll (full or nearly used up) and the ink ribbon spool, the tape's end out at its exit corner."""
    sx, sy, sz = L.CASSETTE
    n = L.NOTCH
    V = [_vm("shell", M.rbox((sx - n, sy, sz - 0.002), 8.0, 48), BLACK, "satin", offset=(-n / 2, 0, 0)),
         _vm("shell2", M.rbox((n + 0.004, sy - n, sz - 0.002), 6.0, 24), BLACK, "satin", offset=(sx / 2 - n / 2 - 0.002, n / 2, 0)),
         _vm("rim", M.rbox((sx - n - 0.002, sy - 0.002, 0.002), 8.0, 48), (0.12, 0.12, 0.13), "satin",
             offset=(-n / 2, 0, sz / 2 - 0.0015))]
    top = sz / 2
    # the label over the front half: yellow band with the tape width, two lines of print
    V.append(_vm("label", M.rbox((0.026, 0.074, 0.0003), 8.0, 32), YELLOW, "satin", offset=(-0.019, 0.0, top + 0.0002)))
    V.append(_vm("width", M.rbox((0.010, 0.016, 0.0003), 6.0, 16), BLACK, "matte", offset=(-0.020, 0.024, top + 0.0004)))
    for k, (w, dx) in enumerate(((0.046, -0.014), (0.034, -0.024))):
        V.append(_vm(f"print{k}", M.rbox((0.0025, w, 0.0002), 4.0, 8), (0.15, 0.15, 0.15), "matte",
                     offset=(dx, -0.008, top + 0.0005)))
    # clear window over the tape roll and the ribbon spool (drawn as thin discs just under the window, over the
    # shell's dark recess, so they read through it)
    wx, wy = 0.014, -0.012
    V.append(_vm("recess", M.rbox((0.034, 0.050, 0.0004), 12.0, 32), (0.10, 0.10, 0.11), "matte",
                 offset=(wx, wy, top + 0.0002)))
    r_tape = 0.0155 if full else 0.0075
    disc = lambda r0, r1: M.revolve([(r0, 0.0), (r1, 0.0), (r1, 0.0007), (r0, 0.0007)], n=40, cap0=True, cap1=True)
    V.append(_vm("tape_roll", disc(0.0045, r_tape), WHITE, "satin", offset=(wx, wy - 0.008, top + 0.0005)))
    V.append(_vm("tape_core", disc(0.0, 0.0045), (0.70, 0.70, 0.72), "satin", offset=(wx, wy - 0.008, top + 0.0006)))
    V.append(_vm("hub_hole", disc(0.0, 0.002), BLACK, "matte", offset=(wx, wy - 0.008, top + 0.0011)))
    V.append(_vm("ribbon", disc(0.003, 0.009 if full else 0.005), (0.03, 0.03, 0.03), "gloss",
                 offset=(wx + 0.004, wy + 0.016, top + 0.0005)))
    V.append(_vm("ribbon_hub", disc(0.0, 0.003), WHITE, "satin", offset=(wx + 0.004, wy + 0.016, top + 0.0006)))
    V.append(_vm("window", M.rbox((0.034, 0.050, 0.0005), 12.0, 32), (0.88, 0.91, 0.95), "acrylic", 0.22,
                 offset=(wx, wy, top + 0.0016)))
    # the tape's end out at the exit corner (+x, -y): the cassette goes in only this way round
    V.append(_vm("tape_end", M.rbox((0.0004, 0.010, 0.022), 2.0, 8), WHITE, "satin",
                 offset=(sx / 2 - n - 0.0003, -sy / 2 + n / 2, 0.0)))
    # everything above is centred on the cassette's middle; its Part reference is CASS_OFF toward -x of that
    return tuple((nm, p + np.array([L.CASS_OFF, 0.0, 0.0]), t, c, m, o) for nm, p, t, c, m, o in V)


# -- carton -------------------------------------------------------------------------------------------
def carton_vmeshes():
    """Open printed carton: white printed outside with a blue band, brown inside, its top flap folded back
    showing a printed tape sample and a picture of the cassette."""
    sx, sy, sz = L.CARTON
    t = L.CARD_T
    V = []
    zc = sz / 2 - t / 2
    for k, (size, off) in enumerate((((t, sy, sz), (-sx / 2 + t / 2, 0, zc)), ((t, sy, sz), (sx / 2 - t / 2, 0, zc)),
                                     ((sx, t, sz), (0, -sy / 2 + t / 2, zc)), ((sx, t, sz), (0, sy / 2 - t / 2, zc)))):
        V.append(_vm(f"wall{k}", M.rbox(size, 3.0, 8), WHITE, "matte", offset=off))
    V.append(_vm("floor", M.rbox((sx, sy, t), 3.0, 8), CARD_IN, "matte"))
    for k, (size, off) in enumerate((((0.0004, sy - 0.004, 0.008), (-sx / 2 - 0.0003, 0, 0.010)),
                                     ((sx - 0.004, 0.0004, 0.008), (0, -sy / 2 - 0.0003, 0.010)))):
        V.append(_vm(f"band{k}", M.rbox(size, 3.0, 8), BLUE, "matte", offset=off))
    # the top flap, hinged along the +x top edge and folded back past upright
    R = _rot_y(math.radians(105))
    hinge = np.array([sx / 2, 0.0, sz - t / 2])
    def on_flap(name, size, along, across, color, lift=0.0):
        # a rectangle laid on the flap: ``along`` measured from the hinge across the flap, ``across`` along y
        p = M.rbox(size, 3.0, 8)
        local = np.array([-along, across, t / 2 + lift])
        return _vm(name, p, color, "matte", offset=hinge + R @ local, R=R)
    V.append(on_flap("flap", (sx, sy, t), sx / 2, 0.0, WHITE))
    V.append(on_flap("flap_band", (0.014, sy - 0.006, 0.0003), 0.012, 0.0, BLUE, 0.0007))
    V.append(on_flap("sample", (0.016, 0.070, 0.0003), 0.034, -0.004, (0.99, 0.99, 0.98), 0.0007))
    for k in range(6):
        V.append(on_flap(f"sample_txt{k}", (0.006, 0.006, 0.0002), 0.034, -0.032 + 0.0105 * k, (0.08, 0.08, 0.08), 0.0011))
    V.append(on_flap("pic", (0.020, 0.026, 0.0003), 0.060, 0.026, (0.10, 0.10, 0.11), 0.0007))
    V.append(on_flap("pic_label", (0.008, 0.022, 0.0003), 0.055, 0.026, YELLOW, 0.0010))
    V.append(on_flap("size_badge", (0.012, 0.016, 0.0003), 0.060, -0.030, YELLOW, 0.0007))
    return tuple(V)


def label_strip_vmeshes():
    """A 24 mm label just printed: white laminated tape with a line of black characters, rounded ends."""
    sx, sy, sz = L.LABEL_STRIP
    V = [_vm("tape", M.rbox((sx, sy, sz), 6.0, 24), (0.98, 0.98, 0.97), "gloss")]
    for k in range(8):
        w = 0.0045 if k % 3 else 0.0060
        V.append(_vm(f"ch{k}", M.rbox((0.012, w * 1.6, 0.0002), 3.0, 8), (0.05, 0.05, 0.05), "matte",
                     offset=(0.0, -0.036 + 0.0102 * k, sz / 2 + 0.0002)))
    return tuple(V)
