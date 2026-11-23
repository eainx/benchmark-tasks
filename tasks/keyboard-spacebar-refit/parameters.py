"""Geometry and measurements for refitting a mechanical keyboard's spacebar onto its switch and both
stabilizer stems.

A standard ANSI mechanical keyboard with Cherry-MX-style switches and plate-mounted wire stabilizers.
The 6.25u spacebar (118.5 x 18.5 mm, 8 g) has been pulled off for cleaning and lies keycap-up on the desk
beside the keyboard. To go back on, its three cross sockets -- the switch's in the middle and the two
stabilizer sliders' 50 mm either side -- must all come down over their cross stems together, square and
level, and be pressed home; the two stabilizer sliders ride on one wire. Seated, the key must still
travel and spring back up.

Frame: metres, x away from the arm bases (the keyboard's back), y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece, Spring

TOP = 0.7475

CASE = (0.20, 0.20, 0.22)              # dark aluminium case
PLATE = (0.55, 0.56, 0.58)             # steel plate
HOUSING = (0.12, 0.12, 0.13)           # black switch housings
CAP = (0.90, 0.89, 0.86)               # PBT alpha keycaps
MOD = (0.55, 0.56, 0.60)               # grey modifiers
STEM_COL = (0.75, 0.10, 0.10)          # red (linear) stem
STAB_COL = (0.94, 0.94, 0.92)

U = 0.01905                            # 1u key pitch
# -- the keyboard (fixed): plate, housings, and the bottom two rows round the spacebar --------------
KX, KY = 0.300, -0.060                 # the spacebar's centre over the board
PLATE_Z = 0.020                        # plate top above the bench (case + standoffs)
HOUSING_TOP = PLATE_Z + 0.0066         # MX housing top 6.6 mm above the plate
CAP_BASE, CAP_TOP_W, CAP_H = 0.0182, 0.0127, 0.0095   # 1u keycap: base, top width, height
SKIRT_GAP = 0.0055                     # keycap skirt bottoms above the plate, keys at rest

# -- stems (MX cross) -----------------------------------------------------------------------------------
ARM_L, ARM_W = 0.0040, 0.00117         # cross arms: 4.0 x 1.17 mm
COLUMN = 0.0040                        # the slider's column above the housing top, key at rest (= travel)
CROSS_H = 0.0036                       # the cross on top of it
TRAVEL = 0.0040
SWITCH_K, SWITCH_PRE = 112.0, 0.0040   # linear switch: 0.45 N at rest to 0.9 N bottomed
STAB_K, STAB_PRE = 40.0, 0.0040
STAB_Y = 0.050                         # 6.25u stabilizer spacing: 100 mm between sliders

# -- the spacebar's sockets ------------------------------------------------------------------------------
BOSS = 0.0056                          # socket boss, square stand-in for the 5.6 mm round boss
SLOT = 0.00133                         # cross slot width (0.08 mm play each side)
BOSS_H = 0.0035                        # socket depth: its bottom meets the column top when seated
TOP_T = 0.0080                         # the spacebar's top: a thick 8 mm block (a tall spacebar, 3 mm proud of its neighbours)
SB_L, SB_W, SB_TOP_W = 0.1185, 0.0182, 0.0140   # 6.25u spacebar: length, base depth, top depth
SEATED = 0.0025                        # a socket counts as on with its cross at least 2.5 mm into the 3.5 mm socket (a real MX cap holds from about 2 mm)
SB_SKIRT = HOUSING_TOP + COLUMN + CROSS_H - PLATE_Z - SKIRT_GAP   # spacebar skirt below its top (8.7 mm)
SB_H = SB_SKIRT + TOP_T                # spacebar height (10.2 mm)
TIP_Z = HOUSING_TOP + COLUMN + CROSS_H # cross tops above the bench, key at rest
SEATED_TOP = TIP_Z + TOP_T             # the spacebar's top face above the bench, seated at rest
CAP0 = (0.200, -0.060)                 # the spacebar lying keycap-up on the desk in front of the keyboard


def _c(x, y, z):
    return (float(KX + x), float(KY + y), float(TOP + z))


def _keycap(x, y, w_units, color, collide=True, depth_units=1.0):
    """A sculpted keycap at rest on its switch (fixed), as two welded blocks: skirt and narrower top."""
    w = w_units * U - 0.0009
    d = depth_units * U - 0.0009
    z0 = PLATE_Z + SKIRT_GAP
    return (Piece("box", (d, w, CAP_H * 0.55), (x, y, z0 + CAP_H * 0.275), color=color, collide=collide),
            Piece("box", (d - (CAP_BASE - CAP_TOP_W), w - (CAP_BASE - CAP_TOP_W), CAP_H * 0.45),
                  (x - 0.0006, y, z0 + CAP_H * 0.775), color=color, collide=collide))


# The ANSI 104-key layout, row by row from the function row to the bottom row: (label, width in u) with
# None for a gap. Rows run left to right (+y to -y here); "SPACE" is the free spacebar, not built in.
ALPHA, MODS = CAP, MOD
_MODS = {"Esc", "Tab", "Caps", "LShift", "RShift", "LCtrl", "RCtrl", "LWin", "RWin", "LAlt", "RAlt", "Menu",
         "Bksp", "Enter", "PrtSc", "ScrLk", "Pause", "Ins", "Home", "PgUp", "Del", "End", "PgDn", "NumLk",
         "Up", "Left", "Down", "Right", "KPEnter", "KP+", "KP/", "KP*", "KP-"}
LAYOUT = (
    (5.25, [("Esc", 1), (None, 1), *[(f"F{i}", 1) for i in range(1, 5)], (None, 0.5), *[(f"F{i}", 1) for i in range(5, 9)],
            (None, 0.5), *[(f"F{i}", 1) for i in range(9, 13)], (None, 0.25), ("PrtSc", 1), ("ScrLk", 1), ("Pause", 1)]),
    (4.0, [*[(k, 1) for k in "`1234567890-="], ("Bksp", 2), (None, 0.25), ("Ins", 1), ("Home", 1), ("PgUp", 1),
           (None, 0.25), ("NumLk", 1), ("KP/", 1), ("KP*", 1), ("KP-", 1)]),
    (3.0, [("Tab", 1.5), *[(k, 1) for k in "QWERTYUIOP[]"], ("\\", 1.5), (None, 0.25), ("Del", 1), ("End", 1),
           ("PgDn", 1), (None, 0.25), ("KP7", 1), ("KP8", 1), ("KP9", 1), ("KP+", 1, 2)]),
    (2.0, [("Caps", 1.75), *[(k, 1) for k in "ASDFGHJKL;'"], ("Enter", 2.25), (None, 3.5), ("KP4", 1), ("KP5", 1),
           ("KP6", 1)]),
    (1.0, [("LShift", 2.25), *[(k, 1) for k in "ZXCVBNM,./"], ("RShift", 2.75), (None, 1.25), ("Up", 1), (None, 1.25),
           ("KP1", 1), ("KP2", 1), ("KP3", 1), ("KPEnter", 1, 2)]),
    (0.0, [("LCtrl", 1.25), ("LWin", 1.25), ("LAlt", 1.25), ("SPACE", 6.25), ("RAlt", 1.25), ("RWin", 1.25),
           ("Menu", 1.25), ("RCtrl", 1.25), (None, 0.25), ("Left", 1), ("Down", 1), ("Right", 1), (None, 0.25),
           ("KP0", 2), ("KP.", 1)]),
)
SPACE_FROM_LEFT = 3 * 1.25 + 6.25 / 2   # the spacebar's centre from the main block's left edge, in u
ROW_GAP = 0.002                        # the rows above the spacebar set 2 mm further back (a slightly roomier board)
SIDE_GAP = 0.00075                     # and the keys either side of the spacebar 0.75 mm further out
LEFT_Y = 0.0 + SPACE_FROM_LEFT * U      # the main block's left edge, from the spacebar's centre (+y)
WIDTH_U = 22.5


def _keys():
    """(label, x, y, width_u, depth_u) of every fixed keycap, relative to the spacebar's centre."""
    out = []
    for row_x, keys in LAYOUT:
        o = 0.0
        for k in keys:
            label, w = k[0], k[1]
            tall = k[2] if len(k) > 2 else 1
            if label is not None and label != "SPACE":
                x = row_x * U - (tall - 1) * U / 2 + (ROW_GAP if row_x > 0 else 0.0)
                y = LEFT_Y - (o + w / 2) * U
                if row_x == 0.0 and label in ("LCtrl", "LWin", "LAlt"):
                    y += SIDE_GAP
                elif row_x == 0.0:
                    y -= SIDE_GAP
                out.append((label, x, y, w, tall))
            o += w
    return out


def keyboard_object() -> Part:
    """Case, plate, the spacebar's switch and stabilizer housings, and every other keycap of a 104-key ANSI
    board (all fixed; the case is the root). Keycaps near the spacebar collide; the rest are visual."""
    left, right = LEFT_Y + 0.012, LEFT_Y - WIDTH_U * U - 0.012
    back, front = 5.25 * U + U / 2 + 0.012, -U / 2 - 0.012
    case = (back - front, left - right, PLATE_Z - 0.0015)
    cx, cy = (back + front) / 2, (left + right) / 2
    rel = lambda x, y, z: (x - cx, y - cy, z - case[2] / 2)      # noqa: E731
    pieces = [Piece("box", (case[0] - 0.010, case[1] - 0.010, 0.0015), rel(cx, cy, PLATE_Z - 0.00075), color=PLATE),
              # the case's raised rim round the plate
              Piece("box", (case[0], 0.006, 0.010), rel(cx, left - 0.003, PLATE_Z + 0.005), color=CASE),
              Piece("box", (case[0], 0.006, 0.010), rel(cx, right + 0.003, PLATE_Z + 0.005), color=CASE),
              Piece("box", (0.006, case[1], 0.010), rel(back - 0.003, cy, PLATE_Z + 0.005), color=CASE),
              Piece("box", (0.006, case[1], 0.006), rel(front + 0.003, cy, PLATE_Z + 0.003), color=CASE)]
    pieces.append(Piece("box", (0.014, 0.014, HOUSING_TOP - PLATE_Z), rel(0.0, 0.0, (PLATE_Z + HOUSING_TOP) / 2),
                        color=HOUSING))
    for s in (1, -1):
        pieces.append(Piece("box", (0.011, 0.007, HOUSING_TOP - PLATE_Z - 0.001),
                            rel(0.0, s * STAB_Y, (PLATE_Z + HOUSING_TOP - 0.001) / 2), color=STAB_COL))
    for label, x, y, w, tall in _keys():
        near = x <= 1.5 * U and abs(y) < 0.11
        color = MODS if label in _MODS or label.startswith("F") and label[1:].isdigit() else ALPHA
        pieces += [Piece(p.shape, p.size, rel(*p.offset), color=p.color, collide=p.collide)
                   for p in _keycap(x, y, w, color, collide=near, depth_units=tall)]
    return Part("keyboard", pos=_c(cx, cy, case[2] / 2), size=case, exact_z=True, mass=None, color=CASE,
                friction=(0.4, 0.3), pieces=tuple(pieces), contact_offset=0.0004)


def _cross(z_top, color, at_y=0.0):
    """An MX cross whose top is at local z_top: two arms, each with a 45-degree rounded ridge on top."""
    h = CROSS_H
    zc = z_top - h / 2
    r = ARM_W / math.sqrt(2)
    return (Piece("box", (ARM_L, ARM_W, h - ARM_W / 2), (0.0, at_y, zc - ARM_W / 4), color=color),
            Piece("box", (ARM_W, ARM_L, h - ARM_W / 2), (0.0, at_y, zc - ARM_W / 4), color=color),
            Piece("box", (ARM_L, r, r), (0.0, at_y, z_top - ARM_W / 2), rot=(math.pi / 4, 0.0, 0.0), color=color),
            Piece("box", (r, ARM_L, r), (0.0, at_y, z_top - ARM_W / 2), rot=(0.0, math.pi / 4, 0.0), color=color))


STEM_UP = COLUMN / 2 + CROSS_H          # cross top above a slider body's root (the column's centre)


def _below(color, at_y=0.0):
    """The slider's lower part inside the housing (visual only)."""
    return (Piece("box", (0.0052, 0.0052, 0.003), (0.0, at_y, -COLUMN / 2 - 0.0015), color=color, collide=False),)


def switch_stem() -> Part:
    """The switch's slider: its root is the 5.2 mm column standing above the housing top, the cross on it."""
    pos = _c(0.0, 0.0, HOUSING_TOP + COLUMN / 2)
    return Part("switch_stem", pos=pos, size=(0.0052, 0.0052, COLUMN), exact_z=True, mass=0.004, color=STEM_COL,
                friction=(0.35, 0.3), contact_offset=0.0003, pieces=_below(STEM_COL) + _cross(STEM_UP, STEM_COL),
                spring=Spring(axis="Z", lower=-TRAVEL, upper=0.0, stiffness=SWITCH_K, damping=0.5, target=SWITCH_PRE))


def stab_bar() -> Part:
    """Both stabilizer sliders and their wire: one body on one slide, as the wire links them. Root: the left
    slider's column."""
    pos = _c(0.0, STAB_Y, HOUSING_TOP + COLUMN / 2)
    wire = (Piece("cylinder", (0.0008, 2 * STAB_Y, 0.0008), (0.009, -STAB_Y, -COLUMN / 2 - 0.004), axis="Y",
                  color=(0.78, 0.79, 0.81), collide=False),
            Piece("cylinder", (0.0008, 0.010, 0.0008), (0.004, 0.0, -COLUMN / 2 - 0.002), axis="X",
                  color=(0.78, 0.79, 0.81), collide=False),
            Piece("cylinder", (0.0008, 0.010, 0.0008), (0.004, -2 * STAB_Y, -COLUMN / 2 - 0.002), axis="X",
                  color=(0.78, 0.79, 0.81), collide=False))
    other = Piece("box", (0.0052, 0.0052, COLUMN), (0.0, -2 * STAB_Y, 0.0), color=STAB_COL)
    return Part("stab_bar", pos=pos, size=(0.0052, 0.0052, COLUMN), exact_z=True, mass=0.004, color=STAB_COL,
                friction=(0.35, 0.3), contact_offset=0.0003,
                pieces=(other,) + _below(STAB_COL) + _below(STAB_COL, -2 * STAB_Y) + _cross(STEM_UP, STAB_COL)
                + _cross(STEM_UP, STAB_COL, at_y=-2 * STAB_Y) + wire,
                spring=Spring(axis="Z", lower=-TRAVEL, upper=0.0, stiffness=STAB_K, damping=0.5, target=STAB_PRE))


def _socket(y):
    """A socket boss with an MX cross slot through it, hanging under the cap's top at y: four corner blocks
    leave the slot open in both directions."""
    q = (BOSS - SLOT) / 2
    z = -(TOP_T / 2 + BOSS_H / 2)
    off = SLOT / 2 + q / 2
    return tuple(Piece("box", (q, q, BOSS_H), (sx * off, y + sy * off, z)) for sx in (1, -1) for sy in (1, -1))


def cap_object() -> Part:
    """The spacebar: root is its top (14 mm deep, sculpted), with a sloped skirt stand-in, three sockets."""
    skirt_h = SB_SKIRT
    sz = -(TOP_T / 2 + skirt_h / 2)
    sw = 0.0020                                   # 2 mm PBT walls: the 14 mm housing fits inside with 0.1 mm to spare
    sw_long = 0.0020                              # the long walls 2 mm too: 14.2 mm inside, over the 14 mm housing
    side = (SB_W - SB_TOP_W) / 2
    skirt = (Piece("box", (sw_long, SB_L, skirt_h), (SB_W / 2 - sw_long / 2, 0.0, sz)),
             Piece("box", (sw_long, SB_L, skirt_h), (-SB_W / 2 + sw_long / 2, 0.0, sz)),
             Piece("box", (SB_W - 2 * sw, 0.0015, skirt_h), (0.0, SB_L / 2 - 0.00075, sz)),
             Piece("box", (SB_W - 2 * sw, 0.0015, skirt_h), (0.0, -SB_L / 2 + 0.00075, sz)))

    socks = _socket(0.0) + _socket(STAB_Y) + _socket(-STAB_Y)
    return Part("spacebar", pos=(CAP0[0], CAP0[1], TOP + SB_SKIRT + TOP_T / 2), size=(SB_W, SB_L, TOP_T),
                exact_z=True, mass=0.008, color=MOD, friction=(0.45, 0.35), contact_offset=0.0015,
                pieces=skirt + socks)


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def stem_tops(poses):
    """World positions of the three cross tops (middle, left, right)."""
    s = np.asarray(poses["switch_stem"], float)
    b = np.asarray(poses["stab_bar"], float)
    up = np.array([0.0, 0.0, STEM_UP])
    return {0.0: s[:3] + up, STAB_Y: b[:3] + up, -STAB_Y: b[:3] + np.array([0.0, -2 * STAB_Y, 0.0]) + up}


def engagement(poses):
    """For each socket: how far its cross is inside it (m; -ve: not in) and its lateral offset (m)."""
    cap = np.asarray(poses["spacebar"], float)
    o, r = cap[:3], quat_matrix(cap[3:7])
    out = {}
    for y, tip in stem_tops(poses).items():
        local = r.T @ (tip - o)
        mouth_z = -TOP_T / 2 - BOSS_H
        out[y] = (float(local[2] - mouth_z), float(math.hypot(local[0], local[1] - y)))
    return out


def roll_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))
