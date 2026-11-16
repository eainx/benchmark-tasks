"""Geometry and measurements for comb-binding a half-letter booklet on a manual comb binder (real size).

Everything is real size; the bench is an office desk top at working height. A manual plastic-comb
binder of the common upright-slot design stands on the desk, its front toward the arms: the paper
stands binding edge down in the punch slot on the top front, against the side guide, and is punched
by pulling the punch handle on the right forward and down; the comb opener on the top behind it
opens a comb laid on it spine down when its lever on the left is pulled forward, and the punched
pages are hung on the open rings, then the lever is let back to close the comb. A half-letter
booklet (15 sheets between a clear front cover and a black back cover, square, unpunched) lies on a
ream of copy paper beside the binder; a 1/2 in black 15-ring comb lies beside its box on the left.
Frame: metres, x away from the arm bases, y left, z up; TOP is the desk top.

Real parts:
  binder     manual comb binder, upright punch slot (GBC CombBind C110 / Fellowes Star+ class),
             400 W x 330 D x 120 H mm body (handles extra), 6.8 kg, light-grey ABS case, charcoal
             top deck, steel punch handle with black rubber grip (right), steel comb-opener lever
             with black knob (left), side paper guide, margin-depth dial, steel comb-opener fingers.
             Punches 19 rectangular 8 x 3 mm holes on 14.3 mm (9/16 in) pitch along a letter edge;
             a 216 mm (8.5 in) edge takes 15.
  booklet    half-letter (140 x 216 mm) report: 15 sheets of 80 g/m2 paper, a 0.2 mm clear PVC
             front cover and a black 250 g/m2 card back cover; 3.0 mm thick, 75 g; bound on its
             216 mm edge.
  comb       1/2 in (12.7 mm) black PVC binding comb cut to 15 rings, 216 mm long, 9 g.
  ream       500 sheets of letter copy paper, 216 x 279 x 52 mm in its paper wrapper, 2.3 kg.
  comb box   carton of 1/2 in combs, 290 x 70 x 60 mm, printed paperboard, lid open.
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace

import numpy as np

import meshlib as M
from simkit import Hinge, Part, Piece, Spring

TOP = 0.7475

# -- colours -----------------------------------------------------------------------------------------
CASE = (0.80, 0.81, 0.80)          # light-grey ABS case
DECK = (0.20, 0.21, 0.23)          # charcoal top deck
SLOT_DARK = (0.04, 0.04, 0.05)
STEEL = (0.78, 0.79, 0.80)
RUBBER = (0.06, 0.06, 0.07)
GUIDE_COL = (0.85, 0.30, 0.12)     # orange side guide
COMB_COL = (0.05, 0.05, 0.06)
PAPER = (0.96, 0.96, 0.94)
COVER_BLACK = (0.07, 0.07, 0.08)
INK = (0.12, 0.20, 0.42)           # navy print
ACCENT = (0.80, 0.30, 0.10)

# -- binder (fixed) ------------------------------------------------------------------------------------
BX0 = 0.250                        # front face (toward the arms)
BD = 0.330                         # depth along x
BX1 = BX0 + BD
BW = 0.400                         # width along y
BYC = 0.000                        # centre along y
BH = 0.120                         # deck top above the desk
DECK_T = 0.015                     # deck plate thickness
DECK_Z = TOP + BH
SLOT_X = 0.285                     # centre line of the upright punch slot
SLOT_GAP = 0.006                   # slot width (x): takes up to about 5 mm of paper
SLOT_HALF = 0.150                  # slot runs y -0.150 .. +0.150
SLOT_FLOOR = DECK_Z - DECK_T       # the paper's binding edge rests here in the slot (15 mm deep)
GUIDE_Y = -0.120                   # the side guide's face: the booklet's right edge stands against it
COMB_X = 0.425                     # comb opener: the comb's spine lies here, along y
COMB_FLOOR = DECK_Z                # on the deck
STOP_X = COMB_X + 0.045            # tall back plate the hung pages lean on (clear of a hand laying the comb)
STOP_H = 0.130

# punch handle (right): pivot on the case side, arm up and forward at rest, grip along -y
PUNCH_PIVOT = (0.470, -(BW / 2) - 0.016, TOP + 0.085)
PUNCH_LEN = 0.190
PUNCH_REST_DEG = 75.0              # arm above horizontal, pointing toward -x (the arms)
PUNCH_STROKE_DEG = 50.0            # pulled forward/down this far: the punch bottoms
GRIP_LEN = 0.085
# comb opener lever (left)
OPEN_PIVOT = (0.480, (BW / 2) + 0.016, TOP + 0.095)
OPEN_LEN = 0.120
OPEN_REST_DEG = 95.0               # just past upright, leaning back
OPEN_STROKE_DEG = 50.0             # pulled forward this far: the comb is fully open

# -- booklet, ream, comb, box --------------------------------------------------------------------------
BOOK = (0.0030, 0.216, 0.140)      # standing: thickness along x, binding edge (216) along y at the bottom
BOOK_MASS = 0.075
SORTER = (0.110, 0.250, 0.090)     # desktop upright file sorter: 3 slots along x, open top
SORTER_C = (0.175, 0.105)
SORT_FLOOR = TOP + 0.005
BOOK_C = (0.175, 0.105)            # standing in the sorter's middle slot
COMB_LEN = 0.216
COMB_RINGS = 15
COMB_PITCH = 0.0143
COMB_R = 0.00635                   # 1/2 in comb: ring radius
COMB_SPINE = (0.009, 0.216, 0.0012)  # along y: width (x), length (y), thickness
COMB_C = (0.205, -0.190)           # lying spine down, rings up, along y
COMB_Y = BOOK_C[1] * 0 - 0.012       # its seat on the opener (centred on the booklet held against the guide)
GUIDE_GAP = 0.0
COMB_MASS = 0.012
BOX = (0.290, 0.070, 0.060)
BOX_C = (0.335, -0.400)


def _hidden(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _box(size, center, color, material=None, e=8.0, name="b", opacity=1.0):
    p, t = M.transform(M.rbox(size, e=e, n=24), offset=center)
    return (name, p, t, color, material, opacity)


# -- binder ----------------------------------------------------------------------------------------
def binder_object() -> Part:
    cx, cy = BX0 + BD / 2, BYC
    cz = TOP + (BH - DECK_T) / 2
    root = (BD, BW, BH - DECK_T)
    P = []
    # deck: front strip ahead of the slot and the rear deck behind it (the slot is the gap between)
    fx0, fx1 = BX0, SLOT_X - SLOT_GAP / 2
    rx0, rx1 = SLOT_X + SLOT_GAP / 2, BX1
    P.append(Piece("box", (fx1 - fx0, BW, DECK_T), ((fx0 + fx1) / 2 - cx, 0.0, BH - DECK_T / 2 - (BH - DECK_T) / 2)))
    P.append(Piece("box", (rx1 - rx0, BW, DECK_T), ((rx0 + rx1) / 2 - cx, 0.0, BH - DECK_T / 2 - (BH - DECK_T) / 2)))
    # the slot's ends beyond its length are closed
    for sgn in (-1, 1):
        w = BW / 2 - SLOT_HALF
        P.append(Piece("box", (SLOT_GAP, w, DECK_T), (SLOT_X - cx, sgn * (SLOT_HALF + w / 2), BH - DECK_T / 2 - (BH - DECK_T) / 2)))
    # side guide standing on the front strip and the rear deck at GUIDE_Y (the booklet's edge stop)
    gz = BH + 0.012 - (BH - DECK_T) / 2
    P.append(Piece("box", (0.050, 0.004, 0.024), (SLOT_X - cx, GUIDE_Y - 0.002, gz)))
    # comb opener: a low channel the comb's spine lies in, and the back stop
    P.append(Piece("box", (0.006, 2 * SLOT_HALF, STOP_H), (STOP_X - cx, 0.0, BH + STOP_H / 2 - (BH - DECK_T) / 2)))
    return Part("binder", pos=(cx, cy, cz), size=root, exact_z=True, color=CASE, friction=(0.5, 0.4),
                pieces=_hidden(P), hide_root=True, vmeshes=binder_vmeshes())


def binder_vmeshes():
    cx, cy = BX0 + BD / 2, BYC
    cz = TOP + (BH - DECK_T) / 2
    o = np.array([cx, cy, cz])
    V = []

    def at(x, y, z):
        return (x - cx, y - cy, z - cz)

    # case: rounded body with a slightly inset waist line
    V.append(_box((BD, BW, BH - DECK_T - 0.004), at(cx, cy, TOP + (BH - DECK_T) / 2 - 0.002), CASE, None, e=10, name="case"))
    V.append(_box((BD + 0.004, BW + 0.004, 0.012), at(cx, cy, TOP + 0.006), (0.42, 0.43, 0.44), None, e=10, name="plinth"))
    # deck plates
    fx0, fx1 = BX0 + 0.002, SLOT_X - SLOT_GAP / 2
    rx0, rx1 = SLOT_X + SLOT_GAP / 2, BX1 - 0.002
    V.append(_box((fx1 - fx0, BW - 0.004, DECK_T), at((fx0 + fx1) / 2, cy, DECK_Z - DECK_T / 2), DECK, None, e=12, name="deck_f"))
    V.append(_box((rx1 - rx0, BW - 0.004, DECK_T), at((rx0 + rx1) / 2, cy, DECK_Z - DECK_T / 2), DECK, None, e=12, name="deck_r"))
    # slot: dark throat, steel lips
    V.append(_box((SLOT_GAP * 0.9, 2 * SLOT_HALF, DECK_T * 0.95), at(SLOT_X, cy, DECK_Z - DECK_T / 2), SLOT_DARK, None, e=4, name="throat"))
    for sx in (-1, 1):
        V.append(_box((0.003, 2 * SLOT_HALF + 0.010, 0.002), at(SLOT_X + sx * (SLOT_GAP / 2 + 0.0015), cy, DECK_Z + 0.001), STEEL, "steel", e=4, name=f"lip{sx}"))
    # printed scale along the front strip: ticks every 14.3 mm (hole pitch) and longer every 5
    for k in range(-10, 11):
        L = 0.010 if k % 5 == 0 else 0.005
        V.append(_box((L, 0.0010, 0.0004), at(SLOT_X - 0.010 - L / 2, k * COMB_PITCH, DECK_Z + 0.0002), (0.92, 0.92, 0.90), None, e=4, name=f"tick{k}"))
    # side guide (orange) with its thumb knob
    V.append(_box((0.050, 0.004, 0.024), at(SLOT_X, GUIDE_Y - 0.002, DECK_Z + 0.012), GUIDE_COL, "abs", e=6, name="guide"))
    kp, kt = M.revolve([(0.0, 0.0), (0.006, 0.0), (0.006, 0.010), (0.0, 0.010)], n=20, axis="Z")
    V.append(("guide_knob", *M.transform((kp, kt), offset=at(SLOT_X - 0.030, GUIDE_Y - 0.010, DECK_Z)), GUIDE_COL, "abs", 1.0))
    # margin-depth dial (front left) with ribs and a white index
    dp, dt = M.revolve([(0.0, 0.0), (0.019, 0.0), (0.020, 0.002), (0.020, 0.010), (0.018, 0.012), (0.0, 0.012)], n=40, axis="Z")
    dx, dy = BX0 + 0.020, 0.150
    V.append(("dial", *M.transform((dp, dt), offset=at(dx, dy, DECK_Z)), (0.28, 0.29, 0.31), "abs", 1.0))
    for k in range(24):
        a = 2 * math.pi * k / 24
        V.append(_box((0.0015, 0.0015, 0.008), at(dx + 0.0203 * math.cos(a), dy + 0.0203 * math.sin(a), DECK_Z + 0.006), (0.22, 0.23, 0.25), "abs", e=4, name=f"rib{k}"))
    V.append(_box((0.014, 0.0025, 0.0006), at(dx - 0.006, dy, DECK_Z + 0.0123), (0.95, 0.95, 0.95), None, e=4, name="index"))
    for k in range(5):                                  # the depth scale printed round it
        a = math.radians(-60 + 30 * k)
        V.append(_box((0.004, 0.0008, 0.0004), at(dx - 0.026 * math.cos(a), dy + 0.026 * math.sin(a), DECK_Z + 0.0002), (0.92, 0.92, 0.90), None, e=4, name=f"dscale{k}"))
    # comb opener: steel fingers in a row, the spine channel and the white back stop
    n_f = 21
    for k in range(n_f):
        y = (k - (n_f - 1) / 2) * COMB_PITCH
        V.append(_box((0.010, 0.0022, 0.016), at(COMB_X - 0.002, y, DECK_Z + 0.008), STEEL, "steel", e=4, name=f"finger{k}"))
        V.append(_box((0.006, 0.0022, 0.004), at(COMB_X + 0.004, y, DECK_Z + 0.014), STEEL, "steel", e=4, name=f"hook{k}"))
    V.append(_box((0.004, 2 * SLOT_HALF, 0.006), at(COMB_X - 0.012, cy, DECK_Z + 0.003), (0.55, 0.56, 0.58), "steel", e=4, name="channel"))
    V.append(_box((0.006, 2 * SLOT_HALF + 0.010, STOP_H), at(STOP_X, cy, DECK_Z + STOP_H / 2), (0.90, 0.90, 0.88), None, e=6, name="backstop"))
    # comb-size gauge (stepped notches) on the deck's back right
    for k in range(6):
        V.append(_box((0.010, 0.006, 0.004 + 0.003 * k), at(BX1 - 0.030, -0.100 - 0.008 * k, DECK_Z + (0.004 + 0.003 * k) / 2), (0.92, 0.92, 0.90), None, e=4, name=f"gauge{k}"))
    # waste drawer at the front, with its finger pull
    V.append(_box((0.004, 0.300, 0.050), at(BX0 - 0.001, cy, TOP + 0.045), (0.66, 0.67, 0.66), None, e=6, name="drawer"))
    V.append(_box((0.006, 0.080, 0.010), at(BX0 - 0.003, cy, TOP + 0.060), (0.30, 0.31, 0.32), None, e=6, name="pull"))
    # rubber feet
    for sx in (-1, 1):
        for sy in (-1, 1):
            V.append(_box((0.030, 0.030, 0.004), at(cx + sx * (BD / 2 - 0.030), sy * (BW / 2 - 0.030), TOP + 0.002), RUBBER, None, e=4, name=f"foot{sx}{sy}"))
    # pivot bosses for both handles
    for piv in (PUNCH_PIVOT, OPEN_PIVOT):
        bp, bt = M.revolve([(0.0, 0.0), (0.022, 0.0), (0.022, 0.012), (0.0, 0.012)], n=32, axis="Y")
        sgn = 1 if piv[1] > 0 else -1
        bp = bp * np.array([1.0, sgn, 1.0])
        V.append(("boss" + str(sgn), *M.transform((bp, bt), offset=at(piv[0], sgn * BW / 2, piv[2])), (0.35, 0.36, 0.37), None, 1.0))
    return tuple(V)


# -- handles (hinged to the world at the case's pivots) ----------------------------------------------
def _lever_mesh(length, rest_deg, side_sign, grip):
    """Lever arm from the pivot (origin) at ``rest_deg`` above the -x horizontal, in the x-z plane."""
    a = math.radians(rest_deg)
    d = np.array([-math.cos(a), 0.0, math.sin(a)])
    tip = d * length
    V = []
    arm = M.tube([np.zeros(3), tip], 0.007, n=14)
    V.append(("arm", *arm, STEEL, "steel", 1.0))
    hub = M.revolve([(0.0, 0.0), (0.016, 0.0), (0.016, 0.010), (0.0, 0.010)], n=28, axis="Y")
    hp = hub[0] * np.array([1.0, side_sign, 1.0])
    V.append(("hub", hp, hub[1], STEEL, "steel", 1.0))
    if grip == "bar":
        g0 = tip
        g1 = tip + np.array([0.0, side_sign * GRIP_LEN, 0.0])
        V.append(("crank", *M.tube([tip - d * 0.01, g0], 0.0075, n=14), STEEL, "steel", 1.0))
        prof = [(0.0, 0.0), (0.013, 0.0), (0.0145, 0.006)] + [(0.0145 - 0.0012 * (k % 2), 0.006 + 0.007 * k) for k in range(1, 10)] + [(0.013, GRIP_LEN), (0.0, GRIP_LEN)]
        gp, gt = M.revolve(prof, n=28, axis="Y")
        gp = gp * np.array([1.0, side_sign, 1.0]) + g0
        V.append(("grip", gp, gt, RUBBER, None, 1.0))
    else:
        kp, kt = M.revolve([(0.0, -0.016)] + [(0.016 * math.sin(math.pi * k / 12), -0.016 * math.cos(math.pi * k / 12)) for k in range(1, 12)] + [(0.0, 0.016)], n=28, axis="Z")
        V.append(("knob", kp + tip + d * 0.012, kt, RUBBER, None, 1.0))
    return tuple(V), tip


def punch_handle_object() -> Part:
    vm, tip = _lever_mesh(PUNCH_LEN, PUNCH_REST_DEG, -1, "bar")
    a = math.radians(PUNCH_REST_DEG)
    mid = np.array([-math.cos(a), 0.0, math.sin(a)]) * PUNCH_LEN / 2
    P = [Piece("capsule", (0.008, PUNCH_LEN - 0.016, 0.008), tuple(mid), rot=(0.0, -(math.pi / 2 - a), 0.0), axis="Z"),
         Piece("cylinder", (0.0145, GRIP_LEN, 0.0145), tuple(tip + np.array([0.0, -GRIP_LEN / 2, 0.0])), axis="Y")]
    return Part("punch_handle", pos=PUNCH_PIVOT, size=(0.005, 0.005, 0.005), shape="sphere", exact_z=True, mass=0.35,
                com=(0.012, 0.0, 0.0),
                color=STEEL, friction=(0.9, 0.8), pieces=_hidden(P), hide_root=True, vmeshes=vm,
                hinge=Hinge(pivot=PUNCH_PIVOT, axis="Y", damping=0.004, lower_deg=-PUNCH_STROKE_DEG - 1.0, upper_deg=0.0,
                            stiffness=0.008, target_deg=15.0))


def opener_lever_object() -> Part:
    vm, tip = _lever_mesh(OPEN_LEN, OPEN_REST_DEG, 1, "knob")
    a = math.radians(OPEN_REST_DEG)
    d = np.array([-math.cos(a), 0.0, math.sin(a)])
    P = [Piece("capsule", (0.007, OPEN_LEN - 0.014, 0.007), tuple(d * OPEN_LEN / 2), rot=(0.0, -(math.pi / 2 - a), 0.0), axis="Z"),
         Piece("sphere", (0.016, 0.016, 0.016), tuple(tip + d * 0.012))]
    return Part("opener_lever", pos=OPEN_PIVOT, size=(0.005, 0.005, 0.005), shape="sphere", exact_z=True, mass=0.20,
                com=(0.010, 0.0, 0.0),
                color=STEEL, friction=(0.9, 0.8), pieces=_hidden(P), hide_root=True, vmeshes=vm,
                hinge=Hinge(pivot=OPEN_PIVOT, axis="Y", damping=0.003, lower_deg=-OPEN_STROKE_DEG - 1.0, upper_deg=0.0,
                            stiffness=0.005, target_deg=10.0))


# -- booklet, sorter, comb, comb box ------------------------------------------------------------------
HOLE = (0.008, 0.003)                # rectangular comb-binding hole: 8 mm along the edge, 3 mm deep
HOLE_IN = 0.0055                     # its centre this far in from the binding edge


def booklet_pos():
    return (BOOK_C[0], BOOK_C[1], SORT_FLOOR + BOOK[2] / 2)


def hole_ys():
    """The punched holes along the binding edge, booklet frame (y): 15 on 14.3 mm pitch, centred."""
    return [(k - (COMB_RINGS - 1) / 2) * COMB_PITCH for k in range(COMB_RINGS)]


def booklet_object() -> Part:
    tx, ly, hz = BOOK
    V = []
    V.append(_box((0.0008, ly, hz), (-tx / 2 + 0.0004, 0.0, 0.0), COVER_BLACK, None, e=10, name="back"))
    V.append(_box((tx - 0.0012, ly - 0.001, hz - 0.001), (0.0, 0.0, 0.0), PAPER, None, e=12, name="pages"))
    for k in range(6):                                   # the page edges' fine lines along the top
        x = -tx / 2 + 0.0010 + k * 0.0003
        V.append(_box((0.00008, ly - 0.0006, 0.0003), (x, 0.0, hz / 2 - 0.0004), (0.80, 0.80, 0.78), None, e=4, name=f"edge{k}"))
    fx = tx / 2 - 0.0003                                 # the title page under the clear front cover, facing +x
    V.append(_box((0.0001, ly - 0.020, 0.030), (fx, 0.0, hz / 2 - 0.028), INK, None, e=4, name="band"))
    V.append(_box((0.0001, 0.110, 0.009), (fx, -0.010, hz / 2 - 0.058), (0.05, 0.05, 0.07), None, e=4, name="title"))
    V.append(_box((0.0001, 0.080, 0.005), (fx, -0.020, hz / 2 - 0.072), (0.35, 0.36, 0.40), None, e=4, name="sub"))
    for k in range(4):
        h = 0.012 + 0.008 * ((k * 7) % 4)
        V.append(_box((0.0001, 0.010, h), (fx, -0.040 + 0.018 * k, -0.030 + h / 2), ACCENT if k % 2 else INK, None, e=4, name=f"bar{k}"))
    V.append(_box((0.0001, 0.060, 0.003), (fx, 0.030, -hz / 2 + 0.020), (0.45, 0.46, 0.50), None, e=4, name="foot"))
    V.append(_box((0.0002, ly, hz), (tx / 2 - 0.0001, 0.0, 0.0), (0.92, 0.95, 0.98), None, e=10, name="cover", opacity=0.12))
    # punched holes: drawn on both faces, hidden until the booklet is punched (task.Dynamics shows them)
    for k, y in enumerate(hole_ys()):
        for sgn in (-1, 1):
            V.append(_box((0.0002, HOLE[0], HOLE[1]), (sgn * (tx / 2 + 0.0002), y, -hz / 2 + HOLE_IN), (0.03, 0.03, 0.04), None,
                          e=4, name=f"hole{k}_{'f' if sgn > 0 else 'b'}"))
    # bound: once the comb closes through its holes it hangs on the comb -- a joint to the comb made at spawn,
    # switched on where it hangs (task.Dynamics sets its frames from the measured poses; no snap)
    return Part("booklet", pos=booklet_pos(), size=BOOK, exact_z=True, mass=BOOK_MASS, color=PAPER, friction=(0.7, 0.6),
                restitution=0.02, contact_offset=0.0010, hide_root=True, vmeshes=tuple(V),
                hinge=Hinge(pivot=booklet_pos(), axis="Z", damping=5.0, lower_deg=-0.01, upper_deg=0.01, parent="comb",
                            parent_pos=comb_pos(), local0=(0.0, 0.0, 0.0), local1=(0.0, 0.0, 0.0), rot0=(1.0, 0.0, 0.0, 0.0),
                            rot1=(1.0, 0.0, 0.0, 0.0), enabled=False))


def sorter_object() -> Part:
    """Desktop upright file sorter, black steel mesh: a floor tray, front, back and two divider walls,
    low side frames; three slots along x, the booklet stands in the middle one. The root is the floor."""
    sx, sy, sz = SORTER
    wall = 0.0025
    fl = 0.005
    P, V = [], []
    slots = 3
    pitch = sx / slots
    for k in range(slots + 1):
        x = -sx / 2 + k * pitch
        h = sz - fl - 0.008 * (1 if 0 < k < slots else 0)
        P.append(Piece("box", (wall, sy, h), (x, 0.0, fl / 2 + h / 2)))
    for sgn in (-1, 1):
        P.append(Piece("box", (sx, wall, 0.040), (0.0, sgn * (sy / 2 - wall / 2), fl / 2 + 0.020)))
    V.append(_box((sx, sy, fl), (0.0, 0.0, 0.0), (0.10, 0.10, 0.11), "steel_painted", e=6, name="floor"))
    for i, pc in enumerate(P):
        V.append(_box(pc.size, pc.offset, (0.10, 0.10, 0.11), "steel_painted", e=6, name=f"panel{i}"))
    for k in range(slots + 1):
        x = -sx / 2 + k * pitch
        for j in range(1, 16):
            V.append(_box((wall + 0.0008, 0.0015, sz - fl - 0.012), (x, -sy / 2 + j * sy / 16, fl / 2 + (sz - fl - 0.012) / 2 + 0.002),
                          (0.18, 0.18, 0.19), None, e=4, name=f"bar{k}_{j}"))
    return Part("sorter", pos=(SORTER_C[0], SORTER_C[1], TOP + fl / 2), size=(sx, sy, fl), exact_z=True, color=(0.1, 0.1, 0.11),
                friction=(0.5, 0.4), pieces=_hidden(P), hide_root=True, vmeshes=tuple(V))


def comb_pos():
    return (COMB_C[0], COMB_C[1], TOP + COMB_SPINE[2] / 2)


def comb_seat():
    """The comb's root on the opener: spine centred on COMB_X, along y at COMB_Y, on the deck."""
    return (COMB_X, COMB_Y, COMB_FLOOR + COMB_SPINE[2] / 2)


def _rings(sweep_deg, tag):
    V = []
    sx, sy, sz = COMB_SPINE
    sweep = math.radians(sweep_deg)
    for k, y in enumerate(hole_ys()):
        c = np.array([sx / 2 - COMB_R, y, COMB_R])      # rings rise from the spine's back (+x) edge, curl toward -x
        pts = []
        for j in range(29):
            th = sweep * j / 28
            pts.append(c + np.array([COMB_R * math.sin(th), 0.0, -COMB_R * math.cos(th)]))
        rp, rt = M.tube(np.asarray(pts), 0.0010, n=10)
        rp[:, 1] = y + (rp[:, 1] - y) * 2.6                 # a flat band ~5 mm wide, not a round wire
        V.append((f"{tag}{k}", rp, rt, COMB_COL, None, 1.0))
    return V


def comb_mesh():
    """The comb in its own frame: spine strip along y on z = 0 and 15 C-rings; a second set, drawn open
    (spread by the opener's fingers), hidden until the comb is opened."""
    sx, sy, sz = COMB_SPINE
    p, t = M.rbox((sx, sy, sz), e=8, n=20)
    V = [("spine", p, t, COMB_COL, None, 1.0)]
    V += _rings(290.0, "ring")
    V += _rings(150.0, "open")
    for k, y in enumerate(hole_ys()):
        bp, bt = M.rbox((0.002, 0.0045, 0.0016), e=6, n=12)
        V.append((f"root{k}", bp + np.array([sx / 2 - 0.001, y, 0.0008]), bt, COMB_COL, None, 1.0))
    return tuple(V)


def comb_object() -> Part:
    sx, sy, sz = COMB_SPINE
    P = [Piece("box", (2 * COMB_R, sy - 0.004, 2 * COMB_R), (sx / 2 - COMB_R, 0.0, COMB_R))]
    seat = comb_seat()
    return Part("comb", pos=comb_pos(), size=COMB_SPINE, exact_z=True, mass=COMB_MASS, color=COMB_COL,
                friction=(0.7, 0.6), restitution=0.02, contact_offset=0.0010, pieces=_hidden(P), hide_root=True,
                vmeshes=comb_mesh(),
                # held on the opener once laid on it square (task.Dynamics switches it on): a locked slide
                spring=Spring(axis="Z", lower=0.0, upper=0.0, stiffness=2000.0, damping=20.0, anchor=seat,
                              local=(0.0, 0.0, 0.0), enabled=False))


def comb_box_object() -> Part:
    V = [_box((BOX[0], BOX[1], BOX[2] - 0.002), (0.0, 0.0, -0.001), (0.93, 0.92, 0.88), None, e=12, name="carton")]
    V.append(_box((BOX[0] + 0.0004, BOX[1] + 0.0004, 0.016), (0.0, 0.0, -BOX[2] / 2 + 0.010), (0.15, 0.40, 0.70), None, e=12, name="band"))
    # the printed picture of a comb on the front face (toward -y), and the size badge
    fy = -BOX[1] / 2 - 0.0003
    V.append(_box((0.200, 0.0004, 0.004), (0.0, fy, 0.004), COMB_COL, None, e=4, name="pic_spine"))
    for k in range(12):
        V.append(_box((0.004, 0.0004, 0.012), (-0.088 + 0.016 * k, fy, 0.012), COMB_COL, None, e=4, name=f"pic_ring{k}"))
    V.append(_box((0.030, 0.0004, 0.022), (0.118, fy, 0.008), ACCENT, None, e=6, name="badge"))
    # open lid folded back, and combs packed inside showing their spines
    V.append(_box((BOX[0], 0.0015, BOX[1]), (0.0, BOX[1] / 2 + 0.0008, BOX[2] / 2 + BOX[1] / 2 - 0.004), (0.93, 0.92, 0.88), None, e=8, name="lid"))
    for k in range(4):
        V.append(_box((0.270, 0.008, 0.012), (0.0, -0.020 + 0.012 * k, BOX[2] / 2 - 0.004), COMB_COL, None, e=6, name=f"packed{k}"))
    return Part("comb_box", pos=(BOX_C[0], BOX_C[1], TOP + BOX[2] / 2), size=BOX, exact_z=True, color=(0.93, 0.92, 0.88),
                friction=(0.5, 0.4), hide_root=True, vmeshes=tuple(V))


# -- measurements ------------------------------------------------------------------------------------
def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
