"""Geometry and parts for stocking a spiral anti-sweep peg hook (real size).

A retail battery bay: a white slatwall panel stands at the back of the bench with a spiral anti-sweep
dispensing hook on it (the kind loss-prevention suppliers sell for razor blades and batteries, e.g. a 9 in
"spiral anti-sweep hook"): a fixed 4.8 mm steel hook rod the cards hang on, and above it, under a clear
polycarbonate hood, a wire dispensing coil that a front knob turns. Each card's hang tab sits in a gap
between coil turns, so turning the knob screws the cards along the rod, one coil pitch per turn. Two cards
of AA batteries already hang on it; a new card stands in the opened cardboard shipper on the bench.
Stocking: hang the new card's euro slot on the rod tip, slide it back under the knob to the coil, and,
steadying the card so it does not swing or tilt, wind the knob backwards so the coil draws the card into
its first gap.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.

Real parts:
  slatwall   white melamine slatwall panel, 3 in (76 mm) slat pitch, aluminium groove inserts, 19 mm thick
  hook       9 in (229 mm) spiral anti-sweep hook on a slatwall bracket: 4.8 mm (3/16 in) chrome steel rod,
             dispensing coil of 3 mm steel wire, 46 mm across, 28 mm pitch, front knob 26 mm across,
             clear polycarbonate hood, yellow scan-label strip on its front
  card       blister card of four AA batteries: 100 x 180 mm, 0.8 mm SBS board, euro slot (30 mm wide,
             9 mm centre hole) 8 mm below its top edge, PETG blister 70 x 66 mm x 18 mm deep, 105 g
  shipper    opened cardboard shipper with a slotted card insert, 140 x 70 x 60 mm
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

import meshlib as M
from simkit import Hinge, Part, Piece

TOP = 0.7475
FLOOR = TOP
TEXTURES = Path(__file__).resolve().parent / "textures"

# The fixture is laid out below in a local frame (hook pointing along local -x, the slatwall's face normal
# -x) and then the whole bay is turned 90 deg about z round ROT_P: the slatwall stands along the bench's left
# side (its face at world y 0.33) and the hook points right (-y), so the hung cards face the room camera.
YAW = math.pi / 2
ROT_P = np.array([0.40, 0.09])
Q_YAW = (math.cos(YAW / 2), 0.0, 0.0, math.sin(YAW / 2))          # w, x, y, z


def W(p):
    """Local -> world."""
    p = np.asarray(p, float)
    c, s_ = math.cos(YAW), math.sin(YAW)
    d = p[:2] - ROT_P
    return np.array([ROT_P[0] + c * d[0] - s_ * d[1], ROT_P[1] + s_ * d[0] + c * d[1], p[2]])


def L(p):
    """World -> local."""
    p = np.asarray(p, float)
    c, s_ = math.cos(YAW), math.sin(YAW)
    d = p[:2] - ROT_P
    return np.array([ROT_P[0] + c * d[0] + s_ * d[1], ROT_P[1] - s_ * d[0] + c * d[1], p[2]])

CHROME = (0.86, 0.87, 0.90)
KNOB_BLACK = (0.07, 0.07, 0.08)
HOOD = (0.86, 0.92, 0.96)
SLAT_WHITE = (0.93, 0.93, 0.91)
KRAFT = (0.72, 0.57, 0.40)

# -- slatwall ------------------------------------------------------------------------------------------
WALL_X = 0.640                         # its front face
WALL_T = 0.019
WALL_W = 0.480                         # (local y)
WALL_YC = -0.070                       # its middle (local y)
WALL_H = 0.305                         # z: four 3 in slats, standing on the bench (a gondola-end section)
SLAT_P = 0.0762                        # 3 in slat pitch

# -- hook: rod (fixed) ---------------------------------------------------------------------------------
HOOK_Y = 0.140                         # local; world x 0.350
ROD_R = 0.0024                         # 4.8 mm (3/16 in) rod
ROD_Z = FLOOR + 0.235                  # the rod's axis height
BRACKET_D = 0.025                      # the slatwall bracket's depth (the rod starts here)
ROD_L = 0.229                          # 9 in hook
ROD_X0 = WALL_X - BRACKET_D            # rod's back end
ROD_X1 = ROD_X0 - ROD_L                # rod's front tip
TIP_UP = 0.0018                        # the tip's slight upturn over its last 10 mm

# -- the dispensing coil (free, on its axle) -------------------------------------------------------------
COIL_Z = ROD_Z + 0.030                 # axle height
COIL_R = 0.023                         # coil radius to the wire centre (46 mm across)
WIRE_R = 0.0015                        # 3 mm wire
PITCH = 0.028
COIL_X0 = ROD_X0 - 0.010               # the coil's back end (behind it: the bracket's bearing)
COIL_L = 0.175                         # 6.25 turns
COIL_X1 = COIL_X0 - COIL_L             # its front end
KNOB_R = 0.013
KNOB_L = 0.016
KNOB_X = COIL_X1 - 0.004 - KNOB_L / 2  # knob centre (its axle stub links it to the coil's front end)
SEG_PER_TURN = 16
COIL_MASS = 0.060
BOTTOM0 = 0.46                         # the coil's set turn: its bottom (where it reaches the cards' tabs) at s = (k + 0.46) p,
PHI0 = -math.pi / 2 + 2 * math.pi * BOTTOM0    # so the front pocket (between the wire's free end and the last bottom) is open

# -- cards -----------------------------------------------------------------------------------------------
CARD_W = 0.100                         # y
CARD_H = 0.180                         # z
CARD_T = 0.0008                        # board (drawn)
CARD_TC = 0.0020                       # its collider: thicker, so the thin board does not tunnel
HOLE_TOP = 0.008                       # euro slot: its top edge this far below the card's top
HOLE_H = 0.009                         # centre hole 9 mm tall ...
HOLE_W = 0.009                         # ... and 9 mm wide
WING_W = 0.030                         # the wings' span
WING_H = 0.003
BLISTER = (0.018, 0.070, 0.066)        # depth (toward -x), width, height
BLISTER_Z = -0.004                     # blister centre below the card centre
CARD_MASS = 0.105                      # 4 x 23 g AA + board and blister
CARD_SWING_DAMPING = 5.0               # 1/s: a hung card's swing settles in about a second (slot friction, air)
CARD_CZ_HUNG = ROD_Z + ROD_R - (CARD_H / 2 - HOLE_TOP)   # card centre when its slot's top edge sits on the rod
CARD_NAMES = ("card_a", "card_b")
NEW_CARD = "card_new"

# -- shipper (fixed) with the new card standing in its slotted insert ------------------------------------
SHIP_C = (0.190, 0.220)                # local; world (0.27, -0.12)
SHIP = (0.070, 0.140, 0.100)           # x, y, z
SHIP_WALL = 0.003
SLOT_W = 0.004                         # the insert's slot (holds the card upright)
RISER = 0.080                          # the insert stands the card on an 80 mm corrugated riser, blister above the rim
NEW_X = SHIP_C[0] + 0.012              # the new card's plane (blister toward the arms, -x)


def _f(v):
    return tuple(float(x) for x in v)


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _quad(name, corners, tex, uv, color=(1.0, 1.0, 1.0)):
    return (name, np.asarray(corners, float), np.array([[0, 1, 2], [0, 2, 3]]), color, "tex:" + str(TEXTURES / tex),
            1.0, np.asarray(uv, float))


def coil_point(s):
    """World point of the coil wire at distance s from its back end (toward the front, -x)."""
    phi = PHI0 - 2 * math.pi * s / PITCH
    return np.array([COIL_X0 - s, HOOK_Y + COIL_R * math.cos(phi), COIL_Z + COIL_R * math.sin(phi)])


def gap_x(k):
    """x of the middle of the coil's k-th gap from the back (where a hung card's plane rests)."""
    return COIL_X0 - (k + BOTTOM0 + 0.5) * PITCH


POCKET_X = COIL_X1 + 0.011                     # the front pocket: behind the wire's free end, ahead of the last bottom


# -------------------------------------------------------------------------------------------- slatwall
def slatwall_object() -> Part:
    V = []
    n = int(WALL_H // SLAT_P) + 1
    for k in range(n):
        z0 = -WALL_H / 2 + k * SLAT_P
        z1 = min(z0 + SLAT_P, WALL_H / 2)
        f = (z1 - z0) / SLAT_P
        x = -WALL_T / 2 - 0.0002
        V.append(_quad(f"slat{k}", [(x, WALL_W / 2, z0), (x, -WALL_W / 2, z0), (x, -WALL_W / 2, z1), (x, WALL_W / 2, z1)],
                       "slat.png", [(0, 1 - f), (3, 1 - f), (3, 1), (0, 1)]))
    V.append(_vm("edge", M.rbox((WALL_T, WALL_W, WALL_H), e=10, axis="Z"), (0.85, 0.85, 0.83), "matte"))
    for k in range(1, n):                                   # each groove: a dark slot between aluminium insert lips
        zg = -WALL_H / 2 + k * SLAT_P
        V.append(_vm(f"slot{k}", M.rbox((0.0012, WALL_W, 0.0050), e=6, axis="Z"), (0.10, 0.10, 0.11), "matte",
                     offset=(-WALL_T / 2 - 0.0004, 0.0, zg)))
        for sgn in (-1, 1):
            V.append(_vm(f"lip{k}{sgn}", M.rbox((0.0016, WALL_W, 0.0016), e=6, axis="Z"), (0.78, 0.79, 0.81), "steel",
                         offset=(-WALL_T / 2 - 0.0008, 0.0, zg + sgn * 0.0033)))
    # a plain 6 in peg hook further along (fixture context) with two USB cable cards on it
    py = -0.170
    zc = ROD_Z - (FLOOR + WALL_H / 2)
    peg = np.array([[-WALL_T / 2, py, zc + 0.012], [-WALL_T / 2 - 0.010, py, zc], [-WALL_T / 2 - 0.150, py, zc],
                    [-WALL_T / 2 - 0.158, py, zc + 0.006]])
    V.append(_vm("peg", M.tube(peg, ROD_R, n=14), CHROME, "steel"))
    V.append(_vm("peg_plate", M.rbox((0.003, 0.022, 0.040), e=6, axis="Z"), CHROME, "steel",
                 offset=(-WALL_T / 2 - 0.0015, py, zc + 0.006)))
    for k, xx in enumerate((-WALL_T / 2 - 0.060, -WALL_T / 2 - 0.075)):
        off = np.array([xx, py, zc + ROD_R - (CARD_H / 2 - HOLE_TOP)])
        for nm, pts, t, col, mat_, op, *rest in card_vmeshes("cable_front.png", "cable"):
            V.append((f"cable{k}_{nm}", pts + off, t, col, mat_, op, *rest))
    # the hook's bracket: a chrome plate hooked into a groove, with the coil's bearing boss
    V.append(_vm("bracket", M.rbox((0.004, 0.050, 0.075), e=6, axis="Z"), CHROME, "steel",
                 offset=(-WALL_T / 2 - 0.002, HOOK_Y - WALL_YC, ROD_Z + 0.020 - (FLOOR + WALL_H / 2))))
    boss = M.revolve([(0.0, 0.0), (0.012, 0.0), (0.012, 0.020), (0.0, 0.020)], n=32, axis="X")
    V.append(_vm("boss", boss, (0.20, 0.20, 0.22), "abs",
                 offset=(-WALL_T / 2 - BRACKET_D, HOOK_Y - WALL_YC, COIL_Z - (FLOOR + WALL_H / 2))))
    return Part("slatwall", pos=_f(W((WALL_X + WALL_T / 2, WALL_YC, FLOOR + WALL_H / 2))), yaw=YAW, size=(WALL_T, WALL_W, WALL_H),
                exact_z=True, color=SLAT_WHITE, friction=(0.5, 0.4), hide_root=False, opacity=0.0, vmeshes=tuple(V))


# ------------------------------------------------------------------------------------------- hook rod
def rod_path():
    """The rod centreline, back to front, with its slight upturned tip (world)."""
    pts = []
    for k in range(25):
        x = ROD_X0 - (ROD_L - 0.010) * k / 24
        pts.append((x, HOOK_Y, ROD_Z))
    for k in range(1, 6):
        u = k / 5
        pts.append((ROD_X0 - ROD_L + 0.010 * (1 - u), HOOK_Y, ROD_Z + TIP_UP * u * u))
    return np.asarray(pts)


def hook_object() -> Part:
    """Fixed: the rod (colliders: a long capsule and the tip) and the clear hood with its scan label."""
    c = np.array([(ROD_X0 + ROD_X1) / 2, HOOK_Y, ROD_Z])
    P = []
    nseg = 8
    seg = (ROD_L - 0.012) / nseg
    for k in range(nseg):
        xm = ROD_X0 - seg * (k + 0.5)
        P.append(Piece("capsule", (ROD_R * 0.96, seg - 2 * ROD_R * 0.96 + 0.001, ROD_R * 0.96), (float(xm - c[0]), 0.0, 0.0),
                       axis="X", opacity=0.0))
    P.append(Piece("capsule", (ROD_R * 0.96, 0.008, ROD_R * 0.96), (float(ROD_X1 + 0.006 - c[0]), 0.0, TIP_UP / 2),
                   rot=(0.0, -math.atan2(TIP_UP, 0.010), 0.0), axis="X", opacity=0.0))
    V = [_vm("rod", M.tube(rod_path() - c, ROD_R, n=16), (0.97, 0.97, 0.99), "steel", 0.999)]
    V.append(_vm("rod_cap", M.revolve([(0.0, -0.0025), (ROD_R * 1.15, -0.002), (ROD_R * 1.15, 0.002), (0.0, 0.0025)],
                                      n=16, axis="X"), CHROME, "steel",
                 offset=(ROD_X1 - c[0], 0.0, TIP_UP)))
    # clear hood over the coil: a rounded top and two sides, open below the coil's lower reach
    hl = COIL_L + 0.012
    hx = COIL_X0 - hl / 2 + 0.006 - c[0]
    # the scan-label strip on the hood's front edge
    lx = COIL_X1 - 0.0005 - c[0]
    z0, z1 = COIL_Z + 0.014 - ROD_Z, COIL_Z + 0.034 - ROD_Z
    V.append(_vm("label_holder", M.rbox((0.004, 0.068, 0.026), e=6, axis="Z"), HOOD, "gloss", 0.35,
                 offset=(lx - 0.004, 0.0, (z0 + z1) / 2)))
    V.append(_quad("label", [(lx - 0.006, 0.030, z0), (lx - 0.006, -0.030, z0), (lx - 0.006, -0.030, z1),
                             (lx - 0.006, 0.030, z1)], "label.png", [(0, 0), (1, 0), (1, 1), (0, 1)]))
    return Part("hook", pos=_f(W(c)), yaw=YAW, size=(0.003, 0.003, 0.003), exact_z=True, color=CHROME,
                friction=(0.35, 0.30), contact_offset=0.0008, pieces=tuple(P), hide_root=False, opacity=0.0, vmeshes=tuple(V))


# ------------------------------------------------------------------------------------------- the coil
def coil_object() -> Part:
    """Free, on a world revolute axle along x (the bracket's bearing and the hood's front bushing). Its
    colliders: the wire as capsule segments, and the knob."""
    piv = np.array([(COIL_X0 + KNOB_X) / 2, HOOK_Y, COIL_Z])
    P = []
    n = int(round(COIL_L / PITCH * SEG_PER_TURN))
    pts = [coil_point(COIL_L * k / n) for k in range(n + 1)]
    for k in range(n):
        a, b = pts[k], pts[k + 1]
        mid = (a + b) / 2 - piv
        d = b - a
        L = float(np.linalg.norm(d))
        d /= L
        pitch = math.asin(max(-1.0, min(1.0, -d[2])))           # rotate local x onto d: yaw then pitch
        yaw = math.atan2(d[1], d[0])
        P.append(Piece("capsule", (WIRE_R * 0.93, max(L - 2 * WIRE_R, 1e-4), WIRE_R * 0.93), _f(mid), rot=(0.0, float(pitch), float(yaw)),
                       axis="X", opacity=0.0))
    P.append(Piece("cylinder", (KNOB_R - 0.0004, KNOB_L - 0.0006, KNOB_R - 0.0004), (float(KNOB_X - piv[0]), 0.0, 0.0), axis="X", opacity=0.0))
    # visuals: the wire as a smooth tube, the axle stub, the knurled knob
    path = np.asarray([coil_point(COIL_L * k / (n * 3)) for k in range(n * 3 + 1)]) - piv
    V = [_vm("coil", M.tube(path, WIRE_R, n=12), (0.97, 0.97, 0.99), "steel", 0.999)]
    end = coil_point(COIL_L) - piv
    stub = np.array([[end[0], end[1], end[2]], [end[0] - 0.002, end[1], end[2]], [KNOB_X - piv[0] + KNOB_L / 2, 0.0, 0.0]])
    V.append(_vm("spoke", M.tube(stub, WIRE_R, n=12), (0.97, 0.97, 0.99), "steel", 0.999))
    back = coil_point(0.0) - piv
    V.append(_vm("back_spoke", M.tube(np.array([back, [COIL_X0 + 0.004 - piv[0], 0.0, 0.0]]), WIRE_R, n=10), CHROME,
                 "steel"))
    V.append(_vm("axle_back", M.revolve([(0.0, 0.0), (0.003, 0.0), (0.003, 0.012), (0.0, 0.012)], n=16, axis="X"),
                 CHROME, "steel", offset=(COIL_X0 - piv[0], 0.0, 0.0)))
    knob = [(0.0, -KNOB_L / 2), (KNOB_R - 0.002, -KNOB_L / 2), (KNOB_R, -KNOB_L / 2 + 0.002), (KNOB_R, KNOB_L / 2 - 0.002),
            (KNOB_R - 0.002, KNOB_L / 2), (0.0, KNOB_L / 2)]
    V.append(_vm("knob", M.revolve(knob, n=48, axis="X"), KNOB_BLACK, "abs", offset=(KNOB_X - piv[0], 0.0, 0.0)))
    for k in range(18):                                     # grip ribs round the knob
        a = 2 * math.pi * k / 18
        rib = M.rbox((KNOB_L * 0.8, 0.0018, 0.0018), e=4, axis="Z")
        V.append(_vm(f"rib{k}", rib, (0.14, 0.14, 0.15), "abs",
                     offset=(KNOB_X - piv[0], KNOB_R * math.cos(a), KNOB_R * math.sin(a))))
    # a white arrow on the knob's face: the dispensing direction
    V.append(_vm("arrow", M.rbox((0.0006, 0.010, 0.003), e=4, axis="Z"), (0.95, 0.95, 0.95), "matte",
                 offset=(KNOB_X - piv[0] - KNOB_L / 2 - 0.0003, 0.0, 0.006)))
    hinge = Hinge(pivot=_f(W(piv)), axis="X", rot0=Q_YAW, rot1=(1.0, 0.0, 0.0, 0.0), damping=0.004, friction=0.004, lower_deg=-1e6, upper_deg=1e6)
    return Part("coil", pos=_f(W(piv)), yaw=YAW, size=(0.003, 0.012, 0.003), shape="cylinder", axis="X", exact_z=True,
                mass=COIL_MASS, color=CHROME, friction=(0.30, 0.25), restitution=0.02, contact_offset=0.0008,
                pieces=tuple(P), hide_root=False, opacity=0.0, vmeshes=tuple(V), hinge=hinge, solver_iters=(32, 4))


# -------------------------------------------------------------------------------------------- the cards
def _card_board_quads(prefix, x, tex):
    """The board's face at x (card frame), with the euro slot left open (8 rectangles round it)."""
    hw, hh = CARD_W / 2, CARD_H / 2
    zt = hh - HOLE_TOP                       # slot top
    zb = zt - HOLE_H                         # slot bottom
    zw0 = zt - HOLE_H / 2 - WING_H / 2       # wing band
    zw1 = zw0 + WING_H
    rects = [(-hw, hw, -hh, zb), (-hw, hw, zt, hh),
             (-hw, -WING_W / 2, zb, zt), (WING_W / 2, hw, zb, zt),
             (-WING_W / 2, -HOLE_W / 2, zb, zw0), (HOLE_W / 2, WING_W / 2, zb, zw0),
             (-WING_W / 2, -HOLE_W / 2, zw1, zt), (HOLE_W / 2, WING_W / 2, zw1, zt)]
    out = []
    front = x < 0
    for i, (y0, y1, z0, z1) in enumerate(rects):
        # seen from -x (front), u runs with -y... the print's left is +y; from the back, with +y
        def uv(y, z):
            u = (hw - y) / CARD_W if front else (y + hw) / CARD_W
            return (u, (z + hh) / CARD_H)
        if front:
            corners = [(x, y1, z0), (x, y0, z0), (x, y0, z1), (x, y1, z1)]
        else:
            corners = [(x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1)]
        out.append(_quad(f"{prefix}{i}", corners, tex, [uv(c[1], c[2]) for c in corners]))
    return out


def card_vmeshes(front="card_front.png", contents="aa"):
    V = _card_board_quads("front", -CARD_T / 2 - 0.0001, front)
    V += _card_board_quads("back", CARD_T / 2 + 0.0001, "card_back.png")
    # the PETG blister bubble and the four AA cells inside
    bd, bw, bh = BLISTER
    V.append(_vm("blister", M.rbox((bd, bw, bh), e=5, axis="X"), (0.92, 0.95, 0.97), "gloss", 0.22,
                 offset=(-CARD_T / 2 - bd / 2, 0.0, BLISTER_Z)))
    V.append(_vm("flange", M.rbox((0.0006, bw + 0.008, bh + 0.008), e=6, axis="X"), (0.93, 0.96, 0.98), "gloss", 0.35,
                 offset=(-CARD_T / 2 - 0.0004, 0.0, BLISTER_Z)))
    if contents == "cable":                                 # a coiled white cable and its two plugs
        for k in range(5):
            ring = M.revolve([(0.020 - 0.0012, -0.0012), (0.020, 0.0), (0.020 - 0.0012, 0.0012), (0.020 - 0.0024, 0.0)],
                             n=40, axis="X")
            V.append(_vm(f"loop{k}", ring, (0.95, 0.95, 0.95), "gloss", offset=(-CARD_T / 2 - 0.004 - 0.0025 * k, 0.0, BLISTER_Z)))
        for k, zz in enumerate((0.022, -0.026)):
            V.append(_vm(f"plug{k}", M.rbox((0.006, 0.009, 0.020), e=4, axis="Z"), (0.92, 0.92, 0.92), "gloss",
                         offset=(-CARD_T / 2 - 0.010, 0.0, BLISTER_Z + zz)))
        return tuple(V)
    for k in range(4):
        y = -0.0216 + 0.0144 * k
        cell = M.revolve([(0.0, -0.025), (0.0070, -0.025), (0.0070, 0.023), (0.0035, 0.024), (0.0028, 0.0255), (0.0, 0.0255)],
                         n=32, axis="Z")
        V.append(_vm(f"cell{k}", cell, (0.06, 0.06, 0.07), "gloss", offset=(-CARD_T / 2 - 0.0085, y, BLISTER_Z - 0.002)))
        band = M.revolve([(0.00705, -0.006), (0.00705, 0.023)], n=32, axis="Z")
        V.append(_vm(f"cellband{k}", band, (0.96, 0.78, 0.08), "gloss", offset=(-CARD_T / 2 - 0.0085, y, BLISTER_Z - 0.002)))
        tip = M.revolve([(0.0, 0.0240), (0.0030, 0.0240), (0.0030, 0.0258), (0.0, 0.0258)], n=16, axis="Z")
        V.append(_vm(f"celltip{k}", tip, (0.80, 0.70, 0.40), "steel",
                     offset=(-CARD_T / 2 - 0.0085, y, BLISTER_Z - 0.002)))
    return tuple(V)


def card_object(name, pos) -> Part:
    """Free. Root collider: the board below the slot; pieces: the strip above it, the board either side of
    the centre hole, and the blister."""
    hw, hh = CARD_W / 2, CARD_H / 2
    zt = hh - HOLE_TOP
    zb = zt - HOLE_H
    low_h = zb + hh
    P = [Piece("box", (CARD_TC, CARD_W / 4, HOLE_TOP), (0.0, -hw + CARD_W / 8 + j * CARD_W / 4, (zt + hh) / 2 - (zb - hh) / 2))
         for j in range(4)]
    P += [Piece("box", (CARD_TC, hw - HOLE_W / 2, HOLE_H), (0.0, (hw + HOLE_W / 2) / 2, (zb + zt) / 2 - (zb - hh) / 2)),
         Piece("box", (CARD_TC, hw - HOLE_W / 2, HOLE_H), (0.0, -(hw + HOLE_W / 2) / 2, (zb + zt) / 2 - (zb - hh) / 2)),
         Piece("box", BLISTER, (-CARD_TC / 2 - BLISTER[0] / 2 + 0.0004, 0.0, BLISTER_Z - (zb - hh) / 2))]
    P = [_replace(p, opacity=0.0) for p in P]
    # the root is the lower board; the part's origin is its centre, so offset everything to it
    root_off = np.array([0.0, 0.0, (zb - hh) / 2])            # lower board centre in the card frame
    V = tuple((n, p - root_off, t, c, m, o, *rest) for n, p, t, c, m, o, *rest in card_vmeshes())
    x, y, z = pos
    return Part(name, pos=_f(W((x, y, z + root_off[2]))), yaw=YAW, size=(CARD_TC, CARD_W, low_h), exact_z=True, mass=CARD_MASS,
                color=(0.15, 0.22, 0.45), friction=(0.45, 0.40), restitution=0.02, contact_offset=0.0008,
                com=_f((-0.006, 0.0, BLISTER_Z - root_off[2])), pieces=tuple(P), hide_root=False, opacity=0.0, vmeshes=V,
                solver_iters=(32, 4), angular_damping=CARD_SWING_DAMPING)


def hung_cards():
    return tuple(card_object(n, (gap_x(k), HOOK_Y, CARD_CZ_HUNG + 0.0004)) for n, k in zip(CARD_NAMES, (0, 1)))


def new_card():
    return card_object(NEW_CARD, (NEW_X, SHIP_C[1], FLOOR + SHIP_WALL + RISER + CARD_H / 2 + 0.0005))


# ------------------------------------------------------------------------------------------ the shipper
def shipper_object() -> Part:
    """Fixed: an opened corrugated shipper with a slotted insert that stands the new card upright."""
    sx, sy, sz = SHIP
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    up = hz - SHIP_WALL / 2
    cx = NEW_X - SHIP_C[0]
    P = [Piece("box", (SHIP_WALL, sy, sz), (-hx + SHIP_WALL / 2, 0.0, up)),
         Piece("box", (SHIP_WALL, sy, sz), (hx - SHIP_WALL / 2, 0.0, up)),
         Piece("box", (sx, SHIP_WALL, sz), (0.0, -hy + SHIP_WALL / 2, up)),
         Piece("box", (sx, SHIP_WALL, sz), (0.0, hy - SHIP_WALL / 2, up)),
         # the insert: a riser the card stands on, and two blocks either side of its slot, 40 mm tall above it
         Piece("box", (sx - 2 * SHIP_WALL, sy - 2 * SHIP_WALL, RISER), (0.0, 0.0, RISER / 2)),
         Piece("box", ((cx - SLOT_W / 2) - (-hx + SHIP_WALL), sy - 2 * SHIP_WALL, 0.040),
               (((cx - SLOT_W / 2) + (-hx + SHIP_WALL)) / 2, 0.0, RISER + 0.020)),
         Piece("box", ((hx - SHIP_WALL) - (cx + SLOT_W / 2), sy - 2 * SHIP_WALL, 0.040),
               (((hx - SHIP_WALL) + (cx + SLOT_W / 2)) / 2, 0.0, RISER + 0.020))]
    P = [_replace(p, opacity=0.0) for p in P]
    V = [_vm("box", M.merge(*(M.transform(M.rbox(p.size, e=8, axis="Z"), None, p.offset) for p in P[:4])), KRAFT, "matte"),
         _vm("insert", M.merge(*(M.transform(M.rbox(p.size, e=8, axis="Z"), None, p.offset) for p in P[4:])),
             (0.80, 0.70, 0.55), "matte")]
    xf = -hx - 0.0004
    V.append(_quad("print", [(xf, hy, 0.020), (xf, -hy, 0.020), (xf, -hy, 0.080), (xf, hy, 0.080)], "shipper.png",
                   [(0, 0), (1, 0), (1, 1), (0, 1)]))
    # the four top flaps, opened and folded outward
    def ry(a_):
        return np.array([[math.cos(a_), 0, math.sin(a_)], [0, 1, 0], [-math.sin(a_), 0, math.cos(a_)]])

    def rx(a_):
        return np.array([[1, 0, 0], [0, math.cos(a_), -math.sin(a_)], [0, math.sin(a_), math.cos(a_)]])
    fl = 0.035                                               # long flaps: half the box depth
    for sgn in (-1, 1):                                      # front/back (long) flaps, hinged on the x faces' top edges
        R = ry(-sgn * math.radians(115))
        V.append(_vm(f"flapx{sgn}", M.rbox((0.003, sy, fl), e=8, axis="Z"), KRAFT, "matte",
                     offset=np.array([sgn * hx, 0.0, sz]) + R @ np.array([0.0, 0.0, fl / 2]), R=R))
        R = rx(sgn * math.radians(110))
        V.append(_vm(f"flapy{sgn}", M.rbox((sx, 0.003, 0.030), e=8, axis="Z"), KRAFT, "matte",
                     offset=np.array([0.0, sgn * hy, sz]) + R @ np.array([0.0, 0.0, 0.015]), R=R))
    for k in range(3):                                       # corrugation flutes showing at the cut top edges
        V.append(_vm(f"edge{k}", M.rbox((sx + 0.001, sy + 0.001, 0.0012), e=8, axis="Z"), (0.62, 0.48, 0.33), "matte",
                     offset=(0.0, 0.0, sz - 0.0008)))
    return Part("shipper", pos=_f(W((SHIP_C[0], SHIP_C[1], FLOOR + SHIP_WALL / 2))), yaw=YAW, size=(sx, sy, SHIP_WALL), exact_z=True,
                color=KRAFT, friction=(0.5, 0.4), pieces=tuple(P), hide_root=False, opacity=0.0, vmeshes=tuple(V))


# ------------------------------------------------------------------------------------- measurements
def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
