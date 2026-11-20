"""Geometry and parts for emptying a commercial-kitchen floor sink's sediment bucket (real size).

A 12 x 12 x 6 in floor sink (the indirect-waste receptor set flush in a kitchen's quarry-tile floor under the
dish table and ice machine; e.g. Zurn Z1900-series / Watts FS-740 / Jay R. Smith 3100 with a 1/2 grate and a
sediment bucket) is set in a cropped section of floor: a 165 mm tiled platform on the bench, faced with red
150 mm quarry tile, with a tiled wall stub on its left. Its near half is covered by a loose stainless slotted
half grate that sits flush in the sink's recess (2 mm clearance all round, nothing to get fingers under);
its far half is open. Inside, a perforated aluminium sediment bucket with a wire bail holds the day's debris
(lemon and lime wedges, bottle caps). The bucket (236 mm square) does not pass the open half (140 mm), so the
grate has to come off: it is lifted with the grate hook (a 6 mm steel rod with a T handle and a 25 mm foot)
lying on the floor, put through a slot and turned under a bar. The scraps go into a grey bus tub standing on
the bench to the right. Real build: the floor sink is set flush in a tiled plywood platform (a floor mock-up
165 mm tall), the way such a fixture would be brought to a robot bench.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench, FLOOR the tiled floor's top.
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace

import numpy as np

import meshlib as M
from simkit import Hinge, Part, Piece

TOP = 0.7475
SLAB_H = 0.165
FLOOR = TOP + SLAB_H
SLAB_X = (0.075, 0.560)
SLAB_Y = (-0.115, 0.44)
WALL_Y = 0.44                   # the wall's face (left side of the floor section)
WALL_T = 0.10
WALL_H = 0.36
CURB_X = 0.4275                 # an equipment curb (housekeeping pad) behind the sink: front face x
CURB_H = 0.0                    # no curb in v3 (the hinged grate rests on its hinge stop), quarry tile faced, from the curb face to the floor section's back

# -- floor sink (all in world coordinates; its top is flush with FLOOR) --------------------------------
SX, SY = 0.300, 0.050           # centre of the sink's top
TOP_HALF = 0.1525               # 305 mm square stainless top frame
REC_HALF = 0.140                # recess for the grate: 280 mm square
INT_HALF = 0.127                # interior: 254 mm square (10 in)
LEDGE_Z = FLOOR - 0.012         # the grate's flange bears on this ledge (and on the centre bar)
BOTTOM_Z = FLOOR - 0.152        # 6 in deep
BAR_X = (SX - 0.008, SX + 0.008)   # centre support bar across the recess, top at LEDGE_Z

# -- half grate: 12 ga (2.7 mm) type 304 stainless, slotted, with a 12 mm turned-down flange -----------
G_T = 0.0019                    # 14 ga (1.9 mm) type 304 stainless
G_FL = 0.012
G_X = (SX - REC_HALF + 0.002, SX - 0.002)          # near half, hinged at its near edge; far edge rests on the side ledges
HINGE = (G_X[0], FLOOR)          # hinge pin line (x, z), along y
LIFT_SLOT = (0.090, 0.045, 0.010)   # hand-hold slot by the free (far) edge: width (y), length (x), 10 mm round hand-hold rod at the edge
G_Y = (SY - REC_HALF + 0.002, SY + REC_HALF - 0.002)
G_C = ((G_X[0] + G_X[1]) / 2, (G_Y[0] + G_Y[1]) / 2, FLOOR - G_FL / 2)
G_LX, G_LY = G_X[1] - G_X[0], G_Y[1] - G_Y[0]      # 136 x 276 mm
SLOT_W = 0.012                  # slots 12 mm wide, running across the grate (along y), 10 mm bars between
SLOT_PITCH = 0.022
SLOT_LEN = 0.110                # two slots per row, either side of a 16 mm middle tie
G_MASS = 0.58

# -- sediment bucket -----------------------------------------------------------------------------------
B_HALF = 0.120                  # 240 mm square
B_H = 0.075                     # shallow basket
SEAT_Z = FLOOR - 0.115          # it hangs on a seat ledge 115 mm down (a ring of 14 mm lugs round the interior)
B_C = (SX, SY, SEAT_Z + B_H / 2)
B_T = 0.0016
B_MASS = 0.38
BAIL_R = 0.0025                 # 5 mm wire
BAIL_RISE = 0.090               # the bail's top stands this far over its pivots when up
BAIL_PIVOT_Z = B_C[2] + B_H / 2 - 0.020
BAIL_DOWN_DEG = 85.0            # folded over toward the right (-y), resting on the rim; ears on the near/far walls
BAIL_STOP_DEG = 88.0            # the bail's legs meet the rim here (joint limit stands in for that contact)
BAIL_MASS = 0.04

# -- grate hook: 6 mm steel rod, 200 mm shaft, 100 mm T handle (red vinyl grip), 25 mm foot, on the floor --
H_R = 0.003
H_LEN = 0.200
H_T = 0.100
H_FOOT = 0.025
H_POS = (0.240, 0.300, FLOOR + H_R)    # the shaft's middle; the shaft lies along x, T handle at the far end
H_MASS = 0.10

# -- bus tub for the scraps: grey polypropylene, 15 x 20 x 5 in (381 x 508 x 127 mm), on the bench right ----
TUB_C = (0.300, -0.312)
TUB_L, TUB_W, TUB_H = 0.508, 0.381, 0.127      # along x, y; height
TUB_MASS = 1.1

TILE = (0.56, 0.21, 0.14)       # red quarry tile
GROUT = (0.42, 0.40, 0.37)
CONC = (0.55, 0.54, 0.52)
WALL_TILE = (0.92, 0.92, 0.90)
SS = (0.78, 0.79, 0.80)
ENAMEL = (0.95, 0.95, 0.94)
ALU = (0.80, 0.81, 0.82)
LEMON = (0.96, 0.84, 0.18)
LIME = (0.46, 0.70, 0.18)
PITH = (0.97, 0.95, 0.80)
CAP_RED = (0.78, 0.10, 0.08)
CAP_BLUE = (0.10, 0.25, 0.70)
BIN_GREY = (0.36, 0.37, 0.38)
STEEL_DK = (0.30, 0.30, 0.31)


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _hid(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def _box(lo, hi):
    """A collider box from world corner lo to hi, as (size, centre)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return tuple(map(float, hi - lo)), tuple(map(float, (hi + lo) / 2))


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


# ======================================================================================================
# fixed: floor section, wall stub and the floor sink body
# ======================================================================================================
def floor_object() -> Part:
    origin = np.array([(SLAB_X[0] + SLAB_X[1]) / 2, (SLAB_Y[0] + SLAB_Y[1]) / 2, TOP])
    rel = lambda p: tuple(map(float, np.asarray(p, float) - origin))
    P, V = [], []

    def box(lo, hi, vis=None, mat=None, e=None):
        s, c = _box(lo, hi)
        P.append(Piece("box", s, rel(c)))
        if vis is not None:
            V.append(_vm(f"b{len(V)}", M.rbox(s, e=e or 30.0), vis, mat, offset=rel(c)))

    # slab: concrete block under the tile, with a hole for the sink body (colliders round the hole)
    z_t = FLOOR - 0.012
    box((SLAB_X[0], SLAB_Y[0], TOP), (SLAB_X[1], SLAB_Y[1], BOTTOM_Z - 0.006), CONC, "matte")    # below the sink
    for lo, hi in (((SLAB_X[0], SLAB_Y[0], BOTTOM_Z - 0.006), (SX - TOP_HALF, SLAB_Y[1], z_t)),
                   ((SX + TOP_HALF, SLAB_Y[0], BOTTOM_Z - 0.006), (SLAB_X[1], SLAB_Y[1], z_t)),
                   ((SX - TOP_HALF, SLAB_Y[0], BOTTOM_Z - 0.006), (SX + TOP_HALF, SY - TOP_HALF, z_t)),
                   ((SX - TOP_HALF, SY + TOP_HALF, BOTTOM_Z - 0.006), (SX + TOP_HALF, SLAB_Y[1], z_t))):
        box(lo, hi, CONC, "matte")
    # quarry tile, 150 mm square, 12 mm thick, 5 mm grout joints, laid from the sink's corner
    pitch = 0.155
    xs = np.arange(SX - TOP_HALF - 4 * pitch, SLAB_X[1] + 1e-6, pitch)
    ys = np.arange(SY - TOP_HALF - 6 * pitch, SLAB_Y[1] + 1e-6, pitch)
    k = 0
    for x0 in xs:
        for y0 in ys:
            x1, y1 = x0 + 0.150, y0 + 0.150
            cx0, cx1 = max(x0, SLAB_X[0]), min(x1, SLAB_X[1])
            cy0, cy1 = max(y0, SLAB_Y[0]), min(y1, WALL_Y)
            if cx1 - cx0 < 0.01 or cy1 - cy0 < 0.01:
                continue
            # cut round the sink's top frame
            if cx1 > SX - TOP_HALF and cx0 < SX + TOP_HALF and cy1 > SY - TOP_HALF and cy0 < SY + TOP_HALF:
                pieces = []
                if cx0 < SX - TOP_HALF:
                    pieces.append((cx0, SX - TOP_HALF - 0.003, cy0, cy1))
                if cx1 > SX + TOP_HALF:
                    pieces.append((SX + TOP_HALF + 0.003, cx1, cy0, cy1))
                xa, xb = max(cx0, SX - TOP_HALF), min(cx1, SX + TOP_HALF)
                if cy0 < SY - TOP_HALF:
                    pieces.append((xa, xb, cy0, SY - TOP_HALF - 0.003))
                if cy1 > SY + TOP_HALF:
                    pieces.append((xa, xb, SY + TOP_HALF + 0.003, cy1))
            else:
                pieces = [(cx0, cx1, cy0, cy1)]
            for a0, a1, b0, b1 in pieces:
                if a1 - a0 < 0.008 or b1 - b0 < 0.008:
                    continue
                shade = 0.92 + 0.16 * ((k * 37) % 11) / 10.0
                col = tuple(min(1.0, c * shade) for c in TILE)
                V.append(_vm(f"tile{k}", M.rbox((a1 - a0, b1 - b0, 0.012), e=14.0), col, "satin",
                             offset=rel(((a0 + a1) / 2, (b0 + b1) / 2, FLOOR - 0.006))))
                k += 1
    P.append(Piece("box", (SX - TOP_HALF - SLAB_X[0], SLAB_Y[1] - SLAB_Y[0], 0.012),
                   rel(((SLAB_X[0] + SX - TOP_HALF) / 2, (SLAB_Y[0] + SLAB_Y[1]) / 2, FLOOR - 0.006))))
    P.append(Piece("box", (SLAB_X[1] - SX - TOP_HALF, SLAB_Y[1] - SLAB_Y[0], 0.012),
                   rel(((SLAB_X[1] + SX + TOP_HALF) / 2, (SLAB_Y[0] + SLAB_Y[1]) / 2, FLOOR - 0.006))))
    P.append(Piece("box", (2 * TOP_HALF, SY - TOP_HALF - SLAB_Y[0], 0.012),
                   rel((SX, (SLAB_Y[0] + SY - TOP_HALF) / 2, FLOOR - 0.006))))
    P.append(Piece("box", (2 * TOP_HALF, SLAB_Y[1] - SY - TOP_HALF, 0.012),
                   rel((SX, (SLAB_Y[1] + SY + TOP_HALF) / 2, FLOOR - 0.006))))
    # grout bed under the tiles (shows in the joints)
    for k, (lo, hi) in enumerate((((SLAB_X[0], SLAB_Y[0]), (SX - TOP_HALF, WALL_Y)), ((SX + TOP_HALF, SLAB_Y[0]), (SLAB_X[1], WALL_Y)),
                                  ((SX - TOP_HALF, SLAB_Y[0]), (SX + TOP_HALF, SY - TOP_HALF)),
                                  ((SX - TOP_HALF, SY + TOP_HALF), (SX + TOP_HALF, WALL_Y)))):
        V.append(_vm(f"grout{k}", M.rbox((hi[0] - lo[0], hi[1] - lo[1], 0.010), e=40.0), GROUT, "matte",
                     offset=rel(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, FLOOR - 0.0075))))
    # wall stub on the left: white 3 x 6 in glazed wall tile over a red quarry cove base
    wl, wh = (SLAB_X[0], WALL_Y, FLOOR), (SLAB_X[1], WALL_Y + WALL_T, FLOOR + WALL_H)
    s, c = _box(wl, wh)
    P.append(Piece("box", s, rel(c)))
    V.append(_vm("wallcore", M.rbox((s[0], s[1] - 0.006, s[2]), e=40.0), GROUT, "matte", offset=rel((c[0], c[1] + 0.003, c[2]))))
    V.append(_vm("cove", M.rbox((s[0], 0.012, 0.150), e=10.0), TILE, "satin", offset=rel((c[0], WALL_Y + 0.006, FLOOR + 0.075))))
    row = 0
    for z0 in np.arange(FLOOR + 0.153, FLOOR + WALL_H - 0.01, 0.0785):
        off = 0.0 if row % 2 == 0 else 0.0785
        for x0 in np.arange(SLAB_X[0] - 0.16 + off, SLAB_X[1], 0.157):
            a0, a1 = max(x0, SLAB_X[0]), min(x0 + 0.154, SLAB_X[1])
            h = min(0.0755, FLOOR + WALL_H - z0)
            if a1 - a0 < 0.01 or h < 0.01:
                continue
            V.append(_vm(f"wt{row}_{x0:.3f}", M.rbox((a1 - a0, 0.008, h), e=12.0, axis="Y"), WALL_TILE, "gloss",
                         offset=rel(((a0 + a1) / 2, WALL_Y + 0.004, z0 + h / 2))))
        row += 1
    # equipment curb behind the sink (the ice machine / dish table stands on such a pad): concrete faced with
    # quarry tile, a bullnose tile cap on top; the lifted grate is leaned back against its front edge
    cl, ch = (CURB_X, SLAB_Y[0], FLOOR), (SLAB_X[1], WALL_Y, FLOOR + CURB_H)
    s, c = _box(cl, ch)
    if CURB_H > 0:
        P.append(Piece("box", s, rel(c)))
    if CURB_H > 0:
        V.append(_vm("curbcore", M.rbox((s[0] - 0.004, s[1], s[2] - 0.004), e=40.0), GROUT, "matte", offset=rel((c[0] + 0.002, c[1], c[2] - 0.002))))
    for k, y0 in enumerate(np.arange(SLAB_Y[0], WALL_Y - 0.01, 0.155) if CURB_H > 0 else ()):
        y1 = min(y0 + 0.150, WALL_Y)
        shade = 0.92 + 0.16 * ((k * 53) % 11) / 10.0
        col = tuple(min(1.0, cc * shade) for cc in TILE)
        V.append(_vm(f"curbface{k}", M.rbox((0.012, y1 - y0, CURB_H - 0.004), e=14.0, axis="X"), col, "satin",
                     offset=rel((CURB_X + 0.006, (y0 + y1) / 2, FLOOR + (CURB_H - 0.004) / 2))))
        V.append(_vm(f"curbcap{k}", M.rbox((SLAB_X[1] - CURB_X, y1 - y0, 0.012), e=14.0), col, "satin",
                     offset=rel(((CURB_X + SLAB_X[1]) / 2, (y0 + y1) / 2, FLOOR + CURB_H - 0.006))))
    # floor sink body: stainless top frame (flush), recess walls, ledge, centre bar, enamelled interior
    for sgn in (-1, 1):
        box((SX - TOP_HALF, SY + sgn * REC_HALF - (0 if sgn > 0 else TOP_HALF - REC_HALF), LEDGE_Z),
            (SX + TOP_HALF, SY + sgn * REC_HALF + (TOP_HALF - REC_HALF if sgn > 0 else 0), FLOOR))
        box((SX + sgn * REC_HALF - (0 if sgn > 0 else TOP_HALF - REC_HALF), SY - REC_HALF, LEDGE_Z),
            (SX + sgn * REC_HALF + (TOP_HALF - REC_HALF if sgn > 0 else 0), SY + REC_HALF, FLOOR))
        # ledge (13 mm) between recess and interior
        box((SX - REC_HALF, SY + sgn * INT_HALF - (0 if sgn > 0 else REC_HALF - INT_HALF), LEDGE_Z - 0.004),
            (SX + REC_HALF, SY + sgn * INT_HALF + (REC_HALF - INT_HALF if sgn > 0 else 0), LEDGE_Z))
        box((SX + sgn * INT_HALF - (0 if sgn > 0 else REC_HALF - INT_HALF), SY - INT_HALF, LEDGE_Z - 0.004),
            (SX + sgn * INT_HALF + (REC_HALF - INT_HALF if sgn > 0 else 0), SY + INT_HALF, LEDGE_Z))
        # interior walls
        box((SX - INT_HALF - 0.004, SY + sgn * INT_HALF - (0 if sgn > 0 else 0.004), BOTTOM_Z),
            (SX + INT_HALF + 0.004, SY + sgn * INT_HALF + (0.004 if sgn > 0 else 0), LEDGE_Z - 0.004))
        box((SX + sgn * INT_HALF - (0 if sgn > 0 else 0.004), SY - INT_HALF, BOTTOM_Z),
            (SX + sgn * INT_HALF + (0.004 if sgn > 0 else 0), SY + INT_HALF, LEDGE_Z - 0.004))
    box((SX - INT_HALF, SY - INT_HALF, BOTTOM_Z - 0.006), (SX + INT_HALF, SY + INT_HALF, BOTTOM_Z))
    # seat ledge for the basket: 14 mm stainless lugs round the interior, 115 mm down
    for sgn in (-1, 1):
        box((SX - INT_HALF, SY + sgn * INT_HALF - (0.014 if sgn > 0 else 0), SEAT_Z - 0.004),
            (SX + INT_HALF, SY + sgn * INT_HALF + (0 if sgn > 0 else 0.014), SEAT_Z), SS, "steel")
        box((SX + sgn * INT_HALF - (0.014 if sgn > 0 else 0), SY - INT_HALF + 0.014, SEAT_Z - 0.004),
            (SX + sgn * INT_HALF + (0 if sgn > 0 else 0.014), SY + INT_HALF - 0.014, SEAT_Z), SS, "steel")
    # visuals: brushed stainless frame ring (flush with the tile), recess and ledge, white enamel interior
    def strip_ring(name, outer, inner, z, h, col, mat):
        w = outer - inner
        for k, sgn in enumerate((-1, 1)):
            V.append(_vm(f"{name}x{k}", M.rbox((w, 2 * outer, h), e=30.0), col, mat, offset=rel((SX + sgn * (inner + w / 2), SY, z))))
            V.append(_vm(f"{name}y{k}", M.rbox((2 * inner, w, h), e=30.0), col, mat, offset=rel((SX, SY + sgn * (inner + w / 2), z))))
    strip_ring("frame", TOP_HALF, REC_HALF, FLOOR - 0.002, 0.004, SS, "steel")
    V.append(_vm("recess", M.loft([(LEDGE_Z - FLOOR, 2 * REC_HALF, 2 * REC_HALF, 40.0), (0.0, 2 * REC_HALF, 2 * REC_HALF, 40.0)],
                                  axis="Z", n=96, cap0=False, cap1=False), SS, "steel", offset=rel((SX, SY, FLOOR))))
    strip_ring("ledge", REC_HALF, INT_HALF, LEDGE_Z - 0.002, 0.004, SS, "steel")
    V.append(_vm("interior", M.loft([(BOTTOM_Z - LEDGE_Z, 2 * INT_HALF, 2 * INT_HALF, 14.0), (0.0, 2 * INT_HALF, 2 * INT_HALF, 30.0)],
                                    axis="Z", n=96, cap0=True, cap1=False), ENAMEL, "gloss", offset=rel((SX, SY, LEDGE_Z))))
    # outlet: a dark round drain opening in the bottom with a dome strainer
    V.append(_vm("outlet", M.revolve([(0.0, 0.0006), (0.040, 0.0006)], n=48), (0.05, 0.05, 0.05), "matte",
                 offset=rel((SX + 0.06, SY + 0.06, BOTTOM_Z))))
    return Part("floor", pos=tuple(map(float, origin)), size=(0.02, 0.02, 0.02), shape="sphere", exact_z=True,
                color=CONC, friction=(0.6, 0.5), restitution=0.1, pieces=_hid(P), hide_root=True, vmeshes=tuple(V))


# ======================================================================================================
# half grate
# ======================================================================================================
def slot_rows():
    """x (world) of each slot row's centre: 12 mm slots on a 22 mm pitch from 16 mm off the hinged edge, stopping
    short of the solid band that carries the hand-hold slot by the free edge."""
    rows, x = [], G_X[0] + 0.016 + SLOT_W / 2
    while x + SLOT_W / 2 < G_X[1] - LIFT_SLOT[1] - LIFT_SLOT[2] - 0.008:
        rows.append(x)
        x += SLOT_PITCH
    return rows


def grate_object() -> Part:
    """Hinged half grate over the sink's near half: slotted plate, turned-down flange, a hand-hold slot by its free
    (far) edge leaving a 10 mm strip to grip (the flange is cut away under that strip), hinge pins at the near edge."""
    c = np.array(G_C)
    rel = lambda p: tuple(map(float, np.asarray(p, float) - c))
    P, V, plate = [], [], []
    top = FLOOR - G_T / 2
    rows = slot_rows()
    y_in = SLOT_LEN + 0.008
    margin = G_LY / 2 - y_in
    sw, sl, st = LIFT_SLOT
    def plate_box(x0, x1, y0, y1, z=top, h=G_T):
        if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
            return
        P.append(Piece("box", (x1 - x0, y1 - y0, h), rel(((x0 + x1) / 2, (y0 + y1) / 2, z))))
        plate.append(M.transform(M.rbox((x1 - x0, y1 - y0, h), e=40.0), None, rel(((x0 + x1) / 2, (y0 + y1) / 2, z))))
    # bars between the slot rows (full width), and the side margins and middle tie across the slotted band
    edges = [G_X[0]] + [x for r in rows for x in (r - SLOT_W / 2, r + SLOT_W / 2)]
    for i in range(0, len(edges) - 1, 2):
        plate_box(edges[i], edges[i + 1], G_Y[0], G_Y[1])
    xs0, xs1 = rows[0] - SLOT_W / 2, rows[-1] + SLOT_W / 2
    for sgn in (-1, 1):
        plate_box(xs0, xs1, *sorted((G_C[1] + sgn * G_LY / 2, G_C[1] + sgn * y_in)))
    plate_box(xs0, xs1, G_C[1] - 0.008, G_C[1] + 0.008)
    # solid band by the free edge with the hand-hold slot, and the 10 mm strip at the edge
    xa, xb, xc = xs1, G_X[1] - st - sl, G_X[1] - st
    plate_box(xa, xb, G_Y[0], G_Y[1])
    plate_box(xb, xc, G_Y[0], G_C[1] - sw / 2)
    plate_box(xb, xc, G_C[1] + sw / 2, G_Y[1])
    plate_box(xc, G_X[1], G_Y[0], G_C[1] - sw / 2)
    plate_box(xc, G_X[1], G_C[1] + sw / 2, G_Y[1])
    # a 10 mm round hand-hold rod across the slot at the free edge, its top flush with the plate
    rod_c = (G_X[1] - st / 2, G_C[1], FLOOR - 0.005)
    P.append(Piece("capsule", (0.005, sw - 0.010, 0.005), rel(rod_c), axis="Y"))
    V.append(_vm("rod", M.tube(np.array([(0.0, -sw / 2, 0.0), (0.0, sw / 2, 0.0)]), 0.005, n=16), SS, "steel", offset=rel(rod_c)))
    # turned-down flange; the free edge's flange is cut away under the strip
    zf = FLOOR - G_FL / 2 - G_T / 2
    hf = G_FL - G_T
    plate_box(G_X[0], G_X[0] + G_T, G_Y[0], G_Y[1], zf, hf)
    plate_box(G_X[1] - G_T, G_X[1], G_Y[0], G_C[1] - sw / 2 - 0.004, zf, hf)
    plate_box(G_X[1] - G_T, G_X[1], G_C[1] + sw / 2 + 0.004, G_Y[1], zf, hf)
    for sgn in (-1, 1):
        plate_box(G_X[0], G_X[1], *sorted((G_C[1] + sgn * G_LY / 2, G_C[1] + sgn * (G_LY / 2 - G_T))), zf, hf)
    V.append(_vm("plate", M.merge(*plate), SS, "steel"))
    # hinge knuckles at the near edge (visual): two short barrels on pins
    for sgn in (-1, 1):
        V.append(_vm(f"knuckle{sgn}", M.tube(np.array([(0.0, -0.025, 0.0), (0.0, 0.025, 0.0)]), 0.004, n=14), SS, "steel",
                     offset=rel((HINGE[0] + 0.002, G_C[1] + sgn * 0.085, FLOOR - 0.006))))
    hinge = Hinge(pivot=(float(HINGE[0]), float(G_C[1]), float(HINGE[1])), axis="Y", damping=0.02, lower_deg=-100.0, upper_deg=0.0)
    return Part("grate", pos=tuple(map(float, c)), size=(0.003, 0.003, 0.003), shape="sphere", exact_z=True, mass=G_MASS,
                color=SS, friction=(0.6, 0.5), restitution=0.2, contact_offset=0.002, pieces=_hid(P),
                hide_root=True, vmeshes=tuple(V), com=(0.0, 0.0, 0.0), solver_iters=(32, 4), hinge=hinge)


# ======================================================================================================
# sediment bucket and its bail
# ======================================================================================================
def bucket_object() -> Part:
    c = np.array(B_C)
    P, V = [], []
    for sgn in (-1, 1):
        P.append(Piece("box", (B_T, 2 * B_HALF, B_H), (sgn * (B_HALF - B_T / 2), 0.0, 0.0)))
        P.append(Piece("box", (2 * B_HALF, B_T, B_H), (0.0, sgn * (B_HALF - B_T / 2), 0.0)))
    P.append(Piece("box", (2 * B_HALF, 2 * B_HALF, 0.003), (0.0, 0.0, -B_H / 2 + 0.0015)))
    # visuals: a square pail of perforated aluminium with a rolled top edge and a solid bottom band
    wall = M.loft([(-B_H / 2, 2 * B_HALF - 0.004, 2 * B_HALF - 0.004, 16.0), (B_H / 2, 2 * B_HALF, 2 * B_HALF, 16.0)],
                  axis="Z", n=80, cap0=True, cap1=False)
    V.append(_vm("pail", wall, ALU, "satin"))
    inner = M.loft([(-B_H / 2 + 0.003, 2 * B_HALF - 0.008, 2 * B_HALF - 0.008, 16.0), (B_H / 2 - 0.001, 2 * B_HALF - 0.004, 2 * B_HALF - 0.004, 16.0)],
                   axis="Z", n=80, cap0=True, cap1=False)
    V.append(_vm("pail_in", (inner[0], inner[1][:, ::-1]), (0.80, 0.81, 0.82), "satin"))   # faces turned inward
    rim = M.loft([(B_H / 2 - 0.004, 2 * B_HALF + 0.004, 2 * B_HALF + 0.004, 16.0), (B_H / 2 + 0.001, 2 * B_HALF + 0.004, 2 * B_HALF + 0.004, 16.0)],
                 axis="Z", n=80, cap0=False, cap1=False)
    V.append(_vm("rim", rim, ALU, "aluminium"))
    # perforations: rows of 6 mm holes (dark discs) on each side above the solid bottom band
    holes = []
    disc = M.revolve([(0.0, 0.0), (0.003, 0.0)], n=10)
    for side in range(4):
        R = _rz(side * math.pi / 2)
        for zz in np.arange(-B_H / 2 + 0.030, B_H / 2 - 0.014, 0.012):
            for t in np.arange(-B_HALF + 0.016, B_HALF - 0.012, 0.012):
                Rd = R @ np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
                holes.append(M.transform(disc, Rd, R @ np.array([B_HALF + 0.0004, t, zz])))
    V.append(_vm("holes", M.merge(*holes), (0.12, 0.12, 0.12), "matte"))
    for sgn in (-1, 1):            # bail ears riveted to the sides
        V.append(_vm(f"ear{sgn}", M.rbox((0.003, 0.022, 0.024), e=6.0, axis="X"), ALU, "aluminium",
                     offset=(sgn * (B_HALF + 0.0015), 0.0, BAIL_PIVOT_Z - B_C[2])))
    return Part("bucket", pos=tuple(map(float, c)), size=(0.003, 0.003, 0.003), shape="sphere", exact_z=True, mass=B_MASS,
                color=ALU, friction=(0.5, 0.4), restitution=0.15, contact_offset=0.003, pieces=_hid(P), hide_root=True,
                vmeshes=tuple(V), solver_iters=(24, 4))


def bail_object() -> Part:
    """The wire bail, its ears on the bucket's near and far walls (it turns about x), folded over toward -y onto the
    rim. Only the gripped top bar (with its 13 mm plastic sleeve) carries a collider: the legs run clear outside the
    bucket's walls and inside the sink's, so they would only ever graze."""
    piv = np.array([B_C[0], B_C[1], BAIL_PIVOT_Z])
    w = B_HALF + 0.004
    pts = [(-w, 0.0, 0.0), (-w, 0.0, BAIL_RISE * 0.75), (-w * 0.55, 0.0, BAIL_RISE), (w * 0.55, 0.0, BAIL_RISE),
           (w, 0.0, BAIL_RISE * 0.75), (w, 0.0, 0.0)]
    path = M.bend(pts, radius=0.03, steps=6)
    Rx = _rx(math.radians(BAIL_DOWN_DEG))
    V = [_vm("wire", M.tube(path, BAIL_R, n=10), (0.75, 0.76, 0.77), "steel", R=Rx)]
    sleeve = M.tube(np.array([(-0.045, 0.0, BAIL_RISE), (0.045, 0.0, BAIL_RISE)]), BAIL_R + 0.004, n=16)
    V.append(_vm("sleeve", sleeve, (0.10, 0.10, 0.10), "satin", R=Rx))
    mid = Rx @ np.array([0.0, 0.0, BAIL_RISE])
    P = [Piece("capsule", (BAIL_R + 0.004, 2 * 0.55 * w, BAIL_R + 0.004), tuple(map(float, mid)), axis="X")]
    hinge = Hinge(pivot=tuple(map(float, piv)), axis="X", damping=0.0004, lower_deg=-BAIL_STOP_DEG - BAIL_DOWN_DEG,
                  upper_deg=BAIL_STOP_DEG - BAIL_DOWN_DEG, parent="bucket", parent_pos=tuple(map(float, B_C)), collide=False)
    return Part("bail", pos=tuple(map(float, piv)), size=(0.004, 0.004, 0.004), shape="sphere", exact_z=True,
                mass=BAIL_MASS, color=(0.75, 0.76, 0.77), friction=(0.7, 0.6), contact_offset=0.002, pieces=_hid(P),
                hide_root=True, vmeshes=tuple(V), hinge=hinge)


# ======================================================================================================
# debris: two lemon wedges, a lime wedge, two bottle caps (rigid plastic food replicas / real caps)
# ======================================================================================================
def _wedge_mesh(length, radius, ang):
    """A citrus wedge: a sector of a prolate spheroid (length along x) between angles -ang/2..ang/2 about x."""
    us = np.linspace(0, math.pi, 22)
    vs = np.linspace(-ang / 2, ang / 2, 9)
    pts, tris = [], []
    for u in us:
        for v in vs:
            r = radius * math.sin(u)
            pts.append((-length / 2 * math.cos(u), r * math.sin(v), -r * math.cos(v)))
    nu, nv = len(us), len(vs)
    for i in range(nu - 1):
        for j in range(nv - 1):
            a, b, cc, d = i * nv + j, (i + 1) * nv + j, (i + 1) * nv + j + 1, i * nv + j + 1
            tris += [(a, b, cc), (a, cc, d)]
    base = len(pts)
    for side, v in ((0, -ang / 2), (1, ang / 2)):        # the two cut faces (flesh)
        k0 = len(pts)
        pts.append((0.0, 0.0, 0.0))
        for u in us:
            r = radius * math.sin(u)
            pts.append((-length / 2 * math.cos(u), r * math.sin(v), -r * math.cos(v)))
        for i in range(nu - 1):
            tris.append((k0, k0 + 1 + i, k0 + 2 + i) if side else (k0, k0 + 2 + i, k0 + 1 + i))
    return np.asarray(pts, float), np.asarray(tris, int)


DEBRIS = (  # name, kind, colour, local position in the bucket (x, y), yaw
    ("lemon1", "wedge", LEMON, (-0.045, 0.040), 0.3),
    ("lime1", "wedge", LIME, (0.040, -0.035), 1.9),
    ("cap1", "cap", CAP_RED, (0.050, 0.060), 0.0),
)
W_LEN, W_RAD, W_ANG = 0.075, 0.028, math.radians(60)
CAP_R, CAP_H = 0.0145, 0.012


def debris_objects():
    out = []
    floor_z = SEAT_Z + 0.003 + 0.004          # dropped in from 4 mm up (no initial overlap with the bucket's floor)
    for name, kind, col, (lx, ly), yaw in DEBRIS:
        R = _rz(yaw)
        if kind == "wedge":
            # lying on its peel, cut faces up: the spheroid's centre is W_RAD*cos(ang/2) above the floor
            m = _wedge_mesh(W_LEN, W_RAD, W_ANG)
            p, t = m
            zc = floor_z + W_RAD + 0.001
            V = [("peel", p @ R.T, t, col, "satin", 1.0)]
            # flesh faces in a paler colour (drawn slightly inside the cut planes)
            flesh = (0.98, 0.93, 0.55) if col == LEMON else (0.72, 0.86, 0.45)
            fp = p.copy()
            V.append(("flesh", (fp * np.array([0.98, 0.9, 0.9])) @ R.T, t, flesh, "satin", 1.0))
            P = [Piece("capsule", (0.010, W_LEN - 0.03, 0.010), tuple(map(float, R @ np.array([0, 0, -0.014]))),
                       rot=(0.0, 0.0, yaw), axis="X", hidden=True),
                 Piece("box", (W_LEN * 0.6, 0.020, 0.010), tuple(map(float, R @ np.array([0, 0, -0.022]))), rot=(0, 0, yaw), hidden=True)]
            out.append(Part(name, pos=(B_C[0] + lx, B_C[1] + ly, zc), size=(0.005, 0.005, 0.005), shape="sphere", exact_z=True,
                            mass=0.018, color=col, friction=(0.45, 0.35), restitution=0.1, contact_offset=0.002, pieces=tuple(P),
                            hide_root=True, vmeshes=tuple(V), max_depenetration=0.3))
        else:
            prof = M.fillet_profile(0.0, CAP_R, -CAP_H / 2, CAP_H / 2, f=0.0015)
            body = M.revolve(prof, n=40)
            V = [("cap", body[0], body[1], col, "satin", 1.0)]
            ribs = []
            for k in range(28):
                a = 2 * math.pi * k / 28
                ribs.append(M.transform(M.rbox((0.0012, 0.0012, CAP_H * 0.75), e=4.0), _rz(a),
                                        (CAP_R * math.cos(a), CAP_R * math.sin(a), -0.0005)))
            V.append(("ribs", *M.merge(*ribs), col, "satin", 1.0))
            out.append(Part(name, pos=(B_C[0] + lx, B_C[1] + ly, floor_z + CAP_H / 2 + 0.001), size=(CAP_R, CAP_H, CAP_R),
                            shape="cylinder", exact_z=True, mass=0.003, color=col, friction=(0.45, 0.35), restitution=0.2,
                            contact_offset=0.002, hide_root=True, vmeshes=tuple(V), max_depenetration=0.3))
    return out


# ======================================================================================================
# grate hook and wastebasket
# ======================================================================================================
def hook_object() -> Part:
    """Lying on the floor left of the sink: shaft along x, T handle at the far (+x) end, the foot at the near end
    pointing -y (all flat on the tile)."""
    c = np.array(H_POS)
    x0, x1 = -H_LEN / 2, H_LEN / 2
    path = M.bend([(x0, -H_FOOT, 0.0), (x0, 0.0, 0.0), (x1, 0.0, 0.0)], radius=0.008, steps=6)
    V = [_vm("rod", M.tube(path, H_R, n=12), STEEL_DK, "steel_painted")]
    V.append(_vm("T", M.tube(np.array([(x1, -H_T / 2, 0.0), (x1, H_T / 2, 0.0)]), H_R + 0.0045, n=16), (0.85, 0.12, 0.08), "satin"))
    P = [Piece("capsule", (H_R, H_LEN, H_R), (0.0, 0.0, 0.0), axis="X", hidden=True),
         Piece("capsule", (H_R + 0.0045, H_T, H_R + 0.0045), (x1, 0.0, 0.0), axis="Y", hidden=True),
         Piece("capsule", (H_R, H_FOOT, H_R), (x0, -H_FOOT / 2, 0.0), axis="Y", hidden=True)]
    return Part("hook", pos=tuple(map(float, c + np.array([0.0, 0.0, 0.0045]))), size=(0.003, 0.003, 0.003), shape="sphere",
                exact_z=True, mass=H_MASS, color=STEEL_DK, friction=(0.7, 0.6), restitution=0.2, contact_offset=0.002,
                pieces=tuple(P), hide_root=True, vmeshes=tuple(V))


def tub_object() -> Part:
    c = np.array([TUB_C[0], TUB_C[1], TOP + TUB_H / 2])
    t = 0.004
    draft = 0.018                                   # the walls lean out this much over the height
    P = [Piece("box", (TUB_L - 2 * draft, TUB_W - 2 * draft, t), (0.0, 0.0, -TUB_H / 2 + t / 2), hidden=True)]
    ang_x = math.atan2(draft, TUB_H)
    for sgn in (-1, 1):
        P.append(Piece("box", (t, TUB_W - draft, TUB_H), (sgn * (TUB_L / 2 - draft / 2), 0.0, 0.0), rot=(0.0, -sgn * ang_x, 0.0), hidden=True))
        P.append(Piece("box", (TUB_L - draft, t, TUB_H), (0.0, sgn * (TUB_W / 2 - draft / 2), 0.0), rot=(sgn * ang_x, 0.0, 0.0), hidden=True))
    outer = M.loft([(-TUB_H / 2, TUB_L - 2 * draft, TUB_W - 2 * draft, 7.0), (TUB_H / 2 - 0.010, TUB_L - 0.004, TUB_W - 0.004, 7.0),
                    (TUB_H / 2, TUB_L + 0.006, TUB_W + 0.006, 7.0)], axis="Z", n=96, cap0=True, cap1=False)
    inner = M.loft([(-TUB_H / 2 + t, TUB_L - 2 * draft - 2 * t, TUB_W - 2 * draft - 2 * t, 7.0),
                    (TUB_H / 2, TUB_L - 2 * t, TUB_W - 2 * t, 7.0)], axis="Z", n=96, cap0=True, cap1=False)
    lip = M.loft([(TUB_H / 2 - 0.002, TUB_L + 0.006, TUB_W + 0.006, 7.0), (TUB_H / 2 + 0.003, TUB_L + 0.006, TUB_W + 0.006, 7.0)], axis="Z", n=96,
                 cap0=False, cap1=False)
    V = [_vm("tub", outer, BIN_GREY, "satin"), _vm("tub_in", inner, (0.40, 0.41, 0.42), "satin"), _vm("lip", lip, BIN_GREY, "satin")]
    return Part("tub", pos=tuple(map(float, c)), size=(0.003, 0.003, 0.003), shape="sphere", exact_z=True, mass=TUB_MASS,
                color=BIN_GREY, friction=(0.6, 0.5), restitution=0.1, contact_offset=0.003, pieces=tuple(P), hide_root=True,
                vmeshes=tuple(V))


WITH_GRATE = False              # v3b: the grate step is cut (its hinged opening stalled on the arm's own links)


def scene_objects():
    return (floor_object(), *((grate_object(),) if WITH_GRATE else ()), bucket_object(), bail_object(), *debris_objects(), tub_object())


def mat(p):
    """3x3 rotation of a root pose (x, y, z, qx, qy, qz, qw)."""
    x, y, z, w = (float(v) for v in p[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
