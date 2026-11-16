"""Geometry and measurements for replacing one cracked tile on a tiled raised step, back-buttered.

On the bench: a raised step (a concrete plinth 136 mm high) tiled with 75 x 150 mm glazed ceramic
tiles, four rows deep and three across, the spacers still in the joints. One tile of the front
row was cracked and has been chipped out and its old adhesive scraped off, leaving a gap boxed in by
its neighbours. On a sheet of cardboard on the floor in front of the step lies a stack of spare tiles
with the new tile leaning against it, and a tub of mixed tile adhesive stands beside the step with a
notched trowel resting across its rim.

Tile adhesive cannot be troweled into a gap boxed in by tiles, so it goes on the back of the tile
("back-buttering"): one hand holds the tile while the other combs adhesive over its back. The
tile is then turned over, set in the gap level with its neighbours and pressed down to bed it.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench (the floor).
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

CONCRETE = (0.58, 0.58, 0.56)
RENDER = (0.74, 0.72, 0.68)          # the step's plastered riser
TILE_FACE = (0.93, 0.93, 0.90)       # glazed white face
TILE_BODY = (0.80, 0.62, 0.48)       # fired biscuit: the unglazed back and edges
ADHESIVE = (0.62, 0.63, 0.62)        # grey cement-based tile adhesive
SPACER = (0.97, 0.97, 0.97)
CARD_COL = (0.70, 0.56, 0.38)
TUB = (0.92, 0.92, 0.92)
TUB_BAND = (0.15, 0.35, 0.70)
STEEL = (0.70, 0.71, 0.73)
HANDLE_COL = (0.72, 0.18, 0.12)

# -- tiles ---------------------------------------------------------------------------------------
TILE = (0.075, 0.150, 0.008)         # x (depth on the step) by y (along it) by thickness
TILE_MASS = 0.20
JOINT = 0.002
BED = 0.006                          # adhesive under a bedded tile

# -- the raised step (fixed) --------------------------------------------------------------------
SUB_TOP = TOP + 0.136                # the concrete plinth's top
FACE_Z = SUB_TOP + BED + TILE[2]     # the tiles' face: 150 mm above the floor
NOSE_X = 0.282                       # front edge of the front row of tiles
OVERHANG = 0.022                     # the front row stands this far past the riser, as nosing
RISER_X = NOSE_X + OVERHANG
ROWS, COLS = 4, 3
ROW_X = tuple(NOSE_X + (i + 0.5) * TILE[0] + i * JOINT for i in range(ROWS))
COL_Y = tuple(-0.046 + j * (TILE[1] + JOINT) for j in range(COLS))
GAP_RC = (0, 1)                      # the chipped-out tile: front row, middle
STEP_X = (RISER_X, ROW_X[-1] + TILE[0] / 2 + 0.010)
STEP_Y = (COL_Y[0] - TILE[1] / 2 - 0.010, COL_Y[-1] + TILE[1] / 2 + 0.010)
GAP = np.array([ROW_X[GAP_RC[0]], COL_Y[GAP_RC[1]], SUB_TOP + BED + TILE[2] / 2])   # where the new tile's centre goes


def _cross(at, x, y):
    """A tile spacer's cross standing in a joint intersection, its legs proud of the face."""
    z = FACE_Z - 0.002
    return [Piece("box", (JOINT, 0.014, 0.006), at(x, y, z), color=SPACER, collide=False),
            Piece("box", (0.014, JOINT, 0.006), at(x, y, z), color=SPACER, collide=False)]


def step_object() -> Part:
    """Concrete plinth (root) with its plastered riser, the laid tiles on their beds, spacers in the
    joints, and scraped traces of old adhesive in the gap."""
    sx = STEP_X[1] - STEP_X[0]
    sy = STEP_Y[1] - STEP_Y[0]
    h = SUB_TOP - TOP
    c = np.array([(STEP_X[0] + STEP_X[1]) / 2, (STEP_Y[0] + STEP_Y[1]) / 2, TOP + h / 2])

    def at(x, y, z):
        return tuple(float(v) for v in np.array([x, y, z]) - c)

    p = [Piece("box", (0.003, sy, h), at(RISER_X - 0.0015, c[1], c[2]), color=RENDER, collide=False)]
    for i, x in enumerate(ROW_X):
        for j, y in enumerate(COL_Y):
            if (i, j) == GAP_RC:
                continue
            bx = x if i else x + OVERHANG / 2           # the nosing row's bed stops at the riser
            p.append(Piece("box", (TILE[0] - (0 if i else OVERHANG), TILE[1], BED), at(bx, y, SUB_TOP + BED / 2),
                           color=ADHESIVE))
            p.append(Piece("box", TILE, at(x, y, FACE_Z - TILE[2] / 2), color=TILE_FACE))
    # spacers at every joint intersection inside the field, and T-legs along the front joints
    for i in range(ROWS - 1):
        xj = ROW_X[i] + TILE[0] / 2 + JOINT / 2
        for j in range(COLS - 1):
            p += _cross(at, xj, COL_Y[j] + TILE[1] / 2 + JOINT / 2)
    for j in range(COLS - 1):
        yj = COL_Y[j] + TILE[1] / 2 + JOINT / 2
        p.append(Piece("box", (0.014, JOINT, 0.006), at(ROW_X[0] - 0.012, yj, FACE_Z - 0.002), color=SPACER,
                       collide=False))
    # old adhesive scraped off the gap's floor, a few traces left
    for dx, dy, w in ((-0.020, -0.04, 0.030), (0.010, 0.02, 0.045), (0.022, -0.01, 0.020)):
        p.append(Piece("box", (0.006, w, 0.0006), at(GAP[0] + dx + 0.009, GAP[1] + dy, SUB_TOP + 0.0003),
                       color=(0.66, 0.66, 0.64), collide=False))
    return Part("step", pos=tuple(float(v) for v in c), size=(sx, sy, h), exact_z=True, color=CONCRETE,
                friction=(0.8, 0.7), restitution=0.0, contact_offset=0.002, pieces=tuple(p))


# -- cardboard on the floor, spare tiles stacked on it, the new tile leaning against them ------------------
CARD_C = (0.195, 0.100)
CARD = (0.150, 0.210, 0.002)
CARD_TOP = TOP + CARD[2]
SPARES_C = (0.215, 0.100)
SPARES_N = 5
STACK_TOP = CARD_TOP + SPARES_N * 0.008
STACK_NEAR = SPARES_C[0] - 0.075 / 2       # the stack's near face

# Tile frame: its centre; +z is the back (unglazed, to be buttered), -z the glazed face, x across its
# 75 mm width, y along its 150 mm length.
LEAN_DEG = 15.0                      # the new tile stands on a long edge, its back against the stack's top edge


def lean_pose():
    """The new tile's pose leaning on the stack: centre and rotation (back toward +x, glaze toward the arms,
    local +x down the tile)."""
    th = math.radians(90.0 + LEAN_DEG)
    r = np.array([[math.cos(th), 0.0, math.sin(th)], [0.0, 1.0, 0.0], [-math.sin(th), 0.0, math.cos(th)]])
    c, s = math.cos(math.radians(LEAN_DEG)), math.sin(math.radians(LEAN_DEG))
    cz = CARD_TOP + 0.0375 * c + 0.004 * s + 0.0003
    cx = STACK_NEAR - (0.004 + s * (STACK_TOP - cz)) / c - 0.0003
    return np.array([cx, SPARES_C[1], cz]), r


TILE_START = lean_pose()[0] + np.array([0.0, 0.0, 0.15])   # spawned clear; Dynamics.reset stands it in place
CELL = 0.0125
NX, NY = int(round(TILE[0] / CELL)), int(round(TILE[1] / CELL))
CELLS = NX * NY
BUTTER = BED                         # buttered layer, combed into ridges (as thick as a bed)
INSET = 0.025                        # the hidden bed collider stops this far in from every edge (clear of a hand's grip)


def cell_xy(k):
    i, j = divmod(k, NY)
    return np.array([-TILE[0] / 2 + (i + 0.5) * CELL, -TILE[1] / 2 + (j + 0.5) * CELL])


def tile_object() -> Part:
    """The tile, root piece the fired body. The glaze is a thin face piece; the adhesive cells on its
    back are hidden until troweled on. A hidden inset slab stands for the buttered layer's body, so a
    buttered tile set in the gap bears on the concrete at the right height."""
    p = [Piece("box", (TILE[0], TILE[1], 0.0008), (0.0, 0.0, -TILE[2] / 2 - 0.0002), color=TILE_FACE, collide=False),
         Piece("box", (TILE[0] - 2 * INSET, TILE[1] - 2 * INSET, BUTTER),
               (0.0, 0.0, TILE[2] / 2 + BUTTER / 2), color=ADHESIVE)]
    for k in range(CELLS):
        x, y = cell_xy(k)
        # a ridge and a furrow per cell, like a notched trowel's comb
        p.append(Piece("box", (CELL * 0.98, CELL * 0.6, BUTTER), (float(x), float(y - CELL * 0.2), TILE[2] / 2 + BUTTER / 2),
                       color=ADHESIVE, collide=False))
    return Part("tile", pos=tuple(float(v) for v in TILE_START), size=TILE, exact_z=True, mass=TILE_MASS,
                color=TILE_BODY, friction=(0.9, 0.8), restitution=0.0, contact_offset=0.0015, pieces=tuple(p))


BED_PIECE = 1                        # the hidden slab's index among the tile's pieces
CELL_PIECE0 = 2

def spares_object() -> Part:
    """The cardboard sheet (root) on the floor, and the stack of spare tiles on it, face-down, slightly skewed."""
    c = np.array([CARD_C[0], CARD_C[1], TOP + CARD[2] / 2])
    p = []
    for k in range(SPARES_N):
        off = (SPARES_C[0] - c[0] + 0.002 * (k % 2), SPARES_C[1] - c[1] - 0.003 * (k % 3), CARD[2] / 2 + (k + 0.5) * TILE[2])
        rot = (0.0, 0.0, 0.03 * ((k % 3) - 1))
        p.append(Piece("box", TILE, off, rot=rot, color=TILE_BODY))
        p.append(Piece("box", (TILE[0], TILE[1], 0.0008), (off[0], off[1], off[2] - 0.0036), rot=rot, color=TILE_FACE,
                       collide=False))
    return Part("spares", pos=tuple(float(v) for v in c), size=CARD, exact_z=True, color=CARD_COL,
                friction=(0.8, 0.7), restitution=0.0, pieces=tuple(p))


# -- tub of mixed adhesive -----------------------------------------------------------------------------
TUB_C = (0.240, -0.200)
TUB_R = 0.060                        # outside radius at the rim: the trowel's 140 mm blade spans it
TUB_WALL = 0.006
TUB_H = 0.150
RIM_Z = TOP + TUB_H - 0.003         # top of the staves
ADH_Z = RIM_Z - 0.030                # the adhesive's surface in the tub


def tub_object() -> Part:
    """An open plastic tub (root: its bottom), its wall as 20 staves, a rolled rim, a blue label band,
    the wire bail lying down, and the mixed adhesive inside, a scoop taken out of it."""
    c = np.array([TUB_C[0], TUB_C[1], TOP + 0.003])
    p = []
    n = 20
    w = 2 * TUB_R * math.tan(math.pi / n) + 0.001
    for k in range(n):
        a = 2 * math.pi * k / n
        r = TUB_R - TUB_WALL / 2
        p.append(Piece("box", (TUB_WALL, w, TUB_H - 0.006), (r * math.cos(a), r * math.sin(a), (TUB_H - 0.006) / 2 + 0.003 - 0.003),
                       rot=(0.0, 0.0, a), color=TUB))
        p.append(Piece("box", (TUB_WALL + 0.002, w, 0.035), (r * math.cos(a), r * math.sin(a), 0.070 - 0.003),
                       rot=(0.0, 0.0, a), color=TUB_BAND, collide=False))
    p.append(Piece("cylinder", (TUB_R - TUB_WALL - 0.001, ADH_Z - TOP - 0.006, TUB_R - TUB_WALL - 0.001),
                   (0.0, 0.0, (ADH_Z - TOP - 0.006) / 2), color=ADHESIVE))
    p.append(Piece("cylinder", (0.030, 0.010, 0.030), (0.018, -0.010, ADH_Z - TOP - 0.003 - 0.003 + 0.004), color=ADHESIVE,
                   collide=False))
    p.append(Piece("box", (0.004, 2 * TUB_R + 0.006, 0.004), (0.0, 0.0, 0.010), color=STEEL, collide=False))  # bail, dropped
    return Part("tub", pos=tuple(float(v) for v in c), size=(TUB_R - 0.002, 0.006, TUB_R - 0.002), shape="cylinder", exact_z=True,
                color=TUB, friction=(0.6, 0.5), restitution=0.0, pieces=tuple(p))


# Trowel frame: blade centre; blade in the local xy plane, its notched working edge along y at +x,
# the handle along y above it.
BLADE = (0.070, 0.140, 0.0012)
NOTCH = 0.006
HANDLE = (0.014, 0.110)              # radius, length
HANDLE_Z = 0.042                     # handle axis above the blade
TROWEL_MASS = 0.26
TROWEL_START = np.array([TUB_C[0], TUB_C[1], RIM_Z + BLADE[2] / 2])   # resting across the tub's rim
TROWEL_YAW = -math.pi / 2           # handle along x, the notched edge on the right (-y)


def trowel_object() -> Part:
    p = [Piece("cylinder", (HANDLE[0], HANDLE[1], HANDLE[0]), (-0.004, 0.0, HANDLE_Z), axis="Y", color=HANDLE_COL,
               friction=(1.2, 1.0)),
         Piece("box", (0.004, 0.012, HANDLE_Z - HANDLE[0]), (-0.004, -0.038, (HANDLE_Z - HANDLE[0]) / 2 + 0.0006), color=STEEL),
         Piece("box", (0.004, 0.012, HANDLE_Z - HANDLE[0]), (-0.004, 0.038, (HANDLE_Z - HANDLE[0]) / 2 + 0.0006), color=STEEL),
         Piece("box", (0.020, 0.090, 0.0016), (-0.004, 0.0, 0.0014), color=STEEL)]       # the tang along the blade
    # the notched edge: 6 mm square notches, drawn as dark gaps cut into the steel
    n = int(BLADE[1] / (2 * NOTCH))
    for k in range(n):
        y = -BLADE[1] / 2 + (2 * k + 1.5) * NOTCH
        if abs(y) < BLADE[1] / 2 - NOTCH / 2:
            p.append(Piece("box", (NOTCH, NOTCH, BLADE[2] + 0.0004), (BLADE[0] / 2 - NOTCH / 2 + 0.0002, y, 0.0),
                           color=(0.10, 0.10, 0.10), collide=False))
    # its load of adhesive, along the plain back edge
    p.append(Piece("box", (0.024, 0.100, 0.007), (-BLADE[0] / 2 + 0.016, 0.0, 0.0042), color=ADHESIVE, collide=False))
    return Part("trowel", pos=tuple(float(v) for v in TROWEL_START), size=BLADE, exact_z=True, mass=TROWEL_MASS,
                color=STEEL, material="steel", friction=(0.5, 0.4), restitution=0.0, contact_offset=0.0015,
                com=(-0.004, 0.0, 0.020), yaw=TROWEL_YAW, pieces=tuple(p))


LOAD_PIECE = None                    # set below: index of the load blob among the trowel's pieces


def _count_notches():
    n = int(BLADE[1] / (2 * NOTCH))
    return sum(1 for k in range(n) if abs(-BLADE[1] / 2 + (2 * k + 1.5) * NOTCH) < BLADE[1] / 2 - NOTCH / 2)


LOAD_PIECE = 4 + _count_notches()


# -- measurements -------------------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def trowel_edge(pose):
    """World ends of the notched working edge and the blade's normal."""
    p = np.asarray(pose, float)
    r = quat_matrix(p[3:7])
    a = p[:3] + r @ np.array([BLADE[0] / 2, -BLADE[1] / 2, 0.0])
    b = p[:3] + r @ np.array([BLADE[0] / 2, BLADE[1] / 2, 0.0])
    return a, b, r[:, 2]


def handle_point(pose):
    p = np.asarray(pose, float)
    return p[:3] + quat_matrix(p[3:7]) @ np.array([-0.004, 0.0, HANDLE_Z])


EDGE_BAND = 0.0065                  # the edge within this of the buttered layer's top combs it on
ANGLE = (25.0, 75.0)                 # blade to the tile's back, degrees


def combed_cells(tile_pose, trowel_pose):
    """Cells of the tile's back the trowel's notched edge is combing adhesive onto now."""
    t = np.asarray(tile_pose, float)
    rt = quat_matrix(t[3:7])
    a, b, n = trowel_edge(trowel_pose)
    back = rt[:, 2]
    if back[2] < 0.7:                                  # only on a back facing up
        return []
    ang = math.degrees(math.acos(min(1.0, abs(float(n @ back)))))
    if not ANGLE[0] <= ang <= ANGLE[1]:
        return []
    out = set()
    for s in np.linspace(0.0, 1.0, 29):
        q = rt.T @ (a + s * (b - a) - t[:3])
        if abs(q[2] - (TILE[2] / 2 + BUTTER)) > EDGE_BAND:
            continue
        i = math.floor((q[0] + TILE[0] / 2) / CELL)
        j = math.floor((q[1] + TILE[1] / 2) / CELL)
        if 0 <= i < NX and 0 <= j < NY:
            out.add(i * NY + j)
    return sorted(out)


def in_heap(trowel_pose):
    """The notched edge dug into the adhesive in the tub: the trowel picks up a fresh load."""
    a, b, _ = trowel_edge(trowel_pose)
    m = (a + b) / 2
    return (math.hypot(m[0] - TUB_C[0], m[1] - TUB_C[1]) < TUB_R - TUB_WALL - 0.010
            and ADH_Z - 0.012 < m[2] < ADH_Z + 0.010)


def tile_set(tile_pose):
    """Errors of the tile in the gap: centre offset (m), yaw (deg), lippage (m, face above or below the
    neighbours' face), and whether it is face-up (buttered back down)."""
    t = np.asarray(tile_pose, float)
    r = quat_matrix(t[3:7])
    face_up = r[2, 2] < -0.97
    off = float(np.hypot(t[0] - GAP[0], t[1] - GAP[1]))
    yaw = abs(math.degrees(math.atan2(r[0, 1], r[1, 1])))          # tile's y axis against world y
    yaw = min(yaw, 180.0 - yaw)
    # the face's four corners against the neighbours' face height
    corners = [t[:3] + r @ np.array([sx * TILE[0] / 2, sy * TILE[1] / 2, -TILE[2] / 2]) for sx in (-1, 1) for sy in (-1, 1)]
    lip = max(abs(float(c[2]) - FACE_Z) for c in corners)
    return off, yaw, lip, bool(face_up)


def matrix_quat_xyzw(r):
    """Rotation matrix to an XYZW quaternion."""
    r = np.asarray(r, float)
    t = r[0, 0] + r[1, 1] + r[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return np.array(q)
