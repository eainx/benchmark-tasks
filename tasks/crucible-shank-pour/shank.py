"""Geometry and measurements for pouring a sand mould from a crucible in a two-man shank.

On the bench: two closed sand moulds (fixed) side by side, each with a ceramic pouring cup and a
riser, and a crucible of molten bronze held in a steel shank -- a ring clamped round the crucible
with a handle out to each side. One mould is wet: its sand is dark with water and it stands in a
puddle. Molten metal poured into a damp mould turns the water to steam and blows metal back out of
the cup, so only the dry mould may be poured. The crucible is hot; it is carried and tipped only by
the shank's two handles.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

SAND = (0.36, 0.31, 0.24)
FLASK = (0.32, 0.34, 0.36)
CERAMIC = (0.86, 0.83, 0.77)
CRUCIBLE = (0.55, 0.50, 0.46)      # clay-graphite
LINING = (0.80, 0.76, 0.68)
METAL = (1.00, 0.56, 0.12)
GLOW = (1.00, 0.42, 0.06)
STEEL = (0.30, 0.31, 0.33)
GRIP = (0.12, 0.10, 0.09)          # heat-proof handle sleeves

# -- molten charge ----------------------------------------------------------------------------
BEAD_R = 0.0045
BEAD_MASS = 0.0018
BEAD_PITCH = 0.010
BEAD_FRICTION = (0.06, 0.04)
BEAD_COUNT = 36                      # 4 x 3 x 3

# -- crucible and shank -----------------------------------------------------------------------
CRU_R = 0.036                        # inside radius at the rim
CRU_H = 0.060
CRU_WALL = 0.004
CRU_BORE = 0.024                     # inside radius at the floor
CRU_FLOOR = 0.004
CRU_STAVES = 24
CRU_PLATES = 20
CRU_MASS = 0.150                     # crucible and shank together
CRU_XY = (0.200, 0.000)
SHANK_Z = 0.048                      # handle axis above the crucible's foot
RING_R = CRU_R + CRU_WALL            # the shank ring clamps the outside of the crucible
RING_W = 0.006
HANDLE_W = 0.018                     # 18 mm square bar: it cannot roll in the hand when the crucible is tipped
HANDLE_IN = RING_R + RING_W          # where the bar leaves the ring
HANDLE_OUT = 0.125
GRIP_Y = 0.098                       # where a hand takes each handle, from the crucible's axis

# -- mould ------------------------------------------------------------------------------------
MOULD_X = 0.365
MOULD_YS = (0.080, -0.080)           # left mould, right mould
WET = 0                              # the left mould is wet
MOULD = (0.160, 0.140)               # each flask outside, x by y
WET_SAND = (0.20, 0.17, 0.13)
PUDDLE = (0.45, 0.55, 0.62)
DRAG_H = 0.020
COPE_H = 0.014
FLASK_T = 0.008
CAVITY_H = 0.013
BASIN_X = 0.315                      # pouring cups, over the sprues, at each mould's near edge
BASIN_TOP_R = 0.045
BASIN_BOT_R = 0.016
BASIN_H = 0.020
SPRUE = 0.028
RISER_DX, RISER_DY = 0.050, 0.035    # riser, from the mould's centre
RISER = 0.020
CUP_TOP = TOP + DRAG_H + COPE_H + BASIN_H

BEADS = tuple(f"metal{i:02d}" for i in range(BEAD_COUNT))


def _ring(count, radius, height, zc, wall, colour=None, collide=True):
    mid = radius + wall / 2
    chord = 2 * math.pi * (radius + wall) / count
    return [Piece("box", (wall, chord, height), (mid * math.cos(2 * math.pi * i / count),
                  mid * math.sin(2 * math.pi * i / count), zc), rot=(0.0, 0.0, 2 * math.pi * i / count),
                  color=colour, collide=collide) for i in range(count)]


def _cone(count, top_r, bottom_r, height, zc, thickness=0.003, colour=None):
    slant = math.hypot(top_r - bottom_r, height)
    half = math.atan2(top_r - bottom_r, height)
    mid = (top_r + bottom_r) / 2
    chord = 2 * math.pi * top_r / count
    return [Piece("box", (slant, chord, thickness), (mid * math.cos(2 * math.pi * i / count),
                  mid * math.sin(2 * math.pi * i / count), zc), rot=(0.0, math.pi / 2 + half, 2 * math.pi * i / count),
                  color=colour) for i in range(count)]


def crucible_object() -> Part:
    """Crucible (root: its floor) and shank, one rigid body. Local z = 0 is the crucible's foot."""
    f = CRU_FLOOR
    body = _ring(CRU_STAVES, CRU_R, CRU_H, CRU_H / 2 - f / 2, CRU_WALL, colour=CRUCIBLE)
    lining = [dataclasses.replace(p, friction=(0.04, 0.03), offset=(p.offset[0], p.offset[1], p.offset[2] - f / 2))
              for p in _cone(CRU_PLATES, CRU_R, CRU_BORE, CRU_H, CRU_H / 2, colour=LINING)]
    zs = SHANK_Z - f / 2
    shank = _ring(18, RING_R, 0.014, zs, RING_W, colour=STEEL)
    for s in (-1, 1):
        length = HANDLE_OUT - HANDLE_IN
        shank.append(Piece("box", (HANDLE_W, length, HANDLE_W), (0.0, s * (HANDLE_IN + length / 2), zs),
                           color=STEEL, friction=(1.6, 1.4)))
        # heat-proof sleeve where the hand goes, and an end knob
        shank.append(Piece("box", (HANDLE_W + 0.003, 0.050, HANDLE_W + 0.003), (0.0, s * GRIP_Y, zs),
                           color=GRIP, collide=False))
        shank.append(Piece("sphere", (0.013, 0.013, 0.013), (0.0, s * HANDLE_OUT, zs), color=STEEL, collide=False))
    return Part("crucible", pos=(CRU_XY[0], CRU_XY[1], TOP + f / 2), size=(RING_R, f, RING_R), shape="cylinder",
                exact_z=True, mass=CRU_MASS, color=CRUCIBLE, friction=(1.2, 1.0), restitution=0.05, contact_offset=0.001,
                com=(0.0, 0.0, 0.020), pieces=tuple(body) + tuple(lining) + tuple(shank))


def bead_objects():
    out = []
    nx, ny = 4, 3
    for i, name in enumerate(BEADS):
        layer, rem = divmod(i, nx * ny)
        row, col = divmod(rem, nx)
        out.append(Part(name, pos=(CRU_XY[0] + (col - (nx - 1) / 2) * BEAD_PITCH,
                                   CRU_XY[1] + (row - (ny - 1) / 2) * BEAD_PITCH,
                                   TOP + CRU_FLOOR + BEAD_R + 0.001 + layer * BEAD_PITCH),
                        size=(BEAD_R,) * 3, shape="sphere", exact_z=True, mass=BEAD_MASS, color=METAL, emissive=GLOW,
                        friction=BEAD_FRICTION, restitution=0.05, contact_offset=0.001, max_depenetration=0.3))
    return out


def _sand_plate(x0, x1, y0, y1, z, t, holes):
    """A sand plate over [x0,x1]x[y0,y1] centred at height z, thickness t, with square holes [(cx, cy, size)]."""
    pieces = []
    xs = sorted({x0, x1, *[cx - s / 2 for cx, cy, s in holes], *[cx + s / 2 for cx, cy, s in holes]})
    for a, b in zip(xs, xs[1:]):
        mid = (a + b) / 2
        cuts = [(cy - s / 2, cy + s / 2) for cx, cy, s in holes if cx - s / 2 <= mid <= cx + s / 2]
        ys = [y0] + [v for c in sorted(cuts) for v in c] + [y1]
        for c, d in zip(ys[::2], ys[1::2]):
            if d - c > 1e-4 and b - a > 1e-4:
                pieces.append(Piece("box", (b - a, d - c, t), ((a + b) / 2, (c + d) / 2, z), color=SAND))
    return pieces


def basin_xy(k):
    return (BASIN_X, MOULD_YS[k])


def mould_object(k) -> Part:
    """One closed mould, fixed: steel flask, drag with its cavity open at the parting line, the cope
    slab over it with sprue and riser through it, and the ceramic pouring cup on top."""
    cx, cy = MOULD_X, MOULD_YS[k]
    BASIN_XY = basin_xy(k)
    RISER_XY = (cx + RISER_DX, cy + RISER_DY)
    lx, ly = MOULD
    floor_t = DRAG_H - CAVITY_H
    p = []
    wall_h = DRAG_H + COPE_H
    for s in (-1, 1):
        p.append(Piece("box", (FLASK_T, ly, wall_h), (s * (lx - FLASK_T) / 2, 0.0, wall_h / 2 - floor_t / 2), color=FLASK))
        p.append(Piece("box", (lx - 2 * FLASK_T, FLASK_T, wall_h), (0.0, s * (ly - FLASK_T) / 2, wall_h / 2 - floor_t / 2),
                       color=FLASK))
    rim = 0.018
    ix, iy = lx / 2 - FLASK_T, ly / 2 - FLASK_T
    zc = floor_t / 2 + CAVITY_H / 2
    for s in (-1, 1):
        p.append(Piece("box", (rim, 2 * iy, CAVITY_H), (s * (ix - rim / 2), 0.0, zc), color=SAND))
        p.append(Piece("box", (2 * ix - 2 * rim, rim, CAVITY_H), (0.0, s * (iy - rim / 2), zc), color=SAND))
    cope_z = DRAG_H + COPE_H / 2 - floor_t / 2
    holes = [(BASIN_XY[0] - cx, BASIN_XY[1] - cy, SPRUE), (RISER_XY[0] - cx, RISER_XY[1] - cy, RISER)]
    p += _sand_plate(-ix, ix, -iy, iy, cope_z, COPE_H, holes)
    bx, by = BASIN_XY[0] - cx, BASIN_XY[1] - cy
    for c in _cone(18, BASIN_TOP_R, BASIN_BOT_R, BASIN_H, cope_z + COPE_H / 2 + BASIN_H / 2, colour=CERAMIC):
        p.append(dataclasses.replace(c, offset=(c.offset[0] + bx, c.offset[1] + by, c.offset[2])))
    rx, ry = RISER_XY[0] - cx, RISER_XY[1] - cy
    for c in _ring(14, RISER / 2 + 0.002, 0.008, cope_z + COPE_H / 2 + 0.004, 0.004, colour=CERAMIC):
        p.append(dataclasses.replace(c, offset=(c.offset[0] + rx, c.offset[1] + ry, c.offset[2])))
    sand = SAND
    if k == WET:
        # the wet mould: sand dark with water, a wet sheen on the cope, standing in a puddle
        sand = WET_SAND
        p = [dataclasses.replace(q, color=WET_SAND) if q.color == SAND else q for q in p]
        p.append(Piece("box", (2 * ix - 0.004, 2 * iy - 0.004, 0.0008), (0.0, 0.0, cope_z + COPE_H / 2 + 0.0005),
                       color=PUDDLE, opacity=0.35, collide=False))
        p.append(Piece("cylinder", (0.105, 0.0012, 0.105), (0.0, 0.030, -floor_t / 2 + 0.0006), color=PUDDLE, opacity=0.6,
                       collide=False))
    return Part(f"mould_{k}", pos=(cx, cy, TOP + floor_t / 2), size=(lx - 2 * FLASK_T, ly - 2 * FLASK_T, floor_t), exact_z=True,
                color=sand, friction=(0.6, 0.5), contact_offset=0.0015, pieces=tuple(p))


# -- measurements ----------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def crucible_local(point, pose):
    """``point`` in the crucible's frame, origin at its foot."""
    pose = np.asarray(pose, float)
    return (np.asarray(point, float) - pose[:3]) @ quat_matrix(pose[3:7]) + np.array([0.0, 0.0, CRU_FLOOR / 2])


def in_crucible(point, pose) -> bool:
    p = crucible_local(point, pose)
    return bool(math.hypot(p[0], p[1]) < CRU_R + 0.001 and -0.002 < p[2] < CRU_H + 0.006)


def on_hot_body(point, pose) -> bool:
    """A hand's TCP (which rides a pad length above the fingertips) on the crucible itself rather than
    out on a handle."""
    p = crucible_local(point, pose)
    return bool(math.hypot(p[0], p[1]) < RING_R + 0.022 and -0.02 < p[2] < CRU_H + 0.09)


def in_cavity(point):
    """Index of the mould whose cavity ``point`` is in, or None."""
    ix, iy = MOULD[0] / 2 - FLASK_T, MOULD[1] / 2 - FLASK_T
    p = np.asarray(point, float)
    for k, cy in enumerate(MOULD_YS):
        if abs(p[0] - MOULD_X) < ix and abs(p[1] - cy) < iy and p[2] < TOP + DRAG_H + 0.002:
            return k
    return None


def in_cup(point) -> bool:
    p = np.asarray(point, float)
    return bool(any(math.dist(p[:2], basin_xy(k)) < BASIN_TOP_R + 0.003 for k in range(len(MOULD_YS)))
                and TOP + DRAG_H - 0.002 < p[2] < CUP_TOP + 0.006)
