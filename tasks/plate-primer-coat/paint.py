"""Geometry and measurements for priming a steel plate with a block brush.

A bare steel plate lies on the bench; beside it, a paint tray of red-oxide primer and a block brush
lying bristles-down. Primer only goes on from a brush that has been dipped: each dip loads enough
for part of the plate, and the coat appears, tile by tile, where the loaded bristles actually pass
over the plate.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

STEEL = (0.42, 0.43, 0.45)          # bare mill-scale plate
PRIMER = (0.55, 0.16, 0.10)         # red-oxide primer
TRAY = (0.80, 0.80, 0.78)
WOOD = (0.62, 0.45, 0.28)
BRISTLE = (0.86, 0.80, 0.66)        # pale natural bristle, so a loaded brush shows

# -- plate and its coat ---------------------------------------------------------------------
PLATE_XY = (0.330, 0.000)
PLATE = (0.120, 0.070, 0.008)
PLATE_TOP = TOP + PLATE[2]
TILE = 0.010
NX, NY = int(round(PLATE[0] / TILE)), int(round(PLATE[1] / TILE))
TILES = NX * NY                     # 12 x 7 = 84 patches of coat


def tile_centres():
    """Tile centres in plate-local x, y."""
    return [((i - (NX - 1) / 2) * TILE, (j - (NY - 1) / 2) * TILE) for i in range(NX) for j in range(NY)]


# -- primer tray ------------------------------------------------------------------------------
TRAY_XY = (0.330, 0.175)
TRAY_IN = (0.070, 0.100, 0.016)     # inside, x by y by depth: shallow, so the handle clears the rim
TRAY_WALL = 0.004
PRIMER_DEPTH = 0.010
LOAD = 45                           # patches of coat one dip carries
RIM_TOP = TOP + TRAY_WALL + TRAY_IN[2]
WIPE_WALL_X = TRAY_XY[0] + TRAY_IN[0] / 2 + TRAY_WALL / 2   # the tray's far wall, where the brush is wiped off
DRIP_EVERY = 4                      # control steps between drips from a brush carried un-wiped
DROP_R = 0.0035
DROP_MASS = 0.00008
DROPS = 24
DROP_NAMES = tuple(f"drip{i:02d}" for i in range(DROPS))

# -- brush ----------------------------------------------------------------------------------
BRISTLE_BLOCK = (0.016, 0.070, 0.020)
BACK = (0.024, 0.074, 0.012)
HANDLE_R = 0.010
HANDLE_L = 0.110
HANDLE_Z = BRISTLE_BLOCK[2] + BACK[2] / 2
GRIP_X = -0.060
BRUSH_MASS = 0.060
BRUSH_XY = (0.220, 0.200)
WET_PIECE = 2                       # index of the brush's primer-soaked tips piece


def plate_object() -> Part:
    """The plate, with its coat drawn as tiles just above its top (hidden until painted)."""
    z = PLATE[2] / 2 + 0.0004
    tiles = tuple(Piece("box", (TILE + 0.0004, TILE + 0.0004, 0.0006), (x, y, z), color=PRIMER, collide=False)
                  for x, y in tile_centres())
    return Part("plate", pos=(PLATE_XY[0], PLATE_XY[1], TOP + PLATE[2] / 2), size=PLATE, exact_z=True, color=STEEL,
                friction=(0.5, 0.4), pieces=tiles)


def tray_object() -> Part:
    tx, ty = TRAY_XY
    lx, ly, d = TRAY_IN[0] + 2 * TRAY_WALL, TRAY_IN[1] + 2 * TRAY_WALL, TRAY_IN[2]
    t = TRAY_WALL
    zc = d / 2
    p = []
    for s in (-1, 1):
        p.append(Piece("box", (t, ly, d + t), (s * (lx - t) / 2, 0.0, zc)))
        p.append(Piece("box", (lx - 2 * t, t, d + t), (0.0, s * (ly - t) / 2, zc)))
    p.append(Piece("box", (TRAY_IN[0], TRAY_IN[1], PRIMER_DEPTH), (0.0, 0.0, t / 2 + PRIMER_DEPTH / 2), color=PRIMER,
                   collide=False))
    return Part("tray", pos=(tx, ty, TOP + t / 2), size=(lx, ly, t), exact_z=True, color=TRAY, friction=(0.6, 0.5),
                pieces=tuple(p))


def brush_object(xy=BRUSH_XY) -> Part:
    """Block brush: bristle block (root), wooden back, round handle back toward the arms, and the
    primer-soaked tips (drawn, shown while it carries primer)."""
    bz = BRISTLE_BLOCK[2] / 2
    p = [Piece("box", BACK, (0.0, 0.0, BRISTLE_BLOCK[2] + BACK[2] / 2 - bz), color=WOOD),
         Piece("cylinder", (HANDLE_R, HANDLE_L, HANDLE_R), (-BACK[0] / 2 - HANDLE_L / 2 + 0.004, 0.0, HANDLE_Z - bz),
               axis="X", color=WOOD, friction=(1.6, 1.4)),
         Piece("box", (BRISTLE_BLOCK[0] + 0.001, BRISTLE_BLOCK[1] + 0.001, 0.010), (0.0, 0.0, -bz + 0.005), color=PRIMER,
               collide=False)]
    return Part("brush", pos=(xy[0], xy[1], TOP + bz), size=BRISTLE_BLOCK, exact_z=True, mass=BRUSH_MASS, color=BRISTLE,
                friction=(0.25, 0.2), restitution=0.0, contact_offset=0.0015, com=(-0.020, 0.0, HANDLE_Z - bz),
                pieces=tuple(p))


# -- measurements ----------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bristle_tips(brush_pose):
    """World centre of the bristle tips, the brush's rotation, and its tilt from upright (degrees)."""
    b = np.asarray(brush_pose, float)
    r = quat_matrix(b[3:7])
    tips = b[:3] + r @ np.array([0.0, 0.0, -BRISTLE_BLOCK[2] / 2])
    return tips, r, math.degrees(math.acos(np.clip(r[2, 2], -1, 1)))


def in_primer(brush_pose) -> bool:
    tips, _, tilt = bristle_tips(brush_pose)
    return bool(tilt < 30 and abs(tips[0] - TRAY_XY[0]) < TRAY_IN[0] / 2 and abs(tips[1] - TRAY_XY[1]) < TRAY_IN[1] / 2
                and tips[2] < TOP + TRAY_WALL + PRIMER_DEPTH - 0.003)


def tiles_under(brush_pose):
    """Indices of coat tiles the bristle tips are on now (tips on the plate's top, brush near flat)."""
    tips, r, tilt = bristle_tips(brush_pose)
    if tilt > 20 or abs(tips[2] - PLATE_TOP) > 0.004:
        return []
    out = []
    for k, (x, y) in enumerate(tile_centres()):
        c = np.array([PLATE_XY[0] + x, PLATE_XY[1] + y, PLATE_TOP])
        loc = r.T @ (c - tips)
        if abs(loc[0]) < BRISTLE_BLOCK[0] / 2 + 0.002 and abs(loc[1]) < BRISTLE_BLOCK[1] / 2:
            out.append(k)
    return out


def on_rim(brush_pose) -> bool:
    """Bristle tips drawn across the top of the tray's far wall (wiping off the excess)."""
    tips, _, tilt = bristle_tips(brush_pose)
    return bool(tilt < 30 and abs(tips[0] - WIPE_WALL_X) < 0.009 and abs(tips[1] - TRAY_XY[1]) < TRAY_IN[1] / 2
                and tips[2] < RIM_TOP + 0.004)


def over_tray(p) -> bool:
    return bool(abs(p[0] - TRAY_XY[0]) < TRAY_IN[0] / 2 + TRAY_WALL and abs(p[1] - TRAY_XY[1]) < TRAY_IN[1] / 2 + TRAY_WALL)


def drop_objects():
    return [Part(n, pos=parking(i), size=(DROP_R,) * 3, shape="sphere", exact_z=True, mass=DROP_MASS, color=PRIMER,
                 friction=(0.6, 0.5), restitution=0.0, contact_offset=0.001, max_depenetration=0.3)
            for i, n in enumerate(DROP_NAMES)]


def parking(i):
    return (0.10 + 0.012 * (i % 12), -0.55 - 0.012 * (i // 12), TOP - 0.30)


def tips_on_bench(brush_pose) -> bool:
    """Loaded bristles flat on the bench, off the plate and the tray (painting the bench)."""
    tips, _, tilt = bristle_tips(brush_pose)
    return bool(tilt < 30 and tips[2] < TOP + 0.004 and not on_plate(tips) and not over_tray(tips))


def grip_point(brush_pose):
    b = np.asarray(brush_pose, float)
    return b[:3] + quat_matrix(b[3:7]) @ np.array([GRIP_X, 0.0, HANDLE_Z - BRISTLE_BLOCK[2] / 2])


def on_plate(p) -> bool:
    return bool(abs(p[0] - PLATE_XY[0]) < PLATE[0] / 2 + 0.01 and abs(p[1] - PLATE_XY[1]) < PLATE[1] / 2 + 0.01)
