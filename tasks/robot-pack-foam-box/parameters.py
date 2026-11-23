"""Geometry and measurements for packing a small desktop robot back into its retail box.

The box is a rigid two-piece (base and telescoping lid) box, the kind small consumer robots ship in. The
base holds a die-cut foam insert with a pocket shaped to the robot; a second foam pad goes on top of the
robot, and the lid slides down over the base, its skirt round the base's walls, until it rests on the pad.
The robot, the top pad and the lid lie on the desk beside the open base.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

BOX_COL = (0.93, 0.93, 0.91)           # white printed board
LID_COL = (0.15, 0.35, 0.65)           # blue printed lid
FOAM_COL = (0.18, 0.18, 0.20)          # black polyethylene foam
BOT_COL = (0.92, 0.92, 0.94)           # the robot's white shell
FACE_COL = (0.05, 0.05, 0.07)
EYE_COL = (0.20, 0.85, 0.95)

# -- the robot (free) ---------------------------------------------------------------------------------
BOT = (0.070, 0.036, 0.045)            # x, y, z: a palm-sized desktop companion robot
BOT_MASS = 0.150
BOT0 = (0.200, -0.210)                 # lying on the desk, right of the box, facing +x

# -- the base with its foam insert (one free body) ------------------------------------------------------
BX, BY = 0.300, -0.050                 # the base's centre
WALL = 0.0025                          # rigid box board
FLOOR = 0.003
INNER = (0.110, 0.051)                 # inside the base
OUTER = (INNER[0] + 2 * WALL, INNER[1] + 2 * WALL)
CLEAR = 0.0040                         # the foam pocket's clearance round the robot, each side (cut generous: foam does not squeeze here)
POCKET = (BOT[0] + 2 * CLEAR, BOT[1] + 2 * CLEAR, 0.020)
FOAM_H = 0.025                         # the bottom insert's height above the floor
RIM_Z = FLOOR + FOAM_H + 0.010                # the walls' top, 38 mm: the seated robot stands 15 mm above it
BASE_MASS = 0.120
POCKET_FLOOR = FLOOR + FOAM_H - POCKET[2]     # the pocket's floor above the bench

# -- the top pad (free) ----------------------------------------------------------------------------------
PAD = (0.100, 0.046, 0.025)            # rests on the robot's top, inside the lid's skirt
PAD_MASS = 0.025
PAD0 = (0.180, -0.125)

# -- the lid (free) --------------------------------------------------------------------------------------
LID_GAP = 0.0020                       # the lid's skirt clears the base's walls by 2 mm each side
LID_IN = (OUTER[0] + 2 * LID_GAP, OUTER[1] + 2 * LID_GAP)
LID_T = 0.002
LID_OUT = (LID_IN[0] + 2 * LID_T, LID_IN[1] + 2 * LID_T)
LID_TOP_T = 0.0025
SKIRT = 0.048                          # the lid's skirt below its top
LID_MASS = 0.100
LID0 = (0.360, -0.205)
FRICTION_BOARD = (0.45, 0.38)

# seated heights (above the bench) of each part's root
BOT_SEATED = POCKET_FLOOR + BOT[2] / 2
BOT_TOP = POCKET_FLOOR + BOT[2]
PAD_SEATED = BOT_TOP + PAD[2] / 2
LID_SEATED = BOT_TOP + PAD[2] + LID_TOP_T / 2


def _w(x, y, z):
    return (float(x), float(y), float(TOP + z))


def base_object() -> Part:
    """Root: the floor. Four walls and the foam insert round its pocket (four blocks and the pocket floor)."""
    ox, oy = OUTER
    h = RIM_Z - FLOOR
    zc = FLOOR / 2 + h / 2                       # walls' centre relative to the floor's centre (floor at FLOOR/2)
    walls = (Piece("box", (ox, WALL, h), (0.0, oy / 2 - WALL / 2, zc)),
             Piece("box", (ox, WALL, h), (0.0, -oy / 2 + WALL / 2, zc)),
             Piece("box", (WALL, oy - 2 * WALL, h), (ox / 2 - WALL / 2, 0.0, zc)),
             Piece("box", (WALL, oy - 2 * WALL, h), (-ox / 2 + WALL / 2, 0.0, zc)))
    ix, iy = INNER
    px, py, pd = POCKET
    fz = FLOOR / 2 + (FOAM_H - pd) / 2           # the solid foam under the pocket
    sz = FLOOR / 2 + FOAM_H - pd / 2             # the foam round the pocket
    sx, sy = (ix - px) / 2, (iy - py) / 2
    foam = (Piece("box", (ix, iy, FOAM_H - pd), (0.0, 0.0, fz), color=FOAM_COL),
            Piece("box", (ix, sy, pd), (0.0, py / 2 + sy / 2, sz), color=FOAM_COL),
            Piece("box", (ix, sy, pd), (0.0, -py / 2 - sy / 2, sz), color=FOAM_COL),
            Piece("box", (sx, py, pd), (px / 2 + sx / 2, 0.0, sz), color=FOAM_COL),
            Piece("box", (sx, py, pd), (-px / 2 - sx / 2, 0.0, sz), color=FOAM_COL))
    return Part("box_base", pos=_w(BX, BY, FLOOR / 2), size=(ox, oy, FLOOR), exact_z=True, mass=BASE_MASS,
                color=BOX_COL, friction=FRICTION_BOARD, restitution=0.0, contact_offset=0.0015,
                com=(0.0, 0.0, 0.015), pieces=walls + foam)


def robot_object() -> Part:
    bx, by, bz = BOT
    face = Piece("box", (0.001, by - 0.008, bz * 0.45), (bx / 2 + 0.0005, 0.0, bz * 0.08), color=FACE_COL,
                 collide=False)
    eyes = tuple(Piece("box", (0.0008, 0.007, 0.004), (bx / 2 + 0.0012, s * 0.009, bz * 0.12), color=EYE_COL,
                       collide=False) for s in (1, -1))
    wheels = tuple(Piece("box", (0.040, 0.002, 0.012), (0.0, s * (by / 2 + 0.0005), -bz / 2 + 0.008),
                         color=FACE_COL, collide=False) for s in (1, -1))
    return Part("robot", pos=_w(BOT0[0], BOT0[1], bz / 2), size=BOT, exact_z=True, mass=BOT_MASS, color=BOT_COL,
                friction=(1.0, 0.9), restitution=0.0, contact_offset=0.0015, pieces=(face,) + eyes + wheels)


def pad_object() -> Part:
    return Part("top_pad", pos=_w(PAD0[0], PAD0[1], PAD[2] / 2), size=PAD, exact_z=True, mass=PAD_MASS,
                color=FOAM_COL, friction=(0.8, 0.7), restitution=0.0, contact_offset=0.0015)


def lid_object() -> Part:
    """Root: the lid's top plate, lying the right way up (skirt down) on the desk."""
    ox, oy = LID_OUT
    skirt = (Piece("box", (ox, LID_T, SKIRT), (0.0, oy / 2 - LID_T / 2, -LID_TOP_T / 2 - SKIRT / 2)),
             Piece("box", (ox, LID_T, SKIRT), (0.0, -oy / 2 + LID_T / 2, -LID_TOP_T / 2 - SKIRT / 2)),
             Piece("box", (LID_T, oy - 2 * LID_T, SKIRT), (ox / 2 - LID_T / 2, 0.0, -LID_TOP_T / 2 - SKIRT / 2)),
             Piece("box", (LID_T, oy - 2 * LID_T, SKIRT), (-ox / 2 + LID_T / 2, 0.0, -LID_TOP_T / 2 - SKIRT / 2)))
    logo = Piece("box", (0.050, 0.030, 0.0004), (0.0, 0.0, LID_TOP_T / 2 + 0.0002), color=BOX_COL, collide=False)
    return Part("lid", pos=_w(LID0[0], LID0[1], SKIRT + LID_TOP_T / 2), size=(ox, oy, LID_TOP_T), exact_z=True,
                mass=LID_MASS, color=LID_COL, friction=FRICTION_BOARD, restitution=0.0, contact_offset=0.0015,
                com=(0.0, 0.0, -0.008), pieces=skirt + (logo,))


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def tilt_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))


def in_base(base_pose, pose):
    """A part's place relative to the base: offset across the base (m, in its frame), height above the bench
    (m), yaw relative to the base (deg, folded to +-90: the parts are symmetric end for end) and tilt."""
    b, p = np.asarray(base_pose, float), np.asarray(pose, float)
    rb = quat_matrix(b[3:7])
    local = rb.T @ (p[:3] - b[:3])
    yaw = (yaw_deg(p) - yaw_deg(b) + 90.0) % 180.0 - 90.0
    return float(math.hypot(local[0], local[1])), float(p[2] - TOP), float(yaw), tilt_deg(p)
