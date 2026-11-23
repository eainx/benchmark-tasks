"""Geometry and measurements for fishing a bottle cap out from under a toaster oven with a wooden spoon.

A plastic soda-bottle cap has rolled under a countertop toaster oven. The oven stands 22 mm clear of the
bench on four rubber feet set 70-80 mm in from its sides, so the cap is out of sight from above and no
finger fits under. A wooden spoon lies on the bench to the right. Its handle slides under; laid across
behind the cap, between the feet, and drawn toward the front, it rakes the cap out.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

STEEL_DARK = (0.16, 0.16, 0.17)
STEEL = (0.62, 0.63, 0.65)
GLASS = (0.08, 0.09, 0.10)
RUBBER = (0.05, 0.05, 0.05)
BEECH = (0.80, 0.64, 0.42)
CAP_COL = (0.85, 0.12, 0.10)

# -- the oven (free, heavy, on rubber feet) -------------------------------------------------------
OX, OY = 0.400, 0.000                 # footprint centre
OD, OW, OH = 0.300, 0.440, 0.210      # depth (x), width (y), body height
GAP = 0.022                           # bench to the oven's underside (foot height)
FOOT_R = 0.012
FOOT_X = (0.050, 0.050)               # feet inset from the front and the back
FOOT_Y = 0.100                        # feet inset from the sides: at y = +-(OW/2 - FOOT_Y)
OVEN_MASS = 6.0
FRONT = OX - OD / 2                   # 0.25
RIGHT = OY - OW / 2                   # -0.22

# -- the cap (free) ------------------------------------------------------------------------------------
CAP_R, CAP_H = 0.015, 0.012           # 30 mm PE soda-bottle cap
CAP_MASS = 0.004
CAP0 = (0.420, -0.190)                # 30 mm in from the right side, 170 mm in from the front

# -- the spoon (free) ----------------------------------------------------------------------------------
HANDLE = (0.014, 0.260, 0.009)        # width (x), length (y), thickness: lies along y
BOWL = (0.050, 0.070, 0.011)
SPOON_MASS = 0.045
SPOON_X = 0.300
SPOON_TIP_Y = -0.270                  # handle tip at the start (outside the oven's right side)
GRIP_FROM_TIP = 0.250                 # where the hand takes the handle


def oven_object() -> Part:
    zc = TOP + GAP + OH / 2
    fx = (FRONT + FOOT_X[0] - OX, OX + OD / 2 - FOOT_X[1] - OX)
    fy = OW / 2 - FOOT_Y
    feet = tuple(Piece("cylinder", (FOOT_R, GAP, FOOT_R), (x, s * fy, -OH / 2 - GAP / 2), color=RUBBER,
                       friction=(1.0, 0.9)) for x in fx for s in (1, -1))
    door = Piece("box", (0.004, OW * 0.68, OH * 0.70), (-OD / 2 - 0.002, 0.04, 0.0), color=GLASS, collide=False)
    panel = Piece("box", (0.004, OW * 0.22, OH * 0.80), (-OD / 2 - 0.002, -OW / 2 + OW * 0.13, 0.0), color=STEEL,
                  collide=False)
    handle = Piece("box", (0.020, OW * 0.55, 0.014), (-OD / 2 - 0.016, 0.04, OH * 0.38), color=STEEL, collide=False)
    return Part("oven", pos=(OX, OY, zc), size=(OD, OW, OH), exact_z=True, mass=OVEN_MASS, color=STEEL_DARK,
                friction=(0.6, 0.5), restitution=0.0, pieces=feet + (door, panel, handle))


def cap_object() -> Part:
    return Part("cap", pos=(CAP0[0], CAP0[1], TOP + CAP_H / 2), size=(CAP_R, CAP_H, CAP_R), shape="cylinder",
                exact_z=True, mass=CAP_MASS, color=CAP_COL, friction=(0.35, 0.30), restitution=0.0,
                contact_offset=0.002, max_depenetration=0.2)


def spoon_object() -> Part:
    """Root is the handle (centre); the bowl is welded at its near (-y) end."""
    cy = SPOON_TIP_Y - HANDLE[1] / 2
    bowl = Piece("box", BOWL, (0.0, -HANDLE[1] / 2 - BOWL[1] / 2 + 0.004, (BOWL[2] - HANDLE[2]) / 2), color=BEECH)
    return Part("spoon", pos=(SPOON_X, cy, TOP + HANDLE[2] / 2), size=HANDLE, exact_z=True, mass=SPOON_MASS,
                color=BEECH, friction=(0.40, 0.35), restitution=0.0, contact_offset=0.002, pieces=(bowl,))


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


def depth_under(xy, oven_pose):
    """How far the point is inside the oven's footprint (m): + inside (distance to the nearest edge),
    - outside (distance out past the footprint)."""
    o = np.asarray(oven_pose, float)
    r = quat_matrix(o[3:7])
    d = np.asarray(xy, float) - o[:2]
    lx, ly = r[0, 0] * d[0] + r[1, 0] * d[1], r[0, 1] * d[0] + r[1, 1] * d[1]
    ex, ey = OD / 2 - abs(lx), OW / 2 - abs(ly)
    if ex >= 0 and ey >= 0:
        return float(min(ex, ey))
    return float(-math.hypot(min(ex, 0.0), min(ey, 0.0)))


START_DEPTH = 0.030                   # the cap centre's depth under at the start (nearest edge: the right side)
