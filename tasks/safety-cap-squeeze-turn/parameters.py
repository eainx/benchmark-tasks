"""Geometry and measurements for opening a squeeze-and-turn child-resistant cap.

A 750 ml bottle of household cleaner (a flat HDPE bottle, its neck at one end) stands on the counter with
a squeeze-and-turn child-resistant cap: two ribbed panels on opposite sides of the cap have to be squeezed
in together to lift its locking lugs out of the ratchet teeth on the neck, and only while they are held in
does the cap turn anticlockwise. Turned half a turn, its thread comes free and the cap lifts off. The
light bottle turns with the cap unless the other hand holds it.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The bottle's length is along y.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece, Spring

TOP = 0.7475

HDPE = (0.15, 0.45, 0.80)              # blue cleaner bottle
LABEL = (0.95, 0.85, 0.20)
CAP_COL = (0.95, 0.95, 0.93)
TAB_COL = (0.85, 0.20, 0.15)           # red squeeze panels

# -- the bottle (free) -----------------------------------------------------------------------------
BODY = (0.056, 0.180, 0.100)           # x, y, z: a flat 750 ml bottle
BX, BY = 0.270, -0.065                 # body centre
NECK_Y = -0.135                        # the neck near the bottle's right end
NECK_R, NECK_H = 0.0140, 0.014
B_MASS = 0.85                          # 750 ml, nearly full
B_FRICTION = (0.45, 0.38)

# -- the cap (on the neck) -------------------------------------------------------------------------
CAP_R, CAP_H = 0.0190, 0.026           # 38 mm cap
CAP_Z = BODY[2] + NECK_H + CAP_H / 2 - 0.010       # the cap's skirt covers the top of the neck
TAB = (0.016, 0.004, 0.018)            # squeeze panels on the cap's +-y sides (cap frame), proud 1.5 mm
TAB_TRAVEL = 0.0030
SQUEEZED = 0.0020                      # both panels this far in: the lugs clear the ratchet
TAB_K, TAB_PRE = 1500.0, 0.0015        # N/m: 2.3 N to start, 6.8 N fully in
FREE_DEG = 170.0                       # just under half a turn anticlockwise and the thread is free
CAP_MASS, TAB_MASS = 0.012, 0.004
LOCK_K = 2.0                           # N m / deg holding the cap when the panels are out (the ratchet)
THREAD_DAMP = 0.002                    # N m s / deg: the thread's drag while it turns


def _w(x, y, z):
    return (float(x), float(y), float(TOP + z))


def bottle_object() -> Part:
    neck = Piece("cylinder", (NECK_R, NECK_H, NECK_R), (0.0, NECK_Y - BY, BODY[2] / 2 + NECK_H / 2))
    shoulder = Piece("box", (BODY[0] - 0.004, 0.040, 0.008), (0.0, NECK_Y - BY - 0.005, BODY[2] / 2 + 0.004))
    label = Piece("box", (BODY[0] + 0.001, 0.110, 0.055), (0.0, 0.020, -0.005), color=LABEL, collide=False)
    return Part("bottle", pos=_w(BX, BY, BODY[2] / 2), size=BODY, exact_z=True, mass=B_MASS, color=HDPE,
                friction=B_FRICTION, restitution=0.0, pieces=(neck, shoulder, label),
                com=(0.0, 0.010, -0.010))


def cap_pos():
    return _w(BX, NECK_Y, CAP_Z)


def cap_object() -> Part:
    ribs = tuple(Piece("box", (0.0012, 0.0012, CAP_H - 0.004),
                       (CAP_R * math.cos(a), CAP_R * math.sin(a), 0.0), rot=(0.0, 0.0, a), collide=False)
                 for a in np.linspace(0, 2 * math.pi, 24, endpoint=False) if abs(math.sin(a)) < 0.8)
    return Part("cap", pos=cap_pos(), size=(CAP_R, CAP_H, CAP_R), shape="cylinder", exact_z=True, mass=CAP_MASS,
                color=CAP_COL, friction=(0.7, 0.6), restitution=0.0, contact_offset=0.0015, pieces=ribs,
                hinge=Hinge(pivot=cap_pos(), axis="Z", damping=THREAD_DAMP, lower_deg=-5.0, upper_deg=FREE_DEG + 30.0,
                            stiffness=LOCK_K, target_deg=0.0, parent="bottle",
                            parent_pos=_w(BX, BY, BODY[2] / 2)))


def tab_object(side) -> Part:
    """A squeeze panel on the cap's +y (side=+1) or -y (side=-1) face, sliding inward on a sprung slide."""
    c = cap_pos()
    pos = (c[0], c[1] + side * (CAP_R - TAB[1] / 2 + 0.0015), c[2] + 0.002)
    lo, hi = (-TAB_TRAVEL, 0.0) if side > 0 else (0.0, TAB_TRAVEL)
    return Part(f"tab_{'a' if side > 0 else 'b'}", pos=pos, size=TAB, exact_z=True, mass=TAB_MASS, color=TAB_COL,
                friction=(1.0, 0.9), restitution=0.0, contact_offset=0.0015,
                spring=Spring(axis="Y", lower=lo, upper=hi, stiffness=TAB_K, damping=2.0,
                              target=side * TAB_PRE, parent="cap", parent_pos=c))


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


def tab_in(cap_pose, tab_pose, side):
    """How far a panel is pressed in from its rest, in the cap's frame (m)."""
    c, t = np.asarray(cap_pose, float), np.asarray(tab_pose, float)
    r = quat_matrix(c[3:7])
    local = r.T @ (t[:3] - c[:3])
    rest = CAP_R - TAB[1] / 2 + 0.0015
    return float(rest - side * local[1])
