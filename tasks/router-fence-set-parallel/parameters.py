"""Geometry and measurements for setting a router-table fence 25 mm from the bit and parallel.

The router hangs under the bench; its 1/2 in straight bit stands up through an insert plate in the
bench top. A 600 mm MDF fence lies on the bench beyond the bit. Each end of its base is held by a
T-bolt in a T-track let into the bench along x, with a star knob on top. With the knobs loose, each end
slides along its track, and the fence turns about the other end. Tightening a knob clamps its end.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The fence runs along y, its
face toward the arms (and the bit); the tracks run along x.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece, Spring

TOP = 0.7475

MDF = (0.62, 0.50, 0.36)
MDF_FACE = (0.70, 0.58, 0.42)
ALU = (0.72, 0.73, 0.75)
STEEL = (0.62, 0.63, 0.66)
CARBIDE = (0.25, 0.25, 0.28)
KNOB_COL = (0.10, 0.10, 0.11)
PLATE_COL = (0.18, 0.20, 0.24)
TAPE_COL = (0.86, 0.82, 0.20)

# -- the bit (fixed) ---------------------------------------------------------------------------
BX, BY = 0.330, 0.000                # bit centre
BIT_R, BIT_H = 0.00635, 0.025        # 1/2 in straight bit, 25 mm above the table
TARGET = 0.025                       # fence face to bit centre

# -- the fence ---------------------------------------------------------------------------------
F_LEN = 0.600                        # along y
F_T = 0.019                          # face board thickness (x)
F_H = 0.075                          # face board height
BASE_D, BASE_T = 0.110, 0.012        # base board: depth behind the face (x), thickness
F_MASS = 1.9                         # 19 mm MDF face and base with the knob blocks
LIFT = 0.0010                        # the fence rides 1 mm clear of the bench on its T-bolts
END_Y = 0.270                        # T-tracks / knobs at y = +-END_Y
KNOB_X = 0.075                       # knobs behind the face (x): clear of the face board for a hand turning them

# -- start: knobs loose, the fence racked ---------------------------------------------------------
START_L = 0.0400                     # face to bit centre line, measured at the left knob (y = +END_Y)
START_R = 0.0330                     # ... at the right knob (y = -END_Y)
YAW0 = -math.atan2(START_L - START_R, 2 * END_Y)  # - : the left end further from the bit

# -- the T-bolt carriers' slides (one per track) -------------------------------------------------
SLIDE_DAMP = 400.0                   # N s/m: the loose T-bolt head and the base dragging in the track
SLIDE_RANGE = (-0.030, 0.030)        # travel along the track from the start, m
CLAMP_K = 4.0e5                      # N/m once the knob is tightened: the end is clamped
LOCK_DEG = 75.0                      # a knob turned this far clockwise from finger-loose clamps its end
UNLOCK_DEG = 60.0


def face_x_at(y, pose):
    """x of the fence face at world y, from the fence's root pose."""
    o, r = frame(pose)
    # the face is the plane x_local = -F_T/2 through the root (the face board's centre)
    n = r @ np.array([1.0, 0.0, 0.0])            # face normal (local +x), into the fence
    p0 = o + r @ np.array([-F_T / 2, 0.0, 0.0])  # a point on the face
    # intersect the line (x, y) with the face plane in xy
    # n.x (x - p0.x) + n.y (y - p0.y) = 0
    return float(p0[0] - n[1] * (y - p0[1]) / n[0])


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame(pose):
    pose = np.asarray(pose, float)
    return pose[:3], quat_matrix(pose[3:7])


def yaw_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def distance_mm(pose, y=BY):
    """Fence face to bit centre along x at world y, mm."""
    return 1000.0 * (face_x_at(y, pose) - BX)


def parallel_mm(pose, span=0.500):
    """How far the face is out of parallel to the bench's x-normal (the mitre slot) over ``span``, mm."""
    return 1000.0 * abs(face_x_at(span / 2, pose) - face_x_at(-span / 2, pose))


def _root0():
    """The fence root (face board centre) at the start."""
    mid = 0.5 * (START_L + START_R)
    return np.array([BX + mid + F_T / 2 * math.cos(YAW0), 0.0, TOP + LIFT + F_H / 2])


ROOT0 = _root0()


def carrier_pos(side):
    """Start position of a T-bolt carrier (the knob's axis at the base's top face)."""
    sy = END_Y if side == "left" else -END_Y
    c, s = math.cos(YAW0), math.sin(YAW0)
    # local offset from the fence root: x behind the face centre, y along the fence
    lx, ly = -F_T / 2 + KNOB_X, sy
    p = ROOT0[:2] + np.array([c * lx - s * ly, s * lx + c * ly])
    return (float(p[0]), float(p[1]), float(TOP + LIFT + BASE_T))


def fence_object() -> Part:
    pl = carrier_pos("left")
    pieces = (
        # base board behind the face, under the knob blocks
        Piece("box", (BASE_D, F_LEN, BASE_T), (-F_T / 2 + BASE_D / 2, 0.0, -F_H / 2 + BASE_T / 2), color=MDF),
        # the bit opening's dust hood (visual)
        Piece("box", (0.050, 0.090, 0.050), (F_T / 2 + 0.025, 0.0, -F_H / 2 + BASE_T + 0.025), color=MDF, collide=False),
        # a strip of yellow tape along the top of the face, for reading it on camera
        Piece("box", (F_T + 0.0004, F_LEN, 0.004), (0.0, 0.0, F_H / 2 - 0.002), color=TAPE_COL, collide=False),
    )
    return Part("fence", pos=tuple(map(float, ROOT0)), size=(F_T, F_LEN, F_H), exact_z=True, mass=F_MASS,
                color=MDF_FACE, yaw=YAW0, friction=(0.45, 0.35), restitution=0.0, contact_offset=0.002,
                pieces=pieces,
                hinge=Hinge(pivot=pl, axis="Z", damping=0.0, lower_deg=-30.0, upper_deg=30.0,
                            parent="carrier_left", parent_pos=carrier_object("left").pos))


def carrier_object(side) -> Part:
    p = carrier_pos(side)
    return Part(f"carrier_{side}", pos=(p[0], p[1], p[2] - BASE_T - LIFT - 0.004), size=(0.030, 0.012, 0.006),
                exact_z=True, mass=0.08, color=STEEL, collision="none",
                spring=Spring(axis="X", lower=SLIDE_RANGE[0], upper=SLIDE_RANGE[1], stiffness=0.0,
                              damping=SLIDE_DAMP, target=0.0))


def slot_object() -> Part:
    """The right end's bolt hole is a short slot along the fence: the right T-bolt pivots in it and may
    slide 3 mm either way, so the two bolts never fight the fence as it turns."""
    p = carrier_pos("right")
    pos = (p[0], p[1], p[2] - 0.003)
    return Part("slot_right", pos=pos, size=(0.012, 0.020, 0.004), exact_z=True, mass=0.03, color=STEEL,
                collision="none", yaw=YAW0,
                spring=Spring(axis="Y", lower=-0.003, upper=0.003, stiffness=0.0, damping=2.0, target=0.0,
                              parent="fence", parent_pos=tuple(map(float, ROOT0)), parent_yaw=YAW0),
                hinge=Hinge(pivot=pos, axis="Z", damping=0.0, lower_deg=-30.0, upper_deg=30.0,
                            parent="carrier_right", parent_pos=carrier_object("right").pos))


LEVER_R = 0.060                      # axis to the middle of the lever's grip end
LEVER_W, LEVER_T = 0.014, 0.012      # the lever arm's width and thickness
HUB_R, HUB_H = 0.013, 0.024


def lever_z():
    """Height of the lever arm's middle (world)."""
    return carrier_pos("left")[2] + 0.005 + LEVER_T / 2


def knob_object(side) -> Part:
    """An adjustable clamping lever on the T-bolt (the 'knob'): a hub with an 80 mm arm, pointing back
    (+x) when finger-loose. Swung a quarter turn clockwise (seen from above) it clamps its end."""
    p = carrier_pos(side)
    c = (p[0], p[1], p[2] + 0.002 + HUB_H / 2)       # the hub stands on a washer on the base
    zc = lever_z() - c[2]
    pieces = (
        Piece("box", (LEVER_R, LEVER_W, LEVER_T), (LEVER_R / 2, 0.0, zc), color=KNOB_COL),
        Piece("cylinder", (0.010, LEVER_T, 0.010), (LEVER_R + 0.004, 0.0, zc), color=KNOB_COL),
        Piece("cylinder", (0.013, 0.002, 0.013), (0.0, 0.0, -HUB_H / 2 - 0.001), color=STEEL, collide=False),
    )
    cpos = (p[0], p[1], p[2] - BASE_T - LIFT - 0.004)
    return Part(f"knob_{side}", pos=c, size=(HUB_R, HUB_H, HUB_R), shape="cylinder", exact_z=True, mass=0.06,
                color=KNOB_COL, friction=(0.8, 0.7), restitution=0.0, pieces=pieces,
                hinge=Hinge(pivot=c, axis="Z", damping=0.0003, friction=0.0, lower_deg=-120.0, upper_deg=20.0,
                            parent=f"carrier_{side}", parent_pos=cpos))


def bit_objects():
    """The bit, its collet nut and the insert plate (all fixed)."""
    return (
        Part("bit", pos=(BX, BY, TOP + BIT_H / 2), size=(BIT_R, BIT_H, BIT_R), shape="cylinder", exact_z=True,
             mass=None, color=CARBIDE, friction=(0.4, 0.3)),
        Part("plate", pos=(BX, BY, TOP + 0.0005), size=(0.230, 0.300, 0.001), exact_z=True, mass=None,
             color=PLATE_COL, collision="none",
             pieces=(Piece("cylinder", (0.024, 0.0012, 0.024), (0.0, 0.0, 0.0001), color=(0.05, 0.05, 0.05),
                           collide=False),)),
        Part("tracks", pos=(BX + 0.05, END_Y, TOP + 0.0004), size=(0.300, 0.019, 0.0008), exact_z=True, mass=None,
             color=ALU, collision="none",
             pieces=(Piece("box", (0.300, 0.019, 0.0008), (0.0, -2 * END_Y, 0.0), color=ALU, collide=False),)),
    )


def knob_turn_deg(knob_pose):
    """How far a knob is turned clockwise (seen from above) from finger-loose, degrees."""
    return -yaw_deg(knob_pose)
