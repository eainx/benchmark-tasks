"""Geometry and measurements for replacing a showerhead on its shower arm (real size).

Everything is real size; the bench is only the working height. On the left stands a section of
shower wall tiled in white 6 x 3 in subway tile; out of it comes a straight chrome 1/2 in shower arm,
screwed behind the tiles into a drop-ear elbow, carrying the old showerhead on its swivel nut. The
arm's thread in the elbow holds only by its tape: twisted hard enough it turns in the elbow (and would
leak), so the arm is held while the head's nut is turned. Under the arm a stainless wire shower basket
hangs on the wall; in it the new head stands face-up in its packaging cup. The old head comes off (4
turns of its nut), goes into the basket without being dropped, and the new one goes onto the arm's
thread square and is run up hand-tight, then its face is aimed down.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The wall's tiled face looks
along -y; the arm comes straight out of it along -y.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

TILE_COL = (0.94, 0.94, 0.92)        # white glazed subway tile
GROUT_COL = (0.72, 0.72, 0.70)
CHROME = (0.82, 0.83, 0.86)
FACE_COL = (0.20, 0.21, 0.23)        # the head's rubber-nozzle face
LIME = (0.80, 0.78, 0.66)            # scale on the old head's nozzles
DARK = (0.06, 0.06, 0.07)

WALL_Y = 0.140                       # the wall's tiled face

# -- tiled end wall (fixed) --------------------------------------------------------------------
WALL_T = 0.030
WALL_X = (0.210, 0.620)              # its run along x
WALL_H = 0.300
TILE = (0.150, 0.075)                # 6 x 3 in subway tile: along x, up z
JOINT = 0.003

# -- shower arm (turns in its elbow if twisted) -------------------------------------------------
# 1/2 in nominal brass arm, 21.3 mm OD, straight out of the wall; its outer end carries a male
# 1/2"-14 NPSM thread (13 mm of it) that the showerhead's swivel nut runs onto against a conical
# washer. Behind the tiles it is screwed into a drop-ear elbow on 1/2"-14 NPT with PTFE tape.
ARM_R = 0.01065
ARM_X = 0.420
ARM_Z = TOP + 0.240
ARM_L = 0.140                        # out of the tile face to its end
ARM_END_Y = WALL_Y - ARM_L
# Joint friction does not act on these single joints, so each resists turning by a damping drive
# (N m s / deg): the stuck old nut five times stiffer than the taped elbow.
ELBOW_DAMP = 0.004

# -- showerheads: 1/2"-14 NPSM female swivel nut, 32 mm knurled, hand-tight on a conical washer ------
# A compact fixed showerhead: knurled nut, ball joint, 76 mm face. The nut takes 7.3 mm of the arm's
# NPSM thread when hand-tight on its conical washer: about 4 turns at 14 threads per inch.
NUT_R = 0.016
NUT_L = 0.022
BALL_R = 0.013
FACE_R = 0.038                       # 76 mm face
FACE_T = 0.024
TPI = 14
THREAD_L = 0.00726                   # engaged length when hand-tight
TURNS_ON = round(THREAD_L / 0.0254 * TPI, 2)   # 4.0 turns
OLD_THREAD_DAMP = 0.02               # the old nut, its washer stuck: five times the elbow's
NEW_THREAD_DAMP = 0.001              # a clean nut running up the thread by hand
OLD_MASS = 0.20
NUT_MASS = 0.06                      # the new head's nut and ball ...
FACE_MASS = 0.13                     # ... and its face, on the ball
BALL_DAMP = 0.05                     # the new head's ball: stiff, so the face stays where it is aimed
BALL_OFF = NUT_L / 2 + BALL_R * 0.7  # nut centre to ball centre
FACE_OFF = BALL_R + FACE_T / 2 + 0.004   # ball centre to face centre


def wall_object() -> Part:
    """Cement-board backer with subway tiles (running bond) on its face, and the arm's escutcheon."""
    x0, x1 = WALL_X
    cx = (x0 + x1) / 2
    cy = WALL_Y + WALL_T / 2
    cz = TOP + WALL_H / 2
    pieces = []
    r = 0
    while True:
        zb = TOP + r * TILE[1]
        if zb >= TOP + WALL_H - 0.005:
            break
        h = min(TILE[1], TOP + WALL_H - zb)
        x = x0 - (TILE[0] / 2 if r % 2 else 0.0)
        while x < x1:
            a, b = max(x, x0), min(x + TILE[0], x1)
            if b - a > 0.01:
                pieces.append(Piece("box", (b - a - JOINT, 0.006, h - JOINT),
                                    ((a + b) / 2 - cx, -WALL_T / 2 - 0.003 + 0.0005, zb + h / 2 - cz),
                                    color=TILE_COL, collide=False))
            x += TILE[0]
        r += 1
    esc = Piece("cylinder", (0.032, 0.006, 0.032), (ARM_X - cx, -WALL_T / 2 - 0.006, ARM_Z - cz), axis="Y",
                color=CHROME, material="steel", collide=False)
    edge = Piece("box", (0.008, WALL_T + 0.008, WALL_H), (x0 - cx - 0.002, -0.004, 0.0), color=CHROME,
                 material="aluminium_brushed", collide=False)           # metal tile-edge trim
    return Part("wall", pos=(cx, cy, cz), size=(x1 - x0, WALL_T, WALL_H), exact_z=True, color=GROUT_COL,
                friction=(0.5, 0.4), pieces=(*pieces, esc, edge))


def arm_pos():
    return (ARM_X, WALL_Y - ARM_L / 2, ARM_Z)


def arm_object() -> Part:
    """The shower arm: turns about its own axis in the elbow behind the wall, damped, held by friction."""
    return Part("arm", pos=arm_pos(), size=(ARM_R, ARM_L, ARM_R), shape="cylinder", axis="Y", exact_z=True,
                mass=0.20, color=CHROME, material="steel", friction=(0.7, 0.6), restitution=0.0, contact_offset=0.002,
                hinge=Hinge(pivot=arm_pos(), axis="Y", damping=ELBOW_DAMP,
                            lower_deg=-1e6, upper_deg=1e6))


def _ribs(axis):
    out = []
    for k in range(18):
        t = 2 * math.pi * k / 18
        if axis == "Y":
            out.append(Piece("box", (0.0012, NUT_L * 0.8, 0.0012), (NUT_R * math.cos(t), 0.0, NUT_R * math.sin(t)),
                             rot=(0.0, -t, 0.0), collide=False))
        else:
            out.append(Piece("box", (0.0012, 0.0012, NUT_L * 0.8), (NUT_R * math.cos(t), NUT_R * math.sin(t), 0.0),
                             rot=(0.0, 0.0, t), collide=False))
    return tuple(out)


def old_pos():
    """The old head's nut: its back face against the arm's end (the thread inside it is not drawn)."""
    return (ARM_X, ARM_END_Y - NUT_L / 2, ARM_Z)


def old_object() -> Part:
    """The old head, run up on the arm's thread and left level: nut, ball, neck and limed face as one
    body, turning on the thread through a hinge to the arm whose friction is the thread's and stuck
    washer's. task.Dynamics lets it go once it has been turned off its last thread."""
    out = np.array([0.0, -1.0, 0.0])
    ball = out * BALL_OFF
    face = ball + out * FACE_OFF
    pieces = (Piece("sphere", (BALL_R, BALL_R, BALL_R), tuple(ball)),
              Piece("cylinder", (0.011, 0.012, 0.011), tuple(ball + out * BALL_R * 0.8), axis="Y"),
              Piece("cylinder", (FACE_R, FACE_T, FACE_R), tuple(face), axis="Y"),
              Piece("cylinder", (FACE_R - 0.007, 0.002, FACE_R - 0.007), tuple(face + out * (FACE_T / 2 + 0.0005)),
                    axis="Y", color=FACE_COL, collide=False),
              Piece("cylinder", (FACE_R - 0.017, 0.0022, FACE_R - 0.017), tuple(face + out * (FACE_T / 2 + 0.0008)),
                    axis="Y", color=LIME, collide=False),
              *_ribs("Y"))
    return Part("old_head", pos=old_pos(), size=(NUT_R, NUT_L, NUT_R), shape="cylinder", axis="Y", exact_z=True,
                mass=OLD_MASS, color=CHROME, material="steel", friction=(1.0, 0.9), restitution=0.05,
                contact_offset=0.0015, com=tuple(ball), pieces=pieces,
                hinge=Hinge(pivot=old_pos(), axis="Y", damping=OLD_THREAD_DAMP, lower_deg=-1e6,
                            upper_deg=1e6, parent="arm", parent_pos=arm_pos()))


# -- stainless wire shower basket on the wall under the arm (fixed) --------------------------------
# 260 x 110 x 55 mm, 4 mm wire on 20 mm centres, hung low on the wall (floor 5 mm off the bench).
BASKET_X = (0.210, 0.470)
BASKET_Y = (WALL_Y - 0.110, WALL_Y)
BASKET_FLOOR = TOP + 0.005
BASKET_H = 0.055
WIRE_R = 0.002
STEEL_COL = (0.75, 0.76, 0.78)


def basket_object() -> Part:
    x0, x1 = BASKET_X
    y0, y1 = BASKET_Y
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    z0 = BASKET_FLOOR
    L, W = x1 - x0, y1 - y0
    r = WIRE_R
    P = []
    # floor: wires across (along y) every 20 mm, two long rails
    n = int(L / 0.020)
    for k in range(n + 1):
        P.append(Piece("cylinder", (r, W, r), (x0 + k * L / n - cx, 0.0, 0.0), axis="Y"))
    for yy in (y0 + 0.005, (y0 + y1) / 2, y1 - 0.005):
        P.append(Piece("cylinder", (r, L, r), (0.0, yy - cy, -r), axis="X"))
    # front (toward -y) and ends: uprights every 20 mm, a top rail and a mid rail
    for k in range(n + 1):
        P.append(Piece("cylinder", (r, BASKET_H, r), (x0 + k * L / n - cx, y0 - cy, BASKET_H / 2), axis="Z"))
    m = int(W / 0.020)
    for k in range(1, m + 1):
        for xx in (x0, x1):
            P.append(Piece("cylinder", (r, BASKET_H, r), (xx - cx, y0 + k * W / m - cy, BASKET_H / 2), axis="Z"))
    for zz in (BASKET_H, BASKET_H / 2):
        P.append(Piece("cylinder", (r * 1.4, L, r * 1.4), (0.0, y0 - cy, zz), axis="X"))
        for xx in (x0, x1):
            P.append(Piece("cylinder", (r * 1.4, W, r * 1.4), (xx - cx, 0.0, zz), axis="Y"))
    # its two wall brackets
    for xx in (x0 + 0.03, x1 - 0.03):
        P.append(Piece("box", (0.020, 0.006, 0.050), (xx - cx, y1 - 0.003 - cy, BASKET_H - 0.025), collide=False))
    return Part("basket", pos=(cx, cy, z0), size=(0.004, 0.004, 0.004), exact_z=True, color=STEEL_COL,
                material="steel", friction=(0.6, 0.5), pieces=tuple(P))


# -- stainless shower shelf on the wall beside the arm, the new head lying on it in its foam cradle --
SHELF_X = (0.225, 0.365)
SHELF_T = 0.006
NEW_X = 0.300
SHELF_TOP = ARM_Z - FACE_R - 0.010            # the head lies with its axis level with the arm's
NEW_NUT_Y = WALL_Y - 0.022 - NUT_L / 2         # nut toward the wall, face out
NEW_BALL_Y = NEW_NUT_Y - BALL_OFF
NEW_FACE_Y = NEW_BALL_Y - FACE_OFF


def shelf_object() -> Part:
    """A 140 x 110 mm stainless shower shelf on two brackets, with the new head's foam cradle on it:
    two pads under the face's rim and one under the nut."""
    x0, x1 = SHELF_X
    cx = (x0 + x1) / 2
    cy = WALL_Y - 0.055
    cz = SHELF_TOP - SHELF_T / 2
    foam = (0.28, 0.29, 0.31)
    P = [Piece("box", (0.020, 0.008, 0.040), (x0 + 0.02 - cx, 0.055 - 0.004, -0.020), collide=False),
         Piece("box", (0.020, 0.008, 0.040), (x1 - 0.02 - cx, 0.055 - 0.004, -0.020), collide=False),
         Piece("cylinder", (0.003, x1 - x0, 0.003), (0.0, -0.055, 0.004), axis="X")]
    pad_h = 0.010 + FACE_R - FACE_R * math.cos(math.radians(30))
    for sx in (-1, 1):
        P.append(Piece("box", (0.022, FACE_T, pad_h), (NEW_X + sx * FACE_R * math.sin(math.radians(30)) + sx * 0.008 - cx,
                                                       NEW_FACE_Y - cy, SHELF_T / 2 + pad_h / 2), color=foam))
    nut_h = ARM_Z - NUT_R - SHELF_TOP
    P.append(Piece("box", (0.030, NUT_L * 0.8, nut_h), (NEW_X - cx, NEW_NUT_Y - cy, SHELF_T / 2 + nut_h / 2), color=foam))
    return Part("shelf", pos=(cx, cy, cz), size=(x1 - x0, 0.110, SHELF_T), exact_z=True, color=STEEL_COL,
                material="steel", friction=(0.7, 0.6), pieces=tuple(P))


def new_nut_pos():
    return (NEW_X, NEW_NUT_Y, ARM_Z)


def thread_local0():
    """Where a nut sits on the arm's thread, in the arm's frame: its back face against the arm's end."""
    return (0.0, -ARM_L / 2 - NUT_L / 2, 0.0)


# joint frames (w, x, y, z): the thread joint turns about its Z, which is +y (toward the wall) both in
# the arm and in the new head (its nut faces +y in its own frame)
ARM_ROT = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)      # Rx(-90): z -> +y
NUT_ROT = ARM_ROT


def new_object() -> Part:
    """The new head's nut, ball and neck (its face is a body of its own on the ball). Its thread joint
    to the arm is made at spawn, switched off; task.Dynamics switches it on once the nut is put to the
    arm's end square."""
    out = np.array([0.0, -1.0, 0.0])
    pieces = (Piece("sphere", (BALL_R, BALL_R, BALL_R), tuple(out * BALL_OFF)),
              Piece("cylinder", (0.011, 0.012, 0.011), tuple(out * (BALL_OFF + BALL_R * 0.8)), axis="Y"),
              *_ribs("Y"))
    return Part("new_head", pos=new_nut_pos(), size=(NUT_R, NUT_L, NUT_R), shape="cylinder", axis="Y", exact_z=True,
                mass=NUT_MASS, color=CHROME, material="steel", friction=(1.0, 0.9), restitution=0.05,
                contact_offset=0.0015, com=tuple(out * BALL_OFF * 0.6), pieces=pieces,
                hinge=Hinge(pivot=new_nut_pos(), axis="Z", damping=NEW_THREAD_DAMP,
                            lower_deg=-1e6, upper_deg=1e6, parent="arm", parent_pos=arm_pos(),
                            local0=thread_local0(), local1=(0.0, 0.0, 0.0), rot0=ARM_ROT, rot1=NUT_ROT,
                            enabled=False))


def new_face_pos():
    return (NEW_X, NEW_FACE_Y, ARM_Z)


def new_face_object() -> Part:
    """The new head's face on its ball: turns about the ball (a damped hinge across the head) to aim it."""
    pivot = (NEW_X, NEW_BALL_Y, ARM_Z)
    nozzles = Piece("cylinder", (FACE_R - 0.007, 0.002, FACE_R - 0.007), (0.0, -(FACE_T / 2 + 0.0005), 0.0),
                    axis="Y", color=FACE_COL, collide=False)
    return Part("new_face", pos=new_face_pos(), size=(FACE_R, FACE_T, FACE_R), shape="cylinder", axis="Y",
                exact_z=True, mass=FACE_MASS, color=CHROME, material="steel", friction=(1.0, 0.9),
                restitution=0.05, contact_offset=0.0015, pieces=(nozzles,),
                hinge=Hinge(pivot=pivot, axis="X", damping=BALL_DAMP, lower_deg=-0.5,
                            upper_deg=0.5, parent="new_head", parent_pos=new_nut_pos()))


def quat_matrix(q):
    """Rotation matrix from a quaternion given (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])



def quat_wxyz_matrix(q):
    w, x, y, z = q
    return quat_matrix((x, y, z, w))


def mat(pose):
    return quat_matrix(pose[3:7])


def thread_target(arm_pose):
    """Where a nut's centre sits on the arm's thread, and the arm's outward axis, from its pose."""
    r = mat(arm_pose)
    return np.asarray(arm_pose[:3], float) + r @ np.asarray(thread_local0()), r @ np.array([0.0, -1.0, 0.0])


def elbow_deg(arm_pose):
    r = mat(arm_pose)
    return math.degrees(math.atan2(float(r[0, 2]), float(r[0, 0])))


def old_twist(arm_pose, head_pose):
    """The old head's angle on the thread about +y (positive: tightening)."""
    rel = mat(arm_pose).T @ mat(head_pose)
    return math.atan2(float(rel[0, 2]), float(rel[0, 0]))


def new_twist(arm_pose, head_pose):
    """The new head's thread-joint angle about +y (positive: tightening), as the joint measures it."""
    f0 = mat(arm_pose) @ quat_wxyz_matrix(ARM_ROT)
    f1 = mat(head_pose) @ quat_wxyz_matrix(NUT_ROT)
    rel = f0.T @ f1
    return math.atan2(float(rel[1, 0]), float(rel[0, 0]))


def face_normal(face_pose):
    return mat(face_pose) @ np.array([0.0, -1.0, 0.0])


def fallen(p):
    """Down on the bench outside the basket: dropped (to the shower floor, in real life)."""
    x0, x1 = BASKET_X
    y0, y1 = BASKET_Y
    inside = x0 - 0.01 < p[0] < x1 + 0.01 and y0 - 0.01 < p[1] < y1 + 0.01
    return (not inside) and p[2] < TOP + 0.06


def in_basket(p):
    x0, x1 = BASKET_X
    y0, y1 = BASKET_Y
    return x0 < p[0] < x1 and y0 < p[1] < y1 and p[2] < BASKET_FLOOR + BASKET_H + 0.01
