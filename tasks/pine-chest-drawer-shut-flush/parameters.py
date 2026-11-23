"""Geometry and measurements for shutting a wide wood-runner chest drawer that has gone in crooked.

A low solid-pine one-drawer chest stands on the bench, its front facing the arms. Its single wide
drawer rides on two hardwood runners, steered only by the chest's side panels, with 1.5 mm of side
play each side. It starts pulled 130 mm out and cocked about 1 degree, its left end further out, as
a drawer is left after being shoved shut by one knob. Pushed off-centre, a drawer like this turns in
its play until a back corner and the opposite front edge bear on the side panels; then the harder it
is pushed the harder they bind. Shut, its inset front sits flush with the chest's face.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The drawer slides along x.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

PINE = (0.80, 0.63, 0.42)
PINE_DARK = (0.70, 0.53, 0.34)
RUNNER_COL = (0.62, 0.44, 0.27)        # beech runners
INSIDE = (0.86, 0.72, 0.52)
TOWEL = (0.36, 0.52, 0.66)             # folded towels in the drawer
KNOB_COL = (0.55, 0.37, 0.22)

# -- the chest (fixed) -------------------------------------------------------------------------
FX = 0.420                             # its front face
CW, CD, CH = 0.620, 0.320, 0.200       # outside width (y), depth (x), height (z)
TP = 0.018                             # 18 mm solid pine panels
OPEN_W = CW - 2 * TP                   # 0.584 between the side panels
BASE_TOP = 0.030                       # bottom board's top face
RUNNER = (0.020, 0.010)                # beech runner strip: width, height; along each side panel
RUN_TOP = BASE_TOP + RUNNER[1]         # the drawer's sides ride on this
RUN_X0 = FX + 0.024                    # runners start just behind the inset front (20 mm board + 4 mm)
RUN_LEN = CD - 0.006 - 0.024
UNDER_TOP = CH - TP                    # underside of the top panel (the kicker above the drawer)
STOP_AT = FX + 0.280                   # the drawer's back hits the stop blocks here when it is shut
WOOD_FRICTION = (0.45, 0.35)           # pine drawer side on pine carcass side (dry)
RUNNER_FRICTION = (0.30, 0.25)         # waxed drawer side on beech runner

# -- the drawer (free) -------------------------------------------------------------------------
PLAY = 0.0015                          # side play, each side
DW = OPEN_W - 2 * PLAY                 # 0.581 outside width
DD = 0.280                             # front face to back, outside
DH = 0.135                             # side height
ST = 0.014                             # side / back thickness
FRONT_T = 0.020                        # inset front board
FRONT_H = UNDER_TOP - RUN_TOP - 0.005     # 3 mm reveal at the top; clears the runners by 2 mm
KNOB_Y = 0.200                         # knobs 400 mm apart
KNOB_R, KNOB_L = 0.015, 0.035          # 30 mm turned knob, standing 35 mm proud
DRAWER_MASS = 1.60                     # 1.1 kg pine drawer + 0.5 kg of folded towels
PULLED = 0.130                         # start: front face this far out of the chest
YAW0 = 0.017                           # start: cocked 0.97 degrees, left end further out
TRAVEL = PULLED

# The drawer's frame: x from its front face into the chest, z above TOP.
SIDE_Z = RUN_TOP + DH / 2              # sides' centre height above TOP
FRONT_Z = RUN_TOP + 0.002 + FRONT_H / 2
KNOB_Z = FRONT_Z


def _start_root_y():
    """The cocked drawer's root y: the left side's back corner and the right side at the chest's mouth
    each 0.15 mm short of their side panels."""
    th = YAW0
    back = DD / 2                             # the back corners, from the drawer's centre
    mouth = PULLED - DD / 2                   # the chest's mouth (x = FX), from the drawer's centre
    y_max = OPEN_W / 2 - 0.00015 - (DW / 2 + back * math.sin(th))
    y_min = -OPEN_W / 2 + 0.00015 + (DW / 2 + mouth * math.sin(th))
    return 0.5 * (y_max + y_min), (y_min, y_max)


ROOT0 = np.array([FX - PULLED + DD / 2, _start_root_y()[0], SIDE_Z])


def chest_object() -> Part:
    """The carcass: bottom board, two side panels, top, back, two runners and two stop blocks."""
    cx = FX + CD / 2
    root_pos = (cx, 0.0, TOP + BASE_TOP / 2)
    side_y = CW / 2 - TP / 2

    def rel(x, y, z):
        return (float(x - cx), float(y), float(z - BASE_TOP / 2))

    pieces = (
        Piece("box", (CD, TP, CH), rel(cx, side_y, CH / 2), color=PINE, friction=WOOD_FRICTION),
        Piece("box", (CD, TP, CH), rel(cx, -side_y, CH / 2), color=PINE, friction=WOOD_FRICTION),
        Piece("box", (CD, CW, TP), rel(cx, 0.0, CH - TP / 2), color=PINE, friction=WOOD_FRICTION),
        Piece("box", (0.006, OPEN_W, CH - TP - BASE_TOP), rel(FX + CD - 0.003, 0.0, (CH - TP + BASE_TOP) / 2),
              color=PINE_DARK),
        Piece("box", (RUN_LEN, RUNNER[0], RUNNER[1]), rel(RUN_X0 + RUN_LEN / 2, OPEN_W / 2 - RUNNER[0] / 2, BASE_TOP + RUNNER[1] / 2),
              color=RUNNER_COL, friction=RUNNER_FRICTION),
        Piece("box", (RUN_LEN, RUNNER[0], RUNNER[1]), rel(RUN_X0 + RUN_LEN / 2, -OPEN_W / 2 + RUNNER[0] / 2, BASE_TOP + RUNNER[1] / 2),
              color=RUNNER_COL, friction=RUNNER_FRICTION),
        Piece("box", (FX + CD - 0.006 - STOP_AT, 0.040, 0.030), rel((STOP_AT + FX + CD - 0.006) / 2, OPEN_W / 2 - 0.050, RUN_TOP + 0.015),
              color=RUNNER_COL),
        Piece("box", (FX + CD - 0.006 - STOP_AT, 0.040, 0.030), rel((STOP_AT + FX + CD - 0.006) / 2, -OPEN_W / 2 + 0.050, RUN_TOP + 0.015),
              color=RUNNER_COL),
    )
    return Part("chest", pos=root_pos, size=(CD, CW, BASE_TOP), exact_z=True, mass=None, color=PINE,
                friction=RUNNER_FRICTION, restitution=0.0, contact_offset=0.002, pieces=pieces)


def drawer_object() -> Part:
    """The drawer: one rigid body. Root piece is its bottom; sides, back, front, knobs and the towels
    are welded to it."""
    zc = SIDE_Z                                   # root centre height above TOP
    def rel(x, y, z):                             # drawer-frame (x from the front face, z above TOP)
        return (float(x - DD / 2), float(y), float(z - zc))

    side_y = DW / 2 - ST / 2
    pieces = (
        Piece("box", (DD - FRONT_T, ST, DH), rel(FRONT_T + (DD - FRONT_T) / 2, side_y, RUN_TOP + DH / 2),
              color=PINE, friction=WOOD_FRICTION),
        Piece("box", (DD - FRONT_T, ST, DH), rel(FRONT_T + (DD - FRONT_T) / 2, -side_y, RUN_TOP + DH / 2),
              color=PINE, friction=WOOD_FRICTION),
        Piece("box", (ST, DW - 2 * ST, DH - 0.010), rel(DD - ST / 2, 0.0, RUN_TOP + 0.010 + (DH - 0.010) / 2), color=PINE),
        Piece("box", (FRONT_T, DW, FRONT_H), rel(FRONT_T / 2, 0.0, FRONT_Z), color=PINE, friction=WOOD_FRICTION),
        Piece("box", (0.15, 0.46, 0.060), rel(0.190, 0.0, RUN_TOP + 0.018 + 0.030), color=TOWEL, collide=False),
        Piece("cylinder", (KNOB_R, KNOB_L, KNOB_R), rel(-KNOB_L / 2, KNOB_Y, KNOB_Z), axis="X", color=KNOB_COL,
              friction=(0.6, 0.5)),
        Piece("cylinder", (KNOB_R, KNOB_L, KNOB_R), rel(-KNOB_L / 2, -KNOB_Y, KNOB_Z), axis="X", color=KNOB_COL,
              friction=(0.6, 0.5)),
    )
    # root: the 6 mm plywood bottom, set in the sides 12 mm above their lower edges
    bottom = (DD - FRONT_T - ST, DW - 2 * ST, 0.006)
    root_local_x = FRONT_T + bottom[0] / 2
    # shift every piece so the root (the bottom) sits at its own centre
    dx = root_local_x - DD / 2
    dz = (RUN_TOP + 0.015) - zc
    pieces = tuple(Piece(p.shape, p.size, (p.offset[0] - dx, p.offset[1], p.offset[2] - dz), p.rot, p.axis, p.color,
                         p.material, p.opacity, p.emissive, p.collide, p.friction, p.restitution) for p in pieces)
    c, s = math.cos(YAW0), math.sin(YAW0)
    # where the bottom's centre starts, given the drawer frame's start pose
    origin = np.array([ROOT0[0] - DD / 2 * c, ROOT0[1] - DD / 2 * s])      # front-face origin in world xy
    p = origin + np.array([root_local_x * c, root_local_x * s])
    pos = (float(p[0]), float(p[1]), float(TOP + RUN_TOP + 0.015))
    # centre of mass: 1.1 kg of drawer about its middle, 0.5 kg of towels at the back (drawer frame x 0.19)
    com_x = (1.1 * (DD / 2) + 0.5 * 0.190) / DRAWER_MASS
    com = (float(com_x - root_local_x), 0.0, 0.030)
    return Part("drawer", pos=pos, size=bottom, exact_z=True, mass=DRAWER_MASS, color=INSIDE, yaw=YAW0,
                friction=RUNNER_FRICTION, restitution=0.0, contact_offset=0.002, pieces=pieces, com=com)


ROOT_Z = RUN_TOP + 0.015                                   # the root's height above TOP
BOTTOM_LOCAL_X = FRONT_T + (DD - FRONT_T - ST) / 2          # the root (bottom) centre in the drawer frame


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def drawer_frame(pose):
    """(origin, R): the drawer frame in world, from its root pose: origin on the front face's centre
    line at the root's height, x into the chest."""
    pose = np.asarray(pose, float)
    r = quat_matrix(pose[3:7])
    origin = pose[:3] - r @ np.array([BOTTOM_LOCAL_X, 0.0, 0.0])
    return origin, r


def to_world(pose, local):
    o, r = drawer_frame(pose)
    return o + r @ np.asarray(local, float)


def front_gaps(pose):
    """How far each end of the drawer front stands proud of the chest face (m, + is out): left, right."""
    o, r = drawer_frame(pose)
    gl = FX - (o + r @ np.array([0.0, DW / 2, 0.0]))[0]
    gr = FX - (o + r @ np.array([0.0, -DW / 2, 0.0]))[0]
    return float(gl), float(gr)


def yaw_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def tilt_deg(pose):
    """How far the drawer is tipped out of level (pitch or roll), degrees."""
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))


def knob_end(pose, side):
    """World position of a knob's end face centre: side 'left' (+y) or 'right'."""
    y = KNOB_Y if side == "left" else -KNOB_Y
    return to_world(pose, (-KNOB_L, y, KNOB_Z - ROOT_Z))


def front_point(pose, y):
    """World position on the drawer front's outer face at drawer-frame y, at the root's height."""
    return to_world(pose, (0.0, y, 0.0))


def start_pose():
    """The drawer root's start pose (x, y, z, qx, qy, qz, qw)."""
    d = drawer_object()
    return np.array([*d.pos, 0.0, 0.0, math.sin(YAW0 / 2), math.cos(YAW0 / 2)])
