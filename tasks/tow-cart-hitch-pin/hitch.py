"""Geometry of the tow-cart pin hitch: the lead cart's rear end with its clevis jaw and drop pin, and the
trailing cart's front end with its pivoting drawbar and eye. Real size; both carts are cropped sections
(their far ends are not modelled) standing on their casters on the bench, which plays the shop floor.

World frame: x away from the arms, y to their left, z up; bench top z = TOP.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece, Spring

TOP = 0.7475

# -- the hitch -------------------------------------------------------------------------------------------
HX, HY = 0.320, -0.040            # the pin's axis (the jaw's hole centre)
PLATE_T = 0.012                   # 1/2 in steel jaw plates
LO_TOP = TOP + 0.080              # lower jaw plate's top face
GAP = 0.045                       # between the plates
UP_BOT = LO_TOP + GAP             # upper plate's underside
UP_TOP = UP_BOT + PLATE_T
HOLE = 0.021                      # square-cut hole through each plate (a 19 mm pin, 1 mm clearance a side)
PLATE_X = 0.100                   # plate size across (x) ...
PLATE_Y0, PLATE_Y1 = HY - 0.070, HY + 0.055   # ... and its extent along the hitch line (welded at y0)

PIN_R = 0.0095                    # 3/4 in pin
PIN_BOT_SEATED = LO_TOP - PLATE_T - 0.015      # its end stands 15 mm out under the lower plate
COLLAR_R, COLLAR_H = 0.017, 0.010
PIN_SHANK = UP_TOP - PIN_BOT_SEATED            # the shank, collar seated on the upper plate
PIN_LIFT_MAX = 0.080              # the pin can be drawn up this far and still stays in the upper plate's hole
PIN_MASS = 0.36
PIN_Z0 = PIN_BOT_SEATED + PIN_SHANK / 2 + 0.0005   # the shank's centre as set down (0.5 mm proud: settles)
HANDLE_Z = UP_TOP + COLLAR_H + 0.032            # the T-handle's crossbar centre, pin seated
HANDLE_LEN, HANDLE_R = 0.080, 0.0075

# -- the drawbar -----------------------------------------------------------------------------------------
EYE_T = 0.025                     # eye plate thickness
EYE_Z = LO_TOP + GAP / 2          # eye (and bar) centre height: the drawbar hangs level at the jaw's middle
EYE_RO, EYE_RI = 0.036, 0.013     # eye outer radius, hole radius (26 mm hole)
BAR_LEN = 0.500                   # pivot to eye centre
PX, PY = HX, HY + BAR_LEN         # the drawbar's pivot on the trailing cart
BAR_R = 0.016                     # 32 mm round bar
START_DEG = -22.0                 # the drawbar as left: swung 22 degrees off line, toward the arms
BAR_MASS = 2.2
BAR_DAMP = 0.02                   # N m s / deg on its pivot: a cart's tongue swings stiffly, it does not coast

# -- colours ---------------------------------------------------------------------------------------------
YELLOW = (0.86, 0.66, 0.06)       # safety-yellow powder coat
BLUE = (0.10, 0.25, 0.52)         # the trailing cart's blue frame
DECK = (0.42, 0.43, 0.44)         # steel deck plate
BLACK = (0.05, 0.05, 0.055)
ZINC = (0.74, 0.75, 0.73)
RUBBER = (0.06, 0.06, 0.06)
DARK_STEEL = (0.20, 0.20, 0.21)


def eye_center(theta_deg):
    """Eye centre (world) with the drawbar at ``theta_deg`` (0: on line, eye on the pin's axis)."""
    t = math.radians(theta_deg)
    return np.array([PX + BAR_LEN * math.sin(t), PY - BAR_LEN * math.cos(t), EYE_Z])


def bar_point(theta_deg, r):
    t = math.radians(theta_deg)
    return np.array([PX + r * math.sin(t), PY - r * math.cos(t), EYE_Z])


def _ring(r_in, r_out, thick, z, n, color, cx=0.0, cy=0.0, collide=True):
    """A flat ring (an eye) of n box segments about (cx, cy) in the Part frame."""
    ps = []
    rm = (r_in + r_out) / 2
    w = r_out - r_in
    seg = 2 * r_out * math.tan(math.pi / n) + 0.002
    for k in range(n):
        a = 2 * math.pi * k / n
        ps.append(Piece("box", (w, seg, thick), (cx + rm * math.cos(a), cy + rm * math.sin(a), z), rot=(0.0, 0.0, a),
                        color=color, collide=collide))
    return ps


def _caster(x, y, color_fork=ZINC):
    """A 4 in swivel caster under a cart frame: rubber wheel, zinc fork and top plate (visual)."""
    ps = [Piece("cylinder", (0.050, 0.032, 0.050), (x, y, TOP + 0.050), axis="X", color=RUBBER, collide=False),
          Piece("cylinder", (0.020, 0.036, 0.020), (x, y, TOP + 0.050), axis="X", color=ZINC, collide=False)]
    for s in (-1, 1):
        ps.append(Piece("box", (0.004, 0.030, 0.070), (x + s * 0.021, y + 0.012, TOP + 0.062), color=color_fork, collide=False))
    ps.append(Piece("box", (0.050, 0.050, 0.010), (x, y + 0.012, TOP + 0.098), color=color_fork, collide=False))
    ps.append(Piece("cylinder", (0.012, 0.012, 0.012), (x, y + 0.012, TOP + 0.090), color=color_fork, collide=False))
    return ps


# -- the lead cart: rear end, clevis jaw ------------------------------------------------------------------
LEAD_X0, LEAD_X1 = 0.150, 0.650   # its frame across (x)
LEAD_Y_REAR = PLATE_Y0            # its rear crossmember's rear face
LEAD_Y_CROP = HY - 0.380          # cropped here (the rest of the cart is not modelled)
FRAME_Z0, FRAME_Z1 = TOP + 0.105, TOP + 0.155   # 50 mm box-section frame


def lead_cart_object() -> Part:
    ps: list[Piece] = []
    zc = (FRAME_Z0 + FRAME_Z1) / 2
    fh = FRAME_Z1 - FRAME_Z0
    # rear crossmember (50 x 50 box tube) and the two side rails
    ps.append(Piece("box", (LEAD_X1 - LEAD_X0, 0.050, fh), ((LEAD_X0 + LEAD_X1) / 2, LEAD_Y_REAR - 0.025, zc), color=YELLOW))
    for x in (LEAD_X0 + 0.025, LEAD_X1 - 0.025):
        ps.append(Piece("box", (0.050, LEAD_Y_REAR - LEAD_Y_CROP, fh), (x, (LEAD_Y_REAR + LEAD_Y_CROP) / 2, zc), color=YELLOW))
    # a mid cross-rail and the deck plate
    ps.append(Piece("box", (LEAD_X1 - LEAD_X0, 0.040, 0.040), ((LEAD_X0 + LEAD_X1) / 2, LEAD_Y_CROP + 0.080, zc), color=YELLOW))
    ps.append(Piece("box", (LEAD_X1 - LEAD_X0 - 0.004, LEAD_Y_REAR - LEAD_Y_CROP - 0.054, 0.005),
                    ((LEAD_X0 + LEAD_X1) / 2, (LEAD_Y_REAR - 0.052 + LEAD_Y_CROP) / 2, FRAME_Z1 + 0.0025), color=DECK))
    # tread pattern on the deck plate
    for i in range(18):
        for j in range(8):
            x = LEAD_X0 + 0.030 + i * 0.025
            y = LEAD_Y_CROP + 0.020 + j * 0.033 + (0.012 if i % 2 else 0.0)
            if y < LEAD_Y_REAR - 0.060:
                ps.append(Piece("box", (0.014, 0.004, 0.0015), (x, y, FRAME_Z1 + 0.0055), rot=(0.0, 0.0, 0.6 if (i + j) % 2 else -0.6),
                                color=(0.50, 0.51, 0.52), collide=False))
    # raised bumper lip on the deck's rear edge
    ps.append(Piece("box", (LEAD_X1 - LEAD_X0, 0.010, 0.030), ((LEAD_X0 + LEAD_X1) / 2, LEAD_Y_REAR - 0.055, FRAME_Z1 + 0.015),
                    color=YELLOW))
    # the clevis jaw: two plates welded to the crossmember's rear face, each with a square-cut pin hole
    xs = (HX - PLATE_X / 2, HX + PLATE_X / 2)
    for ztop in (LO_TOP, UP_TOP):
        zc_p = ztop - PLATE_T / 2
        h = HOLE / 2
        ps.append(Piece("box", (PLATE_X, HY - h - PLATE_Y0, PLATE_T), (HX, (PLATE_Y0 + HY - h) / 2, zc_p), color=BLACK))
        ps.append(Piece("box", (PLATE_X, PLATE_Y1 - HY - h, PLATE_T), (HX, (PLATE_Y1 + HY + h) / 2, zc_p), color=BLACK))
        ps.append(Piece("box", (HX - h - xs[0], HOLE, PLATE_T), ((xs[0] + HX - h) / 2, HY, zc_p), color=BLACK))
        ps.append(Piece("box", (xs[1] - HX - h, HOLE, PLATE_T), ((xs[1] + HX + h) / 2, HY, zc_p), color=BLACK))
        # a rounded nose on each plate's free end
        ps.append(Piece("cylinder", (PLATE_X / 2, PLATE_T, PLATE_X / 2), (HX, PLATE_Y1 - 0.020, zc_p), color=BLACK, collide=False))
    # the plates' gussets down to the crossmember (visual)
    for x in xs:
        ps.append(Piece("box", (0.008, 0.030, GAP), (x - math.copysign(0.004, x - HX), PLATE_Y0 + 0.015, LO_TOP + GAP / 2),
                        color=BLACK, collide=False))
    # weld beads along the plates' roots
    for z in (LO_TOP - PLATE_T, UP_TOP):
        ps.append(Piece("capsule", (0.003, PLATE_X, 0.003), (HX, PLATE_Y0 + 0.002, z), axis="X", color=(0.15, 0.15, 0.14),
                        collide=False))
    # casters under the side rails
    for x in (LEAD_X0 + 0.025, LEAD_X1 - 0.025):
        ps += _caster(x, LEAD_Y_REAR - 0.090)
    root = ((LEAD_X0 + LEAD_X1) / 2, (LEAD_Y_REAR + LEAD_Y_CROP) / 2, TOP + 0.003)
    rel = [Piece(p.shape, p.size, (p.offset[0] - root[0], p.offset[1] - root[1], p.offset[2] - root[2]), rot=p.rot, axis=p.axis,
                 color=p.color, collide=p.collide) for p in ps]
    return Part("lead_cart", pos=root, size=(0.010, 0.010, 0.006), exact_z=True, color=YELLOW, material=None,
                friction=(0.6, 0.5), contact_offset=0.002, pieces=tuple(rel))


# -- the trailing cart: front end and drawbar pivot bracket ---------------------------------------------
TRAIL_Y_FRONT = PY + 0.030
TRAIL_Y_CROP = PY + 0.300


def trail_cart_object() -> Part:
    ps: list[Piece] = []
    zc = (FRAME_Z0 + FRAME_Z1) / 2
    fh = FRAME_Z1 - FRAME_Z0
    ps.append(Piece("box", (LEAD_X1 - LEAD_X0, 0.050, fh), ((LEAD_X0 + LEAD_X1) / 2, TRAIL_Y_FRONT + 0.025, zc), color=BLUE))
    for x in (LEAD_X0 + 0.025, LEAD_X1 - 0.025):
        ps.append(Piece("box", (0.050, TRAIL_Y_CROP - TRAIL_Y_FRONT, fh), (x, (TRAIL_Y_FRONT + TRAIL_Y_CROP) / 2, zc), color=BLUE))
    ps.append(Piece("box", (LEAD_X1 - LEAD_X0 - 0.004, TRAIL_Y_CROP - TRAIL_Y_FRONT - 0.050, 0.005),
                    ((LEAD_X0 + LEAD_X1) / 2, (TRAIL_Y_FRONT + 0.050 + TRAIL_Y_CROP) / 2, FRAME_Z1 + 0.0025), color=DECK))
    for i in range(18):
        for j in range(6):
            x = LEAD_X0 + 0.030 + i * 0.025
            y = TRAIL_Y_FRONT + 0.070 + j * 0.033 + (0.012 if i % 2 else 0.0)
            if y < TRAIL_Y_CROP - 0.010:
                ps.append(Piece("box", (0.014, 0.004, 0.0015), (x, y, FRAME_Z1 + 0.0055), rot=(0.0, 0.0, 0.6 if (i + j) % 2 else -0.6),
                                color=(0.50, 0.51, 0.52), collide=False))
    # the drawbar's pivot bracket: a clevis on the front crossmember's face, a vertical pivot bolt through it
    for z in (EYE_Z + 0.022, EYE_Z - 0.022):
        ps.append(Piece("box", (0.070, 0.060, 0.008), (PX, TRAIL_Y_FRONT - 0.030, z), color=DARK_STEEL, collide=False))
    ps.append(Piece("cylinder", (0.008, 0.070, 0.008), (PX, PY, EYE_Z), color=ZINC, collide=False))
    ps.append(Piece("cylinder", (0.014, 0.008, 0.014), (PX, PY, EYE_Z + 0.030), color=ZINC, collide=False))
    for x in (LEAD_X0 + 0.025, LEAD_X1 - 0.025):
        ps += _caster(x, TRAIL_Y_FRONT + 0.090)
    root = ((LEAD_X0 + LEAD_X1) / 2, (TRAIL_Y_FRONT + TRAIL_Y_CROP) / 2, TOP + 0.003)
    rel = [Piece(p.shape, p.size, (p.offset[0] - root[0], p.offset[1] - root[1], p.offset[2] - root[2]), rot=p.rot, axis=p.axis,
                 color=p.color, collide=p.collide) for p in ps]
    return Part("trail_cart", pos=root, size=(0.010, 0.010, 0.006), exact_z=True, color=BLUE, friction=(0.6, 0.5),
                contact_offset=0.002, pieces=tuple(rel))


# -- the drawbar: a 32 mm steel bar on a vertical pivot, a flat forged eye on its end ---------------------
def drawbar_object() -> Part:
    """Laid out pointing -y from its pivot (on line), then turned START_DEG about the pivot."""
    ps: list[Piece] = []
    L = BAR_LEN - EYE_RO + 0.004 - 0.026
    ps.append(Piece("cylinder", (BAR_R, L, BAR_R), (0.0, -(0.026 + L / 2), 0.0), axis="Y", color=BLACK))
    # the flattened neck into the eye and the eye ring (its 26 mm hole a ring of 12 segments)
    ps.append(Piece("box", (0.040, 0.030, EYE_T), (0.0, -(BAR_LEN - EYE_RO - 0.008), 0.0), color=BLACK))
    ps += _ring(EYE_RI, EYE_RO, EYE_T, 0.0, 12, BLACK, cy=-BAR_LEN)
    # the pivot end: a forged clevis knuckle round the pivot bolt (the root collider is the knuckle)
    # a yellow grip sleeve partway along it
    ps.append(Piece("cylinder", (BAR_R + 0.0025, 0.110, BAR_R + 0.0025), (0.0, -0.300, 0.0), axis="Y", color=(0.80, 0.12, 0.08),
                    collide=False))
    return Part("drawbar", pos=(PX, PY, EYE_Z), size=(0.022, 0.030, 0.022), shape="cylinder", exact_z=True, mass=BAR_MASS,
                yaw=math.radians(START_DEG), color=BLACK, friction=(0.6, 0.5), contact_offset=0.002,
                com=(0.0, -0.20, 0.0),
                hinge=Hinge(pivot=(PX, PY, EYE_Z), axis="Z", damping=BAR_DAMP, lower_deg=-75.0, upper_deg=75.0),
                pieces=tuple(ps))


# -- the hitch pin: 3/4 in zinc-plated drop pin with a T-handle -----------------------------------------
def pin_object() -> Part:
    zc = PIN_Z0
    ps = [Piece("cylinder", (COLLAR_R, COLLAR_H, COLLAR_R), (0.0, 0.0, UP_TOP + 0.0005 + COLLAR_H / 2 - zc), color=ZINC),
          Piece("cylinder", (0.006, 0.026, 0.006), (0.0, 0.0, UP_TOP + 0.0005 + COLLAR_H + 0.013 - zc), color=ZINC),
          Piece("cylinder", (HANDLE_R, HANDLE_LEN, HANDLE_R), (0.0, 0.0, HANDLE_Z + 0.0005 - zc), axis="X", color=ZINC),
          # a chamfered tip
          Piece("cone", (PIN_R, 0.006, PIN_R), (0.0, 0.0, PIN_BOT_SEATED + 0.0005 - 0.003 - zc), rot=(math.pi, 0.0, 0.0), color=ZINC,
                collide=False)]
    return Part("pin", pos=(HX, HY, zc), size=(PIN_R, PIN_SHANK, PIN_R), shape="cylinder", exact_z=True, mass=PIN_MASS,
                color=ZINC, material="steel", friction=(0.45, 0.40), contact_offset=0.002,
                spring=Spring(axis="Z", lower=-0.003, upper=PIN_LIFT_MAX, stiffness=0.0, damping=1.0, target=0.0),
                pieces=tuple(ps))


def quat_matrix(q):
    """Rotation matrix of a quaternion (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bar_angle(bar_pose):
    """The drawbar's swing, degrees (0 on line), from its measured pose."""
    r = quat_matrix(np.asarray(bar_pose, float)[3:7])
    d = r @ np.array([0.0, -1.0, 0.0])
    return math.degrees(math.atan2(d[0], -d[1]))


def eye_from_pose(bar_pose):
    p = np.asarray(bar_pose, float)
    r = quat_matrix(p[3:7])
    return p[:3] + r @ np.array([0.0, -BAR_LEN, 0.0])


def pin_lift(pin_pose):
    """How far the pin is drawn up from seated (m)."""
    return float(np.asarray(pin_pose, float)[2] - (PIN_Z0 - 0.0005))
