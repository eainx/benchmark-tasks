"""Geometry and measurements for making up a shipping carton's bottom.

An 8 x 6 x 2 in regular slotted carton (RSC), opened into a tube and standing upside down: its top
flaps are folded out flat on the bench as its feet, its four bottom flaps stand open, straight up.
The tube is four wall panels joined at their corner creases; the two creases that were folded flat
when the carton was knocked down remember it and rack the tube toward flat unless it is held. The
short (minor) flaps are exactly as wide as the inside of the tube, so they can only be folded in
when it is square, and once folded in their edges bear on the long walls and keep it square.
The long (major) flaps then go over them and meet at the seam, which is taped with a hand tape gun.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The carton's long walls
run along x, so the seam runs along x, away from the arms.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

BOARD = (0.70, 0.55, 0.36)           # kraft corrugated board
BOARD_IN = (0.62, 0.47, 0.29)        # the inside face, a little darker
CREASE = (0.55, 0.42, 0.26)
TAPE_COL = (0.80, 0.62, 0.36)        # tan packing tape
GUN_COL = (0.75, 0.10, 0.10)
RUBBER = (0.08, 0.08, 0.09)
STEEL = (0.70, 0.71, 0.73)

# -- the carton ---------------------------------------------------------------------------------
C = (0.255, 0.000)                   # centre of its footprint
L, W, H = 0.203, 0.152, 0.051        # outside: 8 x 6 x 2 in
T = 0.004                            # 32 ECT C-flute board
FLAP = W / 2                         # RSC: every flap is half the width deep (76 mm)
HL = H + T                           # the long walls' creases sit one board above the short walls': their flaps go over
MINOR_CLEAR = 0.0030                 # a short flap is this much narrower than the inside of the tube
MAJOR_FLAP = (W - 2 * 0.004) / 2 - 0.001   # the long flaps meet at the middle of the inside, 1 mm apart each
SLOT = 0.003                         # the RSC's slots: a long flap stops this far short of each end wall
WALL_MASS = {"long": 0.028, "short": 0.021}
FLAP_MASS = {"major": 0.014, "minor": 0.011}
FRICTION = (0.55, 0.45)              # board on board and on the bench

# Corner creases (vertical revolute joints). Square = 0. The two creases folded flat in the knocked-down
# carton (near-left and far-right) spring it back toward flat: unheld the tube racks about RACK_REST.
RACK_REST_DEG = 15.0
CORNER_K = 0.0030                    # N m per degree, at each corner, toward the racked rest
CORNER_DAMP = 0.0008

# Flap creases. 0 = standing straight up (open); +90 = folded flat, in over the opening.
FLAP_LIMIT = 92.0
MINOR_K, MAJOR_K = 0.0006, 0.0008   # N m per degree toward the crease's current set
SPRING_BACK = {"minor": 20.0, "major": 25.0}   # a crease takes a set: let go, a flap folded to a springs back to a - this

WALLS = ("carton", "wall_far", "wall_left", "wall_right")    # "carton" is the near short wall: the tape's frame
MINORS = ("minor_near", "minor_far")
MAJORS = ("major_left", "major_right")
FLAPS = MINORS + MAJORS


def _c(x, y, z):
    return (float(C[0] + x), float(C[1] + y), float(TOP + z))


def wall_pos(name):
    return {"carton": _c(-L / 2 + T / 2, 0.0, H / 2), "wall_far": _c(L / 2 - T / 2, 0.0, H / 2),
            "wall_left": _c(0.0, W / 2 - T / 2, HL / 2), "wall_right": _c(0.0, -W / 2 + T / 2, HL / 2)}[name]


def corner(xs, ys):
    """A corner crease's pivot: on the outside corner, at mid-height."""
    return _c(xs * (L / 2 - T / 2), ys * (W / 2 - T / 2), H / 2)


# Each wall is hinged to the one before it round the tube; the last hinge closes the loop (the
# manufacturer's glued joint is at the near-right corner, which is simply one of the four creases).
LOOP = {"wall_left": ("carton", corner(-1, 1), +1), "wall_far": ("wall_left", corner(1, 1), -1),
        "wall_right": ("wall_far", corner(1, -1), +1), "carton": ("wall_right", corner(-1, -1), -1)}


def _wall(name) -> Part:
    long = name in ("wall_left", "wall_right")
    size = (L, T, HL) if long else (T, W - 2 * T, H)
    hz = HL if long else H
    pos = wall_pos(name)
    out = {"carton": (-1, 0), "wall_far": (1, 0), "wall_left": (0, 1), "wall_right": (0, -1)}[name]
    # the top flap folded out flat under it, as its foot
    if long:
        foot = Piece("box", (L - 0.004, FLAP, T), (0.0, out[1] * (T / 2 + FLAP / 2), -hz / 2 + T / 2))
    else:
        foot = Piece("box", (FLAP, W - 0.004, T), (out[0] * (T / 2 + FLAP / 2), 0.0, -H / 2 + T / 2))
    inner = Piece("box", (L - 2 * T, 0.0006, hz - 0.002) if long else (0.0006, W - 2 * T - 0.002, H - 0.002),
                  (-out[0] * T / 2, -out[1] * T / 2, 0.0), color=BOARD_IN, collide=False)
    parent, pivot, sign = LOOP[name]
    k = CORNER_K
    return Part(name, pos=pos, size=size, exact_z=True, mass=WALL_MASS["long" if long else "short"], color=BOARD,
                friction=FRICTION, restitution=0.0, contact_offset=0.0015, pieces=(foot, inner),
                hinge=Hinge(pivot=pivot, axis="Z", damping=CORNER_DAMP, lower_deg=-45.0, upper_deg=45.0,
                            stiffness=k, target_deg=sign * RACK_REST_DEG, parent=parent,
                            parent_pos=wall_pos(parent)))


# Flaps: a box standing on its wall's top edge, hinged along that edge.
def _flap_geom(name):
    """(centre, size, pivot, axis, sign, wall) with the flap standing up."""
    if name == "minor_near":
        return (_c(-L / 2 + T / 2, 0.0, H + FLAP / 2), (T, W - 2 * T - MINOR_CLEAR, FLAP), _c(-L / 2 + T, 0.0, H), "Y", +1, "carton")
    if name == "minor_far":
        return (_c(L / 2 - T / 2, 0.0, H + FLAP / 2), (T, W - 2 * T - MINOR_CLEAR, FLAP), _c(L / 2 - T, 0.0, H), "Y", -1, "wall_far")
    if name == "major_left":
        return (_c(0.0, W / 2 - T / 2, HL + MAJOR_FLAP / 2), (L - 2 * T - 2 * SLOT, T, MAJOR_FLAP), _c(0.0, W / 2 - T, HL), "X", +1, "wall_left")
    return (_c(0.0, -W / 2 + T / 2, HL + MAJOR_FLAP / 2), (L - 2 * T - 2 * SLOT, T, MAJOR_FLAP), _c(0.0, -W / 2 + T, HL), "X", -1, "wall_right")


def flap_sign(name):
    """Joint angle sign that folds the flap in: +1 if folding in is a positive joint angle."""
    centre, size, pivot, axis, s, wall = _flap_geom(name)
    # about +Y a near flap (at -x) folds in (+x) for a positive angle; about +X a left flap (at +y)
    # folds in (-y) for a positive angle.
    return s


def _flap(name) -> Part:
    centre, size, pivot, axis, s, wall = _flap_geom(name)
    major = name in MAJORS
    k = MAJOR_K if major else MINOR_K
    lo, hi = (-3.0, FLAP_LIMIT) if s > 0 else (-FLAP_LIMIT, 3.0)     # an open flap leans out at most 3 degrees
    crease = Piece("box", (size[0] + 0.0004, size[1] + 0.0004, 0.003), (0.0, 0.0, -size[2] / 2 + 0.0015),
                   color=CREASE, collide=False)
    return Part(name, pos=centre, size=size, exact_z=True, mass=FLAP_MASS["major" if major else "minor"], color=BOARD,
                friction=FRICTION, restitution=0.0, contact_offset=0.0015, pieces=(crease,),
                hinge=Hinge(pivot=pivot, axis=axis, damping=0.0005, lower_deg=lo, upper_deg=hi, stiffness=k,
                            target_deg=0.0, parent=wall, parent_pos=wall_pos(wall)))


def flap_spawn(name):
    return _flap_geom(name)[0]


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def flap_fold(name, pose):
    """How far the flap is folded in, degrees: 0 standing up, 90 flat over the opening."""
    r = quat_matrix(np.asarray(pose)[3:7])
    up = r @ np.array([0.0, 0.0, 1.0])               # from its crease toward its free edge
    inward = {"minor_near": (1, 0), "minor_far": (-1, 0), "major_left": (0, -1), "major_right": (0, 1)}[name]
    horiz = up[0] * inward[0] + up[1] * inward[1]
    return math.degrees(math.atan2(horiz, up[2]))


def yaw_of(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def rack_deg(poses):
    """How far the tube is out of square: the largest corner's departure from 90 degrees."""
    yaws = {n: yaw_of(poses[n]) for n in WALLS}
    pairs = (("carton", "wall_left"), ("wall_left", "wall_far"), ("wall_far", "wall_right"), ("wall_right", "carton"))
    worst = 0.0
    for a, b in pairs:
        d = (yaws[b] - yaws[a] + 180.0) % 360.0 - 180.0
        worst = max(worst, abs(d))
    return worst


def wall_out_of_plane(poses):
    """The largest tilt of a wall from vertical (a crushed carton), degrees."""
    worst = 0.0
    for n in WALLS:
        r = quat_matrix(np.asarray(poses[n])[3:7])
        worst = max(worst, math.degrees(math.acos(np.clip(r[2, 2], -1, 1))))
    return worst


# -- the tape, as cells along the strip ----------------------------------------------------------
# s runs along the strip from the near top edge: negative down the near end wall, 0..L along the
# seam away from the arms, beyond L down the far end wall. Cells belong to the near wall's body.
TAPE_W = 0.048
CELL = 0.008
TAB_CELLS = 6                        # 48 mm down each 51 mm end wall
TOP_CELLS = int(round(L / CELL))     # 25
STRIP_L = L
CELLS = 2 * TAB_CELLS + TOP_CELLS
LEADER = 0.075                       # tape hanging from the gun's roller before it is started
TAB_MIN = 0.040                      # tab down each end wall for full credit
TOP_SURF = TOP + H + 2 * T           # the folded long flaps' top face (over the short flaps)


def cell_s(k):
    return (k - TAB_CELLS + 0.5) * CELL


def carton_frame(near_pose):
    """Carton centre (at bench level) and rotation, from the near wall's pose."""
    p = np.asarray(near_pose, float)
    r = quat_matrix(p[3:7])
    centre = p[:3] - r @ np.array([-L / 2 + T / 2, 0.0, H / 2])
    return centre, r


def cell_local(k, offset=0.0):
    """Cell k's centre relative to the near wall's body origin, and which face it is on."""
    s = cell_s(k)
    top = TOP_SURF - TOP
    base = np.array([-(-L / 2 + T / 2), 0.0, -H / 2])   # carton centre at bench level, in the near wall's frame
    if s < 0:
        p = np.array([-L / 2 - 0.0003, offset, top + s])
        where = "near"
    elif s > STRIP_L:
        p = np.array([L / 2 + 0.0003, offset, top - (s - STRIP_L)])
        where = "far"
    else:
        p = np.array([-L / 2 + s, offset, top + 0.0003])
        where = "top"
    return base + p, where


TAPE_PIECE0 = 2                      # the near wall's pieces: foot, inside face, then the tape cells


def wall_objects():
    parts = [_wall(n) for n in WALLS]
    cells = []
    for k in range(CELLS):
        c, where = cell_local(k)
        size = (CELL + 0.0004, TAPE_W, 0.0004) if where == "top" else (0.0004, TAPE_W, CELL + 0.0004)
        cells.append(Piece("box", size, tuple(float(v) for v in c), color=TAPE_COL, opacity=0.9, collide=False))
    parts[0] = parts[0].with_pieces(*cells)
    return parts


def flap_objects():
    return [_flap(n) for n in FLAPS]


def seam_frame(poses):
    """The frame the tape is measured in: centred between the long walls (where the long flaps meet, the
    seam) and turned with them, at bench level."""
    a = np.asarray(poses["wall_left"], float)
    b = np.asarray(poses["wall_right"], float)
    ya = math.atan2(*(quat_matrix(a[3:7])[[1, 0], 0]))
    yb = math.atan2(*(quat_matrix(b[3:7])[[1, 0], 0]))
    yaw = math.atan2(math.sin(ya) + math.sin(yb), math.cos(ya) + math.cos(yb))
    c, s = math.cos(yaw), math.sin(yaw)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    centre = (a[:3] + b[:3]) / 2
    centre[2] = TOP
    return centre, r


def to_carton(frame, p):
    centre, r = frame
    return r.T @ (np.asarray(p, float) - centre)


TOP_LOCAL = TOP_SURF - TOP
END_X = L / 2


def roller_contact(frame, roller):
    """Where the roller is pressing tape on: (s along the strip, offset across it) or None."""
    c = to_carton(frame, roller)
    r = ROLLER[0]
    if abs(c[1]) > 0.06:
        return None
    if abs(c[2] - (TOP_LOCAL + r)) < 0.003 and abs(c[0]) <= END_X + 0.002:
        return float(c[0] + END_X), float(c[1])
    return None


def blade_cut(frame, roller, blade, fwd):
    c = to_carton(frame, roller)
    if abs(c[1]) > 0.06 or c[0] < END_X + 0.008 or c[2] < TOP_LOCAL + ROLLER[0] + CUT_LIFT:
        return None
    bl = to_carton(frame, blade)
    return float(STRIP_L + min(MAX_TAIL, float(np.linalg.norm(bl - np.array([END_X, bl[1], TOP_LOCAL])))))


CUT_LIFT = 0.012
MAX_TAIL = TAB_CELLS * CELL


def hand_press(frame, tip):
    """Where a fingertip is smoothing tape down an end wall (s), or None."""
    c = to_carton(frame, tip)
    if abs(c[1]) > 0.04 or not (TOP_LOCAL - 0.07 < c[2] < TOP_LOCAL + 0.002):
        return None
    if END_X - 0.002 < c[0] < END_X + 0.020:
        return float(STRIP_L + TOP_LOCAL - c[2])
    if -END_X - 0.020 < c[0] < -END_X + 0.002:
        return float(-(TOP_LOCAL - c[2]))
    return None


def cell_of(s):
    return int(math.floor(s / CELL)) + TAB_CELLS


# -- tape gun (free): the same gun as the accepted carton-sealing task -----------------------------
GUN_REST = (0.200, -0.200)
GUN_YAW = 0.0
GUN_MASS = 0.35
ROLL_R = 0.040
ROLLER = (0.011, 0.054)
ROLLER_AT = (0.050, -0.042)
BLADE_AT = (0.068, -0.050)
HANDLE = (0.014, 0.045)
HANDLE_AT = (-0.030, 0.062)
PLATE = (0.100, 0.006, 0.090)
PLATE_X = -0.028
FOOT_AT = (-0.050, -0.053)
SIDE = TAPE_W / 2 + 0.006
ROOT_LOCAL = (PLATE_X, SIDE, 0.0)


def gun_ground():
    return -(ROLLER_AT[1] - ROLLER[0])


def gun_rot(yaw=GUN_YAW, pitch=0.0):
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz @ ry


def gun_object() -> Part:
    g = np.array([GUN_REST[0], GUN_REST[1], TOP + gun_ground()])
    r = gun_rot()
    root_local = np.array(ROOT_LOCAL)

    def at(x, y, z):
        return tuple(float(v) for v in np.array([x, y, z]) - root_local)

    pieces = (
        Piece("box", PLATE, at(PLATE_X, -SIDE, 0.0)),
        Piece("box", (0.040, 0.005, 0.010), at(ROLLER_AT[0] - 0.016, SIDE, ROLLER_AT[1] + 0.004), collide=False),
        Piece("box", (0.040, 0.005, 0.010), at(ROLLER_AT[0] - 0.016, -SIDE, ROLLER_AT[1] + 0.004), collide=False),
        Piece("cylinder", (ROLL_R, TAPE_W, ROLL_R), at(-0.012, 0.0, 0.004), axis="Y", color=TAPE_COL, opacity=0.85,
              collide=False),
        Piece("cylinder", (0.019, TAPE_W + 0.002, 0.019), at(-0.012, 0.0, 0.004), axis="Y", color=(0.55, 0.45, 0.32),
              collide=False),
        Piece("box", (0.002, TAPE_W + 0.006, 0.012), at(BLADE_AT[0], 0.0, BLADE_AT[1] + 0.006), color=STEEL,
              collide=False),
        Piece("box", (0.024, TAPE_W + 0.012, 0.006), at(FOOT_AT[0], 0.0, FOOT_AT[1] + 0.003), color=RUBBER),
        Piece("box", (2 * HANDLE[0] + 0.006, 2 * HANDLE[0], HANDLE[1]), at(HANDLE_AT[0], 0.0, HANDLE_AT[1]), color=RUBBER,
              friction=(1.4, 1.2)),
        Piece("box", (0.030, TAPE_W + 0.012, 0.008), at(HANDLE_AT[0] + 0.010, 0.0, HANDLE_AT[1] - HANDLE[1] / 2)),
        Piece("box", (0.002, TAPE_W, 0.040), at(ROLLER_AT[0] + ROLLER[0] + 0.001, 0.0, ROLLER_AT[1] + 0.024),
              color=TAPE_COL, opacity=0.85, collide=False),
    )
    root = g + r @ root_local
    return Part("gun", pos=tuple(float(v) for v in root), size=PLATE, exact_z=True, mass=GUN_MASS, color=GUN_COL,
                friction=(0.6, 0.5), restitution=0.0, contact_offset=0.0015, yaw=GUN_YAW,
                com=at(-0.020, 0.0, 0.010), pieces=pieces)


def roller_object() -> Part:
    g = np.array([GUN_REST[0], GUN_REST[1], TOP + gun_ground()])
    r = gun_rot()
    c = g + r @ np.array([ROLLER_AT[0], 0.0, ROLLER_AT[1]])
    marks = []
    for sy in (-1, 1):
        y = sy * (ROLLER[1] / 2 + 0.0008)
        marks.append(Piece("cylinder", (ROLLER[0] + 0.0015, 0.0016, ROLLER[0] + 0.0015), (0.0, y, 0.0), axis="Y",
                           color=(0.98, 0.78, 0.05), collide=False))
        marks.append(Piece("box", (0.004, 0.0020, 2 * ROLLER[0] + 0.002), (0.0, y + sy * 0.0003, 0.0),
                           color=(0.05, 0.05, 0.05), collide=False))
    return Part("roller", pos=tuple(float(v) for v in c), size=(ROLLER[0], ROLLER[1], ROLLER[0]), shape="cylinder",
                axis="Y", exact_z=True, mass=0.025, color=RUBBER, friction=(1.2, 1.0), restitution=0.0,
                contact_offset=0.0015, pieces=tuple(marks),
                hinge=Hinge(pivot=tuple(float(v) for v in c), axis="Y", damping=0.00002, lower_deg=-1e6, upper_deg=1e6,
                            parent="gun", parent_pos=tuple(float(v) for v in g + r @ np.array(ROOT_LOCAL))))


def gun_points(pose):
    p = np.asarray(pose, float)
    r = quat_matrix(p[3:7])
    centre = p[:3] - r @ np.array(ROOT_LOCAL)
    roller = centre + r @ np.array([ROLLER_AT[0], 0.0, ROLLER_AT[1]])
    blade = centre + r @ np.array([BLADE_AT[0], 0.0, BLADE_AT[1]])
    return roller, blade, r[:, 0], r[:, 2]
