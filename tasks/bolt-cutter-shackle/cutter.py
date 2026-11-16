"""Geometry and measurements for cutting a seized padlock off a storage-cage gate with bolt cutters.

On the bench stands a run of steel-mesh storage-cage wall: the gate's latch stile beside the fixed
post, a hasp across them, and through the hasp's staple a 40 mm brass padlock whose key is lost. A
14 in (355 mm) bolt cutter lies shut on a tool case in front of it. One shackle leg has to be taken
between the cutter's jaws, square to it, with the head held still, and both handles closed together
until the leg is cut through; then the cutter is laid down and the hands stand clear.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
The wall runs along x with its face toward -y; the padlock hangs in front of it, and the arms come
at its nearer shackle leg from -x.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

MESH_COL = (0.30, 0.32, 0.34)        # galvanised then painted cage mesh
FRAME_COL = (0.22, 0.24, 0.26)
STEEL = (0.62, 0.63, 0.65)
BRASS = (0.72, 0.56, 0.24)
SHACKLE_COL = (0.70, 0.71, 0.73)
RED = (0.70, 0.10, 0.08)             # the cutter's painted handles and head
GRIP_COL = (0.08, 0.08, 0.09)        # rubber grips
JAW_COL = (0.38, 0.39, 0.41)         # hardened jaws

# -- the cage wall (fixed) --------------------------------------------------------------------
# A run of storage-cage wall stands on the bench along x, its face toward the right arm (-y): the
# gate's latch stile, the fixed post beside it (a hasp across the two), mesh either side.
WALL_Y = 0.040                       # face of the stile and post (toward -y)
TUBE = 0.040                         # 40 x 40 mm square-tube stile and post
GATE_H = 0.40
POST_X = 0.520                       # fixed post the staple is welded to
STILE_X = POST_X - 0.048             # the gate's latch stile beside it (8 mm gap between)
WALL_X = (0.380, 0.740)              # the wall's run along x
BAR_R = 0.003                        # 6 mm mesh wire ...
BAR_PITCH = 0.050                    # ... on a 50 mm grid

# -- hasp staple and padlock -------------------------------------------------------------------
EYE_X = POST_X
EYE_Y = WALL_Y - 0.040               # the staple's eye: 40 mm proud of the post
EYE_Z = TOP + 0.200
EYE_HOLE = 0.010                     # the shackle's bow passes through it along x
LEG_R = 0.003                        # 6 mm plain-steel shackle (a 14 in cutter's rated size)
LEG_X = 0.014                        # legs either side of the lock's centre: 22 mm inside
LEG_TOP = EYE_Z                      # the bow, through the eye
BODY = (0.040, 0.020, 0.036)         # brass body: x wide, y thick, z tall
BODY_TOP = EYE_Z - 0.032             # 30 mm of each leg showing between bow and body
BODY_C = (EYE_X, EYE_Y, BODY_TOP - BODY[2] / 2)
LOCK_MASS = 0.22
CUT_Z = EYE_Z - 0.019                # the middle of the showing legs: where the jaws go
CUT_SEG = 0.014                      # the length of leg the jaws take (drawn gone once cut)
NEAR_LEG = (EYE_X - LEG_X, EYE_Y, CUT_Z)   # the leg toward the arms, the one the jaws can take

# -- the bolt cutter (lies shut on the tool case) --------------------------------------------------
# Its own frame: pivot at the origin, jaws forward along +x, lying flat. Body "cutter" carries the
# head's cheek plates, jaw A (on the +y side) and handle A (going back to -y: they cross at the
# pivot); body "handle_b" carries jaw B (-y side) and handle B (+y), on the pivot.
JAW_L = 0.047                        # pivot to jaw tip
JAW_CUT = 0.030                      # pivot to where the blades cut
JAW_W = 0.009                        # each jaw's width across (y)
JAW_T = 0.012                        # jaw thickness (z)
CHEEK = (0.072, 0.050, 0.006)        # cheek plates above and below the jaws' roots
CHEEK_X = -0.024                     # their centre: they end 12 mm ahead of the pivot
HANDLE_L = 0.260                     # pivot to the middle of the grip
HANDLE_R = 0.009                     # 18 mm handle tube
GRIP = (0.014, 0.110)                # rubber grip: radius, length
ALPHA0 = math.atan2(0.060, HANDLE_L)  # each handle's angle off the centreline, jaws shut
OPEN_DEG = 26.0                      # handles fully open: the blades 13.6 mm apart where they cut
REST_OPEN = 0.0                      # as it lies on the case: shut
CUTTER_MASS = 0.45                   # head, jaw A and handle A (0.75 kg in all: tubular handles)
HANDLE_B_MASS = 0.30
# it lies on the lid of a steel site tool case, its grips overhanging the case's near edge
CASE = (0.230, 0.190, 0.090)         # x deep, y wide, z tall
CASE_C = (0.515, -0.060)
CASE_TOP = TOP + CASE[2]
CASE_COL = (0.20, 0.33, 0.55)
# where each body balances (its own frame): the forged head and jaws are heavy, the handles hollow
COM_A = (-0.085, -0.018, 0.0)
COM_B = (-0.120, 0.028, 0.0)
LIE_Z = CASE_TOP + JAW_T / 2 + CHEEK[2]   # resting on its head's cheek plate and its handles
PIVOT0 = np.array([0.560, -0.040, LIE_Z])  # beside the lock, not under it

# a cut when the blades close to the leg's width on it (squeezing), square to it, held this long
CUT_GAP = 2 * LEG_R + 0.0005
CUT_HOLD_S = 0.25
SQUARE_DEG = 20.0                    # leg within this of square to the jaws
DRAG_MM = 10.0                       # the lock pulled or pushed more than this from where it hung: wrenched


def _grip_dir(sign):
    """Unit vector from the pivot to the middle of a handle's grip, jaws shut: sign -1 handle A, +1 B."""
    return np.array([-math.cos(ALPHA0), sign * math.sin(ALPHA0), 0.0])


def _handle_pieces(sign, jaw_sign):
    """Jaw (on the ``jaw_sign`` side) and the handle tube and grip (on the ``sign`` side)."""
    d = _grip_dir(sign)
    yaw = math.atan2(d[1], d[0])
    tube_len = HANDLE_L - GRIP[1] / 2 - 0.030
    tube_c = d * (0.030 + tube_len / 2)
    grip_c = d * HANDLE_L
    jaw = Piece("box", (JAW_L - 0.004, JAW_W, JAW_T), (0.004 + (JAW_L - 0.004) / 2, jaw_sign * JAW_W / 2, 0.0), color=JAW_COL)
    tip = Piece("box", (0.010, JAW_W * 0.7, JAW_T * 0.8), (JAW_L - 0.003, jaw_sign * JAW_W * 0.35, 0.0), color=JAW_COL,
                collide=False)
    # the arm from the jaw's root to the handle, inside the head
    root_len = 0.034
    root = Piece("box", (root_len, 0.014, JAW_T), tuple(float(v) for v in d * (root_len / 2)), rot=(0.0, 0.0, yaw), color=RED)
    tube = Piece("cylinder", (HANDLE_R, tube_len, HANDLE_R), tuple(float(v) for v in tube_c), rot=(0.0, 0.0, yaw - math.pi / 2),
                 axis="Y", color=RED)
    grip = Piece("cylinder", (GRIP[0], GRIP[1], GRIP[0]), tuple(float(v) for v in grip_c), rot=(0.0, 0.0, yaw - math.pi / 2),
                 axis="Y", color=GRIP_COL, friction=(1.5, 1.3))
    end = Piece("sphere", (GRIP[0], GRIP[0], GRIP[0]), tuple(float(v) for v in d * (HANDLE_L + GRIP[1] / 2)), color=GRIP_COL,
                collide=False)
    return [jaw, tip, root, tube, grip, end]


def cutter_object() -> Part:
    """Body A: the head's cheek plates, jaw A (+y) and handle A (-y), lying shut on the case."""
    pieces = _handle_pieces(-1, +1)
    for sz in (-1, 1):
        pieces.append(Piece("box", CHEEK, (CHEEK_X, 0.0, sz * (JAW_T + CHEEK[2]) / 2), color=RED))
    pieces.append(Piece("cylinder", (0.008, JAW_T + 2 * CHEEK[2] + 0.004, 0.008), (0.0, 0.0, 0.0), color=STEEL, collide=False))
    return Part("cutter", pos=tuple(float(v) for v in PIVOT0), size=(0.006, 0.004, 0.006), shape="cylinder", exact_z=True,
                color=STEEL, mass=CUTTER_MASS, yaw=math.radians(REST_OPEN / 2), material=None, friction=(0.6, 0.5),
                contact_offset=0.002, pieces=tuple(pieces), com=COM_A)


def handle_b_object() -> Part:
    """Body B: jaw B (-y) and handle B (+y), on the pivot; it opens up to OPEN_DEG off body A."""
    pieces = _handle_pieces(+1, -1)
    return Part("handle_b", pos=tuple(float(v) for v in PIVOT0 + np.array([0.0, 0.0, 0.0])), size=(0.005, 0.003, 0.005),
                shape="cylinder", exact_z=True, color=STEEL, mass=HANDLE_B_MASS, yaw=math.radians(-REST_OPEN / 2),
                friction=(0.6, 0.5), contact_offset=0.002, pieces=tuple(pieces), com=COM_B,
                hinge=Hinge(pivot=tuple(float(v) for v in PIVOT0), axis="Z", damping=0.0004, friction=0.02,
                            lower_deg=-OPEN_DEG, upper_deg=0.5, parent="cutter",
                            parent_pos=tuple(float(v) for v in PIVOT0)))


def cage_object() -> Part:
    """The cage wall: stile and post, mesh either side, the hasp; the staple's eye is drawn."""
    c = np.array([POST_X, WALL_Y + TUBE / 2, TOP + GATE_H / 2])

    def at(x, y, z):
        return tuple(float(v) for v in np.array([x, y, z]) - c)

    pieces = []
    pieces.append(Piece("box", (TUBE, TUBE, GATE_H), at(STILE_X, WALL_Y + TUBE / 2, TOP + GATE_H / 2), color=FRAME_COL))
    my = WALL_Y + TUBE / 2
    for x0, x1 in ((WALL_X[0], STILE_X - TUBE / 2), (POST_X + TUBE / 2, WALL_X[1])):
        for k in range(int((x1 - x0) / BAR_PITCH) + 1):
            x = x0 + 0.012 + k * BAR_PITCH
            if x < x1 - 0.005:
                pieces.append(Piece("cylinder", (BAR_R, GATE_H, BAR_R), at(x, my, TOP + GATE_H / 2), color=MESH_COL))
        for z in np.arange(TOP + 0.03, TOP + GATE_H, BAR_PITCH):
            pieces.append(Piece("cylinder", (BAR_R, x1 - x0, BAR_R), at((x0 + x1) / 2, my - 2 * BAR_R, z), axis="X", color=MESH_COL))
        for z in (TOP + 0.010, TOP + GATE_H - 0.010):
            pieces.append(Piece("box", (x1 - x0, 0.025, 0.020), at((x0 + x1) / 2, my, z), color=FRAME_COL))
    if WALL_X[0] < STILE_X - TUBE / 2:
        pieces.append(Piece("box", (0.025, 0.025, GATE_H), at(WALL_X[0] + 0.0125, my, TOP + GATE_H / 2), color=FRAME_COL))
    pieces.append(Piece("box", (0.025, 0.025, GATE_H), at(WALL_X[1] - 0.0125, my, TOP + GATE_H / 2), color=FRAME_COL))
    # foot rail the wall stands on (behind its face)
    pieces.append(Piece("box", (WALL_X[1] - WALL_X[0], 0.080, 0.006), at((WALL_X[0] + WALL_X[1]) / 2, WALL_Y + 0.040, TOP + 0.003),
                        color=FRAME_COL))
    # the hasp: the staple's base plate on the post, the flap from the stile over the staple
    pieces.append(Piece("box", (0.034, 0.004, 0.050), at(POST_X, WALL_Y - 0.002, EYE_Z), color=STEEL))
    pieces.append(Piece("box", (0.070, 0.003, 0.036), at((STILE_X + POST_X) / 2 - 0.006, WALL_Y - 0.0055, EYE_Z), color=STEEL))
    # the staple's eye, sticking out toward -y: drawn only (the lock hangs on its hinge)
    span = (WALL_Y - 0.007) - (EYE_Y - 0.006)
    for dz in (-1, 1):
        pieces.append(Piece("box", (0.008, span, 0.005), at(POST_X, EYE_Y - 0.006 + span / 2, EYE_Z + dz * (EYE_HOLE / 2 + 0.0025)),
                            color=STEEL, collide=False))
    pieces.append(Piece("box", (0.008, 0.005, EYE_HOLE + 0.010), at(POST_X, EYE_Y - 0.0085, EYE_Z), color=STEEL, collide=False))
    return Part("cage", pos=tuple(float(v) for v in c), size=(TUBE, TUBE, GATE_H), exact_z=True, color=FRAME_COL,
                friction=(0.6, 0.5), pieces=tuple(pieces))


def case_object() -> Part:
    """The steel site tool case the cutter lies on (fixed), its lid's handle and catches drawn."""
    c = (CASE_C[0], CASE_C[1], TOP + CASE[2] / 2)
    pieces = (Piece("box", (0.10, 0.018, 0.010), (0.0, 0.0, CASE[2] / 2 + 0.005), color=(0.1, 0.1, 0.1), collide=False),
              Piece("box", (0.004, 0.030, 0.020), (-CASE[0] / 2 - 0.002, -0.06, CASE[2] / 2 - 0.015), color=STEEL, collide=False),
              Piece("box", (0.004, 0.030, 0.020), (-CASE[0] / 2 - 0.002, 0.06, CASE[2] / 2 - 0.015), color=STEEL, collide=False),
              Piece("box", (CASE[0] + 0.004, CASE[1] + 0.004, 0.006), (0.0, 0.0, CASE[2] / 2 - 0.022), color=(0.15, 0.25, 0.42),
                    collide=False))
    return Part("case", pos=c, size=CASE, exact_z=True, color=CASE_COL, friction=(0.5, 0.4), pieces=pieces)


# leg pieces: per leg (y = -LEG_Y, +LEG_Y): below the cut zone, the cut zone, above it
def lock_object() -> Part:
    """The padlock: brass body, shackle through the staple's eye; it swings on the eye (a hinge)."""
    c = np.array(BODY_C)

    def at(x, y, z):
        return tuple(float(v) for v in np.array([x, y, z]) - c)

    pieces = []
    lo, hi = CUT_Z - CUT_SEG / 2, CUT_Z + CUT_SEG / 2
    for sx in (-1, 1):
        x = EYE_X + sx * LEG_X
        for z0, z1 in ((BODY_TOP - 0.004, lo), (lo, hi), (hi, LEG_TOP)):
            pieces.append(Piece("cylinder", (LEG_R, z1 - z0, LEG_R), at(x, EYE_Y, (z0 + z1) / 2), color=SHACKLE_COL))
    pieces.append(Piece("cylinder", (LEG_R, 2 * LEG_X, LEG_R), at(EYE_X, EYE_Y, LEG_TOP), axis="X", color=SHACKLE_COL))
    for sx in (-1, 1):
        pieces.append(Piece("sphere", (LEG_R, LEG_R, LEG_R), at(EYE_X + sx * LEG_X, EYE_Y, LEG_TOP), color=SHACKLE_COL))
    # keyway and a rust bloom on the face (drawn)
    face = EYE_Y - BODY[1] / 2
    pieces.append(Piece("box", (0.004, 0.001, 0.012), at(EYE_X, face - 0.0004, BODY_C[2] - 0.004),
                        color=(0.15, 0.12, 0.08), collide=False))
    pieces.append(Piece("box", (0.016, 0.0008, 0.010), at(EYE_X + 0.008, face - 0.0003, BODY_C[2] + 0.010),
                        color=(0.45, 0.25, 0.12), collide=False))
    return Part("padlock", pos=tuple(float(v) for v in c), size=BODY, exact_z=True, color=BRASS, mass=LOCK_MASS,
                friction=(0.5, 0.4), contact_offset=0.002, pieces=tuple(pieces),
                hinge=Hinge(pivot=(EYE_X, EYE_Y, EYE_Z), axis="X", damping=0.0003, lower_deg=-60.0, upper_deg=60.0))


CUT_PIECE = {-1: 1, +1: 4}           # index of each leg's cut-zone piece in lock_object().pieces


# -- measurements ------------------------------------------------------------------------------

def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def opening_deg(cutter, handle_b):
    """How far the handles are open (deg, 0 = jaws shut), from the two bodies' measured poses."""
    ra, rb = quat_matrix(cutter[3:7]), quat_matrix(handle_b[3:7])
    rel = ra.T @ rb
    return -math.degrees(math.atan2(rel[1, 0], rel[0, 0]))


def leg_axes(lock):
    """Each shackle leg's (point at the cut height, unit axis), from the lock's measured pose."""
    p = np.asarray(lock, float)
    r = quat_matrix(p[3:7])
    out = {}
    for sx in (-1, 1):
        local = np.array([EYE_X + sx * LEG_X, EYE_Y, CUT_Z]) - np.array(BODY_C)
        out[sx] = (p[:3] + r @ local, r @ np.array([0.0, 0.0, 1.0]))
    return out


def bite(cutter, handle_b, lock):
    """Which leg (if any) is between the blades, square to them, and the blades' gap on it (m)."""
    a = np.asarray(cutter, float)
    ra = quat_matrix(a[3:7])
    open_rad = math.radians(max(0.0, opening_deg(cutter, a if handle_b is None else handle_b)))
    best = None
    for sy, (pt, ax) in leg_axes(lock).items():
        n = ra @ np.array([0.0, 0.0, 1.0])                 # the jaws' plane normal
        square = math.degrees(math.acos(min(1.0, abs(float(np.dot(n, ax))))))
        # where the leg's axis crosses the jaws' plane, in body A's frame
        denom = float(np.dot(n, ax))
        if abs(denom) < 1e-6:
            continue
        s = float(np.dot(n, a[:3] - pt)) / denom
        q = ra.T @ (pt + s * ax - a[:3])
        x, y = float(q[0]), float(q[1])
        # jaw A's blade lies along +x at y = 0; jaw B's opens below it by the handles' opening angle
        lower = -x * math.tan(open_rad)
        inside = 0.016 < x < JAW_L - 0.004 and lower - 0.001 < y < 0.001 and abs(s) < 0.020
        gap = x * math.tan(open_rad)
        if inside and square < SQUARE_DEG and (best is None or gap < best[2]):
            best = (sy, square, gap)
    return best


def grip_points(cutter, handle_b):
    """World positions of the middle of each rubber grip (A, B)."""
    out = []
    for pose, sign in ((cutter, -1), (handle_b, +1)):
        p = np.asarray(pose, float)
        out.append(p[:3] + quat_matrix(p[3:7]) @ (_grip_dir(sign) * HANDLE_L))
    return out


def grip_yaw(sign, open_deg=OPEN_DEG, a_yaw_deg=None):
    """World yaw of handle A (sign -1) or B (+1) when symmetric and ``open_deg`` open."""
    a = math.radians(open_deg / 2 if a_yaw_deg is None else a_yaw_deg)
    d = _grip_dir(sign)
    base = math.atan2(d[1], d[0])
    return base + (a if sign < 0 else a - math.radians(open_deg))
