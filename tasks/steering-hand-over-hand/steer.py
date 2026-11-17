"""Geometry and parts for turning a self-centring steering wheel to full lock, hand over hand (real size).

A 380 mm (15 in) steering wheel -- the size fitted to delivery vans and pickups (Ford Transit, Ram ProMaster) and
sold as a 380 mm aftermarket wheel -- sits on a steering column rising 25 deg toward the driver (wheel plane 65 deg
from horizontal), as in a van cab, mounted on a bench test stand in front of a dash panel with its instrument
cluster. A torsion spring in the column pulls the wheel back to straight ahead, standing in for the self-centring
a moving vehicle gives: about 1.6 N m at full lock. Full lock is 1.25 turns (450 deg) to the left of centre. The job: turn it all the way left, hand over hand
-- one hand holds the rim against the spring while the other takes a new grip further round -- and hold it at
the lock; if both hands let go the spring spins it back.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.

Real parts:
  wheel    380 mm (15 in) steering wheel, three spokes (3, 6 and 9 o'clock), 28 mm leather-wrapped rim, centre
           horn pad; about 1.4 kg (any 380 mm aftermarket wheel on a 6-bolt hub adapter).
  column   steering column on a test stand, 25 deg rake; a torsion (clock) spring between column and shaft
           gives the centring torque, rising to about 1.6 N m at 450 deg; mechanical stops at +/-450 deg.
  dash     a dash panel with a two-dial instrument cluster behind the wheel (recognition only, nothing moves).
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

import meshlib as M
from simkit import Hinge, Part, Piece

TOP = 0.7475
_HERE = Path(__file__).resolve().parent

# -- the wheel ----------------------------------------------------------------------------------------
D = 0.380                       # outside diameter (15 in)
R_TUBE = 0.014                  # 28 mm rim section
R_C = D / 2 - R_TUBE            # rim centreline radius
TILT = math.radians(65.0)       # wheel plane from horizontal (column 25 deg up toward the driver)
C = np.array([0.30, 0.0, TOP + 0.20])            # wheel centre
N = np.array([-math.sin(TILT), 0.0, math.cos(TILT)])   # wheel axis, toward the driver (the arms) and up
U = np.array([math.cos(TILT), 0.0, math.sin(TILT)])    # 12 o'clock direction in the wheel plane
V = np.array([0.0, 1.0, 0.0])                          # 9 o'clock (the driver's left)
WF = np.stack([U, V, N], axis=1)                       # wheel frame: local x = 12 o'clock, y = 9 o'clock, z = axis
DISH = 0.035                    # the hub sits this far behind the rim plane (dished wheel)
WHEEL_MASS = 1.4

# -- centring spring and stops --------------------------------------------------------------------------
LOCK_DEG = 450.0                # full lock, left of centre
K_SPRING = 1.6 / math.radians(LOCK_DEG)       # N m / rad: about 1.6 N m at full lock
DAMP = 0.06                     # N m s / rad
K_STOP = 3.0                    # N m / rad past the lock: the steering gear's end stop (stiff spring)
C_STOP = 0.3                    # N m s / rad at the stop
FRICTION = 0.02                 # column bearing friction, N m

BLACK = (0.045, 0.045, 0.05)
LEATHER = (0.07, 0.065, 0.06)
STITCH = (0.32, 0.30, 0.28)
SPOKE = (0.11, 0.11, 0.12)
SILVER = (0.72, 0.73, 0.75)
DASH = (0.16, 0.16, 0.17)
DASH_DK = (0.09, 0.09, 0.10)
DIAL = (0.93, 0.93, 0.90)
NEEDLE = (0.85, 0.25, 0.12)
STAND = (0.30, 0.31, 0.33)
TAPE = (0.98, 0.80, 0.04)       # yellow vinyl marking tape (the centre mark wheel-alignment shops put on the rim)
TAPE_W = 0.025                  # 25 mm band


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _colliders(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def rpy(R):
    """(roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll), the order simkit's pieces use."""
    R = np.asarray(R, float)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return (roll, pitch, yaw)


def frame_z(z, x_hint):
    """A rotation whose z column is z and whose x column is x_hint made square to it."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - np.dot(x_hint, z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def rim_point(phi_deg, r=R_C):
    """A point on the rim centreline, phi from 12 o'clock toward 9 o'clock (the driver's left), wheel at centre."""
    p = math.radians(phi_deg)
    return C + r * (math.cos(p) * U + math.sin(p) * V)


def _quat_wxyz_x_to(n):
    """(w, x, y, z) of the turn about y that takes the x axis onto n (n in the x-z plane)."""
    beta = math.atan2(-n[2], n[0])
    return (math.cos(beta / 2), 0.0, math.sin(beta / 2), 0.0)


def wheel_vmeshes():
    """The wheel's visual meshes in its body frame (origin at the wheel centre, world-aligned)."""
    V_ = []
    # rim: a torus, its section slightly oval like a moulded grip
    prof = [(R_C + R_TUBE * math.cos(a), 1.08 * R_TUBE * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
    prof.append(prof[0])
    rim = M.revolve(prof, n=96, axis="Z")
    V_.append(_vm("rim", rim, LEATHER, "satin", R=WF))
    # stitching line round the inside of the rim
    st = M.revolve([(R_C - R_TUBE * 0.98, -0.0015), (R_C - R_TUBE * 1.005, 0.0), (R_C - R_TUBE * 0.98, 0.0015)], n=96, axis="Z")
    V_.append(_vm("stitch", st, STITCH, "matte", R=WF))
    # perforated-leather grip zones at the sides (a shade lighter, matte) and moulded thumb grips at 3 and 9
    def arc(rt, a0, a1, sq=1.08):
        pr = [(R_C + rt * math.cos(a), sq * rt * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 20, endpoint=False)]
        pr.append(pr[0])
        return M.revolve(pr, n=40, axis="Z", a0=math.radians(a0), a1=math.radians(a1))
    for k, (a0, a1) in enumerate(((40.0, 140.0), (220.0, 320.0))):
        V_.append(_vm(f"grip{k}", arc(R_TUBE * 1.012, a0, a1), (0.10, 0.095, 0.09), "matte", R=WF))
        for j, (b0, b1) in enumerate(((a0, a0 + 1.5), (a1 - 1.5, a1))):        # stitched seams at each zone's ends
            V_.append(_vm(f"seam{k}{j}", arc(R_TUBE * 1.03, b0, b1), STITCH, "matte", R=WF))
    for k, a0 in enumerate((80.0, 260.0)):
        V_.append(_vm(f"thumb{k}", arc(R_TUBE * 1.14, a0, a0 + 20.0, sq=1.0), LEATHER, "satin", R=WF))
    # centre mark: a 25 mm band of yellow marking tape wrapped round the rim at 12 o'clock when the wheel is
    # straight, as wheel-alignment shops and steering test rigs mark it (visual only: the rim's colliders and
    # mass are unchanged); read against the white index line on the column shroud
    half = math.degrees(TAPE_W / 2 / R_C)
    V_.append(_vm("tape", arc(R_TUBE * 1.045, -half, half), TAPE, "matte", R=WF))
    # hub, dished behind the rim plane
    hub = M.revolve(M.fillet_profile(0.0, 0.062, -DISH - 0.030, -DISH + 0.012, f=0.006), n=48, axis="Z")
    V_.append(_vm("hub", hub, SPOKE, "satin", R=WF))
    # horn pad: a rounded pad on the hub with a plain silver badge (no maker's mark)
    pad = M.loft([(-DISH + 0.008, 0.120, 0.105, 5.0), (-DISH + 0.028, 0.114, 0.099, 5.0), (-DISH + 0.034, 0.100, 0.086, 4.0)],
                 axis="Z", n=48)
    V_.append(_vm("horn", pad, BLACK, "satin", R=WF))
    badge = M.revolve(M.fillet_profile(0.0, 0.016, -DISH + 0.033, -DISH + 0.037, f=0.001), n=32, axis="Z")
    V_.append(_vm("badge", badge, SILVER, "aluminium", R=WF))
    # three spokes: 3, 6 and 9 o'clock, wide and flat, rising from the hub to the rim
    for phi in (90.0, -90.0, 180.0):
        a = math.radians(phi)
        d = np.array([math.cos(a), math.sin(a), 0.0])
        w = 0.070 if phi != 180.0 else 0.055
        secs = []
        for k, r in enumerate(np.linspace(0.050, R_C - 0.004, 6)):
            z = -DISH + (DISH - 0.004) * (r - 0.050) / (R_C - 0.054)
            secs.append((r, w - 0.020 * (r - 0.05) / (R_C - 0.05), 0.016, 4.0, 0.0, z))
        sp = M.loft(secs, axis="X", n=32)          # station along x, section (y, z)
        p, t = sp
        p = np.stack([p[:, 0], p[:, 1], p[:, 2]], axis=1)
        Rl = np.array([[d[0], -d[1], 0.0], [d[1], d[0], 0.0], [0.0, 0.0, 1.0]])   # turn x onto the spoke
        V_.append(_vm(f"spoke{int(phi)}", (p @ Rl.T, t), SPOKE, "satin", R=WF))
    return tuple(V_)


def wheel_colliders():
    """Rim as 32 capsules round its centreline, plus the hub and spokes (so a hand can't pass through)."""
    P = []
    n = 32
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        p0 = R_C * np.array([math.cos(a0), math.sin(a0), 0.0])
        p1 = R_C * np.array([math.cos(a1), math.sin(a1), 0.0])
        mid = WF @ ((p0 + p1) / 2)
        tang = WF @ (p1 - p0)
        Rk = frame_z(tang, WF @ ((p0 + p1) / 2))
        P.append(Piece("capsule", (R_TUBE, float(np.linalg.norm(p1 - p0)), R_TUBE), tuple(map(float, mid)), rot=rpy(Rk), axis="Z"))
    P.append(Piece("cylinder", (0.062, 0.040, 0.062), tuple(map(float, WF @ np.array([0, 0, -DISH - 0.008]))), rot=rpy(WF), axis="Z"))
    for phi in (90.0, -90.0, 180.0):
        a = math.radians(phi)
        mid = WF @ np.array([math.cos(a) * (R_C + 0.05) / 2, math.sin(a) * (R_C + 0.05) / 2, -DISH / 2])
        Rk = frame_z(WF @ np.array([0, 0, 1.0]), WF @ np.array([math.cos(a), math.sin(a), 0.0]))
        P.append(Piece("box", (R_C - 0.06, 0.050, 0.014), tuple(map(float, mid)), rot=rpy(Rk)))
    return _colliders(P)


def wheel_object() -> Part:
    # The joint is a free axle carrying the damping (USD angular drives take damping per degree/s) and bearing
    # friction. PhysX revolute limits and drive targets both wrap at +/-180 deg (checked: limits set to -355/+180
    # stopped the wheel at -180), so neither can hold a 1.25-turn spring or a 450 deg stop: the centring spring and
    # the end stops are applied by the task's Dynamics as torques from the measured turn, unwrapped across turns.
    joint = Hinge(pivot=tuple(map(float, C)), axis="X", damping=DAMP * math.pi / 180.0, friction=FRICTION,
                  lower_deg=-1e6, upper_deg=1e6, stiffness=0.0, target_deg=0.0, frame_rot=_quat_wxyz_x_to(N))
    return Part("wheel", pos=tuple(map(float, C)), size=(0.02, 0.02, 0.02), shape="sphere", exact_z=True, mass=WHEEL_MASS,
                color=LEATHER, friction=(0.9, 0.8), restitution=0.0, contact_offset=0.003, pieces=wheel_colliders(),
                hide_root=True, vmeshes=wheel_vmeshes(), hinge=joint, solver_iters=(32, 4))


# -- column, stand and dash (fixed) -----------------------------------------------------------------------
COL_LEN = 0.24                  # column shroud from behind the hub, down and away along the axis
COL_END = C - N * (DISH + 0.035 + COL_LEN)


def fixture_object() -> Part:
    """Column shroud, its test stand and a dash panel with an instrument cluster: one fixed body."""
    V_ = []
    P = []
    origin = np.array([COL_END[0], 0.0, TOP])            # the stand's foot on the bench
    rel = lambda p: np.asarray(p, float) - origin
    # column shroud: a tapered round-cornered housing along the axis
    a0 = C - N * (DISH + 0.035)
    shroud = M.loft([(0.0, 0.105, 0.115, 4.0), (0.02, 0.120, 0.130, 4.0), (COL_LEN * 0.4, 0.125, 0.120, 4.0), (COL_LEN, 0.095, 0.095, 3.0)], axis="Z", n=40)
    Rs = frame_z(-N, V)
    V_.append(_vm("shroud", shroud, DASH_DK, "satin", R=Rs, offset=rel(a0)))
    P.append(Piece("box", (0.100, 0.110, COL_LEN), tuple(map(float, rel(a0 - N * COL_LEN / 2))), rot=rpy(Rs)))
    # index mark: a white painted line along the top of the shroud (12 o'clock), the fixed reference the rim's
    # tape is read against
    V_.append(_vm("index", M.rbox((0.008, 0.004, 0.050), e=4.0), (0.95, 0.95, 0.93), "matte", R=Rs,
                  offset=rel(a0 - N * 0.050 + U * 0.065)))
    for k, sgn in enumerate((1, -1)):                                      # indicator and wiper stalks
        stalk = M.loft([(0.0, 0.016, 0.012, 3.0), (0.075, 0.012, 0.010, 3.0)], axis="Z", n=16)
        V_.append(_vm(f"stalk{k}", stalk, DASH_DK, "satin", R=frame_z(V * sgn - N * 0.25, U), offset=rel(a0 - N * 0.030 + V * sgn * 0.050)))
    # stand: a steel post and base plate from the bench to the column's far end
    post_h = COL_END[2] - TOP
    V_.append(_vm("post", M.rbox((0.090, 0.090, post_h + 0.03), e=8.0), STAND, "steel_painted", offset=(0.0, 0.0, (post_h + 0.03) / 2)))
    V_.append(_vm("plate", M.rbox((0.26, 0.30, 0.014), e=10.0), STAND, "steel_painted", offset=(0.0, 0.0, 0.007)))
    for gx, gy in ((1, 0), (-1, 0), (0, 1), (0, -1)):                      # gussets round the post's foot
        g = M.rbox((0.070 if gx else 0.008, 0.008 if gx else 0.070, 0.060), e=4.0)
        V_.append(_vm(f"gusset{gx}{gy}", g, STAND, "steel_painted", offset=(gx * 0.080, gy * 0.080, 0.044)))
    for bx in (-0.105, 0.105):                                             # bench bolts through the base plate
        for by in (-0.125, 0.125):
            V_.append(_vm(f"bolt{bx}{by}", M.revolve(M.fillet_profile(0.0, 0.010, 0.0, 0.007, f=0.0015), n=6), SILVER, "steel",
                          offset=(bx, by, 0.014)))
    # clamp block joining the post's top to the column
    V_.append(_vm("clamp", M.rbox((0.12, 0.12, 0.07), e=6.0), STAND, "steel_painted", R=frame_z(-N, V), offset=rel(COL_END + N * 0.03)))
    P.append(Piece("box", (0.090, 0.090, post_h), (0.0, 0.0, post_h / 2)))
    P.append(Piece("box", (0.26, 0.30, 0.014), (0.0, 0.0, 0.007)))
    # dash panel behind the wheel, square to the axis, with a two-dial cluster seen through the wheel's upper half
    dash_c = C - N * 0.200 + U * 0.075
    Rd = frame_z(N, U)          # panel's z toward the driver, x up the panel
    # instrument binnacle: a moulded cluster housing behind the wheel's upper half, its top hooded over the dials
    body = M.loft([(0.022, 0.150, 0.340, 5.0, 0.0, 0.0), (-0.010, 0.165, 0.360, 5.0, 0.0, 0.0),
                   (-0.080, 0.130, 0.330, 4.0, -0.020, 0.0), (-0.130, 0.080, 0.260, 3.0, -0.045, 0.0)], axis="Z", n=64)
    V_.append(_vm("dash", body, DASH, "satin", R=Rd, offset=rel(dash_c + U * 0.040)))
    brow = M.loft([(0.0, 0.026, 0.350, 3.0, 0.0, 0.0), (0.050, 0.020, 0.320, 3.0, -0.006, 0.0)], axis="Z", n=40)
    V_.append(_vm("hood", brow, DASH_DK, "satin", R=Rd, offset=rel(dash_c + U * 0.118 + N * 0.020)))
    hazard = M.rbox((0.018, 0.030, 0.010), e=6.0)
    V_.append(_vm("hazard", hazard, (0.80, 0.10, 0.08), "satin", R=Rd, offset=rel(dash_c - U * 0.012 + N * 0.026)))
    P.append(Piece("box", (0.15, 0.34, 0.040), tuple(map(float, rel(dash_c + U * 0.040))), rot=rpy(Rd)))
    for k, yy in enumerate((0.065, -0.065)):
        cc = dash_c + U * 0.040 + V * yy + N * 0.020
        V_.append(_vm(f"bezel{k}", M.revolve(M.fillet_profile(0.050, 0.060, 0.0, 0.022, f=0.003), n=48), SILVER, "aluminium", R=Rd, offset=rel(cc)))
        V_.append(_vm(f"dial{k}", M.revolve([(0.0, 0.002), (0.051, 0.002)], n=48, cap0=False), DIAL, "matte", R=Rd, offset=rel(cc)))
        for j in range(9):                                   # tick marks
            ang = math.radians(225 - j * 33.75)
            tk = M.rbox((0.010, 0.0025, 0.002), e=4.0)
            Rt = Rd @ np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1.0]])
            V_.append(_vm(f"tick{k}{j}", tk, BLACK, "matte", R=Rt, offset=rel(cc + Rd @ np.array([0.040 * math.cos(ang), 0.040 * math.sin(ang), 0.0035]))))
        ang = math.radians(200 if k == 0 else 150)
        nd = M.rbox((0.034, 0.003, 0.002), e=4.0)
        Rn = Rd @ np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1.0]])
        V_.append(_vm(f"needle{k}", nd, NEEDLE, "matte", R=Rn, offset=rel(cc + Rd @ np.array([0.016 * math.cos(ang), 0.016 * math.sin(ang), 0.0045]))))
    return Part("fixture", pos=tuple(map(float, origin)), size=(0.02, 0.02, 0.02), shape="sphere", exact_z=True,
                color=STAND, pieces=_colliders(P), hide_root=True, vmeshes=tuple(V_))


def scene_objects():
    return (fixture_object(), wheel_object())


def mat(p):
    """3x3 rotation of a root pose (x, y, z, qx, qy, qz, qw)."""
    x, y, z, w = (float(v) for v in p[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wheel_angle_wrapped(pose):
    """The wheel's turn (deg, -180..180) from straight ahead, positive to the left (counter-clockwise as the
    driver sees it): the angle of its 12 o'clock line about the axis."""
    u = mat(pose) @ U
    return math.degrees(math.atan2(float(np.dot(u, V)), float(np.dot(u, U))))
