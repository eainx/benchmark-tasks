"""Geometry and parts for stacking one full bail-arm tote on another (real size).

Two 400 x 300 mm stack-nest bale-arm crates (12 L, 131 mm tall, blue polypropylene, tapered walls so empty
crates nest about 3/4 deep) stand on the bench, long sides toward the arms. Each crate has a zinc-plated
steel wire bail (U-shaped, 6 mm wire) hinged at each short end. With both bails swung out (hanging outside
the end walls) an empty crate nests in the one below; with both swung in (lying across the rim, in the
rim's notches) the crate above stands on them. Tote A, at the back, is full: six kraft cartons fill it to
35 mm under the rim and its bails hang out. Tote B, in front, is full of six light cartons. The job: swing
A's two bails in, one per hand, then lift B with both hands by its hand holes and lower it level onto them.
If a bail is left out, B's base drops past A's rim onto A's cartons and crushes them.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.

Real parts:
  totes    400 x 300 x 131 mm stack-nest bale-arm crate, 12 L (internal 364 x 264 x 121 mm), HDPE/PP,
           e.g. Engels 12 L bale arm crate 400 x 300 x 131 mm; tapered walls (base 330 x 230 mm) under a
           rim lip, a hand hole in each end wall, label holder on a long side; about 0.6 kg empty.
  bails    zinc-plated steel wire bail, 6 mm wire, 312 mm across, legs 70 mm from pivot to bar, about 0.11 kg,
           pivoting on bosses at the ends of the long walls.
  cartons  kraft corrugated cartons 100 x 100 mm, six per crate (A: 90 mm tall, 0.5 kg each;
           B: 70 mm tall, 0.06 kg each).
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace

import numpy as np

import meshlib as M
from simkit import Hinge, Part, Piece

TOP = 0.7475

BLUE = (0.09, 0.27, 0.58)
BLUE_DK = (0.06, 0.19, 0.42)
ZINC = (0.90, 0.91, 0.93)
KRAFT = (0.66, 0.50, 0.33)
TAPE = (0.74, 0.62, 0.44)
LABEL = (0.95, 0.95, 0.93)
INK = (0.12, 0.12, 0.13)

# -- crate (both the same model) --------------------------------------------------------------------
L_TOP, W_TOP = 0.400, 0.300        # external at the rim: y (long) by x (short)
L_BOT, W_BOT = 0.330, 0.230        # external at the base (the walls taper so an empty crate nests)
H = 0.131
WALL = 0.003
RIM_W = 0.018                      # the hollow rim, external to the 364 x 264 opening
RIM_H = 0.014
SLOPE = (0.370 - L_BOT) / 2 / (H - RIM_H)   # the walls lean out this much per metre up (about 9.7 degrees) to
                                   # 370 x 270 under the rim lip, which stands out to 400 x 300
HOLE_W, HOLE_Z0, HOLE_Z1 = 0.120, H - 0.055, H - 0.022    # hand hole in each end wall
CRATE_MASS = 0.60

A_C = (0.480, 0.000)               # tote A (back, full, bails out): its middle
B_C = (0.110, 0.000)               # tote B (front, full): its middle
A_MASS = CRATE_MASS + 6 * 0.50
B_MASS = CRATE_MASS + 6 * 0.06

# -- bails ------------------------------------------------------------------------------------------
R_WIRE = 0.003
X_LEG = W_TOP / 2 + 0.006          # the legs run just outside the long walls' rim
Y_PIV = L_TOP / 2                  # pivot bosses on the long walls at the ends' corners
Z_PIV = H - 0.008
L_LEG = 0.070                      # pivot to bar centre
PHI_OUT = math.radians(-90.0)      # spawn: hanging straight down outside the end wall, below the hand hole
PHI_IN = math.radians(171.0)       # swung in: the bar lies across the rim, in the notches
BAIL_MASS = 0.11
Y_IN = Y_PIV - L_LEG * math.cos(math.pi - PHI_IN)       # the bar's middle when swung in (from the crate's middle)

# -- cartons ----------------------------------------------------------------------------------------
CARTON = (0.100, 0.100, 0.090)
CARTON_B = (0.100, 0.100, 0.070)    # tote B's lighter, lower cartons (they stay under its hand holes)


def mat(pose):
    """3x3 rotation of a root pose (x, y, z, qx, qy, qz, qw)."""
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _colliders(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def half_l(z):
    """External half-length (y) of the crate at height z above its underside."""
    return L_BOT / 2 + SLOPE * z


def half_w(z):
    return W_BOT / 2 + SLOPE * z


# ------------------------------------------------------------------------------------------ crate
def crate_vmeshes(z0, seed):
    """The crate's visual meshes, its underside at z0 in the body frame."""
    rnd = np.random.default_rng(seed)
    V = []
    # outer shell, lofted with slightly rounded corners, up to the rim band
    shell = M.loft([(z0 + 0.006, W_BOT - 0.006, L_BOT - 0.006, 14.0), (z0 + 0.012, W_BOT, L_BOT, 14.0),
                    (z0 + H - RIM_H, 2 * half_w(H - RIM_H), 2 * half_l(H - RIM_H), 14.0)], axis="Z", n=128,
                   cap0=True, cap1=False)
    p, t = shell
    V.append(("shell", p + rnd.normal(0, 0.0001, p.shape), t, BLUE, "satin", 1.0))
    # the rim band: a deep hollow lip, rolled over, flat on top (where the bails lie)
    lip = M.loft([(z0 + H - RIM_H, 2 * half_w(H - RIM_H) + 0.001, 2 * half_l(H - RIM_H) + 0.001, 16.0),
                  (z0 + H - RIM_H + 0.002, W_TOP, L_TOP, 16.0), (z0 + H - 0.0015, W_TOP, L_TOP, 16.0),
                  (z0 + H, W_TOP - 0.003, L_TOP - 0.003, 16.0), (z0 + H, W_TOP - 2 * RIM_W + 0.002, L_TOP - 2 * RIM_W + 0.002, 16.0),
                  (z0 + H - 0.002, W_TOP - 2 * RIM_W, L_TOP - 2 * RIM_W, 16.0)], axis="Z", n=128, cap0=False, cap1=False)
    V.append(_vm("rim", lip, BLUE, "satin"))
    # inside: walls and floor, a shade darker
    inner = M.loft([(z0 + 0.004, W_BOT - 2 * WALL - 0.004, L_BOT - 2 * WALL - 0.004, 14.0),
                    (z0 + H - 0.002, W_TOP - 2 * RIM_W, L_TOP - 2 * RIM_W, 16.0)], axis="Z", n=128, cap0=True, cap1=False)
    p, t = inner
    V.append(("inside", p, t[:, ::-1], BLUE_DK, "satin", 1.0))
    # moulded vertical ribs on the long walls, and on the ends either side of the hand hole
    for sx in (-1, 1):
        for k, yy in enumerate(np.linspace(-0.150, 0.150, 6)):
            zc = 0.012 + (H - RIM_H - 0.012) / 2
            rib = M.rbox((0.004, 0.012, H - RIM_H - 0.014), e=6, axis="Y")
            xo = sx * (half_w(zc) + 0.0015)
            V.append(_vm(f"rib{sx}{k}", rib, BLUE, "satin", offset=(xo, yy, z0 + zc), R=_ry(-sx * math.atan(SLOPE))))
    for sy in (-1, 1):
        for k, xx in enumerate((-0.110, -0.075, 0.075, 0.110)):
            zc = 0.012 + (H - RIM_H - 0.012) / 2
            rib = M.rbox((0.012, 0.004, H - RIM_H - 0.014), e=6, axis="X")
            yo = sy * (half_l(zc) + 0.0015)
            V.append(_vm(f"erib{sy}{k}", rib, BLUE, "satin", offset=(xx, yo, z0 + zc), R=_rx(sy * math.atan(SLOPE))))
        # hand hole: a dark through-slot with rounded ends, a thick moulded grip bar over it
        zc = (HOLE_Z0 + HOLE_Z1) / 2
        yo = sy * (half_l(zc) + 0.0006)
        V.append(_vm(f"hole{sy}", M.rbox((HOLE_W, 0.0025, HOLE_Z1 - HOLE_Z0), e=2.6, axis="Y"), (0.02, 0.03, 0.05), "matte",
                     offset=(0.0, yo, z0 + zc), R=_rx(sy * math.atan(SLOPE))))
        V.append(_vm(f"grip{sy}", M.rbox((HOLE_W + 0.03, 0.010, 0.006), e=3.0, axis="Y"), BLUE, "satin",
                     offset=(0.0, sy * (half_l(HOLE_Z1) - 0.004), z0 + HOLE_Z1 + 0.003)))
    # feet: a moulded stacking foot ring under the base
    V.append(_vm("foot", M.loft([(z0, W_BOT - 0.016, L_BOT - 0.016, 12.0), (z0 + 0.006, W_BOT - 0.008, L_BOT - 0.008, 12.0)],
                                 axis="Z", n=96, cap0=True, cap1=False), BLUE_DK, "satin"))
    # pivot bosses for the bails on the long walls, 25 mm in from each end
    for sx in (-1, 1):
        for sy in (-1, 1):
            boss = M.revolve([(0.0, 0.0), (0.008, 0.0), (0.008, X_LEG - half_w(Z_PIV) + 0.002), (0.0, X_LEG - half_w(Z_PIV) + 0.002)],
                             n=24, axis="X")
            R_b = None if sx > 0 else np.diag([-1.0, 1.0, -1.0])
            V.append(_vm(f"boss{sx}{sy}", boss, BLUE, "satin", offset=(sx * (half_w(Z_PIV) - 0.002), sy * Y_PIV, z0 + Z_PIV), R=R_b))
            # the rim's notch pair either side of where the bar lies when swung in
            for d in (-1, 1):
                V.append(_vm(f"notch{sx}{sy}{d}", M.rbox((0.012, 0.004, 0.004), e=4), BLUE, "satin",
                             offset=(sx * (W_TOP / 2 - RIM_W / 2), sy * (Y_IN + d * 0.0075), z0 + H + 0.002)))
    # label holder on the near (-x) long wall: a clear clip-on frame with a printed card
    zc = 0.070
    xw = -(half_w(zc) + 0.004)
    R = _ry(math.atan(SLOPE))
    V.append(_vm("label_frame", M.rbox((0.002, 0.104, 0.064), e=8, axis="X"), (0.85, 0.88, 0.90), "gloss", 0.55,
                 offset=(xw - 0.0012, 0.075, z0 + zc), R=R))
    V.append(_vm("label_card", M.rbox((0.0008, 0.096, 0.056), e=10, axis="X"), LABEL, "matte", offset=(xw, 0.075, z0 + zc), R=R))
    for k, (dy, w) in enumerate(((0.0, 0.070), (0.0, 0.050), (-0.012, 0.045))):
        V.append(_vm(f"label_line{k}", M.rbox((0.0006, w, 0.006 if k == 0 else 0.003), e=6, axis="X"), INK, "matte",
                     offset=(xw - 0.0006, 0.075 + dy, z0 + zc + 0.016 - 0.013 * k), R=R))
    return V


def crate_pieces(z0):
    """Colliders of the crate shell (floor, four leaning walls, the rim band, the end walls round the
    hand holes, the rim notches), its underside at z0 in the body frame."""
    P = [Piece("box", (W_BOT - 0.01, L_BOT - 0.01, 0.006), (0.0, 0.0, z0 + 0.003))]
    tilt = math.atan(SLOPE)
    hgt = (H - RIM_H) / math.cos(tilt)
    zc = z0 + (H - RIM_H) / 2
    for s in (-1, 1):
        # long walls, the full height under the rim band
        P.append(Piece("box", (0.006, L_BOT, hgt), (s * (half_w((H - RIM_H) / 2) - 0.003), 0.0, zc), rot=(0.0, s * tilt, 0.0)))
        # end walls: under the hand hole, either side of it, and the grip bar above it (part of the rim band)
        h1 = HOLE_Z0 / math.cos(tilt)
        P.append(Piece("box", (W_BOT, 0.006, h1), (0.0, s * (half_l(HOLE_Z0 / 2) - 0.003), z0 + HOLE_Z0 / 2), rot=(-s * tilt, 0.0, 0.0)))
        zh = (HOLE_Z0 + H - RIM_H) / 2
        for sx in (-1, 1):
            wside = (half_w(zh) - HOLE_W / 2)
            P.append(Piece("box", (wside, 0.006, H - RIM_H - HOLE_Z0), (sx * (HOLE_W / 2 + wside / 2), s * (half_l(zh) - 0.003), z0 + zh)))
        # rim band: long sides and ends (the ends' band is the hand hole's grip bar, 22 mm over its top)
        P.append(Piece("box", (RIM_W, L_TOP, RIM_H), (s * (W_TOP / 2 - RIM_W / 2), 0.0, z0 + H - RIM_H / 2)))
        P.append(Piece("box", (W_TOP - 2 * RIM_W, RIM_W, H - HOLE_Z1), (0.0, s * (L_TOP / 2 - RIM_W / 2), z0 + (H + HOLE_Z1) / 2)))
    return P


def notch_pieces(z0):
    """The rim's notch stops (real moulded bumps) that keep a swung-in bar from sliding along the rim."""
    P = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for d in (-1, 1):
                P.append(Piece("box", (0.012, 0.004, 0.004), (sx * (W_TOP / 2 - RIM_W / 2), sy * (Y_IN + d * 0.0075), z0 + H + 0.002)))
    return P


def _cartons(z0, tag, mass_label, box=CARTON):
    """Six kraft cartons standing two by three in the crate (visuals and colliders): top flaps meeting in a
    seam along the middle, brown packing tape over the seam and down both ends, a printed shipping label
    with a barcode on some of them."""
    V, P = [], []
    SEAM = (0.36, 0.26, 0.15)
    rnd = np.random.default_rng(7 if tag == "a" else 8)
    for i, xx in enumerate((-0.0505, 0.0505)):
        for j, yy in enumerate((-0.1015, 0.0, 0.1015)):
            c = np.array([xx, yy, z0 + 0.006 + box[2] / 2])
            k = f"{i}{j}"
            shade = tuple(float(v) * (0.94 + 0.10 * rnd.random()) for v in KRAFT)
            V.append(_vm(f"{tag}carton{k}", M.rbox(box, e=14), shade, "matte", offset=c))
            ztop = c[2] + box[2] / 2
            # the two top flaps: a dark seam where they meet (along y), slight gaps at the flaps' ends
            V.append(_vm(f"{tag}seam{k}", M.rbox((0.0016, box[1] - 0.004, 0.0004), e=6), SEAM, "matte", offset=(c[0], c[1], ztop + 0.0001)))
            for sy in (-1, 1):
                V.append(_vm(f"{tag}flap{k}{sy}", M.rbox((box[0] - 0.006, 0.0012, 0.0004), e=6), SEAM, "matte",
                             offset=(c[0], c[1] + sy * (box[1] / 2 - 0.0012), ztop + 0.0001)))
            # tape over the seam and 50 mm down each end, slightly glossy
            V.append(_vm(f"{tag}tape{k}", M.rbox((0.048, box[1] + 0.0012, 0.0006), e=8), TAPE, "satin", offset=(c[0], c[1], ztop + 0.0004)))
            for sy in (-1, 1):
                V.append(_vm(f"{tag}tapeend{k}{sy}", M.rbox((0.048, 0.0006, 0.050), e=8, axis="Y"), TAPE, "satin",
                             offset=(c[0], c[1] + sy * (box[1] / 2 + 0.0004), ztop - 0.025)))
            # a shipping label with a barcode on three of the six, on the top beside the tape
            if (i + j) % 2 == 0:
                lx = c[0] + (0.028 if i == 0 else -0.028)
                V.append(_vm(f"{tag}lbl{k}", M.rbox((0.030, 0.070, 0.0006), e=10), LABEL, "matte", offset=(lx, c[1], ztop + 0.0004)))
                for n in range(3):
                    V.append(_vm(f"{tag}txt{k}{n}", M.rbox((0.0025, 0.040 - 0.008 * n, 0.0003), e=6), INK, "matte",
                                 offset=(lx - 0.010 + 0.005 * n, c[1] - 0.010, ztop + 0.0008)))
                widths = rnd.uniform(0.0004, 0.0016, 22)
                y0 = c[1] - 0.024
                for n, w in enumerate(widths):
                    V.append(_vm(f"{tag}bc{k}{n}", M.rbox((0.009, float(w), 0.0003), e=4), INK, "matte",
                                 offset=(lx + 0.008, y0 + 0.0022 * n, ztop + 0.0008)))
            P.append(Piece("box", box, tuple(float(v) for v in c)))
    return V, P


# ------------------------------------------------------------------------------------------- bails
def bail_path(phi, s):
    """The bail's wire centreline in the crate frame (its underside at 0) for leg angle phi, the bail at
    the crate's +y (s = 1) or -y (s = -1) end: from one pivot boss along the leg, across the bar, back."""
    d = np.array([0.0, s * math.cos(phi), math.sin(phi)])
    piv = [np.array([sx * X_LEG, s * Y_PIV, Z_PIV]) for sx in (-1, 1)]
    stub = [np.array([sx * (X_LEG - 0.010), s * Y_PIV, Z_PIV]) for sx in (-1, 1)]
    bar = [p + d * L_LEG for p in piv]
    return M.bend([stub[0], piv[0], bar[0], bar[1], piv[1], stub[1]], radius=0.012, steps=6)


def bail_object(s, crate_pos):
    """Tote A's bail at its +y (s = 1) or -y (s = -1) end: a free steel wire U, hinged to the crate at its two
    pivot bosses, colliding with the crate (it leans on the end wall when out, lies on the rim when in).
    The body's origin is the middle of its bar; the hinge axis runs through the two pivots."""
    z_under = crate_pos[2] - H / 2
    piv = np.array([crate_pos[0], crate_pos[1] + s * Y_PIV, z_under + Z_PIV])
    d = np.array([0.0, s * math.cos(PHI_OUT), math.sin(PHI_OUT)])
    bar = piv + d * L_LEG
    path = bail_path(PHI_OUT, s) + np.array([crate_pos[0], crate_pos[1], z_under]) - bar
    V = [_vm(f"bail{s}", M.tube(path, R_WIRE, n=12), ZINC, "gloss")]
    beta = PHI_OUT if s > 0 else math.pi - PHI_OUT              # turns a Y-axis capsule onto the leg's direction
    P = [Piece("capsule", (R_WIRE, L_LEG - 0.024, R_WIRE), tuple(np.array([sx * X_LEG, 0.0, 0.0]) - d * (L_LEG / 2 - 0.004)),
               rot=(beta, 0.0, 0.0), axis="Y") for sx in (-1, 1)]
    # No joint limits: the bail's travel (about 260 deg) is stopped by contact alone -- hanging free outside
    # the end wall, lying on the rim when swung in. (A revolute limit range past 180 deg wraps in PhysX.)
    lo, hi = -1.0e6, 1.0e6
    joint = Hinge(pivot=tuple(float(v) for v in piv), axis="X", damping=0.004, friction=0.002, lower_deg=lo, upper_deg=hi,
                  parent="tote_a", parent_pos=tuple(float(v) for v in crate_pos), collide=True)
    name = "bail_l" if s > 0 else "bail_r"
    return Part(name, pos=tuple(float(v) for v in bar), size=(R_WIRE, 2 * X_LEG, R_WIRE), shape="capsule", axis="X",
                exact_z=True, mass=BAIL_MASS, color=ZINC, friction=(0.40, 0.32), restitution=0.05, contact_offset=0.002,
                pieces=_colliders(P), hide_root=True, vmeshes=tuple(V), hinge=joint, solver_iters=(32, 4), max_depenetration=0.3,
                com=tuple(float(v) for v in -d * 0.035))


def a_pos():
    return (A_C[0], A_C[1], TOP + H / 2)


def b_pos():
    return (B_C[0], B_C[1], TOP + H / 2)


def tote_a_object() -> Part:
    z0 = -H / 2
    V = crate_vmeshes(z0, 1)
    cv, cp = _cartons(z0, "a", 0.5)
    P = crate_pieces(z0) + notch_pieces(z0) + cp
    return Part("tote_a", pos=a_pos(), size=(W_BOT - 0.01, L_BOT - 0.01, 0.006), exact_z=True, mass=A_MASS, color=BLUE,
                friction=(0.40, 0.32), restitution=0.05, contact_offset=0.003, collision="convexHull",
                pieces=_colliders(P), hide_root=True, vmeshes=tuple(V + cv), solver_iters=(32, 4),
                com=(0.0, 0.0, z0 + 0.050))


def tote_b_object() -> Part:
    """Tote B: the same crate, full of light cartons, its bails hanging out (welded: they swing, but play no
    part in lifting it by its hand holes)."""
    z0 = -H / 2
    V = crate_vmeshes(z0, 2)
    cv, cp = _cartons(z0, "b", 0.06, CARTON_B)
    for s in (-1, 1):
        V.append(_vm(f"bbail{s}", M.tube(bail_path(PHI_OUT, s) + np.array([0.0, 0.0, z0]), R_WIRE, n=12), ZINC, "gloss"))
    P = crate_pieces(z0) + cp
    return Part("tote_b", pos=b_pos(), size=(W_BOT - 0.01, L_BOT - 0.01, 0.006), exact_z=True, mass=B_MASS, color=BLUE,
                friction=(0.40, 0.32), restitution=0.05, contact_offset=0.003, collision="convexHull",
                pieces=_colliders(P), hide_root=True, vmeshes=tuple(V + cv), solver_iters=(32, 4),
                com=(0.0, 0.0, z0 + 0.050))


def scene_objects():
    return (tote_a_object(), bail_object(1, a_pos()), bail_object(-1, a_pos()), tote_b_object())
