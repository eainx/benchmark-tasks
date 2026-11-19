"""Geometry and parts for racking two washed-up hi-ball glasses mouth-down in a dish-machine glass rack
(real size).

On a stainless dish table two empty 10 oz heavy-base hi-ball tumblers stand upright, mouth up, as they come
back from the bar. Beside them sits an empty 25-compartment glass rack for the door-type dish machine.
Glasses go through the machine mouth-down, one per compartment, resting on the rack's open floor grid.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.

Real parts:
  glass   generic 10 oz (296 mL) heavy-base hi-ball tumbler, soda-lime glass: 116 mm tall, 66 mm across the
          rim, 60 mm across the foot, 2.5 mm wall, 14 mm solid base; about 0.30 kg.
  rack    25-compartment polypropylene glass rack for door-type dish machines (Cambro Camrack 25S318 type):
          500 x 500 mm, 101 mm tall, 5 x 5 compartments of about 95 mm, open floor grid; 2.3 kg.
  table   brushed stainless dish table top, 3 mm sheet, 640 x 1100 mm.
"""
from __future__ import annotations

import math

import numpy as np

import meshlib as M
from simkit import Part, Piece

TOP = 0.7475

# -- dish table (fixed) ---------------------------------------------------------------------------
TABLE = (0.640, 1.100, 0.003)
TABLE_C = (0.330, 0.000)
FLOOR = TOP + TABLE[2]                   # the table's top: what the glasses and rack stand on

# -- glass (free, x2) ------------------------------------------------------------------------------
H = 0.116                                # foot to rim
R_FOOT = 0.030                           # outside radius at the foot
R_RIM = 0.033                            # outside radius at the rim
WALL = 0.0025
BASE_T = 0.014                           # solid heavy base
MASS = 0.30
COM_Z = 0.040                            # above the foot (heavy base)
GLASSES = {"glass_a": (0.215, 0.215), "glass_b": (0.355, 0.225)}
GLASS_NAMES = tuple(GLASSES)
SOLVER = (32, 4)


def r_out(z):
    """Outside radius at height z above the foot."""
    return R_FOOT + (R_RIM - R_FOOT) * z / H


# -- rack (free, heavy) ------------------------------------------------------------------------------
RACK_C = (0.360, -0.250)                 # centre of the rack's footprint
RACK_W = 0.500                           # outside, square
RACK_H = 0.101                           # overall height (top of walls and dividers above the table)
RACK_WALL = 0.006
GRID_T = 0.009                           # floor grid: bars this tall; glasses rest on its top
GRID_BAR = 0.006
DIV_T = 0.0025                           # compartment dividers
N_CELL = 5
PITCH = (RACK_W - 2 * RACK_WALL) / N_CELL      # 97.6 mm
CELL_IN = PITCH - DIV_T                        # about 95 mm inside each compartment
RACK_MASS = 2.3
RACK_GREY = (0.62, 0.63, 0.62)


def cell_xy(i, j):
    """Compartment (i along x, j along y; 0..4) centre in the rack's frame."""
    return ((i - 2) * PITCH, (j - 2) * PITCH)


def cell_world(i, j):
    x, y = cell_xy(i, j)
    return (RACK_C[0] + x, RACK_C[1] + y)


# ------------------------------------------------------------------------------------------- glass
def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


GLASS_TINT = (0.78, 0.88, 0.90)


def glass_vmeshes():
    h0 = -H / 2                           # part origin: the glass's middle
    outer = [(0.0, 0.0), (R_FOOT - 0.004, 0.0), (R_FOOT - 0.0012, 0.0006), (R_FOOT, 0.003)]
    outer += [(r_out(z), z) for z in np.linspace(0.006, H - 0.0015, 14)]
    rim = [(R_RIM - 0.0003, H - 0.0004), (R_RIM - WALL / 2, H), (R_RIM - WALL + 0.0003, H - 0.0004)]
    inner = [(r_out(z) - WALL, z) for z in np.linspace(H - 0.0015, BASE_T + 0.004, 14)]
    inner += [(r_out(BASE_T) - WALL - 0.004, BASE_T + 0.0005), (0.0, BASE_T - 0.0015)]   # dished inside bottom
    prof = [(r, z + h0) for r, z in outer + rim + inner]
    V = [_vm("glass_body", M.revolve(prof, n=96), GLASS_TINT, "glass", opacity=0.45)]   # translucent, so it reads on camera
    return tuple(V)


def _staves(r0, r1, z0, z1, t, n=24):
    """A ring of n box staves forming the glass's tapered outside wall (outer faces on the cone)."""
    out = []
    phi = math.atan2(r1 - r0, z1 - z0)
    L = math.hypot(r1 - r0, z1 - z0)
    rm = (r0 + r1) / 2 - t / 2 * math.cos(phi)
    w = 2 * min(r0, r1) * 1.01 * math.tan(math.pi / n)
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append(Piece("box", (t, w, L), (rm * math.cos(a), rm * math.sin(a), (z0 + z1) / 2),
                         rot=(0.0, phi, a), hidden=True))
    return out


def glass_object(name) -> Part:
    h0 = -H / 2
    P = [Piece("cylinder", (R_FOOT - 0.0005, BASE_T, R_FOOT - 0.0005), (0.0, 0.0, h0 + BASE_T / 2), hidden=True)]
    P += _staves(R_FOOT, R_RIM, h0, h0 + H, 0.003)
    x, y = GLASSES[name]
    # the root is a slim core inside the base (a collider only); the staves and base make the glass
    return Part(name, pos=(x, y, FLOOR + H / 2), size=(0.010, BASE_T, 0.010), shape="cylinder", exact_z=True,
                mass=MASS, color=GLASS_TINT, friction=(0.60, 0.50), restitution=0.05,
                com=(0.0, 0.0, h0 + COM_Z), contact_offset=0.0015, solver_iters=SOLVER, pieces=tuple(P),
                hide_root=True, vmeshes=glass_vmeshes())


# -------------------------------------------------------------------------------------------- rack
def rack_object() -> Part:
    """The rack: four outer walls with hand-hold slots, the open floor grid and 4 + 4 dividers, one body."""
    P, V = [], []
    half = RACK_W / 2
    zc = RACK_H / 2 - RACK_H / 2          # part origin at the rack's centre, half-height up
    o = -RACK_H / 2                       # the table level in the part frame

    def box(size, c, visible=True, color=RACK_GREY, collide=True):
        P.append(Piece("box", tuple(size), tuple(c), color=color, collide=collide, hidden=not visible))

    # outer walls: solid band at the bottom and top, a lattice of posts between (as moulded racks are)
    for s in (-1, 1):
        for axis in (0, 1):
            c = [0.0, 0.0]
            c[axis] = s * (half - RACK_WALL / 2)
            size = [RACK_W, RACK_W]
            size[axis] = RACK_WALL
            box((size[0], size[1], RACK_H), (c[0], c[1], o + RACK_H / 2), visible=False)       # collider
            # visible: bottom band, top band, posts
            box((size[0], size[1], 0.022), (c[0], c[1], o + 0.011), collide=False)
            box((size[0], size[1], 0.020), (c[0], c[1], o + RACK_H - 0.010), collide=False)
            for k in range(-12, 13):
                p = [c[0], c[1]]
                p[1 - axis] = k * 0.0195
                sz = [RACK_WALL, RACK_WALL]
                sz[1 - axis] = 0.008
                box((sz[0], sz[1], RACK_H - 0.042), (p[0], p[1], o + 0.022 + (RACK_H - 0.042) / 2), collide=False)
            # a hand-hold slot in the middle of each wall's top band (dark recess)
            h = [c[0], c[1]]
            h[axis] = s * (half + 0.0002)
            hs = [0.0015, 0.0015]
            hs[1 - axis] = 0.110
            box((hs[0], hs[1], 0.012), (h[0], h[1], o + RACK_H - 0.010), color=(0.20, 0.20, 0.20), collide=False)
    # floor grid: bars both ways, glasses rest on its top
    inner = RACK_W - 2 * RACK_WALL
    nb = 21
    for k in range(nb):
        u = -inner / 2 + (k + 0.5) * inner / nb
        box((GRID_BAR, inner, GRID_T), (u, 0.0, o + GRID_T / 2))
        box((inner, GRID_BAR, GRID_T), (0.0, u, o + GRID_T / 2))
    # dividers: 4 each way, full height from the grid to the top
    dh = RACK_H - GRID_T
    for k in range(1, N_CELL):
        u = -inner / 2 + k * PITCH
        box((DIV_T, inner, dh), (u, 0.0, o + GRID_T + dh / 2))
        box((inner, DIV_T, dh), (0.0, u, o + GRID_T + dh / 2))
    return Part("rack", pos=(RACK_C[0], RACK_C[1], FLOOR + RACK_H / 2), size=(0.02, 0.02, 0.004), exact_z=True,
                mass=RACK_MASS, color=RACK_GREY, friction=(0.40, 0.33), restitution=0.1,
                contact_offset=0.0015, solver_iters=(16, 2), pieces=tuple(P), hide_root=True)


# ---------------------------------------------------------------------------------------- table
def table_object() -> Part:
    return Part("table", pos=(TABLE_C[0], TABLE_C[1], TOP + TABLE[2] / 2), size=TABLE, exact_z=True,
                color=(0.78, 0.79, 0.80), material="aluminium_brushed", friction=(0.45, 0.38), contact_offset=0.001,
                pieces=(Piece("box", (TABLE[0], TABLE[1], 0.030), (0.0, 0.0, TABLE[2] / 2 - 0.015), hidden=True),))


# ------------------------------------------------------------------------------------ measurements
def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mouth_down_deg(pose):
    """Angle of the glass's foot-to-mouth axis from straight down (0 = mouth down, 180 = upright)."""
    R = mat(pose)
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(R[2, 2])))))


def mouth_centre(pose):
    p = np.asarray(pose, float)
    return p[:3] + mat(p)[:, 2] * (H / 2)


def racked_cell(glass_pose, rack_pose):
    """(i, j) of the compartment a glass stands mouth-down in, on the floor grid, or None."""
    R = mat(rack_pose)
    m = R.T @ (mouth_centre(glass_pose) - np.asarray(rack_pose[:3], float))
    if mouth_down_deg(glass_pose) > 10.0:
        return None
    floor_top = -RACK_H / 2 + GRID_T
    if abs(m[2] - floor_top) > 0.005:
        return None
    for i in range(N_CELL):
        for j in range(N_CELL):
            cx, cy = cell_xy(i, j)
            lim = CELL_IN / 2 - R_RIM + 0.002
            if abs(m[0] - cx) < lim and abs(m[1] - cy) < lim:
                return (i, j)
    return None
