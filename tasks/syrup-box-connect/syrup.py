"""Geometry for connecting a new bag-in-box soda syrup carton (real size).

Fast-food back room. Two 5-gallon (19 L) bag-in-box syrup cartons stand side by side on the bench, their
knock-out flaps toward the arms. The left one is new: its flap has been pushed in and the bag's spout
pulled out through the hole. The right one is empty: its spout still wears the syrup line's quick-
disconnect connector. Behind them, on a plywood board, the syrup pump the line runs back to. The
connector's collar is turned a quarter turn counter-clockwise and the connector drawn off the empty
spout; then, with the new spout held by its flange (it rides on the slack bag and gives way when pushed),
the connector is pushed onto it and its collar turned a quarter turn clockwise to lock.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.

Real parts:
  carton     5-gallon bag-in-box syrup carton, 273 x 273 x 394 mm (10.75 x 10.75 x 15.5 in),
             corrugated kraft board, printed; about 24 kg full. Knock-out flap 90 x 70 mm low on one face.
  spout      BIB bag fitting: polyethylene, 46 mm flange, 25 mm spout with an O-ring land.
  connector  BIB quick-disconnect (QCD) coupler: 32 mm body, 40 mm coloured locking collar with grip
             ribs, 60 mm long, 3/8 in barb with its syrup line; turns a quarter turn to lock.
  pump       gas-driven BIB syrup pump, black glass-filled nylon, 150 x 95 x 80 mm, on a plywood board.
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace
from pathlib import Path

import numpy as np

import meshlib as M
from simkit import Anchor, Hinge, Part, Piece, Spring

TOP = 0.7475

KRAFT = (0.64, 0.47, 0.30)
KRAFT_DARK = (0.48, 0.34, 0.21)
INK_RED = (0.70, 0.08, 0.10)
INK_DARK = (0.12, 0.10, 0.10)
WHITE = (0.93, 0.92, 0.89)
YELLOW = (0.95, 0.78, 0.12)
TAPE = (0.78, 0.66, 0.46)
PE_WHITE = (0.92, 0.92, 0.90)
FILM = (0.74, 0.75, 0.78)
HOLE = (0.06, 0.05, 0.04)
COLLAR = (0.80, 0.10, 0.10)          # red: the cola line's colour code
BODY = (0.90, 0.90, 0.88)
LINE = (0.93, 0.93, 0.90)            # translucent white syrup tubing
PUMP_BLACK = (0.08, 0.08, 0.09)
PLY = (0.78, 0.66, 0.48)
BRASS = (0.78, 0.62, 0.30)

# -- cartons -----------------------------------------------------------------------------------------
BOX = (0.273, 0.273, 0.394)          # x (depth, flap face toward -x), y, z
BOX_X = 0.470                        # centres
NEW_Y = -0.165
OLD_Y = 0.165
FRONT_X = BOX_X - BOX[0] / 2         # the flap face
HOLE_W, HOLE_H = 0.090, 0.070        # knock-out opening (y, z)
HOLE_Z0 = 0.030                      # its bottom edge above the carton's base
HOLE_ZC = HOLE_Z0 + HOLE_H / 2
HOLE_DEPTH = 0.060                   # pushed-in flap: open space behind the face for the spout to ride back into
SHELF_Z = 0.120                      # the wire rack shelf's deck, above the bench
BASE = TOP + SHELF_Z                 # the cartons' bottoms

# -- spout (bag fitting) and connector ------------------------------------------------------------------
FLANGE_R = 0.023
SPOUT_R = 0.0125
SPOUT_L = 0.040                      # flange face to spout tip
SPOUT_OUT = 0.045                    # the flange stands this far out in front of the face (fitting pulled out on the bag's neck)
CONN_R = 0.016
CONN_L = 0.060
COLLAR_R = 0.020
COLLAR_L = 0.022
ENGAGE = 0.024                       # the connector covers this much of the spout when on


def spout_axis(y):
    """Flange-face centre of a spout on the carton at y (the spout points along -x)."""
    return np.array([FRONT_X - SPOUT_OUT, y, BASE + HOLE_ZC])


def connector_pos(y):
    """The connector's centre when it is on the spout of the carton at y."""
    tip = spout_axis(y)[0] - SPOUT_L
    return np.array([tip + ENGAGE - CONN_L / 2, y, BASE + HOLE_ZC])


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _colliders(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces)


# -- carton visuals -------------------------------------------------------------------------------------
def _text_lines(prefix, x, yc, zc, w, rows, color=INK_DARK, dz=0.011, h=0.004, normal=-1.0):
    """Printed lines of text (dark bars of varied length) on a face whose outward normal is +-x."""
    V = []
    rng = np.random.default_rng(abs(hash(prefix)) % 2 ** 32)
    for k in range(rows):
        ww = w * (0.55 + 0.45 * rng.random())
        V.append(_vm(f"{prefix}{k}", M.rbox((0.0008, ww, h), 10), color, "matte",
                     offset=(x + normal * 0.0006, yc - (w - ww) / 2, zc - k * dz)))
    return V


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    " ": ["00000"] * 7,
}


def text_mesh(text, px, nrm, centre, depth=0.0008):
    """Printed block letters (5 x 7 pixels of size px) on a face of outward normal nrm (x, y), centred at
    ``centre`` (a point on the face, carton frame); read left to right from outside."""
    n = np.array([nrm[0], nrm[1], 0.0])
    u = np.array([-nrm[1], nrm[0], 0.0])
    w = len(text) * 6 * px - px
    meshes = []
    for i, ch in enumerate(text):
        rows = FONT[ch]
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "1":
                    off = (np.asarray(centre, float) + n * depth + u * (-w / 2 + (i * 6 + c + 0.5) * px)
                           + np.array([0.0, 0.0, (3 - r) * px]))
                    size = np.abs(n) * 0.0010 + np.abs(u) * px * 1.02 + np.array([0.0, 0.0, px * 1.02])
                    meshes.append(M.transform(M.rbox(tuple(size), 4, n=8), None, off))
    return M.merge(*meshes)


TEXTURES = Path(__file__).resolve().parent / "textures"


def textured_face(name, nrm, image, inset=0.0):
    """A printed face: a quad just outside the carton face of outward normal nrm (x, y), carrying ``image``
    (textures/, made by make_textures.py) the right way up, read from outside."""
    x, y, z = BOX
    n = np.array([nrm[0], nrm[1], 0.0])
    u = np.array([-nrm[1], nrm[0], 0.0])
    half = (x if nrm[0] else y) / 2
    w = (y if nrm[0] else x) - inset
    h = z - inset
    c = n * (half + 0.0007)
    zz = np.array([0.0, 0.0, 1.0])
    pts = np.array([c - u * w / 2 - zz * h / 2, c + u * w / 2 - zz * h / 2, c + u * w / 2 + zz * h / 2, c - u * w / 2 + zz * h / 2])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return (name, pts, tris, (1.0, 1.0, 1.0), "tex:" + str(TEXTURES / image), 1.0, uv)


def carton_vmeshes(empty):
    """A printed corrugated carton, flap face toward -x, in the carton Part's frame (centre at its middle)."""
    x, y, z = BOX
    V = [_vm("board", M.rbox((x, y, z), 40), (0.93, 0.92, 0.89), "matte")]
    fx = -x / 2
    # top flaps' seam, flap edges and the packing tape across them
    V.append(_vm("seam", M.rbox((0.0015, y - 0.004, 0.0008), 10), KRAFT_DARK, "matte", offset=(0.0, 0.0, z / 2 + 0.0004)))
    for sx in (-1, 1):
        V.append(_vm(f"flapedge{sx}", M.rbox((x - 0.004, 0.0015, 0.0008), 10), KRAFT_DARK, "matte",
                     offset=(0.0, sx * (y / 2 - 0.020), z / 2 + 0.0004)))
    V.append(_vm("tape", M.rbox((0.050, y + 0.002, 0.0010), 10), TAPE, "satin", offset=(0.0, 0.0, z / 2 + 0.0008)))
    for sx in (-1, 1):
        V.append(_vm(f"tape_end{sx}", M.rbox((0.050, 0.0016, 0.060), 10), TAPE, "satin",
                     offset=(0.0, sx * (y / 2 + 0.0008), z / 2 - 0.030)))
    for k, (nrm, image) in enumerate((((-1, 0), "carton_front.png"), ((1, 0), "carton_front.png"),
                                      ((0, -1), "carton_side.png"), ((0, 1), "carton_side.png"))):
        V.append(textured_face(f"print{k}", nrm, image, inset=0.004))
    for k in range(4):                                                  # corrugated flute showing at the vertical edges
        sx, sy = ((-1, -1), (-1, 1), (1, -1), (1, 1))[k]
        V.append(_vm(f"edge{k}", M.rbox((0.0030, 0.0030, z - 0.010), 6), KRAFT_DARK, "matte",
                     offset=(sx * (x / 2 - 0.0012), sy * (y / 2 - 0.0012), 0.0)))
    # the round knock-out: dashed perforation circle and the pushed-in opening
    hz = -z / 2 + HOLE_ZC
    V.append(_vm("opening", M.revolve([(0.0, 0.0), (0.041, 0.0), (0.041, 0.0014), (0.0, 0.0014)], n=40, axis="X",
                                      cap0=True, cap1=True), HOLE, "matte", offset=(fx - 0.0012, 0.0, hz)))
    V.append(_vm("film", M.heightfield(0.070, 0.070,
                                       lambda U, W: 0.004 * np.sin(U * 90) * np.cos(W * 70) + (0.010 if empty else 0.0),
                                       -0.002, nx=24, ny=18), FILM, "satin",
                 offset=(fx + 0.030, 0.0, hz), R=np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T))
    if empty:                                                            # an empty carton: slightly caved-in top
        V.append(_vm("dent", M.rbox((0.120, 0.090, 0.0010), 12), KRAFT_DARK, "matte", offset=(0.030, 0.020, z / 2 + 0.0002)))
    return tuple(V)


def _carton_colliders():
    """Fixed carton colliders that leave the knock-out opening open HOLE_DEPTH deep."""
    x, y, z = BOX
    fx = -x / 2
    hz = -z / 2 + HOLE_ZC
    P = [Piece("box", (x - HOLE_DEPTH, y, z), (HOLE_DEPTH / 2, 0.0, 0.0)),                       # behind the opening
         Piece("box", (HOLE_DEPTH, y, z / 2 - (hz + HOLE_H / 2)), (fx + HOLE_DEPTH / 2, 0.0, (z / 2 + hz + HOLE_H / 2) / 2)),
         Piece("box", (HOLE_DEPTH, y, hz - HOLE_H / 2 + z / 2), (fx + HOLE_DEPTH / 2, 0.0, (-z / 2 + hz - HOLE_H / 2) / 2))]
    side_w = (y - HOLE_W) / 2
    for sy in (-1, 1):
        P.append(Piece("box", (HOLE_DEPTH, side_w, HOLE_H), (fx + HOLE_DEPTH / 2, sy * (HOLE_W / 2 + side_w / 2), hz)))
    return _colliders(P)


def carton_object(name, y, empty):
    from simkit import Part as _P
    pos = (BOX_X, y, BASE + BOX[2] / 2)
    pieces = _carton_colliders()
    vm = carton_vmeshes(empty)
    if empty:                                                            # the empty carton's spout is part of it
        vm = vm + tuple(_shift(spout_vmeshes("old_"), spout_axis(y) - np.asarray(pos)))
    return _P(name, pos=pos, size=(0.01, 0.01, 0.01), exact_z=True, color=KRAFT, friction=(0.6, 0.5),
              pieces=pieces, hide_root=True, vmeshes=vm)


def _shift(vms, d):
    d = np.asarray(d, float)
    return [(n, p + d, t, c, m, o) for n, p, t, c, m, o in vms]


# -- spout -----------------------------------------------------------------------------------------------
def spout_vmeshes(prefix=""):
    """A BIB bag fitting, flange face at the origin, its spout pointing along -x."""
    prof = [(0.0, 0.004), (FLANGE_R, 0.004), (FLANGE_R, 0.0), (0.0175, -0.001), (SPOUT_R + 0.001, -0.004),
            (SPOUT_R, -0.010), (SPOUT_R + 0.0012, -0.012), (SPOUT_R + 0.0012, -0.015), (SPOUT_R, -0.017),
            (SPOUT_R, -0.034), (SPOUT_R - 0.0006, -SPOUT_L), (0.0065, -SPOUT_L), (0.0065, -SPOUT_L + 0.004)]
    body = M.revolve([(r, -h) for r, h in prof], n=48, axis="X")
    V = [_vm(prefix + "fitting", body, PE_WHITE, "satin")]
    # a dark O-ring in its land, and the welded film round the flange's back
    V.append(_vm(prefix + "oring", M.revolve([(SPOUT_R + 0.0004, 0.0195), (SPOUT_R + 0.0018, 0.0205),
                                               (SPOUT_R + 0.0004, 0.0215)], n=40, axis="X"), INK_DARK, "satin"))
    V.append(_vm(prefix + "weld", M.revolve([(FLANGE_R - 0.002, -0.006), (FLANGE_R + 0.010, -0.010),
                                              (FLANGE_R + 0.012, -0.011)], n=40, axis="X"), FILM, "satin"))
    return tuple(V)


def new_spout_object():
    """The new carton's spout, pulled out through its opening: rides on the slack bag, so it gives way
    along its axis when pushed (a sprung slide, a disclosed stand-in for the bag's slack)."""
    pos = tuple(float(v) for v in spout_axis(NEW_Y))
    P = [Piece("cylinder", (SPOUT_R, SPOUT_L, SPOUT_R), (-SPOUT_L / 2, 0.0, 0.0), axis="X"),
         Piece("cylinder", (FLANGE_R, 0.004, FLANGE_R), (0.002, 0.0, 0.0), axis="X")]
    return Part("new_spout", pos=pos, size=(0.010, 0.010, 0.010), exact_z=True, mass=0.060, color=PE_WHITE,
                friction=(0.7, 0.6), contact_offset=0.002, pieces=_colliders(P), hide_root=True,
                vmeshes=spout_vmeshes("new_"),
                spring=Spring(axis="X", lower=-0.015, upper=0.045, stiffness=120.0, damping=4.0, target=0.0))


# -- connector -----------------------------------------------------------------------------------------
def connector_vmeshes():
    """QCD coupler, its centre at the origin, its mouth toward +x (onto a spout), barb and line out of -x."""
    h = CONN_L / 2
    body = M.revolve([(0.0, -h - 0.016), (0.0045, -h - 0.016), (0.0060, -h - 0.010), (0.0045, -h - 0.008),
                      (0.0060, -h - 0.004), (0.0080, -h), (CONN_R - 0.002, -h + 0.006), (CONN_R, -h + 0.010),
                      (CONN_R, h - 0.002), (CONN_R - 0.0015, h), (SPOUT_R + 0.0015, h), (SPOUT_R + 0.0015, h - 0.020)],
                     n=48, axis="X")
    V = [_vm("body", body, BODY, "satin")]
    # coloured locking collar with grip ribs and two thumb wings
    V.append(_vm("collar", M.revolve([(CONN_R + 0.0005, -0.006), (COLLAR_R, -0.004), (COLLAR_R, COLLAR_L - 0.004),
                                      (CONN_R + 0.0005, COLLAR_L - 0.002)], n=48, axis="X"), COLLAR, "satin",
                 offset=(h - COLLAR_L, 0.0, 0.0)))
    for k in range(18):
        a = 2 * math.pi * k / 18
        V.append(_vm(f"rib{k}", M.rbox((COLLAR_L - 0.006, 0.0022, 0.0016), 6), COLLAR, "satin",
                     offset=(h - COLLAR_L / 2 - 0.002, (COLLAR_R + 0.0006) * math.cos(a), (COLLAR_R + 0.0006) * math.sin(a)),
                     R=_rx(a + math.pi / 2)))
    for sy in (-1, 1):
        V.append(_vm(f"wing{sy}", M.rbox((0.014, 0.012, 0.004), 6), COLLAR, "satin",
                     offset=(h - COLLAR_L / 2 - 0.002, sy * (COLLAR_R + 0.005), 0.0)))
    # the stainless hose clamp on the barb and the first bit of the line
    V.append(_vm("clamp", M.revolve([(0.0072, 0.0), (0.0080, 0.0), (0.0080, 0.006), (0.0072, 0.006)], n=32, axis="X"),
                 (0.78, 0.79, 0.80), "steel", offset=(-h - 0.014, 0.0, 0.0)))
    return tuple(V)


COLLAR_X = CONN_L / 2 - COLLAR_L / 2 - 0.002     # the collar's centre along the connector (from its centre)


def connector_colliders():
    """Hollow at the mouth (the spout goes in ENGAGE deep), a solid stop behind it, and the collar as a ring
    of staves the fingers can grip."""
    h = CONN_L / 2
    P = [Piece("cylinder", (CONN_R, CONN_L - ENGAGE, CONN_R), (-ENGAGE / 2, 0.0, 0.0), axis="X")]
    n = 10
    rm = (SPOUT_R + 0.0025 + COLLAR_R) / 2
    t = COLLAR_R - (SPOUT_R + 0.0025)
    w = 2 * COLLAR_R * math.tan(math.pi / n) + 0.001
    for k in range(n):
        a = 2 * math.pi * k / n
        P.append(Piece("box", (COLLAR_L - 0.004, w, t), (COLLAR_X, rm * math.sin(a), rm * math.cos(a)), rot=(-a, 0.0, 0.0)))
    return _colliders(P)


def connector_object():
    """On the empty carton's spout: turns on it about the spout's axis (a hinge, -95..+5 degrees) and comes off
    once a quarter turn counter-clockwise (task.Dynamics frees it). A second hinge to the new spout is made at
    spawn and switched on when the connector is pushed onto that spout square."""
    pos = connector_pos(OLD_Y)
    on_new = connector_pos(NEW_Y) - spout_axis(NEW_Y)
    return Part("connector", pos=tuple(float(v) for v in pos), size=(0.005, 0.005, 0.005), shape="sphere",
                exact_z=True, mass=0.070, color=BODY, friction=(0.9, 0.8), contact_offset=0.002,
                pieces=connector_colliders(), hide_root=True, vmeshes=connector_vmeshes(),
                hinge=Hinge(pivot=tuple(float(v) for v in pos), axis="X", damping=0.5, lower_deg=-95.0, upper_deg=5.0),
                hinge2=Hinge(pivot=tuple(float(v) for v in connector_pos(NEW_Y)), axis="X", damping=0.002,
                             lower_deg=-1e6, upper_deg=1e6, parent="new_spout",
                             parent_pos=tuple(float(v) for v in spout_axis(NEW_Y)),
                             local0=tuple(float(v) for v in on_new), local1=(0.0, 0.0, 0.0), enabled=False))


# -- chrome wire BIB rack shelf -------------------------------------------------------------------------------
RACK = (0.340, 0.700, SHELF_Z)       # deck x, y; deck top height
RACK_C = (BOX_X, 0.0)
POST_H = SHELF_Z + 0.012           # a low wire shelf: its posts stop just above the deck
CHROME = (0.84, 0.85, 0.87)


def rack_object():
    """A low chrome wire shelving unit (NSF wire shelf on four 25 mm posts) the cartons stand on."""
    dx, dy, dz = RACK
    pos = (RACK_C[0], RACK_C[1], TOP + dz - 0.010)
    V = []
    posts = [(sx * (dx / 2 - 0.0125), sy * (dy / 2 - 0.0125)) for sx in (-1, 1) for sy in (-1, 1)]
    for k, (px, py) in enumerate(posts):
        V.append(_vm(f"post{k}", M.revolve([(0.0, 0.0), (0.0125, 0.0), (0.0125, POST_H), (0.0, POST_H)], n=24, cap0=True, cap1=True),
                     CHROME, "steel", offset=(px, py, -(dz - 0.010))))
        for g in range(int(POST_H / 0.0254)):                        # the posts' grooves every inch
            V.append(_vm(f"groove{k}_{g}", M.revolve([(0.0127, 0.0), (0.0129, 0.0006), (0.0127, 0.0012)], n=20),
                         (0.60, 0.61, 0.63), "steel", offset=(px, py, -(dz - 0.010) + 0.020 + g * 0.0254)))
        V.append(_vm(f"foot{k}", M.revolve(M.fillet_profile(0.0, 0.016, 0.0, 0.012, 0.003), n=20), (0.15, 0.15, 0.16), "satin",
                     offset=(px, py, -(dz - 0.010))))
        V.append(_vm(f"collar{k}", M.revolve([(0.0125, -0.012), (0.017, -0.010), (0.017, 0.006), (0.0125, 0.008)], n=20),
                     CHROME, "steel", offset=(px, py, 0.0)))
    frame = []
    for sx in (-1, 1):                                                 # frame tubes round the deck
        frame.append(M.transform(M.rbox((0.008, dy - 0.020, 0.012), 8), None, (sx * (dx / 2 - 0.004), 0.0, 0.0)))
    for sy in (-1, 1):
        frame.append(M.transform(M.rbox((dx - 0.020, 0.008, 0.012), 8), None, (0.0, sy * (dy / 2 - 0.004), 0.0)))
    for k in range(int((dy - 0.04) / 0.025) + 1):                      # wires across the deck
        yy = -dy / 2 + 0.020 + k * 0.025
        frame.append(M.transform(M.rbox((dx - 0.012, 0.0032, 0.0032), 6, n=8), None, (0.0, yy, 0.004)))
    for xx in (-0.110, -0.040, 0.040, 0.110):                          # truss wires underneath
        frame.append(M.transform(M.rbox((0.004, dy - 0.020, 0.018), 6, n=8), None, (xx, 0.0, -0.010)))
    V.append(_vm("deck", M.merge(*frame), CHROME, "steel"))
    P = [Piece("box", (dx, dy, 0.020), (0.0, 0.0, 0.0), hidden=True)]
    for (px, py) in posts:
        P.append(Piece("cylinder", (0.0125, POST_H, 0.0125), (px, py, POST_H / 2 - (dz - 0.010)), hidden=True))
    return Part("rack", pos=pos, size=(0.01, 0.01, 0.01), exact_z=True, color=CHROME, friction=(0.6, 0.5),
                pieces=tuple(P), hide_root=True, vmeshes=tuple(V))


# -- syrup line, pump board ---------------------------------------------------------------------------------
# The pump board stands along the bench's left edge (its face toward -y), beside the empty carton, so the
# cameras see past it.
BOARD_Y = 0.445
BOARD = (0.440, 0.018, 0.450)        # x, y, z
BOARD_C = (0.520, BOARD_Y, TOP + BOARD[2] / 2)
PUMP_C = (0.560, BOARD_Y - BOARD[1] / 2 - 0.041, TOP + 0.320)
PUMP = (0.095, 0.080, 0.150)


def board_object():
    """Plywood pump board standing at the bench's left edge, a gas-driven syrup pump screwed to it."""
    pos = BOARD_C
    V = [_vm("ply", M.rbox(BOARD, 6), PLY, "oak")]
    px, py, pz = (PUMP_C[0] - pos[0], PUMP_C[1] - pos[1], PUMP_C[2] - pos[2])
    V.append(_vm("pump", M.rbox(PUMP, 5), PUMP_BLACK, "satin", offset=(px, py, pz)))
    V.append(_vm("pump_cap", M.revolve(M.fillet_profile(0.0, 0.038, -0.010, 0.010, 0.004), n=40, axis="Z"), PUMP_BLACK, "satin",
                 offset=(px, py, pz + PUMP[2] / 2 + 0.008)))
    V.append(_vm("label", M.rbox((0.050, 0.0008, 0.030), 8), WHITE, "matte", offset=(px, py - PUMP[1] / 2 - 0.0005, pz + 0.020)))
    V.append(_vm("bracket", M.rbox((0.120, 0.012, 0.040), 6), (0.70, 0.71, 0.73), "steel",
                 offset=(px, -BOARD[1] / 2 - 0.006, pz - PUMP[2] / 2 - 0.010)))
    ports = {"gas": (-PUMP[0] / 2, py - 0.015, pz + 0.050), "in": (-PUMP[0] / 2, py, pz - 0.030),
             "out": (PUMP[0] / 2, py, pz - 0.030)}
    for k, (name, (ox, oy, oz)) in enumerate(ports.items()):
        col = (0.20, 0.45, 0.80) if name == "gas" else BRASS
        sgn = -1 if ox < 0 else 1
        V.append(_vm(f"port_{name}", M.revolve([(0.0, 0.0), (0.008, 0.0), (0.008, 0.016), (0.0, 0.016)], n=24, axis="X",
                                              cap1=True), col, "satin",
                     offset=(px + ox + (0.0 if sgn > 0 else -0.016), oy, oz)))
    gas = M.bend([(px - PUMP[0] / 2 - 0.016, py - 0.015, pz + 0.050), (px - 0.090, py - 0.015, pz + 0.050),
                  (px - 0.120, py - 0.010, BOARD[2] / 2 - 0.030), (px - 0.150, 0.0, BOARD[2] / 2 + 0.040)], 0.03)
    V.append(_vm("gas", M.tube(gas, 0.0048, 10), (0.20, 0.45, 0.80), "satin"))
    out = M.bend([(px + PUMP[0] / 2 + 0.016, py, pz - 0.030), (px + 0.090, py, pz - 0.030),
                  (px + 0.110, py, BOARD[2] / 2 - 0.020), (px + 0.120, 0.0, BOARD[2] / 2 + 0.040)], 0.03)
    V.append(_vm("outlet", M.tube(out, 0.0048, 10), LINE, "satin", 0.9))
    for k, (xx, zz) in enumerate(((0.19, 0.19), (-0.19, 0.19), (0.19, -0.19), (-0.19, -0.19))):
        V.append(_vm(f"screw{k}", M.revolve(M.fillet_profile(0.0, 0.005, 0.0, 0.002, 0.001), n=16, axis="Y"),
                     (0.70, 0.71, 0.73), "steel", offset=(xx, -BOARD[1] / 2 - 0.001, zz)))
    face = np.array([0.0, -BOARD[1] / 2 - 0.0012, 0.0])
    for name, (ox, oz) in {"GAS": (px - 0.095, pz + 0.075), "IN": (px - 0.095, pz - 0.055), "OUT": (px + 0.095, pz - 0.055)}.items():
        V.append(_vm(f"tag_{name}", M.rbox((0.040, 0.0010, 0.016), 8), WHITE, "matte", offset=face + np.array([ox, 0.0, oz])))
        V.append(_vm(f"tagtxt_{name}", text_mesh(name, 0.0016, (0, -1), face + np.array([ox, -0.0008, oz]), 0.0004),
                     INK_DARK, "matte"))
    # three more flavours' lines from off the board's left end run along its top to the fountain (a bundle with ties)
    for k, col in enumerate(((0.85, 0.10, 0.10), (0.15, 0.60, 0.20), (0.95, 0.75, 0.10))):
        yy = py - 0.010 - 0.012 * k
        path = M.bend([(-BOARD[0] / 2 - 0.040, yy, BOARD[2] / 2 - 0.060 - 0.010 * k), (-BOARD[0] / 2 + 0.040, yy, BOARD[2] / 2 - 0.060 - 0.010 * k),
                       (BOARD[0] / 2 - 0.040, yy, BOARD[2] / 2 - 0.060 - 0.010 * k), (BOARD[0] / 2 + 0.040, yy, BOARD[2] / 2 - 0.020)], 0.03)
        V.append(_vm(f"bundle{k}", M.tube(path, 0.0048, 10), LINE, "satin", 0.9))
        for t in (-0.15, 0.0, 0.15):                                   # coloured ID bands on each line
            V.append(_vm(f"band{k}_{t}", M.revolve([(0.0052, 0.0), (0.0052, 0.008)], n=16, axis="X"), col, "matte",
                         offset=(t, yy, BOARD[2] / 2 - 0.060 - 0.010 * k)))
    for t in (-0.10, 0.05, 0.17):                                      # zip ties round the bundle
        V.append(_vm(f"tie{t}", M.rbox((0.003, 0.040, 0.004), 6), (0.10, 0.10, 0.10), "satin",
                     offset=(t, py - 0.022, BOARD[2] / 2 - 0.070)))
    for nm, (cx, cz) in {"in": (px - PUMP[0] / 2 - 0.020, pz - 0.030), "out": (px + PUMP[0] / 2 + 0.020, pz - 0.030),
                          "gas": (px - PUMP[0] / 2 - 0.020, pz + 0.050)}.items():   # stainless clamps on the barbs
        V.append(_vm(f"clamp_{nm}", M.revolve([(0.0058, 0.0), (0.0066, 0.0), (0.0066, 0.006), (0.0058, 0.006)], n=24, axis="X"),
                     (0.78, 0.79, 0.80), "steel", offset=(cx - 0.003, py if nm != "gas" else py - 0.015, cz)))
    P = [Piece("box", BOARD, (0.0, 0.0, 0.0), hidden=True)]
    return Part("pump_board", pos=pos, size=(0.01, 0.01, 0.01), exact_z=True, color=PLY, pieces=tuple(P),
                hide_root=True, vmeshes=tuple(V))


def pump_inlet():
    return np.array([PUMP_C[0] - PUMP[0] / 2 - 0.016, PUMP_C[1], PUMP_C[2] - 0.030])


LINE_N = 48


def line_path(conn_pose):
    """The syrup line's centreline from the connector's barb to the pump's inlet: a cubic Bezier with a fixed
    number of samples (so the drawn tube can follow the connector)."""
    p = np.asarray(conn_pose, float)
    q = p[3:7]
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    ax = R[:, 0]
    barb = p[:3] - ax * (CONN_L / 2 + 0.022)
    pin = pump_inlet()
    c1 = barb - ax * 0.10 + np.array([0.0, 0.0, 0.03])
    c2 = pin + np.array([-0.18, -0.02, -0.12])
    t = np.linspace(0.0, 1.0, LINE_N)[:, None]
    return (1 - t) ** 3 * barb + 3 * (1 - t) ** 2 * t * c1 + 3 * (1 - t) * t ** 2 * c2 + t ** 3 * pin


def line_mesh(conn_pose, origin):
    return M.transform(M.tube(line_path(conn_pose), 0.0064, 12), None, -np.asarray(origin, float))


def line_object():
    """The syrup line from the pump's inlet to the connector: drawn only (no collision); task.Dynamics redraws it
    every step so it follows the connector."""
    c = connector_pos(OLD_Y)
    pose = (*c, 0.0, 0.0, 0.0, 1.0)
    pos = (float(c[0]), float(c[1]), float(c[2]))
    p, t = line_mesh(pose, pos)
    V = [("line", p, t, LINE, "satin", 0.92)]
    return Part("syrup_line", pos=pos, size=(0.004, 0.004, 0.004), exact_z=True, color=LINE,
                pieces=(Piece("box", (0.004, 0.004, 0.004), (0.0, 0.0, 0.0), collide=False, hidden=True),),
                hide_root=True, vmeshes=tuple(V), collision="none")


def quat_x_axis(q):
    """The local x axis of a body with orientation q (x, y, z, w)."""
    x, y, z, w = (float(v) for v in q)
    return np.array([1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)])
