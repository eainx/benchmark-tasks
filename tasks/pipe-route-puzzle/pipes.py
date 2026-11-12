"""Geometry and measurements for a coolant manifold of rotary pipe fittings.

A 3 x 4 panel of rotary fittings sits between the coolant supply (left, +y) and the line out to the
machine (right, -y). Each fitting is a disc on a vertical swivel carrying a pipe -- straight, elbow,
tee or cross -- with a knob on its hub; it turns by hand and stays where it is left. Coolant runs
through every fitting whose ends meet, starting at the supply: a fitting it reaches is drawn full
(blue). The line is restored when coolant reaches the outlet with no open end on any wet fitting.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

PANEL = (0.30, 0.33, 0.36)
STEEL = (0.86, 0.87, 0.89)          # bright galvanised pipe
WATER = (0.10, 0.45, 0.95)
HUB = (0.85, 0.45, 0.10)            # orange hub knob
SUPPLY = (0.75, 0.12, 0.10)

ROWS, COLS = 3, 4
PITCH = 0.095
X0, Y0 = 0.200, 0.1425              # centre of row 0 (nearest the arms), column 0 (left)
DISC_R = 0.044
DISC_T = 0.008
BOARD_T = 0.008
DISC_Z = TOP + BOARD_T + 0.001 + DISC_T / 2
PIPE_R = 0.013
PIPE_Z = DISC_T / 2 + PIPE_R               # pipe axis above the disc's centre
ARM = DISC_R - 0.002                       # pipe from the hub to just short of the disc's rim
KNOB = (0.016, 0.036, 0.024)               # T-handle bar on the hub: width (gripped), length, height
KNOB_Z = PIPE_Z + PIPE_R + KNOB[2] / 2 - 0.004

# directions: index 0..3 = +x (far), +y (left), -x (near), -y (right); turning +90 deg adds 1
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
SHAPES = {"I": (0, 2), "L": (0, 1), "T": (0, 1, 2), "X": (0, 1, 2, 3)}

# the panel: (shape, quarter turns it starts at). Solution path (row, col):
# supply -> (1,0) -> (1,1) -> (2,1) -> (2,2) -> (2,3) -> (1,3) -> (0,3) -> outlet
LAYOUT = {
    (0, 0): ("T", 0), (0, 1): ("L", 1), (0, 2): ("X", 0), (0, 3): ("L", 3),
    (1, 0): ("I", 0), (1, 1): ("L", 2), (1, 2): ("T", 3), (1, 3): ("I", 1),
    (2, 0): ("L", 3), (2, 1): ("L", 1), (2, 2): ("I", 1), (2, 3): ("L", 0),
}
SUPPLY_CELL, SUPPLY_DIR = (1, 0), 1        # supply enters (1,0) from its left side
OUTLET_CELL, OUTLET_DIR = (0, 3), 3        # outlet leaves (0,3) from its right side


def cell_xy(rc):
    r, c = rc
    return (X0 + r * PITCH, Y0 - c * PITCH)


def name(rc):
    return f"fitting_{rc[0]}{rc[1]}"


CELLS = tuple(sorted(LAYOUT))
NAMES = tuple(name(rc) for rc in CELLS)


def _pipe_pieces(shape, colour, r, collide):
    out = []
    for d in SHAPES[shape]:
        a = d * math.pi / 2
        out.append(Piece("cylinder", (r, ARM, r), (ARM / 2 * math.cos(a), ARM / 2 * math.sin(a), PIPE_Z),
                         rot=(0.0, math.pi / 2, a), color=colour, collide=collide))
    out.append(Piece("sphere", (r * 1.15,) * 3, (0.0, 0.0, PIPE_Z), color=colour, collide=collide))
    return out


def fitting_object(rc) -> Part:
    """Disc, grey pipe, its water-filled twin (drawn, hidden until wet) and the hub knob."""
    shape, turns = LAYOUT[rc]
    x, y = cell_xy(rc)
    grey = _pipe_pieces(shape, STEEL, PIPE_R, True)
    wet = _pipe_pieces(shape, WATER, PIPE_R + 0.0006, False)
    # flat-sided T-handle across the hub (along the fitting's local y): the jaws close on its flats
    knob = Piece("box", (KNOB[0], KNOB[1], KNOB[2]), (0.0, 0.0, KNOB_Z), color=HUB, friction=(1.4, 1.2))
    return Part(name(rc), pos=(x, y, DISC_Z), size=(DISC_R, DISC_T, DISC_R), shape="cylinder", exact_z=True,
                mass=0.090, color=(0.16, 0.17, 0.19), friction=(0.6, 0.5), yaw=turns * math.pi / 2,
                com=(0.0, 0.0, 0.0), hinge=Hinge(pivot=(x, y, DISC_Z), axis="Z", damping=0.03),
                pieces=tuple(grey + wet + [knob]))


def wet_piece_range(rc):
    n = len(SHAPES[LAYOUT[rc][0]]) + 1
    return range(n, 2 * n)


def board_object() -> Part:
    """The panel the fittings turn on, with the fixed supply line coming in and the outlet line going out."""
    cx = X0 + (ROWS - 1) * PITCH / 2
    cy = Y0 - (COLS - 1) * PITCH / 2
    lx, ly = ROWS * PITCH + 0.016, COLS * PITCH + 0.016
    p = []
    # cell outlines, drawn
    for k in range(ROWS + 1):
        p.append(Piece("box", (0.0015, ly - 0.01, 0.0006), (-ROWS * PITCH / 2 + k * PITCH, 0.0, BOARD_T / 2 + 0.0003),
                       color=(0.35, 0.38, 0.42), collide=False))
    for k in range(COLS + 1):
        p.append(Piece("box", (lx - 0.01, 0.0015, 0.0006), (0.0, -COLS * PITCH / 2 + k * PITCH, BOARD_T / 2 + 0.0003),
                       color=(0.35, 0.38, 0.42), collide=False))
    pz = DISC_Z + PIPE_Z - (TOP + BOARD_T / 2)
    # supply: a full pipe from off the panel's left edge to the edge of its cell, with a red valve wheel
    sx, sy = cell_xy(SUPPLY_CELL)
    y_in, y_out = sy + PITCH / 2 + 0.002, sy + PITCH / 2 + 0.090
    p.append(Piece("cylinder", (PIPE_R + 0.0006, y_out - y_in, PIPE_R + 0.0006), (sx - cx, (y_in + y_out) / 2 - cy, pz),
                   axis="Y", color=WATER, collide=False))
    p.append(Piece("cylinder", (0.022, 0.005, 0.022), (sx - cx, y_in + 0.045 - cy, pz), axis="Y", color=SUPPLY,
                   collide=False))
    # outlet: a grey pipe from its cell's right edge off the panel
    ox, oy = cell_xy(OUTLET_CELL)
    y_a, y_b = oy - PITCH / 2 - 0.002, oy - PITCH / 2 - 0.090
    p.append(Piece("cylinder", (PIPE_R, y_a - y_b, PIPE_R), (ox - cx, (y_a + y_b) / 2 - cy, pz), axis="Y", color=STEEL,
                   collide=False))
    p.append(Piece("cylinder", (PIPE_R + 0.0006, y_a - y_b, PIPE_R + 0.0006), (ox - cx, (y_a + y_b) / 2 - cy, pz),
                   axis="Y", color=WATER, collide=False))      # outlet_wet: shown once coolant gets through
    return Part("panel", pos=(cx, cy, TOP + BOARD_T / 2), size=(lx, ly, BOARD_T), exact_z=True, color=PANEL,
                friction=(0.5, 0.4), pieces=tuple(p))


OUTLET_WET_PIECE = (ROWS + 1) + (COLS + 1) + 3     # board pieces: grid lines, supply, wheel, outlet, outlet wet


# -- measurements ----------------------------------------------------------------------------

def yaw_of(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quarter_turns(yaw, tol_deg=15.0):
    """The fitting's quarter-turn position, or None if it is left between positions."""
    k = yaw / (math.pi / 2)
    n = round(k)
    if abs(k - n) * 90 > tol_deg:
        return None
    return int(n) % 4


def open_ends(rc, turns):
    return {(d + turns) % 4 for d in SHAPES[LAYOUT[rc][0]]}


def flow(turns_by_cell):
    """Coolant through the panel. Returns (wet cells, reached the outlet, open ends on wet fittings)."""
    t0 = turns_by_cell[SUPPLY_CELL]
    if t0 is None or SUPPLY_DIR not in open_ends(SUPPLY_CELL, t0):
        return set(), False, 0
    wet, todo, leaks, out = {SUPPLY_CELL}, [SUPPLY_CELL], 0, False
    while todo:
        rc = todo.pop()
        for d in open_ends(rc, turns_by_cell[rc]):
            if rc == SUPPLY_CELL and d == SUPPLY_DIR:
                continue
            if rc == OUTLET_CELL and d == OUTLET_DIR:
                out = True
                continue
            nb = (rc[0] + DIRS[d][0], rc[1] - DIRS[d][1])
            t = turns_by_cell.get(nb)
            if nb not in LAYOUT or t is None or (d + 2) % 4 not in open_ends(nb, t):
                leaks += 1
                continue
            if nb not in wet:
                wet.add(nb)
                todo.append(nb)
    return wet, out, leaks
