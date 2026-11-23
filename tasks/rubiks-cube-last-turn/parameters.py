"""Geometry and measurements for finishing a Rubik's cube with its last turn.

A standard 3x3 speedcube (57 mm, about 90 g) sits on a speedcubing mat, one move from solved: its top
layer is a quarter turn out. Every side face shows its two lower rows in one colour and its top row in
the colour of the next face round, so the only move that solves it is the top layer a quarter turn
anticlockwise seen from above (U' in cube notation). The layer has to end square to the rest of the cube;
a layer left part-way round blocks every other face from turning.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

BODY_COL = (0.06, 0.06, 0.07)          # black plastic
MAT_COL = (0.10, 0.22, 0.45)           # blue speedcubing mat
STICKER = {"white": (0.95, 0.95, 0.93), "yellow": (0.98, 0.85, 0.10), "red": (0.80, 0.10, 0.10),
           "orange": (0.98, 0.45, 0.05), "blue": (0.10, 0.30, 0.85), "green": (0.10, 0.65, 0.25)}
FACE_COLOUR = {(1, 0): "red", (-1, 0): "orange", (0, 1): "blue", (0, -1): "green"}   # solved side faces

# -- mat (fixed) ------------------------------------------------------------------------------------
MAT = (0.200, 0.260, 0.003)            # a speedcubing mat, 3 mm rubber
MAT_FRICTION = (1.0, 0.9)              # rubber-backed cloth

# -- the cube ----------------------------------------------------------------------------------------
S = 0.057                              # 57 mm, the standard 3x3
CELL = S / 3
CX, CY = 0.280, -0.090                 # the cube's centre on the mat
BASE_H = 2 * CELL                      # the lower two layers (one rigid body here: they are not turned)
LAYER_H = CELL - 0.0004                # the top layer, 0.4 mm gap to the middle layer
BASE_MASS, LAYER_MASS = 0.060, 0.030
CUBE_FRICTION = (0.6, 0.5)             # matte plastic and stickers on the mat
START_DEG = -90.0                      # the top layer starts a quarter turn clockwise of solved
TURN_DAMP = 0.0006                     # N m s / rad: a broken-in cube's lubricated core
TURN_FRICTION = 0.004                  # N m: it stays where it is left
STK, STK_T = CELL - 0.003, 0.0006      # sticker size and thickness


def _w(x, y, z):
    return (float(x), float(y), float(TOP + z))


def mat_object() -> Part:
    return Part("mat", pos=_w(CX - 0.020, CY, MAT[2] / 2), size=MAT, exact_z=True, mass=None, color=MAT_COL,
                friction=MAT_FRICTION, restitution=0.0)


def _stickers(rows, colour_of, z0):
    """Stickers on the four side faces for the given rows (row centres above z0), each face's colour by
    colour_of(face normal)."""
    out = []
    for (nx, ny) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        col = STICKER[colour_of((nx, ny))]
        for r in rows:
            for k in (-1, 0, 1):
                along = k * CELL
                pos = (nx * (S / 2 + STK_T / 2) + (0 if nx else along),
                       ny * (S / 2 + STK_T / 2) + (0 if ny else along), z0 + r)
                size = (STK_T, STK, STK) if nx else (STK, STK_T, STK)
                out.append(Piece("box", size, pos, color=col, collide=False))
    return out


BASE_Z = MAT[2] + BASE_H / 2           # the base's centre above the bench
LAYER_Z = MAT[2] + BASE_H + 0.0004 + LAYER_H / 2


def base_object() -> Part:
    """The lower two layers: a free body standing on the mat, solved colours on its sides, white below."""
    st = _stickers((-CELL / 2, CELL / 2), lambda n: FACE_COLOUR[n], 0.0)
    return Part("cube", pos=_w(CX, CY, BASE_Z), size=(S, S, BASE_H), exact_z=True, mass=BASE_MASS,
                color=BODY_COL, friction=CUBE_FRICTION, restitution=0.0, contact_offset=0.0015,
                pieces=tuple(st))


def _rot_n(n, deg):
    a = math.radians(deg)
    x, y = n
    return (int(round(x * math.cos(a) - y * math.sin(a))), int(round(x * math.sin(a) + y * math.cos(a))))


def layer_object() -> Part:
    """The top layer on the core's vertical axis. Its stickers are those of the solved cube turned with it:
    it spawns START_DEG from solved, so each side shows the next face's colour, and yellow on top."""
    # a sticker on the layer's local face n ends on world face rot(n, -START_DEG) when the layer is solved
    st = _stickers((0.0,), lambda n: FACE_COLOUR[_rot_n(n, -START_DEG)], 0.0)
    top = [Piece("box", (STK, STK, STK_T), (i * CELL, j * CELL, LAYER_H / 2 + STK_T / 2),
                 color=STICKER["yellow"], collide=False) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    return Part("top_layer", pos=_w(CX, CY, LAYER_Z), size=(S, S, LAYER_H), exact_z=True, mass=LAYER_MASS,
                color=BODY_COL, friction=CUBE_FRICTION, restitution=0.0, contact_offset=0.0015,
                pieces=tuple(st + top),
                hinge=Hinge(pivot=_w(CX, CY, LAYER_Z), axis="Z", damping=TURN_DAMP, friction=TURN_FRICTION,
                            lower_deg=-400.0, upper_deg=400.0, parent="cube", parent_pos=_w(CX, CY, BASE_Z)))


def quat_matrix(q):
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.atan2(r[1, 0], r[0, 0]))


def tilt_deg(pose):
    r = quat_matrix(np.asarray(pose)[3:7])
    return math.degrees(math.acos(max(-1.0, min(1.0, r[2, 2]))))
