"""Geometry and measurements for closing a large kitchen box of matches.

A kitchen matchbox (the 250-count household size) lies flat on the counter with its tray pulled half out
toward the right. The empty-ish sleeve weighs 17 g: pushed on its own, the tray just drags the sleeve along
the counter. One hand has to pin the sleeve down while the other pushes the tray home square, until its end
is flush with the sleeve's end.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The box's length runs along y.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece

TOP = 0.7475

BOARD = (0.86, 0.80, 0.66)             # sleeve: printed board
STRIKER = (0.35, 0.22, 0.18)           # brown striking strips on the long sides
TRAY_COL = (0.82, 0.70, 0.50)
MATCH = (0.85, 0.75, 0.55)
HEAD = (0.70, 0.12, 0.10)

MX, MY = 0.300, -0.050                  # the sleeve's centre
SL = (0.058, 0.110, 0.027)             # sleeve outside: width (x), length (y), height
T = 0.0015                             # board thickness (stand-in: real board 0.8-1 mm)
PLAY = 0.0006                          # tray to sleeve clearance, each side
TL = SL[1] - 0.002                     # tray length
TW = SL[0] - 2 * T - 2 * PLAY          # tray outside width
TH = SL[2] - 2 * T - 0.0010            # tray outside height
OUT = 0.050                            # start: the tray 50 mm out of the sleeve's right (-y) end
SLEEVE_MASS, TRAY_MASS = 0.017, 0.055   # tray with about 120 matches
FRICTION_BOARD = (0.45, 0.38)
FRICTION_BENCH = (0.40, 0.33)


def sleeve_object() -> Part:
    """The sleeve: a four-walled board tube open at both ends (root: its floor)."""
    h, w = SL[2], SL[0]
    pieces = (Piece("box", (w, SL[1], T), (0.0, 0.0, h - T), friction=FRICTION_BOARD),
              Piece("box", (T, SL[1], h - 2 * T), (w / 2 - T / 2, 0.0, h / 2 - T / 2), friction=FRICTION_BOARD),
              Piece("box", (T, SL[1], h - 2 * T), (-w / 2 + T / 2, 0.0, h / 2 - T / 2), friction=FRICTION_BOARD),
              Piece("box", (0.0006, SL[1] - 0.01, 0.018), (w / 2 + 0.0003, 0.0, h / 2 - T / 2), color=STRIKER,
                    collide=False),
              Piece("box", (0.0006, SL[1] - 0.01, 0.018), (-w / 2 - 0.0003, 0.0, h / 2 - T / 2), color=STRIKER,
                    collide=False))
    return Part("sleeve", pos=(MX, MY, TOP + T / 2), size=(w, SL[1], T), exact_z=True, mass=SLEEVE_MASS,
                color=BOARD, friction=FRICTION_BENCH, restitution=0.0, contact_offset=0.0015, com=(0.0, 0.0, h / 2),
                pieces=pieces)


def tray_object() -> Part:
    """The tray: floor, two long sides, two ends, and the matches as one block (root: its floor)."""
    z0 = T + 0.0005                       # the tray rides on the sleeve's floor
    pieces = (Piece("box", (T, TL, TH - T), (TW / 2 - T / 2, 0.0, TH / 2), friction=FRICTION_BOARD),
              Piece("box", (T, TL, TH - T), (-TW / 2 + T / 2, 0.0, TH / 2), friction=FRICTION_BOARD),
              Piece("box", (TW - 2 * T, 0.002, TH - T), (0.0, TL / 2 - 0.001, TH / 2), friction=FRICTION_BOARD),
              Piece("box", (TW - 2 * T, 0.002, TH - T), (0.0, -TL / 2 + 0.001, TH / 2), friction=FRICTION_BOARD),
              Piece("box", (TW - 2 * T - 0.001, TL - 0.006, 0.010), (0.0, 0.0, T / 2 + 0.005), color=MATCH, collide=False),
              Piece("box", (TW - 2 * T - 0.001, 0.010, 0.0035), (0.0, TL / 2 - 0.009, T / 2 + 0.010), color=HEAD,
                    collide=False))
    return Part("tray", pos=(MX, MY - OUT, TOP + z0 + T / 2), size=(TW, TL, T), exact_z=True, mass=TRAY_MASS,
                color=TRAY_COL, friction=FRICTION_BOARD, restitution=0.0, contact_offset=0.0015,
                com=(0.0, 0.0, 0.006), pieces=pieces)


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


def tray_out(sleeve, tray):
    """How far the tray's right (-y) end stands out of the sleeve's right end, along the sleeve (m)."""
    s, t = np.asarray(sleeve, float), np.asarray(tray, float)
    r = quat_matrix(s[3:7])
    axis = r @ np.array([0.0, 1.0, 0.0])
    end_s = s[:3] - axis * SL[1] / 2
    end_t = t[:3] - (quat_matrix(t[3:7]) @ np.array([0.0, 1.0, 0.0])) * TL / 2
    return float(np.dot(end_s - end_t, axis)) - (SL[1] - TL) / 2
