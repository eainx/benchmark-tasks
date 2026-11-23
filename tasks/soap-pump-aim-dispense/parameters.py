"""Geometry and measurements for dispensing one pump of dish soap from a pump bottle onto a sponge.

A 250 ml pump bottle of dish soap stands on the counter, its pump head's spout turned toward the back.
A sponge lies against the bottle's right side. The spout has to be turned round on its swivel until it
overhangs the sponge, then the head pressed through one full stroke against its spring and let come back
up: one dose, on the sponge, not on the counter.

Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Hinge, Part, Piece, Spring

TOP = 0.7475

PET = (0.30, 0.62, 0.38)               # green dish-soap bottle
WHITE = (0.93, 0.93, 0.92)
SPONGE_COL = (0.95, 0.80, 0.20)
SCOUR = (0.20, 0.45, 0.25)

# -- the bottle (free) ---------------------------------------------------------------------------------
BX, BY = 0.250, -0.080
B_R, B_H = 0.032, 0.115                # body: a 250 ml Boston-round pump bottle
NECK_R, NECK_H = 0.014, 0.018          # neck and pump collar
B_MASS = 0.30                          # 250 ml, nearly full

# -- the pump (stem on a sprung slide in the bottle, head on a swivel on the stem) -------------------
STROKE = 0.016                         # full stroke
FULL = 0.013                           # a dose counts from here down
RETURNED = 0.004                       # the head counts as back up above this
PUMP_K = 220.0                         # N/m: a light dish-soap pump
PRELOAD = 0.012                        # spring target above the top stop: 2.6 N at rest, 6.2 N at the bottom
HEAD_R, HEAD_H = 0.017, 0.016
SPOUT_L, SPOUT_W = 0.046, 0.009        # axis to the spout's tip, and its width/height
HEAD_Z = TOP + B_H + NECK_H + 0.019 + HEAD_H / 2   # 19 mm of stem above the collar: the full stroke clears it
SPOUT0_DEG = 0.0                       # the spout starts pointing straight back (+x)
STEM_Z = TOP + B_H + NECK_H - 0.006

# -- the sponge (free, light) ------------------------------------------------------------------------
SP = (0.095, 0.065, 0.028)             # x, y, z
SPX, SPY = BX, BY - B_R - SP[1] / 2 - 0.002


def bottle_object() -> Part:
    zc = TOP + B_H / 2
    neck = Piece("cylinder", (NECK_R, NECK_H, NECK_R), (0.0, 0.0, B_H / 2 + NECK_H / 2), color=WHITE)
    label = Piece("cylinder", (B_R + 0.0005, 0.060, B_R + 0.0005), (0.0, 0.0, 0.0), color=(0.92, 0.92, 0.85),
                  collide=False)
    return Part("bottle", pos=(BX, BY, zc), size=(B_R, B_H, B_R), shape="cylinder", exact_z=True, mass=B_MASS,
                color=PET, friction=(0.50, 0.40), restitution=0.0, pieces=(neck, label))


def stem_object() -> Part:
    return Part("stem", pos=(BX, BY, STEM_Z), size=(0.005, 0.020, 0.005), shape="cylinder", exact_z=True, mass=0.05,
                color=WHITE, collision="none",
                spring=Spring(axis="Z", lower=-STROKE, upper=0.0, stiffness=PUMP_K, damping=3.0, target=PRELOAD,
                              parent="bottle", parent_pos=(BX, BY, TOP + B_H / 2)))


def head_object() -> Part:
    spout = Piece("box", (SPOUT_L - HEAD_R + 0.004, SPOUT_W, SPOUT_W), ((SPOUT_L + HEAD_R - 0.004) / 2, 0.0, 0.002),
                  color=WHITE)
    nozzle = Piece("cylinder", (0.004, 0.010, 0.004), (SPOUT_L - 0.003, 0.0, -0.004), color=WHITE)
    return Part("head", pos=(BX, BY, HEAD_Z), size=(HEAD_R, HEAD_H, HEAD_R), shape="cylinder", exact_z=True, mass=0.04,
                color=WHITE, friction=(0.9, 0.8), restitution=0.0, pieces=(spout, nozzle),
                hinge=Hinge(pivot=(BX, BY, HEAD_Z), axis="Z", damping=0.0004, lower_deg=-200.0, upper_deg=200.0,
                            parent="stem", parent_pos=(BX, BY, STEM_Z)))


def sponge_object() -> Part:
    scour = Piece("box", (SP[0], SP[1], 0.008), (0.0, 0.0, SP[2] / 2 - 0.004), color=SCOUR)
    return Part("sponge", pos=(SPX, SPY, TOP + SP[2] / 2), size=SP, exact_z=True, mass=0.015, color=SPONGE_COL,
                friction=(0.9, 0.8), restitution=0.0, pieces=(scour,))


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


def spout_tip(head_pose):
    p = np.asarray(head_pose, float)
    return p[:3] + quat_matrix(p[3:7]) @ np.array([SPOUT_L - 0.003, 0.0, -0.009])


def over_sponge(xy, sponge_pose, margin=0.005):
    s = np.asarray(sponge_pose, float)
    r = quat_matrix(s[3:7])
    d = np.asarray(xy, float)[:2] - s[:2]
    lx, ly = r[0, 0] * d[0] + r[1, 0] * d[1], r[0, 1] * d[0] + r[1, 1] * d[1]
    return bool(abs(lx) <= SP[0] / 2 - margin and abs(ly) <= SP[1] / 2 - margin)
