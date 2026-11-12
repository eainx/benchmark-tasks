"""Geometry and measurements for a coolant header with three crossing branch lines.

A supply header runs across the near side of the bench. Three quarter-turn ball valves come off it;
from each valve a grey branch line runs away from the arms, crossing over and under the others,
to a nozzle that drips coolant into the machine's trough. One line has a cracked joint spraying
coolant onto an electrical control box. Which valve feeds that line can only be found by tracing
the pipes: the branches swap sides on the way.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from simkit import Hinge, Part, Piece

TOP = 0.7475

PIPE = (0.62, 0.64, 0.67)
LEVER = (0.85, 0.10, 0.08)          # red ball-valve levers
BRASS = (0.70, 0.55, 0.25)
BOX = (0.55, 0.57, 0.52)
STRIPE = (0.95, 0.80, 0.10)
COOLANT = (0.30, 0.90, 0.70)
TROUGH_GREY = (0.30, 0.32, 0.34)

PIPE_R = 0.009
HEADER_X = 0.190
VALVE_X = 0.255
VALVE_Y = (0.110, 0.000, -0.110)            # valves A, B, C from left to right
VALVE_Z = TOP + 0.030                       # pipe axis through the valves
STEM_H = 0.032
LEVER_L = 0.075
LEVER_W = (0.016, 0.012)                    # bar width, height
LEVER_Z = VALVE_Z + STEM_H
GRIP_AT = 0.060                             # the lever's grip pin, along the lever from the stem
PIN = (0.008, 0.030)                        # radius, height of the round pin standing on the lever's end
CLOSED_DEG = 90.0

OUT_X = 0.500                               # nozzles over the trough
OUT_Y = (-0.110, 0.110, 0.000)              # where each valve's line ends: they swap sides
CROSS_X = (0.360, 0.330, 0.395)             # where each line runs across
LINE_Z = (TOP + 0.022, TOP + 0.042, TOP + 0.062)   # each line at its own height, so they pass over each other
LEAK_LINE = 0                               # valve A's line is the cracked one
LEAK_AT = (0.425, OUT_Y[0])                 # the cracked joint
CONTROL_BOX_XY = (0.425, -0.215)
CONTROL_BOX = (0.090, 0.075, 0.120)

TROUGH_X = 0.500
TOGGLES = {}                                # drawn flow pieces (index in the piping's pieces), shown by task.Dynamics
TROUGH_SIZE = (0.050, 0.320, 0.030)

DROP_R = 0.0035
DROP_MASS = 0.00007
DROPS = 96
PUDDLE_R = (0.025, 0.045, 0.065)             # drawn puddle sizes
PUDDLE_AT = (6, 25, 60)                      # spray drops landed on or by the panel before each is shown
NAMES = ("lever_a", "lever_b", "lever_c")


def _seg(a, b, r, colour, collide=False):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    yaw = math.atan2(d[1], d[0])
    pitch = math.atan2(math.hypot(d[0], d[1]), d[2])
    c = (a + b) / 2
    return Piece("cylinder", (r, L, r), tuple(float(v) for v in c), rot=(0.0, pitch, yaw), color=colour, collide=collide)


def line_points(k):
    """Polyline of branch k from its valve to its nozzle, in the world."""
    y0, y1, xc, z = VALVE_Y[k], OUT_Y[k], CROSS_X[k], LINE_Z[k]
    return [(VALVE_X + 0.028, y0, VALVE_Z), (VALVE_X + 0.045, y0, VALVE_Z), (VALVE_X + 0.045, y0, z),
            (xc, y0, z), (xc, y1, z), (OUT_X, y1, z), (OUT_X, y1, z - 0.012)]


def piping_object() -> Part:
    """Header, valve bodies and stems, the three branch lines, nozzles, the trough and the control
    box -- all fixed. Pieces are given in world coordinates and shifted onto the root below."""
    root = np.array([HEADER_X, 0.0, TOP + 0.004])
    w = []
    # header across the near side, supply coming in from the left edge
    w.append(_seg((HEADER_X, 0.30, VALVE_Z), (HEADER_X, -0.16, VALVE_Z), PIPE_R * 1.3, PIPE))
    for k, y in enumerate(VALVE_Y):
        w.append(_seg((HEADER_X, y, VALVE_Z), (VALVE_X - 0.020, y, VALVE_Z), PIPE_R, PIPE))
        w.append(Piece("box", (0.040, 0.030, 0.030), (VALVE_X, y, VALVE_Z), color=BRASS))
        # the stem stops short of the lever's hub, so the turning lever touches nothing fixed
        w.append(Piece("cylinder", (0.005, STEM_H - 0.010, 0.005), (VALVE_X, y, VALVE_Z + (STEM_H - 0.010) / 2), color=BRASS,
                       collide=False))
        pts = line_points(k)
        for a, b in zip(pts, pts[1:]):
            w.append(_seg(a, b, PIPE_R, PIPE))
        for p in pts[1:-1]:
            w.append(Piece("sphere", (PIPE_R * 1.1,) * 3, p, color=PIPE, collide=False))
        w.append(Piece("cylinder", (0.006, 0.014, 0.006), (OUT_X, OUT_Y[k], LINE_Z[k] - 0.019), color=BRASS, collide=False))
    # the cracked joint: a coupling with a split in it
    lx, ly = LEAK_AT
    w.append(Piece("cylinder", (PIPE_R * 1.35, 0.022, PIPE_R * 1.35), (lx, ly, LINE_Z[LEAK_LINE]), rot=(0.0, math.pi / 2, 0.0),
                   color=(0.45, 0.42, 0.38), collide=False))
    # electrical control panel beside it: a steel enclosure with a door, a lightning-bolt warning
    # sign on its lid, start/stop buttons, a lamp and a cable conduit running off the bench
    bx, by = CONTROL_BOX_XY
    cx_, cy_, cz_ = CONTROL_BOX
    top_z = TOP + cz_
    w.append(Piece("box", CONTROL_BOX, (bx, by, TOP + cz_ / 2), color=BOX))
    w.append(Piece("box", (cx_ - 0.012, 0.002, cz_ - 0.016), (bx, by + cy_ / 2 + 0.001, TOP + cz_ / 2), color=(0.66, 0.68, 0.64),
                   collide=False))                                        # door, facing the lines
    w.append(Piece("box", (0.006, 0.008, 0.024), (bx - cx_ / 2 + 0.014, by + cy_ / 2 + 0.005, TOP + cz_ / 2),
                   color=(0.15, 0.15, 0.16), collide=False))              # door handle
    w.append(Piece("box", (0.046, 0.046, 0.002), (bx - 0.012, by, top_z + 0.001), rot=(0.0, 0.0, math.pi / 4),
                   color=STRIPE, collide=False))                          # warning sign (yellow diamond)
    for dx, dy, a in ((-0.004, 0.006, 0.5), (0.004, -0.006, 0.5)):        # lightning bolt
        w.append(Piece("box", (0.004, 0.020, 0.001), (bx - 0.012 + dx, by + dy, top_z + 0.0025), rot=(0.0, 0.0, a),
                       color=(0.08, 0.08, 0.08), collide=False))
    w.append(Piece("cylinder", (0.008, 0.006, 0.008), (bx + 0.026, by + 0.018, top_z + 0.003), color=(0.15, 0.75, 0.2),
                   collide=False))                                        # start
    w.append(Piece("cylinder", (0.008, 0.006, 0.008), (bx + 0.026, by - 0.006, top_z + 0.003), color=(0.85, 0.12, 0.10),
                   collide=False))                                        # stop
    w.append(Piece("sphere", (0.007, 0.007, 0.007), (bx + 0.026, by - 0.026, top_z + 0.002), color=(0.95, 0.75, 0.1),
                   collide=False))                                        # lamp
    w.append(_seg((bx, by - cy_ / 2, TOP + 0.012), (bx, -0.42, TOP + 0.012), 0.010, (0.08, 0.08, 0.09)))   # conduit
    # machine trough under the nozzles
    tx = TROUGH_X
    lx_, ly_, h = TROUGH_SIZE
    for s in (-1, 1):
        w.append(Piece("box", (0.004, ly_, h), (tx + s * (lx_ / 2 - 0.002), 0.0, TOP + h / 2), color=TROUGH_GREY))
        w.append(Piece("box", (lx_, 0.004, h), (tx, s * (ly_ / 2 - 0.002), TOP + h / 2), color=TROUGH_GREY))
    w.append(Piece("box", (lx_, ly_, 0.004), (tx, 0.0, TOP + 0.002), color=TROUGH_GREY))
    # drawn coolant, shown only while it flows: a stream from each nozzle into the trough, the jet from
    # the cracked joint onto the control box, and the wet sheen it leaves on the box's face
    for k in range(3):
        top = LINE_Z[k] - 0.026
        TOGGLES[f"stream{k}"] = len(w)
        w.append(Piece("cylinder", (0.0035, top - (TOP + 0.004), 0.0035), (OUT_X, OUT_Y[k], (top + TOP + 0.004) / 2),
                       color=COOLANT, opacity=0.65, collide=False))
    face_y = CONTROL_BOX_XY[1] + CONTROL_BOX[1] / 2
    # coolant pooling on the bench at the panel's foot, drawn bigger as more spray lands
    for j, r in enumerate(PUDDLE_R):
        TOGGLES[f"puddle{j}"] = len(w)
        w.append(Piece("cylinder", (r, 0.0012, r), (bx - 0.005, face_y + r * 0.55, TOP + 0.0006 + 0.0002 * j), color=COOLANT,
                       opacity=0.6, collide=False))
    TOGGLES["sheen"] = len(w)
    w.append(Piece("box", (CONTROL_BOX[0] * 0.9, 0.002, CONTROL_BOX[2] * 0.8),
                   (bx, face_y + 0.001, TOP + CONTROL_BOX[2] * 0.45), color=COOLANT, opacity=0.7, collide=False))
    pieces = tuple(dataclasses.replace(p, offset=tuple(float(v) for v in np.asarray(p.offset) - root)) for p in w)
    return Part("piping", pos=tuple(root), size=(0.02, 0.02, 0.008), exact_z=True, color=PIPE, friction=(0.5, 0.4),
                pieces=pieces)


def lever_object(k) -> Part:
    """Ball-valve lever on its stem: open along the line (+x), closed a quarter turn to the left."""
    y = VALVE_Y[k]
    bar = Piece("box", (LEVER_L, LEVER_W[0], LEVER_W[1]), (LEVER_L / 2 - 0.006, 0.0, 0.0), color=LEVER, friction=(1.4, 1.2))
    pin = Piece("cylinder", (PIN[0], PIN[1], PIN[0]), (GRIP_AT, 0.0, LEVER_W[1] / 2 + PIN[1] / 2), color=(0.15, 0.15, 0.16),
                friction=(1.2, 1.0))
    return Part(NAMES[k], pos=(VALVE_X, y, LEVER_Z), size=(0.009, 0.010, 0.009), shape="cylinder", exact_z=True,
                mass=0.030, color=BRASS, friction=(0.6, 0.5), com=(0.0, 0.0, 0.0),
                hinge=Hinge(pivot=(VALVE_X, y, LEVER_Z), axis="Z", damping=0.03, lower_deg=-1.0, upper_deg=CLOSED_DEG + 1.0),
                pieces=(bar, pin))


def drop_objects():
    return [Part(f"drop{i:02d}", pos=parking(i), size=(DROP_R,) * 3, shape="sphere", exact_z=True, mass=DROP_MASS,
                 color=COOLANT, opacity=0.9, friction=(0.2, 0.1), restitution=0.0, contact_offset=0.001,
                 max_depenetration=0.3) for i in range(DROPS)]


def parking(i):
    return (0.10 + 0.012 * (i % 16), -0.55 - 0.012 * (i // 16), TOP - 0.30)


DROP_NAMES = tuple(f"drop{i:02d}" for i in range(DROPS))


# -- measurements ----------------------------------------------------------------------------

def yaw_of(q):
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def lever_deg(q):
    return math.degrees(yaw_of(q))


def flowing(deg):
    """A ball valve passes coolant until it is most of the way shut."""
    return deg < 70.0


def grip_point(k, deg):
    a = math.radians(deg)
    return np.array([VALVE_X + GRIP_AT * math.cos(a), VALVE_Y[k] + GRIP_AT * math.sin(a), LEVER_Z + LEVER_W[1] / 2 + PIN[1] / 2])


def on_box(p) -> bool:
    """On the control panel, or at its foot on the side facing the leak."""
    bx, by = CONTROL_BOX_XY
    return bool(abs(p[0] - bx) < CONTROL_BOX[0] / 2 + 0.010 and by - CONTROL_BOX[1] / 2 - 0.006 < p[1] < by + CONTROL_BOX[1] / 2 + 0.045
                and p[2] < TOP + CONTROL_BOX[2] + 0.006)
