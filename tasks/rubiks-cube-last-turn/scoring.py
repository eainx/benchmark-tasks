"""Stateful scorer for finishing the Rubik's cube: the top layer's turn relative to the lower layers, from
their measured poses (unwrapped step by step); the cube's tilt and travel on the mat; the hand's measured
TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
SOLVED_DEG = 90.0                      # a quarter turn anticlockwise from the start solves it
SQUARE_DEG = 3.0                       # left square: the other faces can turn again
NEAR_DEG = 15.0
STARTED_DEG = 60.0
MOVE_MAX = 0.030


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.cube0 = None
        self.turn = 0.0
        self.last = None

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        cube = np.asarray(root_poses["cube"], float)
        layer = np.asarray(root_poses["top_layer"], float)
        if self.cube0 is None:
            self.cube0 = cube.copy()
        rel = _wrap(B.yaw_deg(layer) - B.yaw_deg(cube))
        if self.last is not None:
            self.turn += _wrap(rel - self.last)
        self.last = rel
        off = _wrap(self.turn - SOLVED_DEG)            # three turns the other way solve it too
        tilt = B.tilt_deg(cube)
        moved = float(np.linalg.norm(cube[:2] - self.cube0[:2]))
        if self.failed is None and (tilt > 30.0 or cube[2] < B.TOP):
            self.failed = "cube knocked over or off the mat"
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:2], cube[:2]) > 0.08 or
                    float(np.asarray(robot[s]["tcp"])[2]) > B.TOP + B.LAYER_Z + 0.08 for s in ("left", "right"))
        score = 0.0
        if abs(off) <= STARTED_DEG:
            score = 0.25
        if abs(off) <= NEAR_DEG:
            score = 0.5
        if abs(off) <= SQUARE_DEG and tilt <= 5.0 and moved <= MOVE_MAX:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "turn_deg": round(self.turn, 2),
                "off_solved_deg": round(off, 2), "cube_tilt_deg": round(tilt, 2),
                "cube_moved_mm": round(1000 * moved, 1), "hands_clear": bool(clear), "failed": self.failed,
                "held_frames": self.held}
