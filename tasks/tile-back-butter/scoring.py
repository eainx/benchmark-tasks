"""Stateful scorer for "Replace the missing tile on the step." From the adhesive cells and bedding
reported by task.Dynamics, the tile's and trowel's measured poses and the robot's measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import tile as T

HZ = 20
HOLD_S = 1.0
HALF = 0.5
FULL = 0.8                            # back-buttered over at least 80 % of the back (ANSI A108.5, dry interiors)
SET_MM = 3.0                          # centre within this of the gap's centre (2 mm joints round it)
YAW_DEG = 3.0
LIP_MM = 1.0                          # lippage against the neighbours' face


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.hollow = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("adhesive measurements missing")
        self.step = step
        m = mechanics
        cov = m["coverage"]
        off, yaw, lip, face_up = T.tile_set(root_poses["tile"])
        placed = face_up and off * 1000 <= SET_MM and yaw <= YAW_DEG and lip * 1000 <= LIP_MM
        if placed and cov < FULL:
            self.hollow = True                 # set with bare patches under it: a hollow tile
        handle = T.handle_point(root_poses["trowel"])
        put_down = all(math.dist(np.asarray(robot[s]["tcp"])[:3], handle) > 0.06 or float(np.mean(robot[s]["fingers"])) < -0.040
                       for s in ("left", "right"))
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:3], T.GAP) > 0.08 for s in ("left", "right"))
        score = 0.0
        if cov >= HALF:
            score = 0.25
        if cov >= FULL:
            score = 0.5
        if cov >= FULL and placed:
            score = 0.75
        done = score == 0.75 and m["bedded"] and put_down and clear
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.hollow:
            score = min(score, 0.5)
        return {"score": score, "success": score == 1.0, "coverage": cov, "set_offset_mm": round(off * 1000, 1),
                "yaw_deg": round(yaw, 1), "lippage_mm": round(lip * 1000, 1), "face_up": bool(face_up),
                "placed": bool(placed), "bedded": bool(m["bedded"]), "hollow": self.hollow,
                "trowel_down": bool(put_down), "hands_clear": bool(clear), "held_frames": self.held}
