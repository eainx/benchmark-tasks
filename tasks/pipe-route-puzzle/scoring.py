"""Stateful scorer for the coolant manifold. Reads the flow measured by task.Dynamics and the hands."""
from __future__ import annotations

import math

import numpy as np

import pipes as P

HZ = 20
HOLD_S = 1.0


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("flow measurements missing")
        self.step = step
        through, leaks, wet = mechanics["through"], mechanics["open_ends"], len(mechanics["wet"])
        score = 0.0
        if wet >= 3:
            score = 0.25
        if through:
            score = 0.5
        if through and leaks == 0:
            score = 0.75
        clear = all(min(math.dist(np.asarray(robot[s]["tcp"])[:2], P.cell_xy(rc)) for rc in P.CELLS) > 0.06
                    or float(np.asarray(robot[s]["tcp"])[2]) > P.DISC_Z + 0.15 for s in ("left", "right"))
        self.held = self.held + 1 if score == 0.75 and clear else 0
        if self.held >= HOLD_S * HZ:
            score = 1.0
        return {"score": score, "success": score == 1.0, "wet_fittings": wet, "through": through,
                "open_ends": leaks, "hands_clear": bool(clear), "held_frames": self.held}
