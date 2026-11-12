"""Stateful scorer for "Stop the coolant spraying on the control box." Lever angles and hands, and the
flow reported by task.Dynamics."""
from __future__ import annotations

import math

import numpy as np

import valves as V

HZ = 20
HOLD_S = 1.0


class TrialScorer:
    def __init__(self):
        self.wrong_shut = False
        self.held = 0
        self.step = 0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("flow measurements missing")
        self.step = step
        deg = mechanics["lever_deg"]
        k = V.LEAK_LINE
        others = [j for j in range(3) if j != k]
        if any(deg[j] > 45.0 for j in others):
            self.wrong_shut = True          # a running machine's coolant was shut off
        shut = not V.flowing(deg[k])          # the same 70-degree point at which the flow model stops
        running = all(deg[j] < 20.0 for j in others)
        clear = all(min(math.dist(np.asarray(robot[s]["tcp"])[:2], V.grip_point(j, deg[j])[:2]) for j in range(3)) > 0.06
                    for s in ("left", "right"))
        score = 0.0
        if deg[k] >= 30.0:
            score = 0.25
        if shut:
            score = 0.5
        if shut and running:
            score = 0.75
        self.held = self.held + 1 if score == 0.75 and clear else 0
        if self.held >= HOLD_S * HZ:
            score = 1.0
        if self.wrong_shut:
            score = 0.0                     # a running machine starved of coolant is a failure
        return {"score": score, "success": score == 1.0, "lever_deg": deg, "leak_shut": bool(shut),
                "other_lines_running": bool(running), "wrong_valve_shut": self.wrong_shut,
                "box_hits": mechanics["box_hits"], "hands_clear": bool(clear), "held_frames": self.held}
