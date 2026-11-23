"""Stateful scorer for setting the router-table fence: the fence face's distance from the bit centre and
its parallelism from the fence's measured pose, the knobs' clamp state from task.Dynamics (their measured
turn), the hands from their measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
CLEAR = 0.030


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None

    def _clear(self, tcp, poses):
        """A hand is clear when its TCP is at least 30 mm from both knobs and the fence's top edge, or
        60 mm above the fence."""
        tcp = np.asarray(tcp, float)[:3]
        if tcp[2] > B.TOP + B.F_H + 0.060:
            return True
        for s in ("left", "right"):
            k = np.asarray(poses[f"knob_{s}"], float)[:3]
            if math.dist(tcp[:2], k[:2]) < B.LEVER_R + 0.010 + CLEAR:
                return False
        fx = B.face_x_at(float(tcp[1]), poses["fence"])
        near = -0.010 - CLEAR < tcp[0] - fx < B.BASE_D + CLEAR and abs(tcp[1]) < B.F_LEN / 2 + CLEAR
        return not (near and tcp[2] < B.TOP + B.F_H + CLEAR)

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("clamp state missing")
        self.step = step
        pose = root_poses["fence"]
        d = B.distance_mm(pose)
        par = B.parallel_mm(pose)
        if self.failed is None and d < 1000 * B.BIT_R + 1.0:
            self.failed = "fence on the bit"
        locked = all(mechanics["locked"].values())
        clear = {s: self._clear(robot[s]["tcp"], root_poses) for s in ("left", "right")}
        err = abs(d - 1000 * B.TARGET)
        score = 0.0
        if err <= 3.0:
            score = 0.25
        if err <= 1.0 and par <= 1.0:
            score = 0.5
        if err <= 0.5 and par <= 0.5:
            score = 0.75
        full = score == 0.75 and locked and all(clear.values())
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "distance_mm": round(d, 3), "parallel_mm": round(par, 3),
                "end_left_mm": round(B.distance_mm(pose, B.END_Y), 3), "end_right_mm": round(B.distance_mm(pose, -B.END_Y), 3),
                "locked": mechanics["locked"], "knob_turn_deg": mechanics["turn_deg"], "hands_clear": clear,
                "failed": self.failed, "held_frames": self.held}
