"""Stateful scorer for fishing the cap out from under the toaster oven: how far the cap is inside or out
of the oven's footprint, from the measured cap and oven poses; the oven's shift and turn; the spoon's
measured pose and the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
OVEN_SHIFT, OVEN_TURN = 0.005, 2.0     # 1.0: oven still where it was
OVEN_FAIL_SHIFT, TILT_FAIL = 0.030, 5.0


def _to_spoon(p, spoon):
    """Distance from a point to the spoon's centre line (handle tip to the end of the bowl), m."""
    r = B.quat_matrix(spoon[3:7])
    axis = r @ np.array([0.0, 1.0, 0.0])
    s = float(np.clip(np.dot(p - spoon[:3], axis), -B.HANDLE[1] / 2 - B.BOWL[1], B.HANDLE[1] / 2))
    return float(np.linalg.norm(p - (spoon[:3] + s * axis)))


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.oven0 = None

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        oven = np.asarray(root_poses["oven"], float)
        cap = np.asarray(root_poses["cap"], float)
        spoon = np.asarray(root_poses["spoon"], float)
        if self.oven0 is None:
            self.oven0 = oven.copy()
        shift = float(np.linalg.norm(oven[:2] - self.oven0[:2]))
        turn = abs(B.yaw_deg(oven) - B.yaw_deg(self.oven0))
        if self.failed is None:
            if shift > OVEN_FAIL_SHIFT or B.tilt_deg(oven) > TILT_FAIL:
                self.failed = "oven shoved or tipped"
            elif cap[2] < B.TOP - 0.02:
                self.failed = "cap off the bench"
        depth = B.depth_under(cap[:2], oven)
        spoon_down = spoon[2] < B.TOP + 0.02
        free = all(_to_spoon(np.asarray(robot[s]["tcp"])[:3], spoon) > 0.05 or
                   float(np.mean(robot[s]["fingers"])) < -0.040 for s in ("left", "right"))
        clear = all(float(np.asarray(robot[s]["tcp"])[2]) > B.TOP + 0.06 or
                    B.depth_under(np.asarray(robot[s]["tcp"])[:2], oven) < -0.05 for s in ("left", "right"))
        score = 0.0
        if depth <= 0.010:
            score = 0.25
        if depth <= 0.0:
            score = 0.5
        if depth <= -B.CAP_R:
            score = 0.75
        full = (depth <= -(B.CAP_R + 0.020) and shift <= OVEN_SHIFT and turn <= OVEN_TURN and spoon_down and free
                and clear)
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "cap_depth_mm": round(1000 * depth, 1),
                "oven_shift_mm": round(1000 * shift, 2), "oven_turn_deg": round(turn, 2),
                "spoon_down": bool(spoon_down), "spoon_free": bool(free), "hands_clear": bool(clear),
                "failed": self.failed, "held_frames": self.held}
