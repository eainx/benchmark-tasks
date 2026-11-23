"""Stateful scorer for closing the matchbox: how far the tray's end stands out of (or in from) the sleeve's
end, from the measured tray and sleeve poses; the sleeve's travel and tilt; the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
FLUSH = 0.0010
SLEEVE_MOVE = 0.010
TIP_FAIL = 20.0


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.sleeve0 = None

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        sleeve = np.asarray(root_poses["sleeve"], float)
        tray = np.asarray(root_poses["tray"], float)
        if self.sleeve0 is None:
            self.sleeve0 = sleeve.copy()
        out = B.tray_out(sleeve, tray)
        moved = float(np.linalg.norm(sleeve[:2] - self.sleeve0[:2]))
        tilt = max(B.tilt_deg(sleeve), B.tilt_deg(tray))
        if self.failed is None and (tilt > TIP_FAIL or sleeve[2] < B.TOP - 0.02 or tray[2] < B.TOP - 0.02):
            self.failed = "knocked over or off the bench"
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:2], sleeve[:2]) > 0.08 or
                    float(np.asarray(robot[s]["tcp"])[2]) > B.TOP + B.SL[2] + 0.05 for s in ("left", "right"))
        score = 0.0
        if out <= B.OUT / 2:
            score = 0.25
        if out <= 0.005:
            score = 0.5
        if abs(out) <= FLUSH and moved <= SLEEVE_MOVE and tilt <= 5.0:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "tray_out_mm": round(1000 * out, 2),
                "sleeve_moved_mm": round(1000 * moved, 1), "tilt_deg": round(tilt, 2), "hands_clear": bool(clear),
                "failed": self.failed, "held_frames": self.held}
