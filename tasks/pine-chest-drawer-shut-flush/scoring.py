"""Stateful scorer for shutting the chest drawer: both ends of the drawer front measured against the
chest's face from the drawer's measured pose, the drawer's yaw and tilt, and the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
HALF = 0.5 * B.TRAVEL                # 0.25: the worse end at most half its start distance out
NINETY = 0.1 * B.TRAVEL              # 0.5: the worse end 90% of the way in
NEAR_FLUSH = 0.005                   # 0.75: both ends within 5 mm of flush
FLUSH = 0.002                        # 1.0: both ends within 2 mm of flush
SQUARE_DEG = 0.3                     # 1.0: drawer within 0.3 degrees of square to the chest
CLEAR = 0.030                        # 1.0: each hand at least this far from the drawer front and knobs
TILT_FAIL_DEG = 6.0                  # tipped this far (pulled off its runners or levered up): failed
OUT_FAIL = B.TRAVEL + 0.060          # pulled this far out: off the runners, failed


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.best = 0.0

    @staticmethod
    def _hand_clear(tcp):
        """A hand is clear when its TCP is 30 mm in front of the knobs' ends or above the chest."""
        tcp = np.asarray(tcp, float)[:3]
        return bool(tcp[0] < B.FX - B.KNOB_L - CLEAR or tcp[2] > B.TOP + B.CH + CLEAR)

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        pose = np.asarray(root_poses["drawer"], float)
        gl, gr = B.front_gaps(pose)
        worst = max(gl, gr)
        yaw = B.yaw_deg(pose)
        tilt = B.tilt_deg(pose)
        if self.failed is None and (tilt > TILT_FAIL_DEG or worst > OUT_FAIL or pose[2] < B.TOP):
            self.failed = "tipped" if tilt > TILT_FAIL_DEG else "pulled off its runners"
        clear = {s: self._hand_clear(robot[s]["tcp"]) for s in ("left", "right")}
        score = 0.0
        if worst <= HALF:
            score = 0.25
        if worst <= NINETY:
            score = 0.5
        if max(abs(gl), abs(gr)) <= NEAR_FLUSH:
            score = 0.75
        full = max(abs(gl), abs(gr)) <= FLUSH and abs(yaw) <= SQUARE_DEG and all(clear.values())
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        self.best = max(self.best, score)
        return {"score": score, "success": score == 1.0, "gap_left_mm": round(1000 * gl, 2),
                "gap_right_mm": round(1000 * gr, 2), "yaw_deg": round(yaw, 3), "tilt_deg": round(tilt, 2),
                "hands_clear": clear, "failed": self.failed, "held_frames": self.held}
