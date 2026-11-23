"""Stateful scorer for fishing the cap out from under the toaster oven: how far the cap is inside or out
of the oven's footprint, from the measured cap and oven poses; the oven's shift and turn; the spoon's
measured pose and the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
OVEN_SHIFT, OVEN_TURN = 0.005, 2.0     # the oven must stay within this of where it started, all trial long
TILT_FAIL = 5.0
CAP_STEP = 0.0002                      # a step in which the cap moves more than this counts as the cap moving
SPOON_TOUCH = 0.004                    # ... and it must be the spoon touching it: handle or bowl within 4 mm


def _to_spoon_surface(p, spoon):
    """Gap between the cap's rim and the spoon (handle or bowl), in the bench plane, m."""
    r = B.quat_matrix(spoon[3:7])
    axis = r @ np.array([0.0, 1.0, 0.0])
    d = np.asarray(p, float)[:2] - spoon[:2]
    a = axis[:2] / max(1e-9, np.linalg.norm(axis[:2]))
    s = float(np.clip(np.dot(d, a), -B.HANDLE[1] / 2 - B.BOWL[1], B.HANDLE[1] / 2))
    half = B.HANDLE[0] / 2 if s > -B.HANDLE[1] / 2 else B.BOWL[0] / 2
    return float(np.linalg.norm(d - s * a)) - half - B.CAP_R


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
        self.cap_prev = None
        self.moved_by_spoon = 0.0
        self.moved_otherwise = 0.0

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
            if shift > OVEN_SHIFT or turn > OVEN_TURN or B.tilt_deg(oven) > TILT_FAIL:
                self.failed = "oven shoved"
            elif cap[2] < B.TOP - 0.02:
                self.failed = "cap off the bench"
        # who moved the cap: count its motion step by step, by the spoon (in contact) or by anything else
        if self.cap_prev is not None:
            dmove = float(np.linalg.norm(cap[:2] - self.cap_prev[:2]))
            if dmove > CAP_STEP:
                gap = _to_spoon_surface(cap[:3], spoon)
                if gap <= SPOON_TOUCH:
                    self.moved_by_spoon += dmove
                else:
                    self.moved_otherwise += dmove
        self.cap_prev = cap.copy()
        if self.failed is None and self.moved_otherwise > 0.010:
            self.failed = "cap moved without the spoon"
        # progress is measured against the oven's footprint where it started (it may not move anyway)
        depth = B.depth_under(cap[:2], self.oven0)
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
        full = depth <= -(B.CAP_R + 0.020) and spoon_down and free and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "cap_depth_mm": round(1000 * depth, 1),
                "cap_moved_by_spoon_mm": round(1000 * self.moved_by_spoon, 1),
                "cap_moved_otherwise_mm": round(1000 * self.moved_otherwise, 1),
                "oven_shift_mm": round(1000 * shift, 2), "oven_turn_deg": round(turn, 2),
                "spoon_down": bool(spoon_down), "spoon_free": bool(free), "hands_clear": bool(clear),
                "failed": self.failed, "held_frames": self.held}
