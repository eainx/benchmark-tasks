"""Stateful scorer for one pump of soap onto the sponge: the pump's stroke from the stem's measured
height in the bottle, the spout's aim from the head's measured pose, where the dose lands (the spout tip
at the bottom of a stroke) against the sponge's measured pose, the bottle's pose, the hands' TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
BOTTLE_SHIFT, TILT_OK, TILT_FAIL = 0.010, 5.0, 30.0


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.bottle0 = None
        self.stem0 = None
        self.strokes = 0
        self.down = False
        self.on_sponge = 0

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        bottle = np.asarray(root_poses["bottle"], float)
        stem = np.asarray(root_poses["stem"], float)
        head = np.asarray(root_poses["head"], float)
        sponge = np.asarray(root_poses["sponge"], float)
        if self.bottle0 is None:
            self.bottle0, self.stem0 = bottle.copy(), stem.copy()
        # the stroke: how far the stem has gone down into the bottle (along the bottle's axis)
        axis = B.quat_matrix(bottle[3:7]) @ np.array([0.0, 0.0, 1.0])
        rel0 = self.stem0[:3] - self.bottle0[:3]
        depth = float(np.dot(rel0, np.array([0.0, 0.0, 1.0])) - np.dot(stem[:3] - bottle[:3], axis))
        tip = B.spout_tip(head)
        aimed = B.over_sponge(tip[:2], sponge)
        if not self.down and depth >= B.FULL:
            self.down = True
            self.strokes += 1
            if aimed:
                self.on_sponge += 1
            elif self.failed is None:
                self.failed = "soap on the counter"
        elif self.down and depth <= B.RETURNED:
            self.down = False
        tilt = B.tilt_deg(bottle)
        shift = float(np.linalg.norm(bottle[:2] - self.bottle0[:2]))
        if self.failed is None and tilt > TILT_FAIL:
            self.failed = "bottle knocked over"
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:2], bottle[:2]) > 0.09 or
                    float(np.asarray(robot[s]["tcp"])[2]) > B.HEAD_Z + 0.08 for s in ("left", "right"))
        score = 0.0
        if aimed:
            score = 0.25
        if self.on_sponge >= 1:
            score = 0.5
        if (self.strokes == 1 and self.on_sponge == 1 and not self.down and depth <= B.RETURNED and shift <= BOTTLE_SHIFT
                and tilt <= TILT_OK):
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "stroke_mm": round(1000 * depth, 2), "strokes": self.strokes,
                "doses_on_sponge": self.on_sponge, "spout_over_sponge": bool(aimed),
                "spout_deg": round(B.yaw_deg(head) - B.yaw_deg(bottle), 1), "bottle_shift_mm": round(1000 * shift, 1),
                "bottle_tilt_deg": round(tilt, 2), "hands_clear": bool(clear), "failed": self.failed,
                "held_frames": self.held}
