"""Stateful scorer for packing the robot: where the robot, the top pad and the lid sit relative to the box
base, from their measured poses; the base's travel and tilt; the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
BASE_MOVE = 0.030


def _placed(base, pose, z_seated, off_max, z_tol, yaw_max, tilt_max):
    off, z, yaw, tilt = B.in_base(base, pose)
    return off <= off_max and abs(z - z_seated) <= z_tol and abs(yaw) <= yaw_max and tilt <= tilt_max, (off, z - z_seated, yaw, tilt)


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.base0 = None

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        base = np.asarray(root_poses["box_base"], float)
        if self.base0 is None:
            self.base0 = base.copy()
        # the robot down in its pocket: centred within the pocket's play, on the pocket floor, square, upright
        bot_in, bot_m = _placed(base, root_poses["robot"], B.BOT_SEATED, 0.006, 0.003, 6.0, 5.0)
        # the top pad flat on the robot and the rim, over the base
        pad_on, pad_m = _placed(base, root_poses["top_pad"], B.PAD_SEATED, 0.010, 0.003, 6.0, 5.0)
        # the lid down over the base, resting on the pad, square and level
        lid_on, lid_m = _placed(base, root_poses["lid"], B.LID_SEATED, 0.004, 0.003, 4.0, 3.0)
        moved = float(np.linalg.norm(base[:2] - self.base0[:2]))
        base_tilt = B.tilt_deg(base)
        if self.failed is None and base_tilt > 20.0:
            self.failed = "box knocked over"
        lid = np.asarray(root_poses["lid"], float)
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:2], base[:2]) > 0.09 or
                    float(np.asarray(robot[s]["tcp"])[2]) > lid[2] + 0.06 for s in ("left", "right"))
        steady = moved <= BASE_MOVE and base_tilt <= 5.0
        score = 0.0
        if bot_in:
            score = 0.25
        if bot_in and pad_on:
            score = 0.5
        if bot_in and pad_on and lid_on and steady:
            score = 0.75
        full = score == 0.75 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        r = lambda m: [round(1000 * m[0], 1), round(1000 * m[1], 1), round(m[2], 1), round(m[3], 1)]  # noqa: E731
        return {"score": score, "success": score == 1.0, "robot_in": bool(bot_in), "pad_on": bool(pad_on),
                "lid_on": bool(lid_on), "robot_off_dz_yaw_tilt": r(bot_m), "pad_off_dz_yaw_tilt": r(pad_m),
                "lid_off_dz_yaw_tilt": r(lid_m), "base_moved_mm": round(1000 * moved, 1),
                "base_tilt_deg": round(base_tilt, 2), "hands_clear": bool(clear), "failed": self.failed,
                "held_frames": self.held}
