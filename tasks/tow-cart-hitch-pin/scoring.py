"""Stateful scorer for the tow-cart pin hitch. From the measured poses of the drawbar and the hitch pin and
the robot's measured TCPs only."""
from __future__ import annotations

import math

import numpy as np

import hitch as H

HZ = 20
HOLD_S = 1.0
EYE_IN_MM = 8.0          # the eye in the jaw, under the pin: its centre within this of the pin's axis
THROUGH_MM = 4.5         # the pin can only pass the 26 mm eye with the eye centred within (26 - 19) / 2 + 1 mm
SEAT_MM = 3.0            # the pin down: its collar within this of resting on the upper jaw plate
HAND_CLEAR = 0.060       # let go: every TCP this far from the pin's handle and from the drawbar


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.best = 0.0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        bar = np.asarray(root_poses["drawbar"], float)
        pin = np.asarray(root_poses["pin"], float)
        eye = H.eye_from_pose(bar)
        off = 1000 * math.hypot(eye[0] - pin[0], eye[1] - pin[1])
        eye_z_mm = 1000 * (eye[2] - H.EYE_Z)
        lift = H.pin_lift(pin)
        pin_bottom = H.PIN_BOT_SEATED + lift
        eye_top = eye[2] + H.EYE_T / 2
        level = abs(eye_z_mm) <= 6.0
        eye_in = level and off <= EYE_IN_MM
        through = eye_in and off <= THROUGH_MM and pin_bottom <= eye_top - 0.003
        seated = lift <= SEAT_MM / 1000
        coupled = through and seated
        tcps = [np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")]
        handle = pin[:3] + np.array([0.0, 0.0, H.HANDLE_Z - H.PIN_Z0])
        pivot = bar[:3]
        clear = all(float(np.linalg.norm(t - handle)) >= HAND_CLEAR and _seg_dist(t, pivot, eye) >= HAND_CLEAR for t in tcps)
        score = 0.0
        if eye_in:
            score = 0.25
        if through:
            score = 0.5
        if coupled:
            score = 0.75
        self.held = self.held + 1 if (coupled and clear) else 0
        if coupled and clear and self.held >= HOLD_S * HZ:
            score = 1.0
        self.best = max(self.best, score)
        return {"score": score, "success": score == 1.0, "eye_offset_mm": round(off, 2), "eye_height_mm": round(eye_z_mm, 2),
                "bar_deg": round(H.bar_angle(bar), 2), "pin_lift_mm": round(1000 * lift, 2), "eye_in_jaw": bool(eye_in),
                "pin_through_eye": bool(through), "pin_seated": bool(seated), "coupled": bool(coupled),
                "hands_clear": bool(clear), "held_frames": self.held, "best": self.best}
