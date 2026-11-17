"""Scorer for "Turn the steering wheel all the way to the left and hold it there.".

The wheel's turn is followed step by step from its measured pose, unwrapped across full turns, positive to the
left (counter-clockwise as the driver sees it). Full lock is 450 deg.

Ladder (milestones latch once reached):
  0.1   each quarter turn reached (within 5 deg): 85, 175, 265 and 355 deg (0.4 in all)
  0.6   full lock reached: 440 deg or more
  0.8   held at full lock: 440 deg or more for 2 s without a break
  1.0   held as above, then let go: at the end both hands are 50 mm or more clear of the rim and the wheel,
        returned by its spring, is at rest (under 1 cm/s and 0.2 rad/s) for 1 s
Hand over hand: while the wheel is being turned (from 10 deg until the hold is reached), a hand counts as on the
rim when its jaws are closed on it: both its finger pads (measured from the joints at station.BASES) lie within
6 mm of the rim's surface (pad mid-thickness, 4 mm behind the pad face) and the jaws are under 16 mm apart a side.
Both hands off the rim for more than 0.15 s, or the wheel slipping back more than 30 deg below the furthest it
has been turned, is a let-go: the hold credit is then not given (score capped at 0.6). These are observations
for the score only; nothing is triggered by them.
A wheel torn off its column (hub moved more than 5 mm) is `dropped` and scores 0.
In a real trial: a mark on the rim read against a protractor ring on the column gives the turn; the hold, the
hands and any slip-back are seen on video.
Returns the runner's `clear` and `dropped` keys."""
from __future__ import annotations

import math

import numpy as np

import steer as C

HZ = 20
HOLD_FRAMES = 40
REST_FRAMES = 20
LOCK_OK = 440.0


def _box_pts(box):
    c, R, h = (np.asarray(x, float) for x in box)
    pts = [c]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts)


def rim_gap(pts):
    """Distance of each point from the rim's (torus) surface, metres; negative inside the rim."""
    rel = np.asarray(pts, float) - C.C
    along = rel @ C.N
    radial = np.linalg.norm(rel - np.outer(along, C.N), axis=1)
    return np.sqrt((radial - C.R_C) ** 2 + along ** 2) - C.R_TUBE


def _field(o, name):
    return o[name] if isinstance(o, dict) else getattr(o, name)


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.turn = 0.0
        self.prev = None
        self.max_turn = 0.0
        self.quarters = [False] * 4
        self.lock = False
        self.held = False
        self.hold_run = 0
        self.off_run = 0
        self.let_go = False
        self.fell = False
        self.rest = 0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        w = np.asarray(root_poses["wheel"], float)
        a = C.wheel_angle_wrapped(w)
        if self.prev is None:
            self.turn = a
        else:
            self.turn += (a - self.prev + 180.0) % 360.0 - 180.0      # under half a turn per 50 ms step
        self.prev = a
        self.max_turn = max(self.max_turn, self.turn)
        hub_off = float(np.linalg.norm(w[:3] - C.C))
        if hub_off > 0.005:
            self.fell = True
        for k in range(4):
            if self.turn >= 90.0 * (k + 1) - 5.0:
                self.quarters[k] = True
        if self.turn >= LOCK_OK:
            self.lock = True
        # hands on the rim (observed from the measured finger pads)
        on = {"left": False, "right": False}
        gap = 1.0
        if mechanics and mechanics.get("pad_boxes"):
            boxes = mechanics["pad_boxes"]
            for i, side in enumerate(("left", "right")):
                g = [float(rim_gap(_box_pts(bx)).min()) for bx in boxes[2 * i:2 * i + 2]]
                half = abs(float(robot[side]["fingers"][0])) if robot else 1.0
                on[side] = max(g) <= 0.006 and half <= 0.016
            pts = np.vstack([_box_pts(bx) for bx in boxes])
            gap = float(rim_gap(pts).min())
        turning = 10.0 <= self.max_turn and not self.held
        if turning:
            self.off_run = 0 if (on["left"] or on["right"]) else self.off_run + 1
            if self.off_run > 3 or self.turn < self.max_turn - 30.0:
                self.let_go = True
        self.hold_run = self.hold_run + 1 if self.turn >= LOCK_OK else 0
        if self.hold_run >= HOLD_FRAMES and not self.let_go:
            self.held = True
        clear = gap >= 0.05
        o = objects["wheel"]
        lv = float(np.linalg.norm(np.asarray(_field(o, "lin_vel"), float)))
        av = float(np.linalg.norm(np.asarray(_field(o, "ang_vel"), float)))
        still = lv < 0.01 and av < 0.2
        self.rest = self.rest + 1 if (self.held and clear and still) else 0
        score = 0.1 * sum(self.quarters)
        if self.lock:
            score = 0.6
        if self.held:
            score = 0.8
        success = bool(self.held and self.rest >= REST_FRAMES and not self.fell)
        if success:
            score = 1.0
        if self.fell:
            score = 0.0
        return {"score": round(float(score), 3), "success": success, "turn_deg": round(self.turn, 1),
                "max_turn_deg": round(self.max_turn, 1), "hold_frames": self.hold_run, "let_go": bool(self.let_go),
                "on_rim": on, "off_frames": self.off_run, "milestones": {"quarters": sum(self.quarters), "lock": self.lock,
                                                                          "held": self.held},
                "wheel_ang_vel": round(av, 3), "hub_off_mm": round(1000 * hub_off, 2), "rest_frames": self.rest,
                "pad_gap_mm": round(1000 * gap, 1), "clear": bool(clear), "dropped": bool(self.fell)}
