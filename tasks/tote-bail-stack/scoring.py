"""Scorer for "Stack the full tote on top of the other full tote.".

Ladder (milestones latch once reached; the final state decides full credit):
  0.15  each of tote A's two bails swung in: leg angle 160-180 deg, i.e. the bar lying across A's rim
  0.10  tote B lifted clear of the bench (underside 20 mm up), within 20 deg of level
  0.20  tote B standing on the bails: its underside 131-145 mm over A's (stacked; nested on A's cartons it
        is about 94 mm), within 3 deg of level, both bails still in
  1.0   all of that at the end, both hands 50 mm or more clear of both totes, and both totes and both bails
        at rest (under 1 cm/s and 0.2 rad/s) for 1 s
A tote knocked off the bench or tipped past 30 deg is `dropped` and scores 0. Tote B ending nested in A (its
underside under 120 mm over A's, i.e. down on A's cartons) is the failure the task guards against: the
stack and level credit are not given and the final score is capped at the bails' credit.
In a real trial: the bails' positions and the stack are seen on video; the stack height (B's underside over
A's, 137 mm stacked against about 94 mm nested) is checked with a tape measure.
Returns the runner's `clear` and `dropped` keys."""
from __future__ import annotations

import math

import numpy as np

import tote as T

HZ = 20
REST_FRAMES = 20


def _box_pts(box):
    c, R, h = (np.asarray(x, float) for x in box)
    pts = [c]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts)


def bail_angle(root_poses, name, s):
    """The bail's leg angle phi (deg) in tote A's frame: -90 hanging out, 171 swung in on the rim."""
    a = np.asarray(root_poses["tote_a"], float)
    b = np.asarray(root_poses[name], float)
    Ra = T.mat(a)
    piv = a[:3] + Ra @ np.array([0.0, s * T.Y_PIV, T.Z_PIV - T.H / 2])
    d = Ra.T @ (b[:3] - piv)
    return math.degrees(math.atan2(d[2], s * d[1]))


def _field(o, name):
    """A measured object state's field (the runner's ObjectState, or a dict in a replay)."""
    return o[name] if isinstance(o, dict) else getattr(o, name)


def _tilt(p):
    up = T.mat(p) @ np.array([0.0, 0.0, 1.0])
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.fell = False
        self.got = {"bail_l": False, "bail_r": False, "lifted": False, "stacked": False}
        self.rest = 0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        a = np.asarray(root_poses["tote_a"], float)
        b = np.asarray(root_poses["tote_b"], float)
        ta, tb = _tilt(a), _tilt(b)
        if min(a[2], b[2]) < T.TOP - 0.05 or max(ta, tb) > 30.0:
            self.fell = True
        ang = {"bail_l": bail_angle(root_poses, "bail_l", 1), "bail_r": bail_angle(root_poses, "bail_r", -1)}
        bails_in = {k: 160.0 <= v <= 180.0 for k, v in ang.items()}
        for k, v in bails_in.items():
            if v:
                self.got[k] = True
        under_b = b[2] - T.H / 2 - T.TOP
        over_a = (b[2] - a[2])
        inside = abs(b[0] - a[0]) < 0.06 and abs(b[1] - a[1]) < 0.06
        nested = inside and over_a < 0.120
        stacked = inside and 0.131 <= over_a <= 0.145 and tb < 3.0 and all(bails_in.values())
        if under_b > 0.020 and tb < 20.0:
            self.got["lifted"] = True
        if stacked and not nested:
            self.got["stacked"] = True
        # hands clear of both totes (finger pads measured from the joints at station.BASES, see task.hand_points)
        gap = 1.0
        if mechanics and mechanics.get("pad_boxes"):
            pts = np.vstack([_box_pts(bx) for bx in mechanics["pad_boxes"]])
            for p in (a, b):
                rel = np.abs(T.mat(p).T @ (pts - p[:3]).T).T
                out = np.maximum(rel - np.array([T.W_TOP / 2, T.L_TOP / 2, T.H / 2]), 0.0)
                gap = min(gap, float(np.linalg.norm(out, axis=1).min()))
        clear = gap >= 0.05
        still = True
        for k in ("tote_a", "tote_b", "bail_l", "bail_r"):
            o = objects[k]
            lv = np.asarray(_field(o, "lin_vel"), float)
            av = np.asarray(_field(o, "ang_vel"), float)
            if np.linalg.norm(lv) > 0.01 or np.linalg.norm(av) > 0.2:
                still = False
        self.rest = self.rest + 1 if (stacked and clear and still) else 0
        score = 0.15 * self.got["bail_l"] + 0.15 * self.got["bail_r"]
        if not nested:
            score += 0.10 * self.got["lifted"] + 0.20 * self.got["stacked"]
        success = bool(stacked and clear and self.rest >= REST_FRAMES and not self.fell)
        if success:
            score = 1.0
        if self.fell:
            score = 0.0
        return {"score": round(float(score), 3), "success": success,
                "bail_l_deg": round(ang["bail_l"], 1), "bail_r_deg": round(ang["bail_r"], 1),
                "b_over_a_mm": round(1000 * over_a, 1), "b_under_mm": round(1000 * under_b, 1),
                "tilt_a_deg": round(ta, 2), "tilt_b_deg": round(tb, 2), "nested": bool(nested),
                "milestones": dict(self.got), "rest_frames": self.rest,
                "pad_gap_mm": round(1000 * gap, 1), "clear": bool(clear), "dropped": bool(self.fell)}
