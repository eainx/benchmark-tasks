"""Stateful scorer for "Put the new tape cassette in the label maker." From the label maker's, cover's and
cassettes' measured poses, the robot's measured pad tips and the latch and click states task.Dynamics reports."""
from __future__ import annotations

import math

import numpy as np

import labeler as L

HZ = 20
HOLD_S = 1.0
OPEN_DEG = 80.0                       # the cover counts as opened once swung up this far
ASIDE_MM = 60.0                       # the old cassette at least this far from the bay, across the desk ...
ASIDE_Z = 0.03                        # ... and lying on the desk (its middle below this above the desk)
SHOVE_MM = 20.0                       # the label maker moved more than this ...
SHOVE_DEG = 10.0                      # ... or turned more than this: score capped at 0.5
FALL_Z = -0.02                        # a cassette's middle below the desk top by this: it fell off; score 0
CLEAR = 0.05                          # every point on both hands' finger pads at least this far from the label maker
REST_V = 0.01                         # "set down": the old cassette slower than this (m/s) ...
REST_W = 0.2                          # ... and turning slower than this (rad/s)


def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def cover_deg(lab, cov):
    """How far the cover is swung up from shut, in degrees, in the label maker's own frame."""
    z = mat(lab).T @ mat(cov)[:, 2]
    return math.degrees(math.atan2(float(z[0]), float(z[2])))


def _box_dist(p, lab):
    """Distance from a point to the label maker's outer box (to the top of its hood), in its own frame."""
    c0 = np.asarray(L.labeler_pos(), float)
    local = mat(lab).T @ (np.asarray(p, float) - np.asarray(lab[:3], float)) + c0
    lo = np.array([L.X0, L.Y0, L.TOP])
    hi = np.array([L.X1, L.Y1, L.SHELL_TOP + L.COVER_H])
    d = np.maximum(np.maximum(lo - local, local - hi), 0.0)
    return float(np.linalg.norm(d))


def _speeds(state):
    """Linear (m/s) and angular (rad/s) speed of a measured object state (an object or a dict)."""
    get = (lambda k: state[k]) if isinstance(state, dict) else (lambda k: getattr(state, k))
    return float(np.linalg.norm(np.asarray(get("lin_vel"), float))), float(np.linalg.norm(np.asarray(get("ang_vel"), float)))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.opened = False
        self.start = None
        self.shoved = False
        self.fell = False
        self.wrong_way = False
        self.removed = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("latch and click states missing")
        self.step = step
        m = mechanics
        lab = np.asarray(root_poses["labeler"], float)
        old = np.asarray(root_poses["old_cassette"], float)
        new = np.asarray(root_poses["new_cassette"], float)
        if self.start is None:
            self.start = lab.copy()
        moved = 1000 * float(np.linalg.norm(lab[:2] - self.start[:2]))
        Rr = mat(self.start).T @ mat(lab)
        turned = abs(math.degrees(math.atan2(Rr[1, 0], Rr[0, 0])))
        if moved > SHOVE_MM or turned > SHOVE_DEG:
            self.shoved = True
        if min(old[2], new[2]) < L.TOP + FALL_Z:
            self.fell = True
        if m["cover_deg"] > OPEN_DEG:
            self.opened = True
        seat = np.asarray(L.SEAT, float)
        old_xy = 1000 * float(np.hypot(old[0] - seat[0], old[1] - seat[1]))
        v, w = _speeds(objects["old_cassette"])
        if old_xy > ASIDE_MM and old[2] < L.TOP + ASIDE_Z and v < REST_V and w < REST_W:
            self.removed = True                # latched: where it lies afterwards does not matter
        aside = self.removed
        # a new cassette sitting in the bay turned round (its yaw far off) is recorded; it can never click
        Rn = mat(lab).T @ mat(new)
        yaw = abs(math.degrees(math.atan2(Rn[1, 0], Rn[0, 0])))
        in_bay = 1000 * float(np.hypot(new[0] - seat[0], new[1] - seat[1])) < 20.0 and new[2] < L.SHELL_TOP + 0.02
        if in_bay and yaw > 90.0:
            self.wrong_way = True
        pads = [np.asarray(p, float) for p in m["pads"]]
        clear = all(_box_dist(p, lab) >= CLEAR for p in pads)
        score = 0.0
        if self.opened and aside:
            score = 0.25
            if m["seated"]:
                score = 0.5
                if m["latched"]:
                    score = 0.75
        done = score == 0.75 and clear
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.shoved:
            score = min(score, 0.5)
        if self.fell:
            score = 0.0
        return {"score": score, "success": score == 1.0, "clear": bool(clear), "dropped": self.fell,
                "cover_opened": self.opened, "old_removed": bool(aside),
                "old_from_bay_mm": round(old_xy, 1), "new_seated": bool(m["seated"]), "latched": bool(m["latched"]),
                "labeler_moved_mm": round(moved, 1), "labeler_turned_deg": round(turned, 1), "shoved": self.shoved,
                "fell": self.fell, "new_turned_round_in_bay": self.wrong_way, "hands_clear": bool(clear),
                "held_frames": self.held}
