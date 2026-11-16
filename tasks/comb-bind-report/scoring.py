"""Stateful scorer for "Bind the booklet with the comb binder." From the booklet's and comb's measured
poses and speeds, the robot's measured joints and finger travel (its pads placed by the URDF's forward
kinematics from station.BASES) and the punch / comb / hang states task.Dynamics reports."""
from __future__ import annotations

import numpy as np

import binder as B

HZ = 20
HOLD_S = 1.0
CLEAR = 0.05                 # every pad point this far from the booklet and from the comb
REST_LIN = 0.01              # m/s: booklet and comb at rest
REST_ANG = 0.10              # rad/s


def pad_points(robot):
    """Sample points on both pads of each hand: measured joints through forward kinematics, the jaw
    opening from the measured finger travel (the jaws close along the gripper's local x)."""
    from yam_arena import ik, station
    from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP
    pts = []
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, np.asarray(station.BASES[side], float))
        half = abs(float(robot[side]["fingers"][0]))
        tip = pos + R[:, 2] * PAD_TIP
        for sgn in (-1.0, 1.0):
            for dz in (0.0, -0.012, -0.024):
                for dw in (-0.008, 0.0, 0.008):
                    pts.append(tip + R[:, 0] * sgn * (half + 0.004) + R[:, 2] * dz + R[:, 1] * dw)
    return np.asarray(pts)


def obb_dist(points, pose, half, offset=(0.0, 0.0, 0.0)):
    """Distance of each point to a box (half extents ``half``, centred ``offset`` in the body frame)."""
    R = B.mat(pose)
    c = np.asarray(pose[:3], float) + R @ np.asarray(offset, float)
    local = (points - c) @ R
    d = np.maximum(np.abs(local) - np.asarray(half, float), 0.0)
    return np.linalg.norm(d, axis=1)


def _speeds(obj):
    if not isinstance(obj, dict):
        return 0.0, 0.0
    return float(np.linalg.norm(obj.get("lin_vel", (0, 0, 0)))), float(np.linalg.norm(obj.get("ang_vel", (0, 0, 0))))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.dropped = False
        self.ruined = False
        self.reached = 0.0          # milestones latch: 0.25 punched square, 0.5 comb opened on the opener, 0.75 hung

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("punch and comb measurements missing")
        self.step = step
        m = mechanics
        book = np.asarray(root_poses["booklet"], float)
        comb = np.asarray(root_poses["comb"], float)
        R = B.mat(book)
        if abs(float(R[2, 2])) < 0.5 and book[2] < B.TOP + 0.03:
            self.dropped = True                      # lying on the desk: fell out of a hand
        if m["punched"] and not m["square"]:
            self.ruined = True                       # holes off the comb's pitch: the booklet is spoilt
        if m["punched"] and m["square"]:
            self.reached = max(self.reached, 0.25)
        if self.reached >= 0.25 and m["comb_seated"] and m["comb_opened"]:
            self.reached = max(self.reached, 0.5)
        if self.reached >= 0.5 and (m["hung"] or m["bound"]):
            self.reached = max(self.reached, 0.75)
        score = self.reached
        pads = pad_points(robot)
        d_book = float(obb_dist(pads, book, np.asarray(B.BOOK) / 2).min())
        d_comb = float(obb_dist(pads, comb, (B.COMB_R, B.COMB_SPINE[1] / 2, B.COMB_R),
                                (B.COMB_SPINE[0] / 2 - B.COMB_R, 0.0, B.COMB_R)).min())
        clear = d_book > CLEAR and d_comb > CLEAR
        lb, ab = _speeds((objects or {}).get("booklet"))
        lc, ac = _speeds((objects or {}).get("comb"))
        still = lb < REST_LIN and ab < REST_ANG and lc < REST_LIN and ac < REST_ANG
        done = self.reached >= 0.75 and m["bound"] and not m["comb_open"] and clear and still
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.ruined:
            score = 0.0
        if self.dropped:
            score = min(score, 0.25)
        return {"score": score, "success": score == 1.0, "clear": bool(clear), "dropped": self.dropped,
                "punched": m["punched"], "square": m["square"], "guide_mm": m["guide_mm"],
                "comb_seated": m["comb_seated"], "comb_opened": m["comb_opened"], "hung": m["hung"], "bound": m["bound"],
                "comb_open": m["comb_open"], "pad_book_mm": round(1000 * d_book, 1), "pad_comb_mm": round(1000 * d_comb, 1),
                "still": bool(still), "ruined": self.ruined, "held_frames": self.held}
