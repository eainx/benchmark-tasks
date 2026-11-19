"""Stateful scorer for "Rack both glasses mouth-down for the dish machine." From the measured poses and
velocities of the rack and the two glasses and the finger pads (oriented boxes from the measured joints,
task.hand_points)."""
from __future__ import annotations

import math

import numpy as np

import glass as G

HZ = 20
HOLD_S = 1.0
TURNED_DEG = 30.0        # a glass counts as turned over once its mouth points within this of straight down
KNOCKED_DEG = 35.0       # a glass not held, lying past this from upright or mouth-down, has been knocked over
CLEAR = 0.020            # finger pads at least this far from both glasses at the end
REST_V = 0.02
REST_W = 0.2


def _speed(o, key):
    v = o.get(key, (0, 0, 0)) if isinstance(o, dict) else getattr(o, key, (0, 0, 0))
    return float(np.linalg.norm(np.asarray(v, float)))


def _box_pts(box):
    c, R, h = (np.asarray(x, float) for x in box)
    pts = [c]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts)


def pad_gap(pts, pose):
    """Smallest distance (approximately) from the pad corner points to a glass's outside."""
    p = np.asarray(pose, float)
    R = G.mat(p)
    rel = pts - p[:3]
    ax = rel @ R[:, 2]
    rad = np.linalg.norm(rel - np.outer(ax, R[:, 2]), axis=1)
    return float(np.maximum(rad - G.R_RIM, np.abs(ax) - G.H / 2).min())


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.turned = {n: False for n in G.GLASS_NAMES}
        self.best_racked = 0
        self.knocked = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("finger measurements missing")
        self.step = step
        rack = np.asarray(root_poses["rack"], float)
        pts = np.vstack([_box_pts(b) for b in mechanics["pad_boxes"]])
        cells, gaps = {}, {}
        for n in G.GLASS_NAMES:
            p = np.asarray(root_poses[n], float)
            md = G.mouth_down_deg(p)
            gaps[n] = pad_gap(pts, p)
            if md < TURNED_DEG:
                self.turned[n] = True
            held = gaps[n] < 0.004
            lowest = p[2] - G.H / 2 * abs(G.mat(p)[2, 2]) - G.R_RIM * math.sqrt(max(0.0, 1 - G.mat(p)[2, 2] ** 2))
            lying = KNOCKED_DEG < md < 180.0 - KNOCKED_DEG
            if p[2] < G.TOP - 0.05 or (lying and not held and lowest < G.FLOOR + G.RACK_H + 0.01):
                self.knocked = True
            c = G.racked_cell(p, rack)
            if c is not None and c not in cells.values():
                cells[n] = c
        released = {n: gaps[n] >= CLEAR for n in G.GLASS_NAMES}
        racked = [n for n in cells if released[n]]
        self.best_racked = max(self.best_racked, len(racked))
        rest = all(_speed(objects.get(n, {}), "lin_vel") < REST_V and _speed(objects.get(n, {}), "ang_vel") < REST_W
                   for n in G.GLASS_NAMES) if isinstance(objects, dict) else False
        done = len(racked) == 2 and rest
        self.held = self.held + 1 if done else 0
        score = 0.15 * sum(self.turned.values()) + 0.30 * self.best_racked
        if self.held >= HOLD_S * HZ:
            score = 1.0
        if self.knocked:
            score = 0.0
        return {"score": round(score, 3), "success": score == 1.0,
                "turned_over": dict(self.turned), "racked_now": {n: list(c) for n, c in cells.items()},
                "released": released, "best_racked": self.best_racked,
                "mouth_down_deg": {n: round(G.mouth_down_deg(root_poses[n]), 1) for n in G.GLASS_NAMES},
                "pad_gap_mm": {n: round(1000 * g, 1) for n, g in gaps.items()}, "at_rest": bool(rest),
                "held_frames": self.held, "knocked_over": bool(self.knocked)}
