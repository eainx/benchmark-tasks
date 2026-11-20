"""Scorer for "Empty the floor sink's basket into the bus tub."

Everything is read from measured body poses; nothing is triggered by a hand being near something.

Ladder (the first two milestones latch once reached; the scraps are counted where they lie):
  0.25  the basket has been lifted clear of the sink: its base above the floor. (With drain.WITH_GRATE, which is off in
        this version, the hinged grate must first have been swung open more than 60 deg, worth 0.20.)
  +0.20 per scrap (a lemon wedge, a lime wedge, a bottle cap) lying in the bus tub (centre inside the tub's inner
        outline less 30 mm and below its rim), counted only once the basket has been out: up to 1.00, which also
        needs the three scraps at rest (under 10 mm/s) in the tub for 1 s.
Caps: a scrap anywhere but the tub or the basket (on the floor, in the sink, on the bench) caps the score at 0.5;
the bus tub pushed more than 50 mm or tipped over 15 deg caps it at 0.5.
In a real trial: the basket clearing the floor is seen on video; the scraps are counted in the tub.
"""
from __future__ import annotations

import math

import numpy as np

import drain as D

HZ = 20
REST_FRAMES = 20
SCRAPS = tuple(d[0] for d in D.DEBRIS)
T0 = np.array([D.TUB_C[0], D.TUB_C[1], D.TOP + D.TUB_H / 2])


def _field(o, name):
    return o[name] if isinstance(o, dict) else getattr(o, name)


def in_tub(p, tub_pose):
    R = D.mat(tub_pose)
    q = R.T @ (np.asarray(p, float) - np.asarray(tub_pose[:3], float))
    return (abs(q[0]) < D.TUB_L / 2 - 0.030 and abs(q[1]) < D.TUB_W / 2 - 0.030
            and -D.TUB_H / 2 - 0.005 < q[2] < D.TUB_H / 2)


def in_bucket(p, b_pose):
    R = D.mat(b_pose)
    q = R.T @ (np.asarray(p, float) - np.asarray(b_pose[:3], float))
    return abs(q[0]) < D.B_HALF - 0.002 and abs(q[1]) < D.B_HALF - 0.002 and -D.B_H / 2 < q[2] < D.B_H / 2 + 0.01


def tilt_deg(pose):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(D.mat(pose)[2, 2])))))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.opened = False
        self.bucket_out = False
        self.tub_moved = False
        self.rest = 0
        self.info = {}

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        g = np.asarray(root_poses["grate"], float) if "grate" in root_poses else None
        b = np.asarray(root_poses["bucket"], float)
        tub = np.asarray(root_poses["tub"], float)
        if g is None or tilt_deg(g) > 60.0:
            self.opened = True
        base_z = b[2] - (D.mat(b) @ np.array([0.0, 0.0, D.B_H / 2]))[2]
        if self.opened and base_z > D.FLOOR:
            self.bucket_out = True
        if np.linalg.norm(tub[:2] - T0[:2]) > 0.050 or tilt_deg(tub) > 15.0:
            self.tub_moved = True
        n_tub, n_lost = 0, 0
        for name in SCRAPS:
            p = np.asarray(root_poses[name], float)[:3]
            if in_tub(p, tub):
                n_tub += 1 if self.bucket_out else 0
            elif not in_bucket(p, b):
                n_lost += 1
        speeds = [float(np.linalg.norm(_field(objects[n], "lin_vel"))) for n in SCRAPS if n in objects]
        still = max(speeds) < 0.010 if speeds else True
        done = self.bucket_out and n_tub == len(SCRAPS) and still
        self.rest = self.rest + 1 if done else 0
        score = 0.0
        if self.opened and g is not None:
            score = 0.20  # grate variant only; no free credit when the grate is cut
        if self.bucket_out:
            score = 0.25 + 0.20 * n_tub
            if n_tub == len(SCRAPS) and self.rest >= REST_FRAMES:
                score = 1.0
        if n_lost or self.tub_moved:
            score = min(score, 0.5)
        self.info = {"grate_opened": self.opened, "grate_angle_deg": round(tilt_deg(g), 1) if g is not None else None, "bucket_out": self.bucket_out,
                     "scraps_in_tub": n_tub, "scraps_lost": n_lost, "tub_moved": self.tub_moved}
        return {"score": round(float(score), 3), "success": bool(score >= 1.0), **self.info, "rest_frames": self.rest,
                "clear": True, "dropped": False}
