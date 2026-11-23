"""Stateful scorer for refitting the spacebar: how far each of the three stems is inside its socket and
its offset, from the measured cap and stem poses; the cap's tilt; whether the key is back at rest height;
and the hands from their measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
LEVEL_DEG = 1.5
REST_TOL = 0.0008                      # stems back within 0.8 mm of their rest height: the key has come back up


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.rest = None

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        if self.rest is None:
            self.rest = {n: float(np.asarray(root_poses[n])[2]) for n in ("switch_stem", "stab_bar")}
        eng = B.engagement(root_poses)
        depths = [d for d, _ in eng.values()]
        lat = [l for _, l in eng.values()]
        cap = np.asarray(root_poses["spacebar"], float)
        tilt = B.roll_deg(cap)
        up = all(abs(float(np.asarray(root_poses[n])[2]) - self.rest[n]) <= REST_TOL for n in self.rest)
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:3], cap[:3]) > 0.05 for s in ("left", "right"))
        # on its stem: the cross at least SEATED into the socket but not past its floor (no pass-through)
        inside = [B.SEATED <= d <= B.BOSS_H + 0.0006 and l <= 0.0012 for d, l in zip(depths, lat)]
        score = 0.0
        if any(inside):
            score = 0.25
        if all(d >= 0.0015 and l <= 0.0015 for d, l in zip(depths, lat)):
            score = 0.5
        if all(inside) and tilt <= LEVEL_DEG:
            score = 0.75
        full = score == 0.75 and up and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        return {"score": score, "success": score == 1.0,
                "depth_mm": {k: round(1000 * v[0], 2) for k, v in zip(("mid", "left", "right"), eng.values())},
                "offset_mm": {k: round(1000 * v[1], 2) for k, v in zip(("mid", "left", "right"), eng.values())},
                "tilt_deg": round(tilt, 2), "key_up": bool(up), "hands_clear": bool(clear), "held_frames": self.held}
