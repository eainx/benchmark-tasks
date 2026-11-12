"""Stateful scorer for "Pour the metal into the mould." Object poses and the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import shank as S

HZ = 20
SPILL_LIMIT = 3
HOLD_S = 1.0


class TrialScorer:
    def __init__(self):
        self.entered = {}               # bead -> mould it reached
        self.spilled = set()
        self.burnt = False
        self.lifted = False
        self.held = 0
        self.step = 0

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        cru = np.asarray(root_poses["crucible"], float)
        if cru[2] > S.TOP + 0.03:
            self.lifted = True
        for s in ("left", "right"):
            if S.on_hot_body(np.asarray(robot[s]["tcp"], float)[:3], cru):
                self.burnt = True
        moving = 0
        for name in S.BEADS:
            pos = np.asarray(root_poses[name], float)[:3]
            speed = float(np.linalg.norm(objects[name].lin_vel))
            k = S.in_cavity(pos)
            if k is not None:
                self.entered.setdefault(name, k)
                continue
            if S.in_crucible(pos, cru) or S.in_cup(pos):
                moving += S.in_cup(pos) and speed > 0.03
                continue
            if speed < 0.05 or pos[2] < S.TOP + 0.005:
                self.spilled.add(name)
            else:
                moving += 1
        tilt = math.degrees(math.acos(np.clip(S.quat_matrix(cru[3:7])[2, 2], -1, 1)))
        released = all(float(np.mean(robot[s]["fingers"])) < -0.043 or
                       float(np.linalg.norm(np.asarray(robot[s]["tcp"])[:3] - cru[:3])) > 0.16 for s in ("left", "right"))
        down = tilt < 8 and cru[2] < S.TOP + 0.01 and float(np.linalg.norm(objects["crucible"].lin_vel)) < 0.02
        counts = [sum(1 for m in self.entered.values() if m == k) for k in range(len(S.MOULD_YS))]
        dry = counts[1 - S.WET] / S.BEAD_COUNT
        wet_poured = counts[S.WET] > 0        # metal into a damp mould: a steam blow-out
        spilled = len(self.spilled - set(self.entered))
        score = 0.0
        if self.lifted:
            score = 0.25
        if dry >= 0.5:
            score = 0.5
        if dry >= 0.9 and spilled <= SPILL_LIMIT:
            score = 0.75
        full = score == 0.75 and down and released and moving == 0
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.burnt or wet_poured:
            score = 0.0
        return {"score": score, "success": score == 1.0, "in_dry_mould": counts[1 - S.WET], "in_wet_mould": counts[S.WET], "poured_wet_mould": bool(wet_poured), "spilled": spilled,
                "lifted": self.lifted, "hand_on_hot_crucible": self.burnt, "crucible_tilt_deg": tilt,
                "set_down": bool(down), "released": bool(released), "held_frames": self.held}
