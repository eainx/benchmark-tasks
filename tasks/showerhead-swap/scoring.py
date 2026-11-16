"""Stateful scorer for "Replace the showerhead with the new one." From the measured poses of the arm,
both heads and the robot's TCPs, and the thread state task.Dynamics reports."""
from __future__ import annotations

import math

import numpy as np

import shower as S

HZ = 20
HOLD_S = 1.0
CLEAR = 0.08                     # m: both hands this far from the new head and its face
ELBOW_LIMIT = 15.0               # deg the arm may have turned in its elbow (more: it would leak there)
STILL = 0.002                    # m per step: the old head lying still


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.dropped = False
        self.prev_old = None

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("thread state missing")
        self.step = step
        old = np.asarray(root_poses["old_head"], float)
        new = np.asarray(root_poses["new_head"], float)
        face = np.asarray(root_poses["new_face"], float)
        tcps = [np.asarray(robot[s]["tcp"], float)[:3] for s in ("left", "right")]
        if any(S.fallen(p) for p in (old[:3], new[:3], face[:3])):
            self.dropped = True
        moved = float(np.linalg.norm(old[:3] - self.prev_old)) if self.prev_old is not None else 1.0
        self.prev_old = old[:3].copy()
        old_out = not mechanics["old_on_thread"]
        old_in = old_out and S.in_basket(old[:3]) and moved < STILL and all(
            math.dist(t, old[:3]) > 0.06 for t in tcps)
        started = mechanics["new_on_thread"] and mechanics["new_turns"] >= 0.5
        seated = mechanics["seated"]
        elbow_ok = mechanics["elbow_max_deg"] <= ELBOW_LIMIT
        clear = all(min(math.dist(t, new[:3]), math.dist(t, face[:3])) > CLEAR for t in tcps)
        score = 0.0
        if old_in:
            score = 0.25
            if started:
                score = 0.5
                if seated and elbow_ok:
                    score = 0.75
        done = score == 0.75 and clear
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if not elbow_ok:
            score = min(score, 0.5)
        if self.dropped:
            score = 0.0
        return {"score": score, "success": score == 1.0, "old_off": bool(old_out), "old_in_basket": bool(old_in),
                "new_started": bool(started), "new_seated": bool(seated), "elbow_max_deg": mechanics["elbow_max_deg"],
                "hands_clear": bool(clear),
                "dropped": self.dropped, "held_frames": self.held}
