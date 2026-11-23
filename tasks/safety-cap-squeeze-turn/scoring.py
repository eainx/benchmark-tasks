"""Stateful scorer for opening the squeeze-and-turn cap: the cap's turn on the neck from the measured cap and
bottle poses, the panels' squeeze and the thread's release from task.Dynamics, where the cap ends up, the
bottle's pose, the hands' measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.bottle0 = None
        self.unlocked_once = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("cap mechanics missing")
        self.step = step
        bottle = np.asarray(root_poses["bottle"], float)
        cap = np.asarray(root_poses["cap"], float)
        if self.bottle0 is None:
            self.bottle0 = bottle.copy()
        self.unlocked_once |= bool(mechanics["squeezed"])
        turn = mechanics["turn_deg"]
        free = mechanics["free"]
        neck = bottle[:3] + B.quat_matrix(bottle[3:7]) @ np.array([0.0, B.NECK_Y - B.BY, 0.0])
        off_neck = math.dist(cap[:2], neck[:2]) > 0.040
        cap_down = cap[2] < B.TOP + B.CAP_H / 2 + 0.010 and B.tilt_deg(cap) < 30.0
        moved = float(np.linalg.norm(bottle[:2] - self.bottle0[:2]))
        if self.failed is None and B.tilt_deg(bottle) > 25.0:
            self.failed = "bottle knocked over"
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:2], cap[:2]) > 0.06 and
                    math.dist(np.asarray(robot[s]["tcp"])[:2], bottle[:2]) > 0.10 or
                    float(np.asarray(robot[s]["tcp"])[2]) > B.TOP + B.BODY[2] + 0.08 for s in ("left", "right"))
        score = 0.0
        if self.unlocked_once:
            score = 0.25
        if turn >= 90.0 or free:
            score = 0.5
        if free:
            score = 0.75
        full = free and off_neck and cap_down and moved <= 0.020 and B.tilt_deg(bottle) <= 5.0 and clear
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "turn_deg": round(turn, 1), "free": bool(free),
                "squeezed": bool(mechanics["squeezed"]), "panels_mm": mechanics["panels_mm"],
                "cap_off_neck": bool(off_neck), "cap_down": bool(cap_down), "bottle_moved_mm": round(1000 * moved, 1),
                "hands_clear": bool(clear), "failed": self.failed, "held_frames": self.held}
