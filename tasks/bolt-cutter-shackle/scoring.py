"""Stateful scorer for "Cut the padlock off the cage gate." From the bolt cutter's two bodies' and the
padlock's measured poses, the robot's measured TCPs/fingers and the cut reported by task.Dynamics."""
from __future__ import annotations

import math

import numpy as np

import cutter as C

HZ = 20
HOLD_S = 1.0
CLEAR = 0.10                          # hands this far from the lock at the end
LET_GO = 0.06                         # no closed hand this close to either grip: the cutter is let go of


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.max_drag = 0.0
        self.lock0 = None

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("cut measurements missing")
        self.step = step
        cutter, handle_b, lock = (np.asarray(root_poses[k], float) for k in ("cutter", "handle_b", "padlock"))
        if self.lock0 is None:
            self.lock0 = lock[:3].copy()
        drag = float(np.linalg.norm(lock[:3] - self.lock0))
        if not mechanics.get("cut"):
            self.max_drag = max(self.max_drag, drag)      # up to and during the cut
        b = C.bite(cutter, handle_b, lock)
        cut = bool(mechanics.get("cut"))
        score = 0.0
        if b is not None or cut:
            score = 0.25
        if cut:
            score = 0.5
        steady = self.max_drag <= C.DRAG_MM / 1000
        if cut and steady:
            score = 0.75
        grips = C.grip_points(cutter, handle_b)
        down = float(cutter[2]) < C.LIE_Z + 0.015 and float(handle_b[2]) < C.LIE_Z + 0.015   # laid back down
        let_go = all(not (float(np.mean(robot[s]["fingers"])) > -0.040 and
                          min(math.dist(np.asarray(robot[s]["tcp"])[:3], g) for g in grips) < LET_GO)
                     for s in ("left", "right"))
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:3], lock[:3]) > CLEAR for s in ("left", "right"))
        done = score == 0.75 and down and let_go and clear
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        return {"score": score, "success": score == 1.0, "jaws_on_leg": b is not None,
                "blade_gap_mm": None if b is None else round(1000 * b[2], 1), "cut": cut,
                "lock_drag_mm": round(1000 * drag, 1), "max_drag_mm": round(1000 * self.max_drag, 1),
                "handles_open_deg": round(C.opening_deg(cutter, handle_b), 1), "cutter_down": bool(down),
                "let_go": bool(let_go), "hands_clear": bool(clear), "held_frames": self.held}
