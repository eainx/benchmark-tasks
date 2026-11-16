"""Stateful scorer for "Seal the bag." From the bag's measured pose, the robot's measured TCPs and the
track cells reported closed by task.Dynamics."""
from __future__ import annotations

import math

import numpy as np

import bag as B

HZ = 20
HOLD_S = 1.0
CLEAR = 0.08                          # both hands this far from the track at the end
DRAG_MM = 10.0                        # the bag dragged over the block by more than this ...
TURN_DEG = 5.0                        # ... or slewed round by more than this: not held while sealed
SPILL_DEG = 15.0                      # the bag tipped up past this with its mouth open: the load spills


def _seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip((p - a) @ ab / max(1e-12, ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.start = None
        self.max_drag = 0.0
        self.max_turn = 0.0
        self.max_tilt = 0.0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("track measurements missing")
        self.step = step
        bag = np.asarray(root_poses["bag"], float)
        yaw, tilt = B.yaw_tilt(bag)
        if self.start is None:
            self.start = (bag[:3].copy(), yaw)
        drag = 1000 * float(np.linalg.norm(bag[:2] - self.start[0][:2]))
        turn = abs((yaw - self.start[1] + 180) % 360 - 180)
        self.max_drag = max(self.max_drag, drag)
        self.max_turn = max(self.max_turn, turn)
        self.max_tilt = max(self.max_tilt, tilt)
        closed = int(mechanics["closed"])
        a, b = B.track_world(bag)
        clear = all(_seg_dist(np.asarray(robot[s]["tcp"], float)[:3], a, b) >= CLEAR for s in ("left", "right"))
        held_still = self.max_drag <= DRAG_MM and self.max_turn <= TURN_DEG
        spilled = self.max_tilt > SPILL_DEG and closed < B.N_CELLS
        score = 0.0
        if closed >= 4:
            score = 0.25
        if closed >= 12:
            score = 0.5
        if closed == B.N_CELLS and held_still and not spilled:
            score = 0.75
        done = score == 0.75 and clear
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if not held_still or spilled:
            score = min(score, 0.5)
        return {"score": score, "success": score == 1.0, "closed_cells": closed, "cells": mechanics["closed_cells"],
                "drag_mm": round(drag, 1), "max_drag_mm": round(self.max_drag, 1), "max_turn_deg": round(self.max_turn, 2),
                "max_tilt_deg": round(self.max_tilt, 2), "spilled": bool(spilled), "hands_clear": bool(clear),
                "held_frames": self.held}
