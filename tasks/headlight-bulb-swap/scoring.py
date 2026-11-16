"""Stateful scorer for "Replace the headlight bulb." From the bulbs' and connector's measured poses,
the robot's measured joints and finger travel (its pads placed by the URDF's forward kinematics) and the
latch / bayonet states task.Dynamics reports."""
from __future__ import annotations

import numpy as np

import headlamp as H

HZ = 20
HOLD_S = 1.0
OUT_MM = 20.0                         # the old bulb's grip at least this far back out of the port
CLEAR = 0.08                          # hands this far from the new bulb's grip and from the connector
GLASS_MM = 2.5                        # a pad point this close to the new bulb's quartz capsule: touched


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.touched = False
        self.fell = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("latch and bayonet measurements missing")
        self.step = step
        m = mechanics
        old = np.asarray(root_poses["old_bulb"], float)
        new = np.asarray(root_poses["new_bulb"], float)
        conn = np.asarray(root_poses["connector"], float)
        seat = np.asarray(H.bulb_seated_pos(), float)
        pads = np.asarray(m["pads"], float).reshape(-1, 3)
        # the new bulb's glass: distance of every pad point to its capsule's axis segment
        a0, a1 = H.capsule_segment(new)
        d = _seg_dist(pads, a0, a1)
        if float(d.min()) < H.CAPSULE_R + GLASS_MM / 1000:
            self.touched = True
        for p in (old, new, conn):
            if p[2] < H.TOP - 0.02:                    # off the bench
                self.fell = True
        old_out = (not m["old_in_port"]) and float(seat[1] - old[1]) >= OUT_MM / 1000
        score = 0.0
        if old_out:
            score = 0.25
        if old_out and m["new_locked"]:
            score = 0.5
        if old_out and m["new_locked"] and m["connector_on_new"]:
            score = 0.75
        hands = np.asarray(m["tcp_tips"], float).reshape(-1, 3)
        far_new = float(np.min(np.linalg.norm(hands - new[:3], axis=1)))
        far_conn = float(np.min(np.linalg.norm(hands - conn[:3], axis=1)))
        far_old = float(np.min(np.linalg.norm(hands - old[:3], axis=1)))
        clear = far_new >= CLEAR and far_conn >= CLEAR and far_old >= 0.05
        o = objects["old_bulb"]
        v = o.get("lin_vel", (0, 0, 0)) if isinstance(o, dict) else getattr(o, "lin_vel", (0, 0, 0))
        still = float(np.linalg.norm(np.asarray(v, float))) < 0.02
        done = score == 0.75 and clear and still
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.touched:
            score = min(score, 0.5)
        if self.fell:
            score = 0.0
        return {"score": score, "success": score == 1.0, "old_out": bool(old_out), "old_turn_deg": m["old_deg"],
                "new_turn_deg": m["new_deg"], "new_locked": bool(m["new_locked"]),
                "connector_on_new": bool(m["connector_on_new"]), "connector_unlatched": bool(m["connector_free"]),
                "glass_touched": self.touched, "glass_mm": round(1000 * (float(d.min()) - H.CAPSULE_R), 1),
                "fell": self.fell, "hands_clear": bool(clear), "held_frames": self.held}


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
