"""Stateful scorer for "Replace the headlight bulb." From the bulbs' and connector's measured poses and
velocities, the robot's finger pads (oriented boxes placed from the measured joints and jaw opening by
the URDF's forward kinematics, task.hand_points) and the latch / bayonet states task.Dynamics reports."""
from __future__ import annotations

import numpy as np

import headlamp as H

HZ = 20
HOLD_S = 1.0
OUT_MM = 20.0                         # the old bulb drawn at least this far off its seat: withdrawn
AWAY_MM = 15.0                        # ... and kept at least this far off it (not put back into the port)
CLEAR_NEW = 0.08                      # finger pads at least this far from the new bulb's and the connector's surfaces
CLEAR_OLD = 0.05                      # ... and from the old bulb's
GLASS_MM = 2.5                        # a finger pad this close to the new bulb's quartz capsule: touched
REST_V = 0.02                         # m/s: the old bulb at rest ...
REST_W = 0.20                         # rad/s: ... and not rolling or spinning
BULB_R = 0.020                        # a bulb's extent about its axis (flange radius)
CONN_R = 0.017                        # the connector's extent about its centre (22 x 18 x 22 mm block)


def _segment_box(a, b, box, n=81):
    c, R, h = box
    t = np.linspace(0.0, 1.0, n)[:, None]
    local = (a + t * (b - a) - c) @ R
    return float(np.linalg.norm(np.maximum(np.abs(local) - h, 0.0), axis=1).min())


def _bulb_axis(pose):
    """A bulb's extent as a segment along its axis (+y in its Part frame): socket side of the grip to the
    capsule's tip."""
    p = np.asarray(pose, float)
    a = H.mat(p) @ np.array([0.0, 1.0, 0.0])
    return p[:3] - a * 0.012, p[:3] + a * 0.052


def _speed(o, key):
    v = o.get(key, (0, 0, 0)) if isinstance(o, dict) else getattr(o, key, (0, 0, 0))
    return float(np.linalg.norm(np.asarray(v, float)))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.touched = False
        self.fell = False
        self.withdrawn = False
        self.glass_min = float("inf")

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
        boxes = [(np.asarray(c, float), np.asarray(R, float), np.asarray(h, float)) for c, R, h in m["pad_boxes"]]
        # the new bulb's glass: the pads' full surfaces against its capsule (axis segment, radius)
        g0, g1 = H.capsule_segment(new)
        glass = min(_segment_box(g0, g1, b) for b in boxes) - H.CAPSULE_R
        self.glass_min = min(self.glass_min, glass)
        if glass < GLASS_MM / 1000:
            self.touched = True
        for p in (old, new, conn):
            if p[2] < H.TOP - 0.02:                    # off the bench
                self.fell = True
        # the old bulb: withdrawn once (a latched milestone), and still away from the port now --
        # wherever it was set down
        off_seat = float(np.linalg.norm(old[:3] - seat))
        if not m["old_in_port"] and off_seat >= OUT_MM / 1000:
            self.withdrawn = True
        old_out = self.withdrawn and not m["old_in_port"] and off_seat >= AWAY_MM / 1000
        score = 0.0
        if old_out:
            score = 0.25
        if old_out and m["new_locked"]:
            score = 0.5
        if old_out and m["new_locked"] and m["connector_on_new"]:
            score = 0.75
        # hand clearance: the pads' surfaces against the parts' extents
        n0, n1 = _bulb_axis(new)
        o0, o1 = _bulb_axis(old)
        far_new = min(_segment_box(n0, n1, b) for b in boxes) - BULB_R
        far_conn = min(_segment_box(conn[:3], conn[:3], b) for b in boxes) - CONN_R
        far_old = min(_segment_box(o0, o1, b) for b in boxes) - BULB_R
        clear = far_new >= CLEAR_NEW and far_conn >= CLEAR_NEW and far_old >= CLEAR_OLD
        o = objects["old_bulb"]
        still = _speed(o, "lin_vel") < REST_V and _speed(o, "ang_vel") < REST_W
        done = score == 0.75 and clear and still
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.touched:
            score = min(score, 0.5)
        if self.fell:
            score = 0.0
        return {"score": score, "success": score == 1.0, "old_withdrawn": self.withdrawn, "old_out": bool(old_out),
                "old_turn_deg": m["old_deg"], "new_turn_deg": m["new_deg"], "new_locked": bool(m["new_locked"]),
                "connector_on_new": bool(m["connector_on_new"]), "connector_unlatched": bool(m["connector_free"]),
                "glass_touched": self.touched, "glass_mm": round(1000 * glass, 1),
                "glass_min_mm": round(1000 * self.glass_min, 1), "fell": self.fell, "dropped": self.fell,
                "hands_clear": bool(clear), "clear": bool(clear),
                "clear_mm": [round(1000 * far_new, 1), round(1000 * far_conn, 1), round(1000 * far_old, 1)],
                "old_at_rest": bool(still), "held_frames": self.held}
