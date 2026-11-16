"""Stateful scorer for "Connect the new syrup box." From the connector's and the new spout's measured poses,
the robot's measured TCPs and the connector states task.Dynamics reports."""
from __future__ import annotations

import math

import numpy as np

import syrup as S

HZ = 20
HOLD_S = 1.0
OFF_MM = 30.0                         # the connector at least this far from where it sat on the empty spout
CLEAR = 0.05                          # every pad point of both hands this far from the connector's surface
CONN_RADIUS = 0.035                   # the connector with its barb fits in a sphere this big about its centre
REST_LIN = 0.01                       # m/s: the connector at rest ...
REST_ANG = 0.2                        # ... rad/s
PUSHED_IN_MM = 20.0                   # the new spout pushed further than this back into its carton
DROP_Z = S.BASE - 0.020               # the connector below the shelf's deck: dropped


def pad_points(robot):
    """Both hands' pad sample points from measured joints and jaw opening (the URDF's forward kinematics, the
    arms at station.BASES)."""
    from yam_arena import ik
    from yam_arena.constants import PAD_TIP_PAST_TCP
    from yam_arena.station import BASES
    pts = []
    for side in ("left", "right"):
        pos, R = ik.tcp_pose(np.asarray(robot[side]["joints"], float), np.asarray(BASES[side], float))
        half = abs(float(robot[side]["fingers"][0]))
        tip = pos + R[:, 2] * PAD_TIP_PAST_TCP
        for sgn in (-1.0, 1.0):
            for dz in (0.0, -0.012, -0.030):
                for dw in (-0.009, 0.0, 0.009):
                    pts.append(tip + R[:, 0] * sgn * half + R[:, 2] * dz + R[:, 1] * dw)
    return np.asarray(pts)


def _field(state, key):
    """An object state's field, from the live ObjectState or from its JSON record."""
    v = state[key] if isinstance(state, dict) else getattr(state, key)
    return np.asarray(v, float)


def roll_deg(pose):
    """The connector's turn about its own axis, degrees (atan2 of its local z axis' y and z components)."""
    q = np.asarray(pose, float)[3:7]
    x, y, z, w = q
    zy = 2 * (y * z - x * w)
    zz = 1 - 2 * (x * x + y * y)
    return -math.degrees(math.atan2(zy, zz))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.pushed_in = False
        self.dropped = False
        self.was_off = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("connector measurements missing")
        self.step = step
        m = mechanics
        conn = np.asarray(root_poses["connector"], float)
        spout = np.asarray(root_poses["new_spout"], float)
        spout_in = 1000 * float(spout[0] - S.spout_axis(S.NEW_Y)[0])
        if spout_in > PUSHED_IN_MM:
            self.pushed_in = True
        if conn[2] < DROP_Z:
            self.dropped = True
        off = (not m["on_old"]) and 1000 * float(np.linalg.norm(conn[:3] - S.connector_pos(S.OLD_Y))) >= OFF_MM
        self.was_off = self.was_off or off or bool(m["engaged"])          # latched: wherever it is set down later
        score = 0.0
        if self.was_off:
            score = 0.25
        if m["engaged"]:
            score = 0.5
        if m["locked"]:
            score = 0.75
        gap = float(np.min(np.linalg.norm(pad_points(robot) - conn[:3], axis=1))) - CONN_RADIUS
        clear = gap >= CLEAR
        st = objects["connector"]
        lin = float(np.linalg.norm(_field(st, "lin_vel")))
        ang = float(np.linalg.norm(_field(st, "ang_vel")))
        still = lin < REST_LIN and ang < REST_ANG
        done = score == 0.75 and clear and still
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.pushed_in:
            score = min(score, 0.5)
        if self.dropped:
            score = min(score, 0.25)
        return {"score": score, "success": score == 1.0, "off_empty": bool(off), "engaged": bool(m["engaged"]),
                "locked": bool(m["locked"]), "spout_in_mm": round(spout_in, 1), "pushed_in": self.pushed_in,
                "dropped": self.dropped, "clear": bool(clear), "hands_clear": bool(clear), "pad_gap_mm": round(1000 * gap, 1),
                "at_rest": bool(still), "held_frames": self.held}
