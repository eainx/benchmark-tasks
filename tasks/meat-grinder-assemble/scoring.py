"""Scorer for "Assemble the meat grinder." From the measured poses of the auger, knife, plate and ring,
the robot's TCPs and the fit state task.Dynamics reports (each fit read from the parts' measured poses)."""
from __future__ import annotations

import math

import numpy as np

import grinder as G

HZ = 20
HOLD_S = 1.0
CLEAR = 0.08                     # m: both hands at least this far from the mouth
HOME_MM = 1.0                    # plate and knife within this of fully home (pressed by the ring)
PAD_TIP = 0.013                  # m: finger pad tips past the TCP (yam_arena PAD_TIP_PAST_TCP)
PAD_HALF = 0.012                 # m: half a pad's width
REST_V, REST_W = 0.01, 0.3       # m/s, rad/s: parts at rest
DROP_Z = G.TOP - 0.02            # a part this low has fallen off the bench
RING_EXT = (G.RING_HOME[0] - G.RING_LIP / 2, G.RING_HOME[0] - G.RING_LIP / 2 + G.RING_DEPTH, G.RING_R + 0.009)


def _get(state, key):
    """A field of a part's measured state, from the runner's state object or a trajectory's dict."""
    v = state[key] if isinstance(state, dict) else getattr(state, key)
    return np.asarray(v, float)


def _to_mouth(p):
    """Distance from a point to the fitted ring's outer extent (a cylinder on the barrel's axis)."""
    x0, x1, r = RING_EXT
    dx = max(x0 - p[0], 0.0, p[0] - x1)
    dr = max(math.hypot(p[1] - G.AXIS_Y, p[2] - G.AXIS_Z) - r, 0.0)
    return math.hypot(dx, dr)


def hand_clearance(arm):
    """How far a hand's pads can be from the ring at most-reaching: the TCP's distance less the pads'
    reach (tips past the TCP, the measured jaw opening and half a pad)."""
    tcp = np.asarray(arm["tcp"], float)[:3]
    opening = max(abs(float(f)) for f in arm["fingers"])
    return _to_mouth(tcp) - math.hypot(PAD_TIP, opening + PAD_HALF)


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.dropped = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("fit state missing")
        self.step = step
        m = mechanics
        knife_right = m["knife_on"] and not m["knife_backwards"]
        score = 0.0
        if m["auger_seated"]:
            score = 0.25
            if knife_right:
                score = 0.5
                if m["plate_on"]:
                    score = 0.75
        home = (m["plate_on"] and abs(m["plate_axial_mm"]) <= HOME_MM and knife_right and m["knife_axial_mm"] >= -HOME_MM)
        full = score == 0.75 and m["ring_on"] and m["ring_turns"] >= 1.5 and home
        clear = all(hand_clearance(robot[s]) >= CLEAR for s in ("left", "right"))
        at_rest = all(np.linalg.norm(_get(objects[k], "lin_vel")) < REST_V and np.linalg.norm(_get(objects[k], "ang_vel")) < REST_W
                      for k in ("auger", "knife", "plate", "ring"))
        if any(float(root_poses[k][2]) < DROP_Z for k in ("auger", "knife", "plate", "ring")):
            self.dropped = True                  # a part off the bench: it is not fitted, so its stage is not reached
        self.held = self.held + 1 if (full and clear and at_rest) else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if m["knife_backwards"] or m["plate_before_knife"]:
            score = min(score, 0.25)             # wrong way round or out of order: the grinder cannot cut
        return {"score": score, "success": score == 1.0, "auger_seated": bool(m["auger_seated"]),
                "knife_right_way": bool(knife_right), "knife_backwards": bool(m["knife_backwards"]),
                "plate_on": bool(m["plate_on"]), "plate_before_knife": bool(m["plate_before_knife"]),
                "ring_turns": m["ring_turns"], "plate_axial_mm": m["plate_axial_mm"], "knife_axial_mm": m["knife_axial_mm"],
                "home": bool(home), "hands_clear": bool(clear), "at_rest": bool(at_rest), "held_frames": self.held,
                "clear": bool(clear), "dropped": bool(self.dropped)}
