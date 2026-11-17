"""Scorer for stocking the spiral anti-sweep hook.

Everything is measured from the bodies' poses: each card's euro-slot top point (on the rod or not, and how far
back along the hook), its tilt, its speed, and the finger pads' clearance (pad boxes from the measured joints at
the runner's arm bases, station.BASES). The card can only get behind the coil's front pitch by the coil turning:
the wire crossings block it otherwise.

Ladder (milestones latch; the final step needs the end state):
  0.2  the new card lifted out of the carton
  0.4  the new card hanging on the rod by its slot
  0.6  ... and in the coil, behind its front end
  0.8  ... and wound back at least 3/4 pitch (21 mm) behind the coil's front end (held by the coil)
  1.0  ... with all three cards hanging on the rod, both hands clear (60 mm) and every card at rest
       (linear < 0.05 m/s and angular < 0.5 rad/s) for 20 consecutive frames (1 s)
A card on the bench (dropped) scores 0. Returns the runner's `clear` and `dropped` keys.
"""
from __future__ import annotations

import math

import numpy as np

import spiral as C

HZ = 20
ALL_CARDS = C.CARD_NAMES + (C.NEW_CARD,)
CLEAR = 0.060
REST_LIN, REST_ANG, REST_FRAMES = 0.05, 0.5, 20
IN_COIL = C.COIL_X1 + 0.004                 # the slot behind the coil's front end (local x)
WOUND = C.COIL_X1 + 0.75 * C.PITCH          # ... three quarters of a pitch behind it (past the wire's free end)


def _box_pts(box):
    c, R, h = (np.asarray(x, float) for x in box)
    pts = [c]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts)


def _speed(o, key):
    v = o.get(key, (0, 0, 0)) if isinstance(o, dict) else getattr(o, key, (0, 0, 0))
    return float(np.linalg.norm(np.asarray(v, float)))


def card_state(pose):
    """The card's slot-top point (world) and its tilt from hanging plumb (deg)."""
    p = np.asarray(pose, float)
    R = C.mat(p)
    root_off = -(C.HOLE_TOP + C.HOLE_H) / 2                     # lower board centre (card frame z)
    slot_top = p[:3] + R @ np.array([0.0, 0.0, C.CARD_H / 2 - C.HOLE_TOP - root_off])
    up = R @ np.array([0.0, 0.0, 1.0])
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    return slot_top, tilt


def on_rod(slot_top):
    loc = C.L(slot_top)
    dz = slot_top[2] - (C.ROD_Z + C.ROD_R)
    return (-0.0025 <= dz <= 0.004 and abs(loc[1] - C.HOOK_Y) <= 0.006
            and C.ROD_X1 <= loc[0] <= C.ROD_X0), loc


def coil_turn(pose):
    """The coil's turn about its axle (rad, from its set position)."""
    Rw = C.mat(np.asarray(pose, float))
    c, s = math.cos(C.YAW), math.sin(C.YAW)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    R = Rz.T @ Rw
    return math.atan2(R[2, 1], R[1, 1])


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.lifted = self.hung = self.in_coil = self.wound = False
        self.fell = False
        self.rest = 0
        self.turn = 0.0
        self.prev_ang = None
        self.best = 0.0

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        cards, hung_all, still = {}, True, True
        for n in ALL_CARDS:
            st, tilt = card_state(root_poses[n])
            hung, loc = on_rod(st)
            if st[2] < C.FLOOR + 0.03 or (n != C.NEW_CARD and not hung and tilt > 60.0):
                self.fell = True
            hung_all = hung_all and hung
            o = objects.get(n, {}) if isinstance(objects, dict) else {}
            lin, ang = _speed(o, "lin_vel"), _speed(o, "ang_vel")
            still = still and lin < REST_LIN and ang < REST_ANG
            cards[n] = {"on_rod": bool(hung), "back_of_coil_front_mm": round(1000 * (loc[0] - C.COIL_X1), 1),
                        "slot_above_rod_mm": round(1000 * (st[2] - (C.ROD_Z + C.ROD_R)), 1), "tilt_deg": round(tilt, 1)}
            if n == C.NEW_CARD:
                new_st, new_hung, new_loc = st, hung, loc
        # milestones (latched)
        if new_st[2] > C.FLOOR + C.SHIP_WALL + C.RISER + C.CARD_H - C.HOLE_TOP + 0.050:     # card bottom over the rim
            self.lifted = True
        if self.lifted and new_hung:
            self.hung = True
            if new_loc[0] >= IN_COIL:
                self.in_coil = True
            if new_loc[0] >= WOUND:
                self.wound = True
        # the coil's accumulated turn (for the record)
        a = coil_turn(root_poses["coil"])
        if self.prev_ang is not None:
            d = (a - self.prev_ang + math.pi) % (2 * math.pi) - math.pi
            self.turn += d
        self.prev_ang = a
        gap = 1.0
        if mechanics and mechanics.get("pad_boxes"):
            pts = np.vstack([_box_pts(b) for b in mechanics["pad_boxes"]])
            knob = C.W((C.KNOB_X, C.HOOK_Y, C.COIL_Z))
            targets = [knob, new_st]
            gap = float(min(np.linalg.norm(pts - t, axis=1).min() for t in targets))
        clear = gap >= CLEAR
        final_ok = (self.wound and hung_all and new_loc[0] >= WOUND and clear and still)
        self.rest = self.rest + 1 if final_ok else 0
        score = 0.2 * self.lifted + 0.2 * self.hung + 0.2 * self.in_coil + 0.2 * self.wound
        success = self.rest >= REST_FRAMES
        if success:
            score = 1.0
        if self.fell:
            score = 0.0
            success = False
        self.best = max(self.best, score)
        return {"score": round(score, 3), "success": bool(success), "cards": cards,
                "milestones": {"lifted": self.lifted, "hung": self.hung, "in_coil": self.in_coil, "wound": self.wound},
                "coil_turn_deg": round(math.degrees(self.turn), 1), "pad_gap_mm": round(1000 * gap, 1),
                "rest_frames": self.rest, "clear": bool(clear), "dropped": bool(self.fell)}
