"""Stateful scorer for "Sort the letters into their room boxes." From the measured poses of the four
envelopes and the wire organizer and the robot's measured tool centres only.

An envelope is *filed* in a pigeonhole when all eight corners of its 241 x 105 x 4 mm body lie inside
that compartment (between its side boards, above its floor and below the shelf over it, and at least
2 mm behind the opening's plane): no part of it sticks out. It is *dropped* if it lies on the desk
(its centre within 30 mm of the desk top) outside both the organizer and the sorter.

Score: 0.2 for each envelope that has been filed in the pigeonhole labelled with its own room (latched,
0 to 0.8); 1.0 when all four are filed correctly at once, every envelope at rest (below 10 mm/s and
0.3 rad/s), and both hands' measured pads (each jaw's pad, at its measured opening) at least 30 mm from
every envelope's body and 30 mm out of the sorter's openings, held for 1 s. Caps at 0.4: any envelope filed in, or resting inside,
a pigeonhole of another room; any envelope dropped; the organizer displaced more than 10 mm or tilted
more than 10 degrees from where it stood at any time (not steadied while letters were drawn out).
"""
from __future__ import annotations

import math

import numpy as np

import mail as T

HZ = 20
HOLD_S = 1.0
PER = 0.2
CAP = 0.4
CLEAR = 0.03                          # pads at least this far from every envelope and out of the openings
REST_LIN = 0.010
REST_ANG = 0.3
PAD_TIP = 0.013                       # pad tips past the tool centre (yam_arena PAD_TIP_PAST_TCP)
PAD_LEN = 0.036
INSIDE_MM = 2.0
ORG_MOVE_MM = 10.0
ORG_TILT_DEG = 10.0
_HALF = np.array(T.ENV) / 2


def corners(pose):
    pose = np.asarray(pose, float)
    r = T.quat_matrix(pose[3:7])
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append(pose[:3] + r @ (_HALF * np.array([sx, sy, sz])))
    return np.asarray(out)


def compartment(p):
    """The room whose compartment volume holds world point p (inside the boards), or None."""
    for r, row in enumerate(T.ROOMS):
        for c, room in enumerate(row):
            o = T.slot_centre(r, c)
            if (abs(p[0] - o[0]) <= T.SLOT_W / 2 and o[2] - 0.001 <= p[2] <= o[2] + T.SLOT_H
                    and T.FRONT_Y < p[1] <= T.FRONT_Y + T.SLOT_D):
                return room
    return None


def filed_in(pose):
    """The room an envelope is filed in (every corner inside one compartment, 2 mm behind the opening)."""
    cs = corners(pose)
    rooms = {compartment(c) for c in cs}
    if len(rooms) == 1 and None not in rooms and float(cs[:, 1].min()) >= T.FRONT_Y + INSIDE_MM / 1000:
        return rooms.pop()
    return None


def obb_distance(p, pose):
    """Distance from world point p to an envelope's body (0 inside)."""
    pose = np.asarray(pose, float)
    r = T.quat_matrix(pose[3:7])
    q = r.T @ (np.asarray(p, float) - pose[:3])
    d = np.maximum(np.abs(q) - _HALF, 0.0)
    return float(np.linalg.norm(d))


def pad_points(robot):
    """Sample points on both jaws' pads of both hands, from the measured joints and finger travel through
    the URDF's forward kinematics at the runner's arm bases (station.BASES)."""
    from yam_arena import ik, station
    pts = []
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, np.asarray(station.BASES[side], float))
        a, c, w = R[:, 2], R[:, 0], R[:, 1]
        half = abs(float(np.asarray(robot[side]["fingers"], float)[0]))
        for sgn in (-1, 1):
            for k in np.linspace(0.0, PAD_LEN, 4):
                for sw in (-0.009, 0.0, 0.009):
                    pts.append(pos + a * (PAD_TIP - k) + c * sgn * (half + 0.004) + w * sw)
        pts.append(pos)
    return np.asarray(pts)


def _speed(objects, name):
    st = objects.get(name) if isinstance(objects, dict) else None
    if st is None:
        return 0.0, 0.0
    lv = np.asarray(getattr(st, "lin_vel", np.zeros(3)), float)
    av = np.asarray(getattr(st, "ang_vel", np.zeros(3)), float)
    return float(np.linalg.norm(lv)), float(np.linalg.norm(av))


def _tilt_deg(q):
    r = T.quat_matrix(q)
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.held = 0
        self.org0 = None
        self.org_bad = False
        self.wrong = False
        self.dropped = False
        self.filed_ok = set()

    def update(self, step, objects, root_poses, robot, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        org = np.asarray(root_poses["organizer"], float)
        if self.org0 is None:
            self.org0 = org.copy()
        moved = 1000 * float(np.linalg.norm(org[:2] - self.org0[:2]))
        tilt = _tilt_deg(org[3:7])
        if moved > ORG_MOVE_MM or tilt > ORG_TILT_DEG:
            self.org_bad = True
        correct_now = 0
        rest = True
        report = {}
        pads = pad_points(robot)
        clear = True
        for name, (room, _, _) in zip(T.ENV_NAMES, T.LETTERS):
            pose = np.asarray(root_poses[name], float)
            where = filed_in(pose)
            inside = compartment(pose[:3])
            if where == room:
                correct_now += 1
                self.filed_ok.add(name)
            if (where is not None and where != room) or (inside is not None and inside != room):
                self.wrong = True
            on_desk = pose[2] - T.TOP < 0.030
            in_org = (abs(pose[0] - T.ORG_X) <= T.ORG_D / 2 and abs(pose[1] - T.ORG_Y) <= T.ORG_L / 2)   # turned 90 degrees
            in_sorter = (T.X0 <= pose[0] <= T.X0 + T.SORT_W and T.FRONT_Y <= pose[1] <= T.FRONT_Y + T.SORT_D)
            if on_desk and not in_org and not in_sorter:
                self.dropped = True
            lin, ang = _speed(objects, name)
            if lin > REST_LIN or ang > REST_ANG:
                rest = False
            if min(obb_distance(p, pose) for p in pads) < CLEAR:
                clear = False
            report[name] = {"filed_in": where, "trailing_mm": round(1000 * float(corners(pose)[:, 1].min()), 1)}
        for p in pads:                    # out of the openings: not in front of (within 30 mm) or inside any compartment
            if (T.FRONT_Y - CLEAR < p[1] < T.FRONT_Y + T.SLOT_D and T.X0 + T.BOARD < p[0] < T.X0 + T.SORT_W - T.BOARD
                    and T.TOP + T.BOARD < p[2] < T.TOP + T.SORT_H - T.BOARD):
                clear = False
        correct = len(self.filed_ok)
        score = PER * correct
        done = correct_now == len(T.LETTERS) and clear and rest
        self.held = self.held + 1 if done else 0
        if done and self.held >= HOLD_S * HZ:
            score = 1.0
        capped = self.wrong or self.dropped or self.org_bad
        if capped:
            score = min(score, CAP)
        return {"score": round(score, 3), "success": score == 1.0, "correct": correct, "wrong_slot": self.wrong,
                "dropped": self.dropped, "organizer_moved_mm": round(moved, 1), "organizer_tilt_deg": round(tilt, 1),
                "organizer_bad": self.org_bad, "hands_clear": clear, "clear": clear, "at_rest": rest,
                "correct_now": correct_now, "held_frames": self.held, "envelopes": report}
