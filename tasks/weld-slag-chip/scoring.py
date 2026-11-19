"""Scorer for "Clean the slag off the weld.".

The bead's slag crust is cracked across into six pieces (zones 25 mm long). A piece is off when it has been
knocked loose (moved 2 mm or more, so its breakable joint has let go) and its centre now lies 9 mm or more to
the side of the bead line or past the bead's ends: judged on the current state each step, so a loose piece
still lying on the bead does not count.

Ladder:
  0.15  each slag piece off the bead (0.90 for all six)
  1.00  all six off, and the hammer put down: lying still on the bench or fixture (under 2 cm/s), with no
        finger pad within 30 mm of its handle
A piece that first moves while a finger pad is within 15 mm of it was knocked off by hand, not with the tool:
it earns nothing (the hot, sharp crust is never touched by hand). A hammer that ends up below the bench top
(dropped off the bench) is `dropped`: no credit for putting it down. These are observations only; nothing is
triggered by them. Real trial: the six zones are marked on the coupon with soapstone; a zone counts when its
bead metal is bare (photo); the hammer and the hands are seen on video.
Returns the runner's `clear` and `dropped` keys."""
from __future__ import annotations

import numpy as np

import slag as S

OFF_M = 0.009
MOVED_M = 0.002
HAND_NEAR = 0.015
CLEAR_M = 0.030
STILL = 0.02


def _box_pts(box):
    c, R, h = (np.asarray(x, float) for x in box)
    pts = [c]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                pts.append(c + R @ (h * np.array([sx, sy, sz])))
    return np.asarray(pts)


def _field(o, name):
    return o[name] if isinstance(o, dict) else getattr(o, name)


def _quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def handle_points(pose):
    p = np.asarray(pose, float)
    R = _quat_mat(p[3:7])
    xs = np.linspace(S.HANDLE[1], S.SHANK[0], 9)
    return p[:3] + (R @ np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)])).T


class TrialScorer:
    def __init__(self):
        self.step = 0
        self.home = None
        self.moved = [False] * S.N_SLAG
        self.by_hand = [False] * S.N_SLAG
        self.off = [False] * S.N_SLAG
        self.dropped = False

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        self.step = step
        pads = None
        if mechanics and mechanics.get("pad_boxes"):
            pads = np.vstack([_box_pts(b) for b in mechanics["pad_boxes"]])
        pos = [np.asarray(root_poses[n], float)[:3] for n in S.SLAGS]
        if self.home is None:
            self.home = [np.array([S.BX, S.slag_y(i), S.SLAG_Z]) for i in range(S.N_SLAG)]
        for i in range(S.N_SLAG):
            d = float(np.linalg.norm(pos[i] - self.home[i]))
            if not self.moved[i] and d >= MOVED_M:
                self.moved[i] = True
                if pads is not None and float(np.min(np.linalg.norm(pads - pos[i], axis=1))) < HAND_NEAR:
                    self.by_hand[i] = True
            on_bead = abs(pos[i][0] - S.BX) < OFF_M and S.BEAD_Y0 - 0.004 < pos[i][1] < S.BEAD_Y1 + 0.004
            self.off[i] = bool(self.moved[i] and not on_bead)
        h = np.asarray(root_poses["hammer"], float)
        if h[2] < S.TOP - 0.02:
            self.dropped = True
        o = objects["hammer"]
        v = float(np.linalg.norm(np.asarray(_field(o, "lin_vel"), float)))
        gap = 1.0
        if pads is not None:
            hp = handle_points(h)
            gap = float(np.min(np.linalg.norm(pads[:, None, :] - hp[None, :, :], axis=2)))
        clear = gap >= CLEAR_M
        down = bool(clear and v < STILL and h[2] < S.TOP + 0.05 and not self.dropped)
        good = [self.off[i] and not self.by_hand[i] for i in range(S.N_SLAG)]
        n = sum(good)
        score = 0.15 * n
        success = bool(n == S.N_SLAG and down)
        if success:
            score = 1.0
        return {"score": round(float(score), 3), "success": success, "pieces_off": n,
                "off": self.off, "by_hand": self.by_hand, "hammer_down": down, "hammer_z": round(float(h[2]), 4),
                "pad_gap_mm": round(1000 * gap, 1), "clear": bool(clear), "dropped": bool(self.dropped)}
