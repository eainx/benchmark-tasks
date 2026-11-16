"""Office: "Bind the booklet with the comb binder."

A manual plastic-comb binder of the upright-slot design stands on an office desk, its front toward the
arms. A half-letter booklet (15 sheets between a clear front cover and a black back cover) stands in a
desktop file sorter beside it; a 1/2 in black 15-ring binding comb lies on the desk in front of it. The
comb is laid spine down on the comb opener. The booklet is stood binding edge down in the punch slot,
held square against the side guide, and punched by pulling the punch handle (right) forward and down.
The opener lever (left) is pulled forward to spread the comb, the punched booklet is hung on the open
rings, and the lever is let back so the comb closes through the holes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import simkit  # noqa: E402
import binder as B  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, LEFT_BASE_POS, RIGHT_BASE_POS  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "comb-bind-report"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
SCENE_ONLY = False
MAX_STEPS = 80 if SCENE_ONLY else 2400
OBJECT_NAMES = ("booklet", "comb", "punch_handle", "opener_lever")

DOWN = simkit.patch_demo()
BASES = {"left": np.asarray(LEFT_BASE_POS, float), "right": np.asarray(RIGHT_BASE_POS, float)}
SHARED = {}

# rules (disclosed stand-ins), all from measured poses
SEAT_MM = 2.5            # booklet in the slot: its binding edge this close to the slot floor, centre to the slot line
SEAT_DEG = 6.0           # ... and this close to upright and square to the slot
SQUARE_MM = 1.5          # punched square: its right edge this close to the side guide
PUNCH_FRAC = 0.95        # punch handle pulled this far through its stroke: the punch goes through
COMB_SEAT_MM = 12.0      # comb laid on the opener: within this of its place, square, on the deck
COMB_SEAT_DEG = 6.0
OPEN_FRAC = 0.90         # opener lever this far through its stroke: the comb is spread open
CLOSE_FRAC = 0.10        # ... let back to within this of rest: the comb closes
HANG_X_MM = 9.0          # hung: the booklet's binding edge over the open rings (along x)
HANG_Y_MM = 3.0          # ... its holes in line with the rings (along y)
HANG_Z_MM = 8.0          # ... resting on (within this above) the spread comb
HANG_DEG = 25.0


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.binder_object(), B.sorter_object(), B.comb_box_object(), B.punch_handle_object(), B.opener_lever_object(),
             B.booklet_object(), B.comb_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_comb_bind_report", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def lever_deg(pose):
    """A lever's turn about +y from rest (negative: pulled forward), from its measured orientation."""
    R = B.mat(pose)
    return math.degrees(math.atan2(R[0, 2], R[0, 0]))


def book_geometry(pose):
    """Binding edge's centre, right edge's centre (y), up axis and tilt off upright (deg) of the booklet."""
    R = B.mat(pose)
    p = np.asarray(pose[:3], float)
    bottom = p + R @ np.array([0.0, 0.0, -B.BOOK[2] / 2])
    right = p + R @ np.array([0.0, -B.BOOK[1] / 2, 0.0])
    up = R[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(up[2], -1, 1))))
    along = R[:, 1]
    skew = math.degrees(math.atan2(abs(along[0]), abs(along[1])))
    return bottom, right, tilt, skew


class Dynamics:
    """What rigid bodies cannot do by themselves, from measured poses only (disclosed stand-ins).

    * Punch: when the booklet stands in the slot (binding edge within 2.5 mm of the slot floor, within
      2.5 mm of the slot line, within 6 deg of upright and square) and the punch handle passes 95% of its
      stroke, the booklet is punched: its 15 holes appear. Square if its right edge is within 1.5 mm of
      the side guide; otherwise the holes are off the comb's pitch (ruined).
    * Comb on the opener: laid within 12 mm and 6 deg of its place, spine on the deck, the opener's
      fingers take it -- a locked slide joint switched on where it lies (no snap).
    * Comb open/closed: drawn spread while the opener lever is past 90% of its stroke, closed again once
      let back within 10% of rest.
    * Hung: a square-punched booklet whose binding edge stands within 9 mm (x) and 3 mm (y) of the spread
      rings and rests within 8 mm above them, within 25 deg of upright (leaning on the back plate); bound once the comb closes with it
      hung: then the closed rings hold it on the comb (a joint to the comb switched on where it hangs, no snap).
    """

    def __init__(self, env):
        import omni.usd
        self.stage = omni.usd.get_context().get_stage()
        self.lock = None
        self.vis = {"hole": [], "ring": [], "open": []}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith("/comb/spring_joint"):
                self.lock = prim
            name = path.rsplit("/", 1)[-1]
            if "/booklet/" in path and name.startswith("hole") and not name.endswith("_look"):
                self.vis["hole"].append(prim)
            if "/comb/" in path and not name.endswith("_look"):
                if name.startswith("ring") and name[4:].isdigit():
                    self.vis["ring"].append(prim)
                if name.startswith("open") and name[4:].isdigit():
                    self.vis["open"].append(prim)
        if self.lock is None:
            raise RuntimeError("comb lock joint not found on the stage")
        self.bind = None
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).endswith("/booklet/hinge_joint"):
                self.bind = prim
        if self.bind is None:
            raise RuntimeError("booklet bind joint not found on the stage")
        self.levers = [p for p in self.stage.Traverse()
                       if str(p.GetPath()).endswith(("/punch_handle/hinge_joint", "/opener_lever/hinge_joint"))]
        for p in self.levers:
            attrs = {a.GetName(): a.Get() for a in p.GetAttributes() if "drive" in a.GetName() or "Limit" in a.GetName()}
            print("[lever]", p.GetPath(), attrs, flush=True)

    def _show(self, key, on):
        from pxr import UsdGeom
        for prim in self.vis[key]:
            im = UsdGeom.Imageable(prim)
            im.MakeVisible() if on else im.MakeInvisible()

    def reset(self):
        self.punched = False
        self.square = False
        self.seated = False
        self.opened_ever = False
        self.open = False
        self.hung = False
        self.bound = False
        self.lock.GetAttribute("physics:jointEnabled").Set(False)
        self.bind.GetAttribute("physics:jointEnabled").Set(False)
        col = self.bind.GetAttribute("physics:collisionEnabled")
        if col:
            col.Set(True)                         # the booklet still rests on the comb's collider until it is bound
        self._show("hole", False)
        self._show("open", False)
        self._show("ring", True)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        from pxr import Gf
        book = np.asarray(root_poses["booklet"], float)
        comb = np.asarray(root_poses["comb"], float)
        punch = lever_deg(root_poses["punch_handle"])
        opener = lever_deg(root_poses["opener_lever"])
        pf = max(0.0, -punch / B.PUNCH_STROKE_DEG)
        of = max(0.0, -opener / B.OPEN_STROKE_DEG)
        bottom, right, tilt, skew = book_geometry(book)
        # -- punch
        in_slot = bool(abs(bottom[2] - B.SLOT_FLOOR) < SEAT_MM / 1000 and abs(bottom[0] - B.SLOT_X) < SEAT_MM / 1000
                   and tilt < SEAT_DEG and skew < SEAT_DEG)
        guide_mm = float(1000 * (right[1] - B.GUIDE_Y))
        if not self.punched and in_slot and pf >= PUNCH_FRAC:
            self.punched = True
            self.square = bool(abs(guide_mm) <= SQUARE_MM)
            self._show("hole", True)
        # -- comb on the opener
        seat = np.asarray(B.comb_seat(), float)
        Rc = B.mat(comb)
        cyaw = math.degrees(math.atan2(abs(Rc[1, 0]), abs(Rc[0, 0])))
        ctilt = math.degrees(math.acos(float(np.clip(Rc[2, 2], -1, 1))))
        cd = comb[:3] - seat
        if not self.seated and math.hypot(cd[0], cd[1]) < COMB_SEAT_MM / 1000 and abs(cd[2]) < 0.003 \
                and cyaw < COMB_SEAT_DEG and ctilt < COMB_SEAT_DEG:
            j = self.lock
            j.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*(float(v) for v in comb[:3])))
            x, y, z, w = (float(v) for v in comb[3:7])
            j.GetAttribute("physics:localRot0").Set(Gf.Quatf(w, x, y, z))
            j.GetAttribute("physics:jointEnabled").Set(True)
            self.seated = True
        # -- comb open / closed with the lever
        if self.seated and not self.open and of >= OPEN_FRAC:
            self.open = True
            self.opened_ever = True
            self._show("ring", False)
            self._show("open", True)
        if self.open and of <= CLOSE_FRAC:
            self.open = False
            self._show("open", False)
            self._show("ring", True)
            if self.hung:
                self.bound = True
                # the closed rings through its holes hold it on the comb: a joint switched on where it hangs
                Rc_ = B.mat(comb)
                rel_p = Rc_.T @ (book[:3] - comb[:3])
                rel_R = Rc_.T @ B.mat(book)
                w, x, y, z = _quat_wxyz(rel_R)
                j = self.bind
                j.GetAttribute("physics:localPos0").Set(Gf.Vec3f(*(float(v) for v in rel_p)))
                j.GetAttribute("physics:localRot0").Set(Gf.Quatf(w, x, y, z))
                j.GetAttribute("physics:localPos1").Set(Gf.Vec3f(0.0, 0.0, 0.0))
                j.GetAttribute("physics:localRot1").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
                j.GetAttribute("physics:jointEnabled").Set(True)
        # -- hung on the spread rings
        comb_top = B.COMB_FLOOR + 2 * B.COMB_R
        line_x = float(comb[0]) if self.seated else B.COMB_X
        on_rings = bool(abs(bottom[0] - line_x) < HANG_X_MM / 1000 and abs(bottom[1] - float(comb[1])) < HANG_Y_MM / 1000
                    and -0.002 < bottom[2] - comb_top < HANG_Z_MM / 1000 and tilt < HANG_DEG)
        if self.open and self.punched and self.square and on_rings:
            self.hung = True
        if self.hung and not on_rings and not self.bound:
            self.hung = False
        SHARED.update(book=book.copy(), comb=comb.copy(), punched=self.punched, seated=self.seated, open=self.open,
                      hung=self.hung, left_q=np.asarray(robot["left"]["joints"], float),
                      right_q=np.asarray(robot["right"]["joints"], float))
        return {"punch_deg": round(punch, 1), "opener_deg": round(opener, 1), "in_slot": bool(in_slot),
                "slot_floor_mm": round(1000 * (bottom[2] - B.SLOT_FLOOR), 1), "slot_line_mm": round(1000 * (bottom[0] - B.SLOT_X), 1),
                "tilt_deg": round(tilt, 1), "skew_deg": round(skew, 1), "guide_mm": round(guide_mm, 1),
                "punched": self.punched, "square": self.square, "comb_seated": self.seated,
                "comb_seat_mm": round(1000 * math.hypot(cd[0], cd[1]), 1), "comb_open": self.open,
                "comb_opened": self.opened_ever, "on_rings": bool(on_rings), "hung": self.hung, "bound": self.bound,
                "ring_x_mm": round(1000 * (bottom[0] - line_x), 1), "ring_y_mm": round(1000 * (bottom[1] - float(comb[1])), 1),
                "ring_z_mm": round(1000 * (bottom[2] - comb_top), 1),
                "ik_left": SHARED.get("left_ik"), "ik_right": SHARED.get("right_ik")}


def _quat_wxyz(R):
    t = float(np.trace(R))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    if i == 1:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------------------
def frame_x(approach):
    """TCP rotation whose jaws close along world x, approaching along ``approach``."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def frame_in_xz(approach):
    """TCP rotation approaching along ``approach`` (in the x-z plane), jaws closing in the x-z plane."""
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _tcp(tip, rot, label, seconds, finger):
    tip = np.asarray(tip, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def _seq(items):
    marks, out, t = {}, [], 0.0
    for w in items:
        if isinstance(w, str):
            marks[w] = t
        else:
            out.append(w)
            t += w.duration
    return out, marks, t


REST = {"left": np.array([0.16, 0.16, B.TOP + 0.20]), "right": np.array([0.16, -0.16, B.TOP + 0.20])}
SIDE_L = frame_x((0.0, -0.87, -0.5)) @ np.diag([-1.0, -1.0, 1.0])   # left hand on the upright booklet, from its left side (jaws turned over: its reachable roll)
SIDE_R = frame_x((0.0, 0.87, -0.5))       # right hand, from its right side
COMB_GRIP = None                          # set below
G_SHEET = FINGER_CLOSED
PUNCH_PITCH = 25.0        # the hand on the punch grip, tilted forward this far from vertical; it turns with the handle
OPEN_PITCH = 35.0
G_COMB = 0.004
G_KNOB = 0.008
G_HANDLE = 0.006


COMB_GRIP = frame_x((0.0, 0.5, -0.87))   # right hand on the comb from its right side, jaws across its width


def booklet_targets():
    """Desired booklet centres (identity orientation: upright, along y)."""
    hz = B.BOOK[2]
    yc = B.GUIDE_Y + B.BOOK[1] / 2 + 0.0015            # its right edge 1.5 mm off the guide, then pushed on
    above_slot = np.array([B.SLOT_X, yc, B.DECK_Z + 0.030 + hz / 2])
    in_slot = np.array([B.SLOT_X, yc, B.SLOT_FLOOR + hz / 2 + 0.0005])
    pushed = in_slot + np.array([0.0, -0.0030, 0.0])
    comb_top = B.COMB_FLOOR + 2 * B.COMB_R
    over_comb = np.array([B.COMB_X + 0.001, B.COMB_Y, comb_top + 0.020 + hz / 2])
    on_comb = np.array([B.COMB_X + 0.001, B.COMB_Y, comb_top + 0.0015 + hz / 2])
    return above_slot, in_slot, pushed, over_comb, on_comb


TOP_OFF = B.BOOK[2] / 2 - 0.014                    # pinch this far below the booklet's top edge
GRIP_DY = {"left": 0.030, "right": -0.030}          # each hand's pinch point along the top edge


def sheet_tip(center, side):
    return np.asarray(center, float) + np.array([0.0, GRIP_DY[side], TOP_OFF])


def _book(center, side, label, seconds, finger, extra=(0.0, 0.0, 0.0)):
    """A waypoint for a hand holding (or about to hold) the upright booklet whose centre is ``center``."""
    rot = SIDE_L if side == "left" else SIDE_R
    return _tcp(sheet_tip(center, side) + np.asarray(extra, float), rot, label, seconds, finger)


def handle_tip(deg):
    a = math.radians(B.PUNCH_REST_DEG)
    piv = np.asarray(B.PUNCH_PIVOT, float)
    r0 = np.array([-math.cos(a), 0.0, math.sin(a)]) * B.PUNCH_LEN + np.array([0.0, -B.GRIP_LEN * 0.72, 0.0])
    return piv + ry(math.radians(deg)) @ r0


def knob_tip(deg):
    a = math.radians(B.OPEN_REST_DEG)
    piv = np.asarray(B.OPEN_PIVOT, float)
    r0 = np.array([-math.cos(a), 0.0, math.sin(a)]) * (B.OPEN_LEN + 0.012)
    return piv + ry(math.radians(deg)) @ r0


def demo():
    if SCENE_ONLY:
        return {side: [_tcp(REST[side], DOWN, "look", 1.5, FINGER_OPEN), _tcp(REST[side], DOWN, "hold", MAX_STEPS / HZ, FINGER_OPEN)]
                for side in ("left", "right")}
    above_slot, in_slot, pushed, over_comb, on_comb = booklet_targets()
    hover_slot = np.array([above_slot[0], above_slot[1], B.DECK_Z + 0.006 + B.BOOK[2] / 2])
    up = np.array([0.0, 0.0, 1.0])
    # -- right: comb onto the opener, then the punch ------------------------------------------------------
    c0 = np.array([B.COMB_C[0] + B.COMB_SPINE[0] / 2 - B.COMB_R, B.COMB_C[1], B.TOP + B.COMB_R])
    c1 = np.array([B.COMB_X + B.COMB_SPINE[0] / 2 - B.COMB_R + 0.006, B.COMB_Y, B.COMB_FLOOR + B.COMB_R])   # +6 mm: it sits that far forward in the jaws
    h_rot0 = frame_in_xz((math.sin(math.radians(PUNCH_PITCH)), 0.0, -math.cos(math.radians(PUNCH_PITCH))))
    right = [
        _tcp(REST["right"], DOWN, "look", 1.0, FINGER_OPEN),
        _tcp(c0 + up * 0.06, COMB_GRIP, "over the comb", 2.0, FINGER_OPEN),
        _tcp(c0 + up * 0.002, COMB_GRIP, "down round the comb", 1.2, FINGER_OPEN),
        _tcp(c0 + up * 0.002, COMB_GRIP, "take the comb", 0.6, G_COMB),
        _tcp(c0 + up * 0.06, COMB_GRIP, "lift it", 1.0, G_COMB),
        _tcp(np.array([c0[0], c0[1], B.TOP + 0.200]), COMB_GRIP, "lift it high", 1.2, G_COMB),
        _tcp(np.array([c1[0] - 0.06, c1[1], B.TOP + 0.200]), COMB_GRIP, "over the deck", 1.8, G_COMB),
        _tcp(c1 + up * 0.05, COMB_GRIP, "over the opener", 1.4, G_COMB),
        _tcp(c1 + up * 0.004, COMB_GRIP, "lay it on the opener", 1.4, G_COMB),
        _tcp(c1 + up * 0.004, COMB_GRIP, "let go of it", 0.6, FINGER_OPEN),
        _tcp(c1 + up * 0.06, COMB_GRIP, "up off it", 1.0, FINGER_OPEN),
        "MARK_COMB_DONE",
    ]
    _, mr, tr = _seq(right)
    # -- left: booklet out of the sorter into the slot ----------------------------------------------------
    b0 = np.asarray(B.booklet_pos(), float)
    left = [
        _tcp(REST["left"], DOWN, "look", 1.0, FINGER_OPEN),
        _book(b0, "left", "B:approach the booklet", 2.0, FINGER_OPEN, (0.0, 0.06, 0.035)),
        _book(b0, "left", "B:onto its top edge", 1.2, FINGER_OPEN),
        _book(b0, "left", "B:pinch it", 0.6, G_SHEET),
        "MARK_L_GRASP",
        _book(b0 + up * 0.160, "left", "C:lift it out", 1.8, G_SHEET),
        _book(np.array([B.SLOT_X - 0.050, above_slot[1] + 0.030, above_slot[2] + 0.010]), "left", "C:carry it over", 1.6, G_SHEET),
        _book(above_slot, "left", "C:over the slot", 1.6, G_SHEET),
        _book(hover_slot, "left", "C:line it up over the slot", 1.0, G_SHEET),
        _book(hover_slot, "left", "C:line it up over the slot", 1.6, G_SHEET),
        _book(in_slot, "left", "C:down into the slot", 2.0, G_SHEET),
        _book(pushed, "left", "C:against the side guide", 1.0, G_SHEET),
        "MARK_IN_SLOT",
    ]
    _, ml, tl = _seq(left)
    # right: punch once the booklet is in (and the comb is done)
    t_go = max(tr, ml["MARK_IN_SLOT"] + 0.4)
    right.append(_tcp(REST["right"] + np.array([0.08, -0.04, 0.06]), h_rot0, "toward the punch handle", max(1.0, t_go - tr), FINGER_OPEN))
    g0 = handle_tip(0.0)
    right += [
        _tcp(g0 + np.array([-0.04, 0.0, 0.03]), h_rot0, "over the punch grip", 1.6, FINGER_OPEN),
        _tcp(g0, h_rot0, "onto the punch grip", 1.0, FINGER_OPEN),
        _tcp(g0, h_rot0, "take the punch grip", 0.6, G_HANDLE),
    ]
    for k in range(1, 6):
        d = -B.PUNCH_STROKE_DEG * k / 5 - (1.5 if k == 5 else 0.0)
        right.append(_tcp(handle_tip(d), ry(math.radians(d)) @ h_rot0, "pull the punch handle", 0.7, G_HANDLE))
    right.append(_tcp(handle_tip(-B.PUNCH_STROKE_DEG - 1.5), ry(math.radians(-B.PUNCH_STROKE_DEG - 1.5)) @ h_rot0, "punch through", 0.6, G_HANDLE))
    for k in range(4, -1, -1):
        d = -B.PUNCH_STROKE_DEG * k / 5
        right.append(_tcp(handle_tip(d), ry(math.radians(d)) @ h_rot0, "let it back up", 0.6, G_HANDLE))
    right += [
        _tcp(g0, h_rot0, "let go of the grip", 0.5, FINGER_OPEN),
        _tcp(g0 + np.array([-0.05, 0.0, 0.04]), h_rot0, "off the grip", 1.0, FINGER_OPEN),
        "MARK_PUNCHED",
        _book(pushed, "right", "R:to the booklet's top edge", 2.0, FINGER_OPEN, (0.0, -0.06, 0.035)),
        _book(pushed, "right", "R:onto its top edge", 1.0, FINGER_OPEN),
        _book(pushed, "right", "R:pinch it", 0.6, G_SHEET),
        "MARK_R_GRASP",
    ]
    _, mr, tr = _seq(right)
    # left: hold it in the slot until the right hand has it, then let go and go to the opener lever
    _, ml, tl = _seq(left)
    left.append(_book(pushed, "left", "C:hold it square", max(0.2, mr["MARK_R_GRASP"] + 0.3 - tl), G_SHEET))
    left += [
        _book(pushed, "left", "C:let go of it", 0.6, FINGER_OPEN),
        _book(pushed, "left", "off it", 1.0, FINGER_OPEN, (0.0, 0.08, 0.06)),
    ]
    k0 = knob_tip(0.0)
    k_rot0 = frame_in_xz((math.sin(math.radians(OPEN_PITCH)), 0.0, -math.cos(math.radians(OPEN_PITCH))))
    left += [
        _tcp(k0 + np.array([-0.05, 0.0, 0.04]), k_rot0, "over the opener knob", 2.0, FINGER_OPEN),
        _tcp(k0, k_rot0, "onto the knob", 1.0, FINGER_OPEN),
        _tcp(k0, k_rot0, "take the knob", 0.6, G_KNOB),
        "MARK_KNOB",
    ]
    for k in range(1, 6):
        d = -B.OPEN_STROKE_DEG * k / 5 - (1.5 if k == 5 else 0.0)
        left.append(_tcp(knob_tip(d), ry(math.radians(d)) @ k_rot0, "pull the opener lever", 0.6, G_KNOB))
    left.append("MARK_OPEN")
    _, ml, tl = _seq(left)
    # right: lift the booklet out, over the comb; hang it once the comb is open
    right += [
        _book(in_slot + up * 0.035, "right", "C:lift it out of the slot", 1.4, G_SHEET),
        _book(over_comb, "right", "C:over the comb", 2.2, G_SHEET),
    ]
    _, mr, tr = _seq(right)
    right.append(_book(over_comb, "right", "C:wait for the comb to open", max(0.2, ml["MARK_OPEN"] + 0.4 - tr), G_SHEET))
    right += [
        _book(on_comb, "right", "C:hang it on the rings", 1.6, G_SHEET),
        _book(on_comb, "right", "C:hang it on the rings", 0.8, G_SHEET),
        "MARK_HUNG",
    ]
    _, mr, tr = _seq(right)
    # left: hold the lever open till it is hung, then let it back slowly; right lets go after
    left.append(_tcp(knob_tip(-B.OPEN_STROKE_DEG - 1.5), ry(math.radians(-B.OPEN_STROKE_DEG - 1.5)) @ k_rot0, "hold it open",
                     max(0.2, mr["MARK_HUNG"] + 0.3 - tl), G_KNOB))
    for k in range(4, -1, -1):
        d = -B.OPEN_STROKE_DEG * k / 5
        left.append(_tcp(knob_tip(d), ry(math.radians(d)) @ k_rot0, "let the lever back", 0.6, G_KNOB))
    left += [
        _tcp(k0, k_rot0, "let go of the knob", 0.5, FINGER_OPEN),
        _tcp(k0 + np.array([-0.05, 0.0, 0.05]), k_rot0, "off the knob", 1.0, FINGER_OPEN),
        "MARK_CLOSED",
        _tcp(REST["left"], DOWN, "rest", 2.0, FINGER_OPEN),
        _tcp(REST["left"], DOWN, "rest", 3.0, FINGER_OPEN),
    ]
    _, ml, tl = _seq(left)
    right.append(_book(on_comb, "right", "C:hold it while the comb closes", max(0.2, ml["MARK_CLOSED"] - tr), G_SHEET))
    right += [
        _book(on_comb, "right", "C:let go of it", 0.6, FINGER_OPEN),
        _book(on_comb, "right", "off it", 1.0, FINGER_OPEN, (0.0, -0.07, 0.05)),
        _tcp(REST["right"], DOWN, "rest", 2.0, FINGER_OPEN),
        _tcp(REST["right"], DOWN, "rest", 3.0, FINGER_OPEN),
    ]
    left, _, _ = _seq(left)
    right, _, _ = _seq(right)
    return {"left": left, "right": right}


class DemoController:
    """Follows the waypoint plans by IK each step. "B:"/"R:" waypoints steer the hand onto the booklet's
    measured top edge. "C:" waypoints are planned for a booklet held upright at its nominal pinch; the hand
    pose is recomputed from the hold measured when it was pinched, so the booklet itself goes upright to the
    planned place, and its measured centre trims the last millimetres (what a person does by eye)."""

    def __init__(self, plans):
        self.plans = plans
        self.hold = {}
        self.corr = {"left": np.zeros(3), "right": np.zeros(3)}
        self.phase = {"left": None, "right": None}

    def action(self, t):
        from yam_arena import ik
        out = []
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, label = plan.target(t)
            pos = np.array(pos, float)
            rot = np.asarray(rot, float)
            tip_nom = pos + rot @ np.array([0.0, 0.0, PAD_TIP])
            if label[:2] in ("B:", "R:") and "book" in SHARED:
                book = SHARED["book"]
                Rb = B.mat(book)
                c_nom = tip_nom - np.array([0.0, GRIP_DY[side], TOP_OFF])
                extra = c_nom - (np.asarray(B.booklet_pos(), float) if label[:2] == "B:" else booklet_targets()[2])
                tip = np.asarray(book[:3], float) + Rb @ np.array([0.0, GRIP_DY[side], TOP_OFF]) + extra
                pos = tip - rot @ np.array([0.0, 0.0, PAD_TIP])
                if "pinch" in label and side + "_q" in SHARED:
                    p_h, R_h = ik.tcp_pose(SHARED[side + "_q"], plan.base_pos)
                    self.hold[side] = (R_h.T @ Rb, R_h.T @ (np.asarray(book[:3], float) - p_h))
            elif label[:2] == "C:" and side in self.hold and "book" in SHARED:
                c_nom = tip_nom - np.array([0.0, GRIP_DY[side], TOP_OFF])
                Rhb, phb = self.hold[side]
                R_h = Rhb.T
                fine = None
                if any(w in label for w in ("over the slot", "into the slot", "guide", "square")):
                    fine = "slot"
                elif any(w in label for w in ("over the comb", "wait for the comb", "rings", "while the comb")):
                    fine = "comb"
                if fine != self.phase[side]:
                    self.corr[side] = np.zeros(3)             # a new fine phase starts its trim afresh
                    self.phase[side] = fine
                if fine is not None:
                    bk = SHARED["book"]
                    bottom_now = np.asarray(bk[:3], float) + B.mat(bk) @ np.array([0.0, 0.0, -B.BOOK[2] / 2])
                    goal = c_nom - np.array([0.0, 0.0, B.BOOK[2] / 2])
                    if fine == "comb" and "comb" in SHARED:
                        goal[0] = float(SHARED["comb"][0]) + 0.001  # over the rings where the comb actually lies
                        goal[1] = float(SHARED["comb"][1])          # its holes in line with them
                    err = goal - bottom_now
                    err[2] = 0.0
                    if "guide" in label or "square" in label:
                        err[1] = min(err[1], 0.0)
                    self.corr[side] = np.clip(self.corr[side] + 0.04 * err, -0.008, 0.008)
                pos = c_nom + self.corr[side] - R_h @ phb
                rot = R_h
            quat = ik.matrix_to_quat_xyzw(rot)
            q, pe, re = ik.solve(pos, quat, plan.q, base_pos=plan.base_pos, iters=200)
            if pe > 0.002 or re > 0.03:                      # stuck in a local minimum: try fresh starts, keep the nearest good one
                best = (pe + 0.05 * re, q, pe, re)
                for seed in range(1, 7):
                    q0 = np.asarray(plan.q, float) + np.random.RandomState(seed).uniform(-0.6, 0.6, len(plan.q))
                    q2, pe2, re2 = ik.solve(pos, quat, q0, base_pos=plan.base_pos, iters=300)
                    cost = pe2 + 0.05 * re2 + 0.002 * float(np.linalg.norm(np.asarray(q2) - np.asarray(plan.q)))
                    if cost < best[0]:
                        best = (cost, q2, pe2, re2)
                _, q, pe, re = best
            SHARED[side + "_ik"] = (round(1000 * pe, 1), round(re, 3), label)
            plan.q = q
            g = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            out.append(np.r_[q, g])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
