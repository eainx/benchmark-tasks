"""Office labelling: "Put the new tape cassette in the label maker."

A desktop label maker stands on an office desk, keyboard toward the arms, its cassette bay latched shut
under a flip-up hood cover with the used-up 24 mm tape cassette inside. The new cassette lies face up on
the desk beside its opened carton. The cover's latch tab is pressed and the cover swung up past upright
(it stays open on its stop), the old cassette lifted out by its ends and set down on the desk, the new one
put in the one way it fits and pressed down until it clicks, and the cover swung shut until it latches.
The label maker (1.09 kg) is free on its rubber feet, so the left hand steadies it on its front deck.
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
import labeler as L  # noqa: E402
from scoring import TrialScorer, mat, cover_deg  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "label-maker-cassette-swap"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = ("labeler", "cover", "old_cassette", "new_cassette", "carton")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (L.labeler_object(), L.cover_object(), L.old_cassette_object(), L.carton_object(),
             L.new_cassette_object(), L.label_strip_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_label_maker_cassette_swap", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                                          # measured state the scripted demo steers by

TAB_MM = 12.0                                        # a closed hand's pad tips this close to the latch tab's top: pressing it
TAB_STEPS = 6                                        # ... for 0.3 s: the latch lets go
SHUT_DEG = 1.5                                       # the cover back down within this of shut for 0.2 s: it latches
SEAT_XY_MM = 2.5                                     # the new cassette this close to its seat across the bay ...
SEAT_Z_MM = 3.0                                      # ... and this close in height
SEAT_DEG = 6.0                                       # ... square (tilt and turn) within this
SEAT_STEPS = 4                                       # ... for 0.2 s: it clicks down


def _bases():
    from yam_arena import station
    return {side: np.asarray(station.BASES[side], float) for side in ("left", "right")}


def hand_tips(robot):
    """Per hand: the pad tips' centre and the jaw half-opening, from the measured joints and finger travel."""
    from yam_arena import ik
    bases = _bases()
    out = {}
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, bases[side])
        tip = pos + R[:, 2] * PAD_TIP
        half = abs(float(robot[side]["fingers"][0]))
        # sample points on both finger pads: across the jaw opening, along the pad's length and width
        pads = [tip + R[:, 0] * sgn * (half + 0.004) - R[:, 2] * dl + R[:, 1] * dw
                for sgn in (-1.0, 1.0) for dl in (0.0, 0.02, 0.04) for dw in (-0.009, 0.009)]
        out[side] = (tip, half, pads)
    return out


def seat_world(lab):
    """The new cassette's seat (its reference point) where the label maker now stands."""
    return np.asarray(lab[:3], float) + mat(lab) @ (np.asarray(L.SEAT, float) - np.asarray(L.labeler_pos(), float))


class Dynamics:
    """What rigid bodies cannot do by themselves, from measured poses only (disclosed stand-ins).

    * Cover latch: while latched, the cover's hinge stops at 0.5 degrees. A closed hand's pad tips held within
      12 mm of the latch tab's top for 0.3 s release it (the hinge then opens to its 110-degree stop, past
      upright, where the cover stays). Swung back down to within 1.5 degrees of shut for 0.2 s, it latches again.
    * Cassette click: the new cassette set into the bay within 2.5 mm of its seat across the bay, 3 mm in height
      and 6 degrees square, for 0.2 s, is held down there by a joint to the body (switched on, no position
      target). Turned round, its cut-away corner meets the print head post and it cannot get down to its seat.
    """

    def __init__(self, env):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.j = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for key, end in (("cover", "/cover/hinge_joint"), ("seat", "/new_cassette/hinge_joint")):
                if path.endswith(end):
                    self.j[key] = prim
        if len(self.j) != 2:
            raise RuntimeError(f"joints not found on the stage: {sorted(self.j)}")

    def _upper(self, deg):
        self.j["cover"].GetAttribute("physics:upperLimit").Set(float(deg))

    def reset(self):
        self.latched = True
        self.was_open = False
        self.seated = False
        self.tab_n = self.shut_n = self.seat_n = 0
        self._upper(0.5)
        self.j["seat"].GetAttribute("physics:jointEnabled").Set(False)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        lab = np.asarray(root_poses["labeler"], float)
        cov = np.asarray(root_poses["cover"], float)
        new = np.asarray(root_poses["new_cassette"], float)
        hands = hand_tips(robot)
        deg = cover_deg(lab, cov)
        Rc = mat(cov)
        tab_top = cov[:3] + Rc @ (np.asarray(L.TAB_C) - np.asarray(L.COVER_POS) + np.array([0.0, 0.0, L.TAB[2] / 2]))
        tab_mm = min(1000 * float(np.linalg.norm(h[0] - tab_top)) for h in hands.values())
        pressed = any(h[1] < 0.004 and float(np.linalg.norm(h[0] - tab_top)) < TAB_MM / 1000 for h in hands.values())
        if self.latched:
            self.tab_n = self.tab_n + 1 if pressed else 0
            if self.tab_n >= TAB_STEPS:
                self.latched = False
                self._upper(L.COVER_OPEN_DEG)
        else:
            if deg > 60.0:
                self.was_open = True
            self.shut_n = self.shut_n + 1 if (self.was_open and deg < SHUT_DEG) else 0
            if self.shut_n >= SHUT_STEPS:
                self.latched = True
                self._upper(0.5)
        # the cassette's click
        seat = seat_world(lab)
        Rl, Rn = mat(lab), mat(new)
        rel = Rl.T @ (new[:3] - seat)
        Rrel = Rl.T @ Rn
        tilt = math.degrees(math.acos(float(np.clip(Rrel[2, 2], -1.0, 1.0))))
        yaw = math.degrees(math.atan2(Rrel[1, 0], Rrel[0, 0]))
        xy = 1000 * float(np.hypot(rel[0], rel[1]))
        if not self.seated:
            ok = xy < SEAT_XY_MM and abs(1000 * rel[2]) < SEAT_Z_MM and tilt < SEAT_DEG and abs(yaw) < SEAT_DEG
            self.seat_n = self.seat_n + 1 if ok else 0
            if self.seat_n >= SEAT_STEPS:
                self.seated = True
                self.j["seat"].GetAttribute("physics:jointEnabled").Set(True)
        SHARED.update(lab=lab.copy(), cov=cov.copy(), new=new.copy(), seat=seat, deg=deg, latched=self.latched,
                      seated=self.seated)
        return {"tcp_tips": [list(map(float, h[0])) for h in hands.values()],
                "pads": [list(map(float, p)) for h in hands.values() for p in h[2]],
                "cover_deg": round(deg, 1), "latched": self.latched, "was_open": self.was_open,
                "seated": self.seated, "tab_mm": round(tab_mm, 1), "seat_xy_mm": round(xy, 1),
                "seat_z_mm": round(1000 * float(rel[2]), 1), "seat_tilt_deg": round(tilt, 1), "seat_yaw_deg": round(yaw, 1)}


SHUT_STEPS = 4


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

def _rot(z, x):
    """TCP rotation from its approach axis z and its jaw axis x (made square to z)."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - np.dot(x, z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _tcp(tip, rot, label, seconds, finger):
    tip = np.asarray(tip, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


REST = {"left": (0.16, 0.24, L.TOP + 0.20), "right": (0.16, -0.24, L.TOP + 0.20)}
CASS_GRIP = -0.022                                  # jaws closed well inside the cassette's 66 mm: they stop on it (force-limited)
EDGE_GRIP = -0.0060                                  # jaws across the cover's front edge (plate top to tab foot, 15 mm)
PIVOT = np.asarray(L.HINGE_PIVOT, float)
EDGE_LOCAL = np.array([L.TAB_C[0] + 0.004 - PIVOT[0], 0.0, 0.008])   # pinch point on the front edge, from the hinge
PHI = math.radians(30.0)                             # the pinch comes in 30 degrees down from the cover's plane


def edge_pose(theta_deg):
    """Pad-tip point and TCP rotation pinching the cover's front edge with the cover open theta degrees."""
    R = _ry(math.radians(theta_deg))
    tip = PIVOT + R @ EDGE_LOCAL
    a = R @ np.array([math.cos(PHI), 0.0, -math.sin(PHI)])           # in toward the hinge, tilted down
    jaw = R @ np.array([math.sin(PHI), 0.0, math.cos(PHI)])          # across the edge's thickness
    return tip, _rot(a, jaw)


A_HOOK = np.array([0.7, 0.0, -0.7])                 # the right hand works the cover coming in 45 degrees down from the front
R_HOOK = _rot(A_HOOK, (0.7, 0.0, 0.7))
A_CASS = np.array([0.0, -0.7, -0.7])                # the left hand takes a cassette from the +y side, jaws along x (across 66 mm)
R_CASS = _rot(A_CASS, (1.0, 0.0, 0.0))
STEADY_L = np.array([0.362, 0.118, L.SHELL_TOP - 0.002])   # the left hand's closed pads pressing on the front deck
STEADY_R = np.array([0.356, -0.030, L.SHELL_TOP - 0.002])  # the right hand's, on the deck's other end
DROP = np.array([0.250, 0.095])                     # where the old cassette is set down
CASS_TIP_ABOVE_CENTRE = -0.002                      # pad tips this far above a gripped cassette's middle (in the bay: 3 mm over the deck)
HOOK = np.array([L.TAB_C[0] + 0.002 - L.HINGE_PIVOT[0], 0.0, -0.003])    # under the tab, from the hinge
OUTER = np.array([L.COVER_X[0] + 0.008 - L.HINGE_PIVOT[0], 0.0, L.COVER_H + 0.004])   # just off the hood's top, at its front
PLAN_T = {}


def _seq(items):
    marks, out, t = {}, [], 0.0
    for w in items:
        if isinstance(w, str):
            marks[w] = t
        else:
            out.append(w)
            t += w.duration
    return out, marks, t


def _arc(local, th):
    return PIVOT + _ry(math.radians(th)) @ local


def demo():
    O, C = FINGER_OPEN, FINGER_CLOSED
    bay = np.array([L.BAY_C[0], L.BAY_C[1], L.FLOOR_Z + L.CASSETTE[2] / 2 + CASS_TIP_ABOVE_CENTRE])
    new = np.array([L.NEW_C[0], L.NEW_C[1], L.TOP + L.CASSETTE[2] / 2 + CASS_TIP_ABOVE_CENTRE])
    tab = np.array([L.TAB_C[0], L.TAB_C[1], L.TAB_C[2] + L.TAB[2] / 2])
    drop_z = L.TOP + L.CASSETTE[2] / 2 + CASS_TIP_ABOVE_CENTRE + 0.002
    hook0 = _arc(HOOK, 0.0)
    # -- right hand: unlatch and swing the cover open, steady the label maker, swing the cover shut
    r = [_tcp(REST["right"], R_HOOK, "start", 0.5, O),
         _tcp(tab + [-0.02, 0, 0.05], R_HOOK, "over the latch tab", 2.0, C),
         _tcp(tab + [0, 0, 0.004], R_HOOK, "onto the tab", 1.0, C),
         _tcp(tab + [0, 0, -0.002], R_HOOK, "press the tab", 0.8, C),
         _tcp(tab + [-0.014, 0, 0.012], R_HOOK, "off the tab", 0.6, C),
         _tcp([hook0[0] - 0.014, hook0[1], hook0[2] - 0.001], R_HOOK, "fingers into the recess", 1.2, C),
         _tcp(hook0, R_HOOK, "under the tab", 0.8, C)]
    for th in range(10, 110, 10):
        r.append(_tcp(_arc(HOOK, th), R_HOOK, "swing the cover up", 0.35, C))
    up = _arc(HOOK, 100.0)
    r += [_tcp(up + [-0.03, 0, 0.02], R_HOOK, "off the cover", 0.8, C),
          "cover open",
          _tcp(STEADY_R + [0, -0.02, 0.05], R_HOOK, "over the front deck", 1.6, C),
          _tcp(STEADY_R, R_HOOK, "steady the label maker", 0.6, C),
          _tcp(STEADY_R, R_HOOK, "steady the label maker", 26.4, C),
          _tcp(STEADY_R + [0, -0.02, 0.05], R_HOOK, "off the deck", 0.8, C),
          _tcp(_arc(OUTER, 108.0) + [0.0, 0.0, 0.02], R_HOOK, "over the open cover", 1.4, C),
          _tcp(_arc(OUTER, 108.0), R_HOOK, "behind its front edge", 0.6, C)]
    for th in range(100, 0, -10):
        r.append(_tcp(_arc(OUTER, th), R_HOOK, "swing the cover shut", 0.35, C))
    r += [_tcp(_arc(OUTER, 0.0) + [-0.03, 0, 0.03], R_HOOK, "off the cover", 0.8, C),
          _tcp(REST["right"], R_HOOK, "clear", 2.0, O),
          _tcp(REST["right"], R_HOOK, "clear", 2.5, O)]
    right, mr, tr = _seq(r)
    # -- left hand: steady the label maker, swap the cassettes, steady it again
    su = STEADY_L + [0, 0.04, 0.05]
    l_items = [_tcp(REST["left"], R_CASS, "start", 0.5, O),
               _tcp(su, R_CASS, "over the front deck", 1.8, C),
               _tcp(STEADY_L, R_CASS, "steady the label maker", 1.0, C),
               _tcp(STEADY_L, R_CASS, "steady the label maker", 10.9, C),
               _tcp(su, R_CASS, "off the deck", 0.8, O),
               _tcp(bay + [0, 0.03, 0.07], R_CASS, "over the old cassette", 2.0, O),
               _tcp(bay + [0, 0, 0.012], R_CASS, "down round it", 1.2, O),
               _tcp(bay, R_CASS, "down round it", 0.8, O),
               _tcp(bay, R_CASS, "take the old cassette", 0.6, CASS_GRIP),
               _tcp(bay + [0, 0, 0.08], R_CASS, "lift it out", 1.4, CASS_GRIP),
               _tcp([DROP[0], DROP[1], L.TOP + 0.10], R_CASS, "over the desk", 2.0, CASS_GRIP),
               _tcp([DROP[0], DROP[1], drop_z], R_CASS, "set it down", 1.4, CASS_GRIP),
               _tcp([DROP[0], DROP[1], drop_z], R_CASS, "let go", 0.5, O),
               _tcp([DROP[0], DROP[1], L.TOP + 0.08], R_CASS, "up", 0.8, O),
               _tcp(new + [0, 0, 0.06], R_CASS, "over the new cassette", 1.8, O),
               _tcp(new, R_CASS, "down round it", 1.2, O),
               _tcp(new, R_CASS, "take the new cassette", 0.6, CASS_GRIP),
               _tcp(new + [0, 0, 0.09], R_CASS, "lift it", 1.2, CASS_GRIP),
               _tcp(bay + [0, 0, 0.05], R_CASS, "over the bay", 2.4, CASS_GRIP),
               _tcp(bay + [0, 0, 0.012], R_CASS, "lower it in", 1.4, CASS_GRIP),
               _tcp(bay + [0, 0, -0.001], R_CASS, "press it down", 1.0, CASS_GRIP),
               _tcp(bay + [0, 0, -0.001], R_CASS, "press it down", 0.6, CASS_GRIP),
               _tcp(bay + [0, 0, -0.001], R_CASS, "let go", 0.5, O),
               _tcp(bay + [0, 0.02, 0.06], R_CASS, "up out of the bay", 1.0, O),
               _tcp(su, R_CASS, "over the front deck", 1.8, C),
               _tcp(STEADY_L, R_CASS, "steady the label maker", 1.0, C),
               _tcp(STEADY_L, R_CASS, "steady the label maker", 7.3, C),
               _tcp(su, R_CASS, "off the deck", 1.0, O),
               _tcp(REST["left"], R_CASS, "clear", 2.0, O)]
    lt = sum(w.duration for w in l_items)
    l_items.append(_tcp(REST["left"], R_CASS, "clear", max(0.5, tr - lt), O))
    left, ml, tl = _seq(l_items)
    PLAN_T.update(mr)
    return {"left": left, "right": right}


STEER = ("lower it in", "press it down")


class DemoController:
    """Follows the waypoint plans by IK each step, steering the new cassette's last centimetre over its seat by
    its measured pose (what a person does by eye)."""

    def __init__(self, plans):
        self.plans = plans
        self.corr = np.zeros(3)

    def action(self, t):
        from yam_arena import ik
        out = []
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, label = plan.target(t)
            pos = np.array(pos, float)
            if side == "left" and label in STEER and "new" in SHARED and not SHARED.get("seated"):
                err = np.asarray(SHARED["seat"], float) - np.asarray(SHARED["new"][:3], float)
                err[2] = 0.0
                self.corr = np.clip(self.corr + 0.08 * err, -0.012, 0.012)
            if side == "left" and label in STEER:
                pos = pos + self.corr
            q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), plan.q, base_pos=plan.base_pos, iters=200)
            SHARED[side + "_ik"] = (round(1000 * pe, 1), round(re, 3), label)
            plan.q = q
            g = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            out.append(np.r_[q, g])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
