"""Bathroom repair: "Replace the showerhead with the new one."

A real-size section of tiled shower wall: out of it comes a straight chrome shower arm, screwed into a
drop-ear elbow behind the tiles, carrying the old showerhead on its swivel nut; under it a wire shower
basket hangs on the wall, and beside the arm a stainless shower shelf carries the new head in its foam
cradle. The old head's nut has to come off the arm's 1/2"-14 NPSM thread (4 turns) without the head
being dropped, the head goes into the basket, and the new head goes onto the thread square and is run
up hand-tight. The stuck old nut takes more torque than the taped elbow holds, so the arm is held
while the nut turns.
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
import shower as S  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "showerhead-swap"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 2600
OBJECT_NAMES = ("arm", "old_head", "new_head", "new_face")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (S.wall_object(), S.basket_object(), S.shelf_object(), S.arm_object(), S.old_object(), S.new_object(),
             S.new_face_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_showerhead_swap", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                                          # measured state the scripted demo steers by

ENGAGE_RADIAL = 0.003                                # m: nut centre off the arm's axis
ENGAGE_AXIAL = 0.004                                 # m: nut's back face off the arm's end
ENGAGE_DEG = 6.0                                     # nut axis off the arm's axis
LOCK_D = 2.0                                         # N m s / deg: seated on its washer (no position target,
HOLD_D = 0.3                                         # so nothing snaps); a started nut left alone, held by its thread
BALL_K = 0.05                                        # N m / deg: the new ball, stiff till the head is on


class Dynamics:
    """What rigid bodies cannot do by themselves, from measured poses only (a disclosed stand-in).

    * Thread: a nut on the arm's thread is a hinge about the arm's axis whose damping stands for the
      thread's drag (the stuck old nut five times the taped elbow's; a new nut running up, little).
      The nut's travel along the thread (7.3 mm over 4 turns) is not modelled: it turns where it sits.
    * The old head comes off when it has been turned 4 turns back (its hinge is switched off).
    * The new head's nut starts on the thread when it is put to the arm's end square (centre within
      3 mm of the axis, back face within 4 mm of the arm's end, axis within 6 degrees): its hinge is
      switched on. While no hand grips it, a started nut stays where it was left (a holding drive
      standing for the thread's friction). Turned 4 turns on it is seated on its washer: the hinge is then held stiff, so
      turning it further turns the arm. Turned back off past the start, it comes off again.
    * The arm turns in its elbow, about its own axis, against a damping drive (its taped thread).
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        self.old_joint = self.new_joint = self.ball = None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith("/old_head/hinge_joint"):
                self.old_joint = prim
            if path.endswith("/new_head/hinge_joint"):
                self.new_joint = prim
            if path.endswith("/new_face/hinge_joint"):
                self.ball = prim
        if self.old_joint is None or self.new_joint is None or self.ball is None:
            raise RuntimeError("thread joints not found on the stage")

    def reset(self):
        self.old_on = True
        self.old_prev = None
        self.old_turns = 0.0                         # turned back (loosening) since the start
        self.new_on = False
        self.new_prev = None
        self.new_turns = 0.0
        self.seated = False
        self.elbow_max = 0.0
        self.old_joint.GetAttribute("physics:jointEnabled").Set(True)
        self._new(False)
        self._ball(True)

    def _ball(self, straight):
        """The new head's ball holds the face straight out of the box (a stiff spring)."""
        self.ball.GetAttribute("drive:angular:physics:stiffness").Set(BALL_K if straight else 0.0)
        self.ball.GetAttribute("drive:angular:physics:targetPosition").Set(0.0)

    def _new(self, on, lock_deg=None):
        j = self.new_joint
        j.GetAttribute("physics:jointEnabled").Set(bool(on))
        j.GetAttribute("drive:angular:physics:stiffness").Set(0.0)
        j.GetAttribute("drive:angular:physics:damping").Set(LOCK_D if lock_deg is not None else S.NEW_THREAD_DAMP)
        self.new_on = bool(on)
        self.seated = lock_deg is not None

    def before_step(self, step, action):
        a = np.asarray(action, float).ravel()
        self.right_open = a.size >= 14 and float(a[13]) > 0.9          # hand opened (a grip on the face is ~0.68)

    def after_step(self, step, objects, root_poses, robot):
        arm = np.asarray(root_poses["arm"], float)
        old = np.asarray(root_poses["old_head"], float)
        new = np.asarray(root_poses["new_head"], float)
        face = np.asarray(root_poses["new_face"], float)
        elbow = S.elbow_deg(arm)
        self.elbow_max = max(self.elbow_max, abs(elbow))
        target, out = S.thread_target(arm)
        SHARED.update(target=target.copy(), new_nut=new[:3].copy(), old_nut=old[:3].copy(), face=face[:3].copy(),
                      right_tcp=np.asarray(robot["right"]["tcp"], float)[:3], left_tcp=np.asarray(robot["left"]["tcp"], float)[:3])
        # -- the old head coming off
        if self.old_on:
            tw = S.old_twist(arm, old)
            if self.old_prev is not None:
                d = (tw - self.old_prev + math.pi) % (2 * math.pi) - math.pi
                self.old_turns = max(0.0, self.old_turns - d / (2 * math.pi))
            self.old_prev = tw
            if self.old_turns >= S.TURNS_ON:
                self.old_joint.GetAttribute("physics:jointEnabled").Set(False)
                self.old_on = False
        # -- the new head going on
        rel = new[:3] - target
        axial = float(np.dot(rel, out))
        radial = float(np.linalg.norm(rel - axial * out))
        n_axis = S.mat(new) @ np.array([0.0, 1.0, 0.0])             # nut toward the wall
        ang = math.degrees(math.acos(float(np.clip(np.dot(n_axis, -out), -1.0, 1.0))))
        SHARED.update(new_err=(target - new[:3]).copy(), new_axial=axial, new_radial=radial, new_ang=ang)
        if not self.new_on and radial < ENGAGE_RADIAL and abs(axial) < ENGAGE_AXIAL and ang < ENGAGE_DEG:
            self._new(True)
            self.new_prev = S.new_twist(arm, new)
            self.new_turns = 0.0
            self.new_start_deg = math.degrees(self.new_prev)
            self.hold_deg = self.new_start_deg
        elif self.new_on and not self.seated:
            tw = S.new_twist(arm, new)
            d = (tw - self.new_prev + math.pi) % (2 * math.pi) - math.pi
            self.new_turns += d / (2 * math.pi)
            self.new_prev = tw
            acc = self.new_start_deg + 360.0 * self.new_turns     # unwrapped joint angle (deg)
            if self.new_turns >= S.TURNS_ON:
                self._new(True, lock_deg=acc)
            elif self.new_turns < -0.25:
                self._new(False)
            else:
                # the thread's friction: the nut stays where it was left while no hand turns it
                held = getattr(self, "right_open", False)
                if not held:
                    self.hold_deg = acc                          # turning by hand: follow it
                j = self.new_joint
                j.GetAttribute("drive:angular:physics:damping").Set(HOLD_D if held else S.NEW_THREAD_DAMP)
        SHARED.update(new_on=self.new_on, seated=self.seated, old_on=self.old_on, new_turns=self.new_turns)
        n = S.face_normal(face)
        return {"elbow_deg": round(elbow, 1), "elbow_max_deg": round(self.elbow_max, 1),
                "old_on_thread": bool(self.old_on), "old_turns_back": round(self.old_turns, 2),
                "new_on_thread": bool(self.new_on), "new_turns": round(self.new_turns, 2), "seated": bool(self.seated),
                "new_start_radial_mm": round(1000 * radial, 1), "new_start_axial_mm": round(1000 * axial, 1),
                "new_axis_deg": round(ang, 1), "face_down_deg": round(math.degrees(math.asin(float(np.clip(-n[2], -1, 1)))), 1),
                "ik_right": SHARED.get("right_ik_err"), "ik_left": SHARED.get("left_ik_err")}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

def _axis_rot(v, a):
    v = np.asarray(v, float) / np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def _frame(approach, up):
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def grip(approach, roll_deg, up=(0.0, 0.0, 1.0)):
    """TCP rotation: approach axis, turned ``roll_deg`` about it (jaws close along the local x)."""
    return _axis_rot(approach, math.radians(roll_deg)) @ _frame(approach, up)


Y = np.array([0.0, 1.0, 0.0])
ROLL0 = 265.0                                       # wrist's reachable roll window about +y: 95..265
ROLL1 = 95.0
FACE_GRIP = -(2 * S.FACE_R - 0.012) / 2
PICK_OPEN = FINGER_OPEN
ARM_GRIP = -(2 * S.ARM_R - 0.008) / 2
REST = {"left": (0.16, 0.24, S.TOP + 0.20), "right": (0.16, -0.24, S.TOP + 0.20)}
HOLD_AP = np.array([0.5, 0.0, -0.87])               # the left hand comes down onto the arm from in front
STROKE = 3.4                                         # s per regrip stroke


def _tcp(tip, rot, label, seconds, finger):
    tip = np.asarray(tip, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def _old_face_tip():
    c = np.asarray(S.old_pos()) - Y * (S.BALL_OFF + S.FACE_OFF)
    return c + Y * (S.FACE_T / 2 - 0.002)


def _strokes(tip, sign, n, last_deg, label, first_from=None):
    """Regrip strokes about +y at ``tip``: ``sign`` +1 tightens (roll 95 -> 265), -1 loosens. The first
    stroke may start part-way (``first_from``)."""
    a, b = (ROLL1, ROLL0) if sign > 0 else (ROLL0, ROLL1)
    out = []
    for k in range(n):
        start = first_from if (k == 0 and first_from is not None) else a
        span = (b - start) if k < n - 1 else math.copysign(last_deg, b - a)
        mid = start + span / 2
        end = start + span
        out += [_tcp(tip, grip(Y, mid), label, STROKE * 0.25, FACE_GRIP),
                _tcp(tip, grip(Y, end), label, STROKE * 0.25, FACE_GRIP)]
        if k < n - 1:
            out += [_tcp(tip, grip(Y, end), "let go", 0.4, FINGER_OPEN),
                    _tcp(tip, grip(Y, (end + a) / 2), "turn the open hand back", 0.6, FINGER_OPEN),
                    _tcp(tip, grip(Y, a), "turn the open hand back", 0.6, FINGER_OPEN),
                    _tcp(tip, grip(Y, a), "grip again", 0.4, FACE_GRIP)]
    return out


ENGAGE_ROLL = 180.0                                  # the new head goes on held with the jaws level


def _pick_roll():
    """The roll of the downward grip on the new face that turns, carried round 90 degrees about x,
    into the grip it goes onto the thread with (roll 180 about +y: jaws along x at the pick, clear of
    the wall)."""
    want = grip(Y, ENGAGE_ROLL)
    rx = _axis_rot((1.0, 0.0, 0.0), math.pi / 2)
    best = min(range(0, 360, 1), key=lambda r: np.linalg.norm(rx @ grip((0, 0, -1), r, (1, 0, 0)) - want))
    return float(best)


def demo():
    tip_old = _old_face_tip()
    target = np.asarray(S.old_pos())                  # the nut's place on the thread
    tip_new_on = target - Y * (S.BALL_OFF + S.FACE_OFF) + Y * (S.FACE_T / 2 - 0.002)
    face0 = np.array(S.new_face_pos())
    tip_pick = face0 + Y * (S.FACE_T / 2 - 0.002)
    ap_drop = np.array([0.7071, 0.0, -0.7071])
    r_drop = _axis_rot(np.cross(Y, ap_drop), math.pi / 2)
    r_mid = _axis_rot(np.cross(Y, ap_drop), math.pi / 4)
    fc_drop = np.array([0.400, S.WALL_Y - 0.060, S.TOP + 0.180])          # the old face's centre, over the basket
    tip_drop = fc_drop + ap_drop * (S.FACE_T / 2 - 0.002)
    tip_off = tip_old - Y * 0.05
    fc_drop2 = np.array([0.445, S.WALL_Y - 0.075, S.TOP + 0.120])      # old face's centre over the basket
    tip_drop2 = fc_drop2 + Y * (S.FACE_T / 2 - 0.002)
    r_pick = grip(Y, ENGAGE_ROLL)
    rx = lambda deg: _axis_rot((1.0, 0.0, 0.0), math.radians(deg))
    g_off = grip(Y, ROLL0 - 130.0)                    # holding the old head after it comes off
    right = [
        Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
        _tcp(tip_old - Y * 0.07, grip(Y, ROLL0), "in front of the old head", 2.5, FINGER_OPEN),
        _tcp(tip_old, grip(Y, ROLL0), "round its face", 1.2, FINGER_OPEN),
        _tcp(tip_old, grip(Y, ROLL0), "grip the old head", 0.5, FACE_GRIP),
        _tcp(tip_old, grip(Y, ROLL0), "wait for the arm to be held", 1.5, FACE_GRIP + 1e-6),
        *_strokes(tip_old, -1, 9, 130.0, "unscrew it"),
        "MARK_OFF",
        _tcp(tip_off, g_off, "draw it off the arm", 1.2, FACE_GRIP),
        _tcp(tip_off + np.array([0.03, 0.0, -0.06]), g_off, "lower it to the basket", 1.5, FACE_GRIP),
        _tcp(tip_drop2, g_off, "over the basket", 1.5, FACE_GRIP),
        _tcp(tip_drop2, g_off, "let it go", 0.5, FINGER_OPEN),
        _tcp(tip_drop2 - Y * 0.08, g_off, "back out", 1.0, FINGER_OPEN),
        _tcp(tip_pick - Y * 0.08 + np.array([0, 0, 0.02]), r_pick, "in front of the new head", 2.2, FINGER_OPEN),
        _tcp(tip_pick - Y * 0.06, r_pick, "in front of the new head", 1.0, PICK_OPEN + 1e-6),
        _tcp(tip_pick, r_pick, "round its face", 1.2, PICK_OPEN + 1e-6),
        _tcp(tip_pick, r_pick, "grip the new head", 0.5, FACE_GRIP),
        _tcp(tip_pick + np.array([0, 0, 0.012]), r_pick, "lift it off the shelf", 1.0, FACE_GRIP + 1e-6),
        _tcp(tip_pick + np.array([0, 0, 0.012]), r_pick, "bring it to the arm", 0.2, FACE_GRIP),
        _tcp(tip_pick + np.array([0, -0.08, 0.012]), r_pick, "bring it to the arm", 1.5, FACE_GRIP),
        _tcp(tip_new_on - Y * 0.08, r_pick, "bring it to the arm", 1.8, FACE_GRIP),
        _tcp(tip_new_on - Y * 0.05, r_pick, "line it up with the arm", 1.2, FACE_GRIP),
        "MARK_ON",
        _tcp(tip_new_on - Y * 0.012, r_pick, "line it up with the arm", 1.2, FACE_GRIP),
        _tcp(tip_new_on, r_pick, "onto the thread", 1.5, FACE_GRIP),
        _tcp(tip_new_on, r_pick, "onto the thread", 0.8, FACE_GRIP),
        *_strokes(tip_new_on, +1, 12, 170.0, "screw it on", first_from=ENGAGE_ROLL),
        "MARK_LETGO",
        _tcp(tip_new_on, grip(Y, ROLL0), "let go of the new head", 1.2, FINGER_OPEN),
    ]
    marks = {}
    t = 0.0
    for w in right:
        if isinstance(w, str):
            marks[w] = t
        else:
            t += w.duration
    right = [w for w in right if not isinstance(w, str)]
    PLAN_T.update(marks)
    t_tight = sum(w.duration for w in right)
    end_rot = right[-1].yaw
    right += [
        _tcp(tip_new_on - Y * 0.07, end_rot, "back off", 1.5, FINGER_OPEN),
        _tcp(tip_new_on - Y * 0.14 + np.array([-0.04, 0.0, 0.04]), end_rot, "away from the new head", 1.5, FINGER_OPEN),
        Waypoint(2.0, REST["right"], FINGER_OPEN, "stand clear", DOWN),
    ]
    hold = np.array([S.ARM_X, S.WALL_Y - 0.035, S.ARM_Z]) + HOLD_AP * 0.010
    r_hold = grip(HOLD_AP, 90.0)
    left = [
        Waypoint(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
        _tcp(np.array([0.17, 0.03, S.TOP + 0.33]), r_hold, "round the wall's edge", 1.5, FINGER_OPEN),
        _tcp(hold - HOLD_AP * 0.06, r_hold, "over the shower arm", 1.5, FINGER_OPEN),
        _tcp(hold, r_hold, "round the arm", 1.2, FINGER_OPEN),
        _tcp(hold, r_hold, "hold the arm", 0.6, ARM_GRIP),
    ]
    t_l = sum(w.duration for w in left)
    left += [_tcp(hold, r_hold, "hold the arm still", max(0.5, marks["MARK_OFF"] - t_l), ARM_GRIP + 1e-6),
             _tcp(hold, r_hold, "let go of the arm", 0.5, FINGER_OPEN),
             _tcp(hold - HOLD_AP * 0.07, r_hold, "off the arm", 1.0, FINGER_OPEN),
             _tcp(np.array([0.17, 0.03, S.TOP + 0.33]), r_hold, "round the wall's edge", 1.2, FINGER_OPEN),
             Waypoint(1.5, REST["left"], FINGER_OPEN, "out of the way", DOWN)]
    t_l = sum(w.duration for w in left)
    left += [Waypoint(max(0.5, marks["MARK_ON"] - 4.5 - t_l), REST["left"], FINGER_OPEN + 1e-6, "wait", DOWN),
             _tcp(np.array([0.17, 0.03, S.TOP + 0.33]), r_hold, "round the wall's edge", 1.5, FINGER_OPEN + 1e-6),
             _tcp(hold - HOLD_AP * 0.06, r_hold, "over the shower arm", 1.2, FINGER_OPEN + 1e-6),
             _tcp(hold, r_hold, "round the arm again", 1.0, FINGER_OPEN + 1e-6),
             _tcp(hold, r_hold, "hold the arm again", 0.6, ARM_GRIP)]
    t_l = sum(w.duration for w in left)
    left += [_tcp(hold, r_hold, "hold the arm still", max(0.5, t_tight - t_l), ARM_GRIP + 1e-6),
             _tcp(hold, r_hold, "let go of the arm", 0.5, FINGER_OPEN),
             _tcp(hold - HOLD_AP * 0.07, r_hold, "off the arm", 1.0, FINGER_OPEN),
             _tcp(np.array([0.17, 0.03, S.TOP + 0.33]), r_hold, "round the wall's edge", 1.2, FINGER_OPEN),
             Waypoint(1.5, REST["left"], FINGER_OPEN, "stand clear", DOWN)]
    t_r = sum(w.duration for w in right)
    t_l = sum(w.duration for w in left)
    if t_l < t_r:
        left.append(Waypoint(t_r - t_l + 1.5, REST["left"], FINGER_OPEN, "done", DOWN))
    right.append(Waypoint(max(1.5, t_l - t_r + 1.5), REST["right"], FINGER_OPEN, "done", DOWN))
    return {"left": left, "right": right}


PLAN_T = {}                                          # plan times of marked points (s)
STROKE_LABELS = ("screw it on", "let go", "turn the open hand back", "grip again")
ALIGN = ("line it up with the arm", "onto the thread")
PICK = ()


class DemoController:
    """The scripted plans, with the right hand nudged while it puts the new head to the arm by the
    measured offset of the nut from its place on the thread (a scripted demo may steer by sim state)."""

    def __init__(self, plans):
        self.plans = plans
        self.corr = np.zeros(3)
        self.pick = np.zeros(3)
        self.pause = 0.0                                # plan time held back by the grasp check
        self.jumped = False

    def action(self, t):
        from yam_arena import ik
        lab = self.plans["right"].target(t - self.pause)[3]
        if lab == "line it up with the arm" and "new_nut" in SHARED and not SHARED.get("new_on") and \
                float(np.linalg.norm(np.asarray(SHARED["new_nut"]) - np.asarray(S.new_nut_pos()))) < 0.02:
            self.pause += 1.0 / HZ                       # grasp check: the head has not come with the hand -- wait
        t = t - self.pause
        if SHARED.get("seated") and not self.jumped and lab in STROKE_LABELS and t < PLAN_T.get("MARK_LETGO", 0.0):
            # hand-tight: no more strokes -- straight on to letting go, the wrist staying as it is
            self.jump_rot = np.asarray(self.plans["right"].target(t)[1], float)
            self.pause += t - PLAN_T["MARK_LETGO"]
            t = PLAN_T["MARK_LETGO"]
            self.jumped = True
        FC, FO = FINGER_CLOSED, FINGER_OPEN
        out = []
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, label = plan.target(t)
            pos = np.array(pos, float)
            if side == "right" and self.jumped and label in ("let go of the new head", "back off"):
                rot = self.jump_rot                        # let go where the wrist is: do not wrench the seated head
            if side == "right" and label in ALIGN and "new_err" in SHARED and not SHARED.get("new_on"):
                err = np.asarray(SHARED["new_err"], float)
                gain = 0.08 if label == "onto the thread" else 0.05
                e = err.copy()
                if label == "line it up with the arm":
                    e[1] = 0.0                                   # square it up before pushing it on
                self.corr = np.clip(self.corr + gain * e, -0.02, 0.02)
            if side == "right" and label in PICK and "right_tcp" in SHARED:
                if label != "grip the new head":
                    e = pos - np.asarray(SHARED["right_tcp"], float)
                    e[2] = 0.0                                   # level only: down it may be resting on something
                    self.pick = np.clip(self.pick + 0.06 * e, -0.03, 0.03)
                pos = pos + self.pick
            if side == "right" and (label in ALIGN or label in ("screw it on", "let go", "turn the open hand back",
                                                                  "grip again", "let go of the new head")):
                if SHARED.get("new_on") or label in ALIGN:
                    pos += self.corr
            q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), plan.q, base_pos=plan.base_pos, iters=200)
            SHARED[side + "_ik_err"] = (round(1000 * pe, 1), round(re, 3), label)
            plan.q = q
            grip_v = float(np.clip(finger / FO, 0.0, 1.0)) if FO != FC else 1.0
            out.append(np.r_[q, grip_v])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
