"""Retail battery bay: "Stock the new card on the spiral hook: hang it on the hook and wind it in until the coil holds it."

A spiral anti-sweep dispensing hook stands out of a white slatwall: a fixed chrome rod the cards hang on, and
above it a wire coil that a front knob turns, each card's hang tab sitting in a gap between coil turns. Two
cards of AA batteries hang in the coil; a new card stands in the opened shipper on the bench. Stocking it:
hang its euro slot on the rod tip, slide it back under the knob to the coil, and, steadying the card so it
neither swings nor tilts, wind the knob backwards so the coil draws the card into its first gap.
Unlike a plain peg or scan hook (a straight push along the wire), the card only moves by the turning coil,
so one hand guides the card while the other turns, in step.
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
import spiral as C  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "spiral-hook-card"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1800
OBJECT_NAMES = ("coil",) + C.CARD_NAMES + (C.NEW_CARD,)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (C.slatwall_object(), C.hook_object(), C.coil_object(), C.shipper_object()) + C.hung_cards() + (C.new_card(),)
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_spiral_hook_card", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


# -- measured hands --------------------------------------------------------------------------------
PAD_T = 0.008
PAD_W = 0.024
PAD_L = 0.045


def _bases():
    from yam_arena import station
    return {side: np.asarray(station.BASES[side], float) for side in ("left", "right")}


def hand_points(robot):
    """Per hand: the finger pads as oriented boxes, from the measured joints and finger travel by the URDF's
    forward kinematics at the runner's arm bases (station.BASES)."""
    from yam_arena import ik
    bases = _bases()
    out = {}
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, bases[side])
        half = abs(float(robot[side]["fingers"][0]))
        tip = pos + R[:, 2] * PAD_TIP
        boxes = []
        for sgn in (-1.0, 1.0):
            c = tip + R[:, 0] * sgn * (half + PAD_T / 2) - R[:, 2] * (PAD_L / 2)
            boxes.append((c, R.copy(), np.array([PAD_T / 2, PAD_W / 2, PAD_L / 2])))
        out[side] = (tip, half, boxes)
    return out


class Dynamics:
    """No rules: the coil turns on its axle and the cards move only by contact with the rod and the coil.
    This only reports the finger pads (from the measured joints) to the scorer."""

    def __init__(self, env):
        pass

    def reset(self):
        pass

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        hands = hand_points(robot)
        return {"pad_boxes": [[list(map(float, c)), [list(map(float, r)) for r in R], list(map(float, hh))]
                              for h in hands.values() for c, R, hh in h[2]]}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


def _ready_pose(side):
    from yam_arena import ik, station
    from yam_arena.constants import READY
    p, R = ik.tcp_pose(np.asarray(READY, float), np.asarray(station.BASES[side], float))
    return np.asarray(p, float), np.asarray(R, float)


REST = {side: _ready_pose(side)[0] for side in ("left", "right")}
REST_R = {side: _ready_pose(side)[1] for side in ("left", "right")}
X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def frame(a, close):
    """TCP rotation: approach a (local z), fingers closing along ``close`` (local x)."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - np.dot(c, a) * a
    c = c / np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], axis=1)


def rot_about(R, axis, ang):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K) @ R


# -- world geometry of the bay (the fixture is built in a local frame and turned; see spiral.py) ----------------
HOOK_XW = float(C.W((0.0, C.HOOK_Y, 0.0))[0])        # the rod's world x
AXIS = Y                                             # the hook points -y; the knob's axle runs along +y


def along(xl):
    """World y of a point at local x (along the hook)."""
    return float(C.W((xl, C.HOOK_Y, 0.0))[1])


SLOT_TOP_HUNG = C.ROD_Z + C.ROD_R                    # the slot's top edge resting on the rod
CARD_TOP_OFF = C.HOLE_TOP                            # card top above the slot top

# -- the card grip (right hand): its blister bubble pinched top and bottom, from the front ----------------------
DEPTH = 0.015                                        # pad tips this far onto the blister from its front face
BZ_OFF = (C.CARD_H / 2 - C.HOLE_TOP) - C.BLISTER_Z   # the slot's top edge above the blister's centre
GRIP_TILT = 0.0                                      # level: square to the card
A_CARD = np.array([0.0, math.cos(GRIP_TILT), -math.sin(GRIP_TILT)])
R_CARD = frame(A_CARD, Z)
BLISTER_OPEN = FINGER_OPEN * 0.88                   # jaws just wider than the 66 mm blister


HOLD_DZ = 0.0723      # the held card's slot top above the pad tips' line, as it settles in the pinch (measured)


def card_grip(y_card, slot_top_z, back=0.0):
    """TCP holding a card whose plane is at world y and whose slot top is at z (slot on the rod's line)."""
    tip = np.array([HOOK_XW, y_card - C.CARD_TC / 2 - C.BLISTER[0] + DEPTH, slot_top_z - HOLD_DZ])
    return tip - A_CARD * (PAD_TIP + back)


# the new card in the carton: its plane at world y, slot top at its height there
NEW_W = C.W((C.NEW_X, C.SHIP_C[1], 0.0))
NEW_SLOT_Z = C.FLOOR + C.SHIP_WALL + C.RISER + C.CARD_H - C.HOLE_TOP


def carton_grip(lift=0.0, back=0.0):
    tip = np.array([NEW_W[0], NEW_W[1] - C.CARD_TC / 2 - C.BLISTER[0] + DEPTH, NEW_SLOT_Z - BZ_OFF + lift])
    return tip - A_CARD * (PAD_TIP + back)


# -- the knob grip (right hand): from the front along +y, pads across the knob ----------------------------------
KNOB_W = C.W((C.KNOB_X, C.HOOK_Y, C.COIL_Z))
KNOB_DEPTH = 0.010                                   # pad tips this far onto the knob from its front face
KNOB_FACE_Y = along(C.KNOB_X - C.KNOB_L / 2)


def knob_R(roll_deg):
    """Approach +y; roll about it (0: fingers closing along z)."""
    return rot_about(frame(AXIS, Z), AXIS, math.radians(roll_deg))


def knob_tcp(roll_deg, back=0.0):
    tip = np.array([KNOB_W[0], KNOB_FACE_Y + KNOB_DEPTH - back, KNOB_W[2]])
    return tip - knob_R(roll_deg)[:, 2] * PAD_TIP


# -- the left hand steadies the hung card: closed fingertips against its back face, low ---------------------------
A_STEADY = X                                         # in from the arms' side, level, just behind the card
R_STEADY = frame(A_STEADY, Z)
STEADY_DX = -0.030                                   # fingertips 30 mm in from the hook's line (the card's near half)
STEADY_DOWN = 0.060                                  # 60 mm below the rod: well under the coil
STEADY_GAP = 0.002                                   # a guard 2 mm behind the card's back face


def steady_tcp(y_card, back=0.0):
    tip = np.array([HOOK_XW + STEADY_DX, y_card + C.CARD_TC / 2 + STEADY_GAP + 0.008, C.ROD_Z - STEADY_DOWN])
    return tip - A_STEADY * (PAD_TIP + back)


class Plan:
    def __init__(self, side):
        self.side = side
        self.t = 0.0
        self.w = []
        self.p, self.R, self.f = np.asarray(REST[side], float), REST_R[side], FINGER_OPEN

    def go(self, dt, p, R, f=None, label=""):
        f = self.f if f is None else f
        self.w.append(_w(dt, p, f, label, R))
        self.t += dt
        self.p, self.R, self.f = np.asarray(p, float), R, f

    def grip(self, dt, f, label):
        self.go(dt, self.p, self.R, f, label)

    def wait_until(self, t, label="wait"):
        if t > self.t + 1e-6:
            self.go(t - self.t, self.p, self.R, self.f, label)


POCKET_Y = along(C.POCKET_X)                         # where the new card's plane goes: the coil's front pocket
TIP_Y = along(C.ROD_X1)
PRE_Y = TIP_Y - 0.035                                # in front of the rod tip
SLOT_LIFT = 0.0090                                   # commanded slot clearance over the rod (the arm sags ~5 mm: about 3.5 mm actual)
STROKE = 150.0                                       # each wind: 150 deg of wrist roll, then regrip
STROKES = 3
ROLL_HI = 165.0


def demo():
    Rt, Lf = Plan("right"), Plan("left")
    Rt.go(1.0, REST["right"], REST_R["right"], label="look")
    Lf.go(1.0, REST["left"], REST_R["left"], label="look")
    # right hand: take the new card out of the carton by its near top corner
    Rt.go(2.4, carton_grip(back=0.05), R_CARD, BLISTER_OPEN, "in front of the card's blister")
    Rt.go(1.4, carton_grip(), R_CARD, label="jaws over and under the blister")
    Rt.go(0.4, Rt.p, R_CARD, label="settle")
    Rt.grip(0.8, FINGER_CLOSED, "pinch the blister")
    Rt.go(1.6, carton_grip(lift=0.14), R_CARD, label="lift it out of the carton")
    # line its slot up with the rod in front of the tip, then thread it on and back under the knob
    Rt.go(2.4, card_grip(PRE_Y, SLOT_TOP_HUNG + SLOT_LIFT + 0.03), R_CARD, label="over in front of the rod tip")
    Rt.go(1.4, card_grip(PRE_Y, SLOT_TOP_HUNG + SLOT_LIFT), R_CARD, label="slot level with the rod")
    Rt.go(0.4, Rt.p, R_CARD, label="settle")
    for y, lab in ((TIP_Y + 0.010, "slot onto the rod tip"), (POCKET_Y - 0.030, "along the rod, under the knob"),
                   (POCKET_Y, "into the coil's front pocket")):
        Rt.go(1.6, card_grip(y, SLOT_TOP_HUNG + SLOT_LIFT), R_CARD, label=lab)
    Rt.go(0.8, card_grip(POCKET_Y, SLOT_TOP_HUNG - 0.0015), R_CARD, label="slot down on the rod")
    Rt.go(0.4, Rt.p, R_CARD, label="hang")
    t_hung = Rt.t
    # left hand comes in behind the card and steadies it
    Lf.wait_until(t_hung - 3.5)
    Lf.go(2.4, steady_tcp(POCKET_Y, back=0.07), R_STEADY, FINGER_CLOSED, "level with the card, behind it")
    Lf.go(1.5, steady_tcp(POCKET_Y), R_STEADY, label="fingertips just behind the card")
    t_steady = Lf.t
    Rt.wait_until(t_steady, "hold until it is steadied")
    Rt.grip(0.6, FINGER_OPEN, "let go of the card")
    Rt.go(1.0, Rt.p - A_CARD * 0.06, R_CARD, label="hand back off the card")
    # right hand: the knob, wound backward in strokes, regripping
    roll = ROLL_HI
    Rt.go(2.2, knob_tcp(roll, back=0.05), knob_R(roll), FINGER_OPEN, "in front of the knob")
    for k in range(STROKES):
        Rt.go(1.0, knob_tcp(roll), knob_R(roll), FINGER_OPEN, "jaws over the knob")
        Rt.grip(0.6, FINGER_CLOSED, "grip the knob")
        n = 5
        for j in range(1, n + 1):
            Rt.go(0.9, knob_tcp(roll - STROKE * j / n), knob_R(roll - STROKE * j / n), label=f"wind back (stroke {k + 1})")
        Rt.grip(0.6, FINGER_OPEN, "let go of the knob")
        Rt.go(0.5, Rt.p, Rt.R, label="settle")
        if k == 0:
            t_held = Rt.t
        if k < STROKES - 1:
            Rt.go(0.6, knob_tcp(roll - STROKE, back=0.030), knob_R(roll - STROKE), label="open jaws back off the knob")
            for j in range(1, n + 1):
                Rt.go(0.5, knob_tcp(roll - STROKE + STROKE * j / n, back=0.030), knob_R(roll - STROKE + STROKE * j / n),
                      label="turn the hand back, clear of the knob")
    Rt.go(1.0, knob_tcp(Rt_roll_end(roll), back=0.06), Rt.R, label="hand off the knob")
    # left hand: once the coil holds the card (the first stroke done), it lets go
    Lf.wait_until(t_held, "steady the card while the coil takes it")
    Lf.go(1.2, steady_tcp(POCKET_Y, back=0.08), R_STEADY, label="fingertips out from behind the card")
    Lf.go(2.0, REST["left"], REST_R["left"], FINGER_OPEN, "rest")
    end = max(Rt.t, Lf.t)
    for P in (Rt, Lf):
        P.go(2.0, REST[P.side], REST_R[P.side], FINGER_OPEN, "rest")
        P.wait_until(end + 2.0 + 6.0, "hold")
    return {"left": Lf.w, "right": Rt.w}


def Rt_roll_end(roll):
    return roll - STROKE


def plan_poses():
    out = []
    for side, ws in demo().items():
        for w in ws:
            out.append((side, np.asarray(w.pos, float), np.asarray(w.yaw, float) if np.ndim(w.yaw) == 2 else DOWN, w.label))
    return out


class DemoController:
    """Joint targets for the scripted demo: each control step's TCP target (the plan's spline) solved by IK
    warm-started from the last step with more iterations than ArmPlan's 60, re-seeded from several postures
    when the warm start misses (it can lock onto a poor elbow branch)."""

    def __init__(self, plans):
        from yam_arena import ik
        self.ik = ik
        self.plans = plans
        self.q = {side: np.asarray(p.q, float) for side, p in plans.items()}
        self.worst = {side: 0.0 for side in plans}
        self.tick, self.last = {}, {}
        rng = np.random.default_rng(0)
        self.seeds = [np.asarray(x, float) for x in ([0, 1, 1, -0.6, 0, 0], [0.5, 1.2, 1.0, -0.3, 0, 0],
                                                     [-0.5, 1.2, 1.0, -0.3, 0, 0], [0, 1.6, 1.4, -0.4, 0, 0])]
        self.seeds += [rng.uniform(ik.LOWER[:6], ik.UPPER[:6]) for _ in range(2)]

    def _solve(self, side, pos, rot):
        ik, base = self.ik, self.plans[side].base_pos
        quat = ik.matrix_to_quat_xyzw(rot)
        q, pe, re = ik.solve(pos, quat, self.q[side], base_pos=base, iters=60, rot_weight=0.6)
        self.tick[side] = self.tick.get(side, 0) + 1
        if (pe > 0.004 or re > 0.05) and self.tick[side] - self.last.get(side, -99) >= 20:
            self.last[side] = self.tick[side]
            for sd in self.seeds:
                q2, pe2, re2 = ik.solve(pos, quat, sd, base_pos=base, iters=60, rot_weight=0.6)
                jump = float(np.abs(q2 - self.q[side]).max())
                if pe2 + 0.05 * re2 < pe + 0.05 * re - 0.001 and jump < 1.0:
                    q, pe, re = q2, pe2, re2
        self.worst[side] = max(self.worst[side], pe * 1000)
        self.q[side] = q
        return q

    def action(self, t):
        out = []
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, _ = plan.target(t)
            q = self._solve(side, np.asarray(pos, float), np.asarray(rot, float))
            grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            out.append(np.r_[q, grip])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
