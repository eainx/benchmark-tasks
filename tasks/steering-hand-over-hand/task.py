"""Van / truck cab: "Turn the steering wheel all the way to the left, hold it there for two seconds, then let go and let it spring back to straight ahead."

A 380 mm steering wheel on a raked column (a bench test stand with a dash panel behind it) springs back to
straight ahead: a torsion spring in the column stands in for a moving vehicle's self-centring, about 1.6 N m at
full lock. Full lock is 1.25 turns (450 deg) left. Turning it there is done hand over hand: one hand holds the
rim against the spring while the other lets go and takes a new grip further round, and so on, never both hands
off at once (the wheel would spin back); at the lock it is held for 2 s.
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
import steer as C  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "steering-hand-over-hand"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1400
OBJECT_NAMES = ("wheel",)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = C.scene_objects()
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_steering_hand_over_hand", scene=Scene(objects), embodiment=embodiment,
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
    """The centring spring. A torsion spring in the column pulls the wheel back toward straight ahead with
    K_SPRING times its turn (about 1.6 N m at the 450 deg lock): applied here each control step as a torque about
    the column axis from the wheel's measured turn, unwrapped across full turns. Nothing else is applied: the
    wheel turns on its joint (damping, bearing friction, stops at +/-450 deg) only by the fingers' contacts.
    Also reports the finger pads (from the measured joints) to the scorer."""

    def __init__(self, env):
        self.base = env.unwrapped
        self.body = self.base.scene["wheel"]
        self.turn = 0.0
        self.prev = None
        self.omega = 0.0

    def _measure(self, pose):
        a = C.wheel_angle_wrapped(pose)
        if self.prev is None:
            self.turn = a
        else:
            self.turn += (a - self.prev + 180.0) % 360.0 - 180.0
        self.prev = a

    def reset(self):
        self.turn = 0.0
        self.prev = None
        self.omega = 0.0
        self.before_step(0, None)

    def before_step(self, step, action):
        import torch
        tq = -C.K_SPRING * math.radians(self.turn)
        over = abs(self.turn) - C.LOCK_DEG
        if over > 0:                                   # the end stop at 450 deg either way
            tq -= math.copysign(1.0, self.turn) * C.K_STOP * math.radians(over) + C.C_STOP * self.omega
        t = torch.tensor(np.asarray(C.N, float) * tq, dtype=torch.float32, device=self.base.device).reshape(1, 1, 3)
        self.body.set_external_force_and_torque(torch.zeros_like(t), t, is_global=True)
        self.torque = float(tq)

    def after_step(self, step, objects, root_poses, robot):
        self._measure(np.asarray(root_poses["wheel"], float))
        o = objects["wheel"]
        av = o["ang_vel"] if isinstance(o, dict) else getattr(o, "ang_vel")
        self.omega = float(np.dot(np.asarray(av, float), C.N))
        hands = hand_points(robot)
        return {"spring_torque": self.torque, "dyn_turn": self.turn,
                "pad_boxes": [[list(map(float, c)), [list(map(float, r)) for r in R], list(map(float, hh))]
                              for h in hands.values() for c, R, hh in h[2]]}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


REST = {"left": (0.14, 0.22, C.TOP + 0.30), "right": (0.14, -0.22, C.TOP + 0.30)}

# -- hand over hand -------------------------------------------------------------------------------------
# Each hand pinches the 28 mm rim across its section (jaws closing radially, the pads centred on the rim, the
# hand coming in from the driver's side along the column axis, leaned a little outward), then carries its grip
# round with the wheel. The left hand works the lower-left arc, the right hand the right side: per quarter turn
# both carry the rim 90 deg together, then the left hand lets go and takes a new grip 90 deg back while the
# right holds the wheel against the spring, then the right does the same while the left holds. The grips sit
# 45 / 30 deg off the quarter points, so a spoke (at 3, 6 and 9 o'clock of the wheel) is never where a hand takes hold.
GRIP = {"left": 135.0, "right": 300.0}                     # rim angle (from 12 o'clock toward 9) where each hand takes hold
STROKES = (93.0, 95.0, 97.0, 100.0, 100.0)                 # how far the hands carry the rim each stroke: the wheel lags the
                                                           # hands more as the spring winds up (about 1 deg at the first
                                                           # quarter, 6 by the fourth), and the last stroke ends with the
                                                           # wheel pressed onto its 450 deg stop
QUARTERS = len(STROKES)                                    # 5 strokes = full lock (450 deg)
OPEN = -0.036                                              # jaws 72 mm apart round the 28 mm rim
PINCH = (-0.0100, -0.0099)                                 # closed on the rim (about 4 mm into it a side); two
                                                           # values only so each phase is a move of its own
STEP_DEG = 15.0
LEANS = (15.0, 0.0, 30.0, -15.0, 45.0)
FLIP = np.diag([-1.0, -1.0, 1.0])
# where each hand waits at the end, open, out beside the wheel and toward the robot (about 200 mm from the rim):
# TCP, approach axis, jaw closing axis
EXIT_BACK = 0.100                   # how far in front of the rim (toward the robot) the hands go back round
CLEAR = {"left": ((0.15, 0.32, C.TOP + 0.22), (0.744, 0.345, -0.572), (-0.636, 0.105, -0.765)),
         "right": ((0.15, -0.32, C.TOP + 0.22), (0.711, -0.385, -0.589), (-0.266, -0.922, 0.281))}


def frame(a, close):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - np.dot(c, a) * a
    c = c / np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], axis=1)


def grip_pose(phi, lean, flip):
    """TCP and rotation pinching the rim at world rim angle phi (deg)."""
    p = math.radians(phi)
    radial = math.cos(p) * C.U + math.sin(p) * C.V
    a = -C.N * math.cos(math.radians(lean)) - radial * math.sin(math.radians(lean))
    R = frame(a, radial)
    if flip:
        R = R @ FLIP
    tcp = C.rim_point(phi) + a * (0.0225 - PAD_TIP)        # pads (45 mm long, ending at the tip) centred on the rim
    return tcp, R


SEEDS = ([0, 1, 1, -0.6, 0, 0], [0.5, 1.5, 1.2, -1, 0, 0], [-0.5, 1.5, 1.2, -1, 0, 0], [0, 2, 2, -1.5, 0, 0],
         [0.8, 1.2, 0.8, -0.5, 1, 0], [-0.8, 1.2, 0.8, -0.5, -1, 0], [0.8, 1.2, 0.8, -0.5, -1, 0],
         [-0.8, 1.2, 0.8, -0.5, 1, 0], [0, 1.2, 1.0, -0.3, 0, 1.5], [0, 1.2, 1.0, -0.3, 0, -1.5])


def _chain_cost(side, phis, lean, flip):
    """Worst IK error along a stroke solved warm-started from one point to the next (a continuous joint path)."""
    from yam_arena import ik, station
    base = station.BASES[side]
    tcp, R = grip_pose(phis[0], lean, flip)
    best = None
    for sd in SEEDS:
        q, pe, re = ik.solve(tcp, ik.matrix_to_quat_xyzw(R), np.asarray(sd, float), base_pos=base, iters=200, rot_weight=1.0)
        if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
            best = (q, pe, re)
    q, worst = best[0], best[1] + 0.05 * best[2]
    for phi in phis[1:]:
        tcp, R = grip_pose(phi, lean, flip)
        q2, pe, re = ik.solve(tcp, ik.matrix_to_quat_xyzw(R), q, base_pos=base, iters=200, rot_weight=1.0)
        jump = float(np.max(np.abs(np.asarray(q2[:6]) - np.asarray(q[:6]))))
        worst = max(worst, pe + 0.05 * re + (1.0 if jump > 0.6 else 0.0))
        q = q2
    return worst


_CHOICE = {}


def choice(side):
    """The lean and jaw turn that carry this hand's whole stroke on one continuous joint path."""
    if side not in _CHOICE:
        a0, a1 = GRIP[side], GRIP[side] + max(STROKES)
        phis = list(np.arange(a0, a1 + 1e-6, STEP_DEG))
        best = None
        for lean in LEANS:
            for flip in (False, True):
                c = _chain_cost(side, phis, lean, flip)
                if best is None or c < best[0]:
                    best = (c, lean, flip)
        _CHOICE[side] = best[1:]
        print(f"[steer] {side} grip lean {best[1]} flip {best[2]} cost {best[0]:.4f}", flush=True)
    return _CHOICE[side]


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def demo():
    T_STROKE, T_HOLD = 3.6, 3.0
    plans = {"left": [], "right": []}
    pose = {}
    for side in ("left", "right"):
        lean, flip = choice(side)
        pose[side] = lambda phi, lean=lean, flip=flip: grip_pose(phi, lean, flip)
    pk = {"left": 0, "right": 0}

    def pinch(side):
        pk[side] ^= 1
        return PINCH[pk[side]]

    def back(side, phi):
        tcp, R = pose[side](phi)
        return tcp - R[:, 2] * 0.055, R

    # reach in and take hold: both hands, wheel straight ahead
    for side in ("left", "right"):
        tcp, R = pose[side](GRIP[side])
        b, _ = back(side, GRIP[side])
        plans[side] += [_w(1.0, REST[side], FINGER_OPEN, "look", DOWN),
                        _w(2.4, b, OPEN, "in front of the rim", R),
                        _w(1.2, tcp, OPEN, "jaws round the rim", R),
                        _w(0.6, tcp, pinch(side), "pinch the rim", R)]
    turned = 0.0
    for k in range(QUARTERS):
        d = STROKES[k]
        # both carry the rim round d degrees together
        for side in ("left", "right"):
            a0 = GRIP[side]
            f = pinch(side)
            tcp, R = pose[side](a0)
            plans[side].append(_w(0.05, tcp, f, "set", R))
            n = max(2, int(round(d / STEP_DEG)))
            for j in range(1, n + 1):
                phi = a0 + d * j / n
                tcp, R = pose[side](phi)
                plans[side].append(_w(T_STROKE / n, tcp, f, f"turn {turned + d * j / n:.0f} deg", R))
        turned += d
        if k == QUARTERS - 1:
            break
        # left hand regrips 90 deg back while the right holds; then the right while the left holds
        for mover, holder in (("left", "right"), ("right", "left")):
            a0, a1 = GRIP[mover], GRIP[mover] + d
            tcp1, R1 = pose[mover](a1)
            tcp0, R0 = pose[mover](a0)
            b1, _ = back(mover, a1)
            b0, _ = back(mover, a0)
            plans[mover] += [_w(0.5, tcp1, OPEN, "let go", R1),
                             _w(0.6, b1, OPEN, "back off the rim", R1),
                             _w(1.2, b0, OPEN, "round to a new grip", R0),
                             _w(0.8, tcp0, OPEN, "jaws round the rim", R0),
                             _w(0.5, tcp0, pinch(mover), "pinch the rim", R0)]
            hold_tcp, hold_R = pose[holder](GRIP[holder] if mover == "right" else GRIP[holder] + d)
            f = pinch(holder)
            plans[holder].append(_w(0.05, hold_tcp, f, "hold the wheel", hold_R))
            plans[holder].append(_w(3.55, hold_tcp, f, "hold the wheel", hold_R))
    # at full lock: hold, then let go together and take the hands well away; the spring returns the wheel.
    # The hands end crossed over (the left at the lower right, the right at the upper left). Both open their
    # jaws fully, back straight off the rim along the approach, and go back round in front of the wheel,
    # EXIT_BACK toward the robot (the left under it, the right over it, so the arms pass one above the other), to
    # where they first took hold; then each goes out beside the wheel on its own side, about 200 mm from the
    # rim, clear of it in every view.
    T_ROUND = (0.6, 0.45, 0.45, 0.45, 0.45)
    T_OUT = 1.0
    moving = {}
    for side in ("left", "right"):
        a1 = GRIP[side] + STROKES[-1]
        tcp, R = pose[side](a1)
        f = pinch(side)
        plans[side] += [_w(0.05, tcp, f, "hold at full lock", R),
                        _w(T_HOLD, tcp, f, "hold at full lock", R),
                        _w(0.4, tcp, FINGER_OPEN, "let go: jaws open", R),
                        _w(0.5, tcp - R[:, 2] * 0.080, FINGER_OPEN, "straight back off the rim", R)]
        mv = []
        for j, phi in enumerate(np.linspace(a1, GRIP[side], 5)):
            t2, R2 = pose[side](phi)
            mv.append(_w(T_ROUND[j], t2 - R2[:, 2] * EXIT_BACK, FINGER_OPEN, "back round in front of the wheel", R2))
        cp, ca, cc = CLEAR[side]
        Rc = frame(ca, cc)
        mv.append(_w(T_OUT, cp, FINGER_OPEN, "out to the side, clear of the wheel", Rc))
        moving[side] = mv
    for side in ("left", "right"):
        plans[side] += moving[side]
    for side in ("left", "right"):
        w = plans[side][-1]
        plans[side].append(_w(5.0, w.pos, w.finger, "watch it return", w.yaw))
    return plans


def plan_poses():
    out = []
    for side, wps in demo().items():
        for w in wps:
            out.append((side, np.asarray(w.pos, float), np.asarray(w.yaw, float), w.label))
    return out


class DemoController:
    """The waypoint plans, each arm's warm-started IK first seeded on the branch that reaches its first grip."""

    def __init__(self, plans):
        from yam_arena import ik
        self.plans = plans
        for side, plan in plans.items():
            w = plan.waypoints[1]
            pos, rot = np.asarray(w.pos, float), np.asarray(w.yaw, float)
            best = None
            for sd in SEEDS:
                q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), np.asarray(sd, float), base_pos=plan.base_pos, iters=200)
                if best is None or pe + 0.02 * re < best[0]:
                    best = (pe + 0.02 * re, q)
            plan.q = best[1]

    def action(self, clock):
        return np.concatenate([self.plans[side].action7(clock)[0] for side in ("left", "right")])


def make_demo_controller(env, plans):
    return DemoController(plans)
