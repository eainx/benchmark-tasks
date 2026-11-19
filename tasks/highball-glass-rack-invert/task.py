"""Bar back / dish station: "Rack both glasses mouth-down for the dish machine."

Two empty 10 oz heavy-base hi-ball tumblers stand mouth-up on a stainless dish table beside an empty
25-compartment dish-machine glass rack. Each glass has to end mouth-down in its own compartment, resting on
the rack's floor grid. A YAM gripper can only take a standing glass from the side near its rim (lower down
the wrist meets the table), and only from above by its foot to lower it between the 101 mm compartment
walls. Turning the glass over between those two grips cannot be done in one hand, so one hand picks and
rolls the glass over and the other takes it by the foot and lowers it in. Everything happens by contact:
there are no joints, attachments or triggers.
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
import glass as G  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402,F401
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "highball-glass-rack-invert"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
GLASS_NAMES = G.GLASS_NAMES
OBJECT_NAMES = ("rack",) + GLASS_NAMES

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (G.table_object(), G.rack_object()) + tuple(G.glass_object(n) for n in GLASS_NAMES)
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_highball_glass_rack_invert", scene=Scene(objects), embodiment=embodiment,
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
    """Per hand: the finger pads as oriented boxes (centre, axes, half-sizes), from the measured joints and
    finger travel by the URDF's forward kinematics at the runner's arm bases."""
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
    """No rules: the glasses, the rack and the table only meet by contact. This only reports the finger
    pads (from the measured joints) to the scorer."""

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
F = G.FLOOR


def frame(a, close):
    """TCP rotation: approach a (local z), fingers closing along ``close`` (local x)."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - np.dot(c, a) * a
    c = c / np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], axis=1)


def rodrigues(k, ang):
    k = np.asarray(k, float) / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


# -- left hand: side grip near the rim, approach from its own side tipped 15 degrees down -----------------
TILT = math.radians(15.0)
A_L = np.array([0.0, -math.cos(TILT), -math.sin(TILT)])
R_UP = frame(A_L, X)                         # glass mouth-up between the jaws
GH = 0.090                                   # grip point above the glass's foot
PAST = 0.020                                 # pad tips this far past the glass's axis
L_GRIP = -(G.r_out(GH) - 0.006)              # 6 mm squeeze
LIFT_Z = 0.150                               # grip point over the table while carried upright
HAND = np.array([0.300, 0.080, F + 0.100])   # grip point where the glass is handed over (mouth down)
ROLL_FROM = np.array([0.300, 0.100, F + LIFT_Z])

# -- right hand: down onto the foot, tipped like the turned glass, then straightened over the rack -----------
G2 = rodrigues(A_L, math.pi) @ Z             # the turned glass's foot-to-mouth axis (30 deg off straight down)
FOOT_IN = 0.028                              # pad tips this far below the foot's face
R_GRIP = -(G.r_out(0.012) - 0.006)
CARRY_TIP = 0.205                            # pad tips over the table while the glass crosses the rack's walls
CELLS = {"glass_a": (0, 3), "glass_b": (0, 4)}


def left_tcp(grip_point, R):
    a = R[:, 2]
    return np.asarray(grip_point, float) + a * PAST - a * PAD_TIP


def right_tcp(tip, R):
    return np.asarray(tip, float) - R[:, 2] * PAD_TIP


def handover_tip():
    """The right pad tips' target on the turned glass held at HAND."""
    foot = HAND - GH * G2
    return foot + FOOT_IN * G2


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


def left_fetch_and_turn(L, name):
    """Take glass ``name`` from the side near its rim, lift it and roll it over (mouth down) at HAND."""
    gx, gy = G.GLASSES[name]
    g = np.array([gx, gy, F + GH])
    L.go(2.4, left_tcp(g - A_L * 0.01 + Z * 0.05, R_UP), R_UP, FINGER_OPEN, f"over {name}")
    L.go(1.2, left_tcp(g, R_UP), R_UP, label=f"jaws round {name}")
    L.grip(0.7, L_GRIP, f"grip {name}")
    L.go(1.0, left_tcp(np.array([gx, gy, F + LIFT_Z]), R_UP), R_UP, label=f"lift {name}")
    L.go(1.6, left_tcp(ROLL_FROM, R_UP), R_UP, label="carry to the hand-over")
    n = 6
    for k in range(1, n + 1):
        f = k / n
        R = rodrigues(A_L, math.pi * f) @ R_UP
        L.go(0.55, left_tcp(ROLL_FROM + (HAND - ROLL_FROM) * f, R), R, label=f"roll the glass over ({30 * k} deg)")
    L.go(0.4, L.p, L.R, label="hold it mouth down")


def left_release(L, name):
    L.grip(0.6, FINGER_OPEN, f"let go of {name}")
    R = L.R
    p0 = HAND - A_L * 0.06 + Z * 0.04
    L.go(1.0, left_tcp(p0, R), R, label="hand back off the glass")
    p1 = np.array([0.300, 0.130, F + 0.200])
    n = 4
    for k in range(1, n + 1):                      # roll the wrist back the way it came, up and clear
        f = k / n
        Rk = rodrigues(A_L, math.pi * (1 - f)) @ R_UP
        L.go(0.5, left_tcp(p0 + (p1 - p0) * f, Rk), Rk, label="wrist rolled back")


def right_take_and_rack(R_, name, t_turned, t_left_free=None):
    """Take the turned glass by its foot at the hand-over, straighten it over the rack and lower it into its
    compartment."""
    Rt = frame(G2, np.cross(X, G2))            # jaws across the tilt, so straightening turns the glass with them
    tip = handover_tip()
    R_.wait_until(t_turned - 2.6, "wait")
    R_.go(2.6, right_tcp(tip - G2 * 0.07, Rt), Rt, FINGER_OPEN, "over the turned glass's foot")
    R_.go(1.2, right_tcp(tip, Rt), Rt, label="jaws round the foot")
    R_.grip(0.7, R_GRIP, f"grip {name}'s foot")
    R_.go(0.3, R_.p, Rt, label="hold")
    t_gripped = R_.t
    return t_gripped


def right_carry(R_, name):
    i, j = CELLS[name]
    cx, cy = G.cell_world(i, j)
    Rv = frame(-Z, Y)
    Rm = frame(G2 - Z, np.cross(X, G2 - Z))                           # half straightened
    tip0 = handover_tip()
    R_.go(1.2, right_tcp(tip0 + np.array([0.0, -0.010, 0.030]), R_.R), R_.R, label="lift out of the left hand")
    R_.go(1.4, right_tcp(np.array([0.5 * (tip0[0] + cx), 0.010, F + CARRY_TIP - 0.010]), Rm), Rm,
          label="straighten it over the rack's edge")
    R_.go(1.4, right_tcp(np.array([cx, cy, F + CARRY_TIP]), Rv), Rv, label=f"over compartment {i},{j}")
    seat = G.GRID_T + G.H - FOOT_IN
    R_.go(2.0, right_tcp(np.array([cx, cy, F + seat + 0.001]), Rv), Rv, label="lower it in, mouth down")
    R_.go(0.3, R_.p, Rv, label="seated")
    R_.grip(0.6, -(G.r_out(0.012) + 0.004), f"let go of {name}")
    R_.go(1.2, right_tcp(np.array([cx, cy, F + CARRY_TIP]), Rv), Rv, label="hand up")


def demo():
    L, R_ = Plan("left"), Plan("right")
    L.go(1.0, REST["left"], REST_R["left"], label="look")
    R_.go(1.0, REST["right"], REST_R["right"], label="look")
    for name in GLASS_NAMES:
        L.wait_until(R_.t - 6.0 if name != GLASS_NAMES[0] else 0.0, "wait for the right hand")
        left_fetch_and_turn(L, name)
        t_turned = L.t
        t_gripped = right_take_and_rack(R_, name, t_turned)
        L.wait_until(t_gripped, "hold until the foot is gripped")
        left_release(L, name)
        R_.wait_until(L.t, "until the left hand is clear")
        right_carry(R_, name)
    end = max(R_.t, L.t)
    for P in (R_, L):
        P.go(2.0, REST[P.side], REST_R[P.side], FINGER_OPEN, "rest")
        P.wait_until(end + 2.0 + 2.5, "hold")
    return {"left": L.w, "right": R_.w}


def plan_poses():
    """Every pose the demo commands (for reach checks): (side, position, rotation, label)."""
    out = []
    for side, ws in demo().items():
        for w in ws:
            out.append((side, np.asarray(w.pos, float), np.asarray(w.yaw, float) if np.ndim(w.yaw) == 2 else DOWN, w.label))
    return out


class DemoController:
    """Joint targets for the scripted demo: each control step's TCP target (the plan's spline) solved by IK
    warm-started from the last step, re-seeded from several postures when the warm start misses."""

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
        if (pe > 0.004 or re > 0.05) and self.tick[side] - self.last.get(side, -99) >= 40:
            self.last[side] = self.tick[side]
            for sd in self.seeds:
                q2, pe2, re2 = ik.solve(pos, quat, sd, base_pos=base, iters=60, rot_weight=0.6)
                jump = float(np.abs(q2 - self.q[side]).max())
                if pe2 + 0.05 * re2 < pe + 0.05 * re - 0.001 and jump < 0.6:
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
