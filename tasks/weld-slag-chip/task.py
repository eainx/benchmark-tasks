"""Welding bay: "Clean the slag off the weld."

A 150 mm stick-weld stringer bead on a 3/8 in steel coupon, tacked to a steel fixture plate, is still covered by
its slag, cracked across into six pieces. A welder's chipping hammer lies on the bench beside it. The welder
picks the hammer up by its spring handle, turns it chisel end down and drives the chisel into the toe of the
slag, piece by piece along the bead, so each piece pops off; then lays the hammer back on the bench.
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
import slag as S  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "weld-slag-chip"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = ("hammer",) + S.SLAGS

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in S.scene_objects()]
    cfg = IsaacLabArenaEnvironment(name="yam_weld_slag_chip", scene=Scene(objects), embodiment=embodiment,
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
        out[side] = boxes
    return out


class Dynamics:
    """Reports the finger pads (from the measured joints) to the scorer. Nothing here moves a body: each slag
    piece is held by its own breakable PhysX fixed joint (slag.slag_objects) and comes off only when something
    loads it past SLAG_BREAK_N / SLAG_BREAK_NM."""

    def __init__(self, env):
        pass

    def reset(self):
        pass

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        hands = hand_points(robot)
        return {"pad_boxes": [[list(map(float, c)), [list(map(float, r)) for r in R], list(map(float, hh))]
                              for boxes in hands.values() for c, R, hh in boxes]}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- scripted demo -----------------------------------------------------------------------------------
# Right hand only. It takes the hammer's spring handle straight down (jaws across it, 75 mm from the head),
# lifts it and rolls it 60 deg about the handle so the chisel end points down and out, then for each slag piece
# drives the chisel edge along the line from the grip to the edge (so the blow puts no twist on the grip) into
# the toe of the slag on the near side of the bead, down onto the plate. Then the hammer is rolled back and laid
# where it was, and the hand goes back to rest. The left arm stays at rest.
Z = np.array([0.0, 0.0, 1.0])
ROLL = math.radians(60.0)
REST = {"left": (0.20, 0.24, S.TOP + 0.22), "right": (0.20, -0.24, S.TOP + 0.22)}
TIP_LOCAL = np.array([0.0, -S.CHISEL_TIP + 0.001, 0.0])     # chisel edge middle, hammer frame
GRIP_LOCAL = np.array([S.GRIP_X, 0.0, 0.0])
PRE = 0.020                                                  # the blow starts this far back (along +x)
THROUGH = 0.040                                              # and is aimed this far past the toe (into the plate)
# The arms are soft (joint PD) and the 0.4 kg hammer hangs 0.1 m out: with the chisel down the hand sits about
# 10 mm short, about 6 mm right and 16 mm low of its target (measured in the first run), so the targets are offset.
SAG = np.array([0.010, 0.006, 0.016])


def rx(t):
    c, s_ = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def T(R, p):
    M_ = np.eye(4)
    M_[:3, :3] = R
    M_[:3, 3] = p
    return M_


def inv(M_):
    R, p = M_[:3, :3], M_[:3, 3]
    return T(R.T, -R.T @ p)


def grasp_tcp():
    """Straight down over the handle, jaws across it (closing along y), pad tips 4 mm over the bench."""
    return np.array([S.HAMMER_C[0] + S.GRIP_X, S.HAMMER_C[1], S.TOP + 0.004 + PAD_TIP])


def hammer_in_hand():
    return inv(T(DOWN, grasp_tcp())) @ T(np.eye(3), np.asarray(S.HAMMER_C, float))


def tip_in_hand():
    H = hammer_in_hand()
    return H[:3, :3] @ TIP_LOCAL + H[:3, 3], H[:3, :3] @ GRIP_LOCAL + H[:3, 3]


def blow_dir(R):
    tip_h, grip_h = tip_in_hand()
    d = R @ (tip_h - grip_h)
    return d / np.linalg.norm(d)


def tcp_for_tip(tip_w, R):
    tip_h, _ = tip_in_hand()
    return np.asarray(tip_w, float) - R @ tip_h


def toe_target(i):
    """Where the chisel edge's middle is aimed for slag piece i: level with the crust on the bead's crown (the
    edge's lowest end 4 mm over the plate, 1 mm over the bead's crown), just short of its near (-x) edge."""
    x = S.BX - S.CAP_W / 2 - 0.003 - 0.001          # the blade is 6 mm thick across the blow
    z = S.CT + 0.0005 + S.HEAD_S / 2 * math.cos(ROLL)
    return np.array([x, S.slag_y(i), z])


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def demo():
    left = [_w(1.0, REST["left"], FINGER_OPEN, "rest", DOWN)]
    right = [_w(1.0, REST["right"], FINGER_OPEN, "look", DOWN)]
    g = grasp_tcp()
    right += [_w(2.0, g + Z * 0.08, FINGER_OPEN, "over the hammer's handle", DOWN),
              _w(1.4, g, FINGER_OPEN, "jaws round the handle", DOWN),
              _w(0.8, g, FINGER_CLOSED, "grip the handle", DOWN),
              _w(1.2, g + Z * 0.12, FINGER_CLOSED, "lift the hammer", DOWN)]
    R = rx(ROLL) @ DOWN
    d = blow_dir(R)
    t0 = toe_target(0)
    right += [_w(1.6, tcp_for_tip(t0 - np.array([PRE, 0.0, 0.0]) + Z * 0.05, R) + SAG, FINGER_CLOSED, "chisel end down, over the bead", R)]
    for i in range(S.N_SLAG):
        for k, dy in enumerate((-0.005, 0.005)):        # two blows a piece, 10 mm apart along it
            t = toe_target(i) + np.array([0.0, dy, 0.0])
            X = np.array([1.0, 0.0, 0.0])
            a = tcp_for_tip(t - X * PRE, R) + SAG
            b = tcp_for_tip(t + X * THROUGH - Z * 0.002, R) + SAG
            up = tcp_for_tip(t - X * PRE + Z * 0.02, R) + SAG
            n = f"piece {i + 1}, blow {k + 1}"
            right += [_w(0.5, up, FINGER_CLOSED, f"over the near side of {n}", R),
                      _w(0.6, a, FINGER_CLOSED, f"chisel level with {n}", R),
                      _w(0.25, b, FINGER_CLOSED, f"drive it in ({n})", R),
                      _w(0.25, b, FINGER_CLOSED, "follow through", R),
                      _w(0.25, b + Z * 0.020, FINGER_CLOSED, "lift off", R),
                      _w(0.4, up, FINGER_CLOSED, "draw back", R)]
    right += [_w(1.6, g + Z * 0.12, FINGER_CLOSED, "hammer back over the bench", DOWN),
              _w(1.2, g + Z * 0.006, FINGER_CLOSED, "lay it down", DOWN),
              _w(0.6, g + Z * 0.006, FINGER_OPEN, "let go", DOWN),
              _w(1.0, g + Z * 0.10, FINGER_OPEN, "hand up", DOWN),
              _w(1.6, REST["right"], FINGER_OPEN, "rest", DOWN),
              _w(2.0, REST["right"], FINGER_OPEN, "hold", DOWN)]
    tr = sum(w.duration for w in right)
    left += [_w(tr - 1.0, REST["left"], FINGER_OPEN, "rest", DOWN)]
    return {"left": left, "right": right}


def plan_poses():
    out = []
    for side, wps in demo().items():
        for w in wps:
            out.append((side, w.label, np.asarray(w.pos, float), np.asarray(w.yaw, float)))
    return out


SEEDS = ([0, 1, 1, -0.6, 0, 0], [0.5, 1.5, 1.2, -1, 0, 0], [-0.5, 1.5, 1.2, -1, 0, 0], [0, 2, 2, -1.5, 0, 0],
         [0.8, 1.2, 0.8, -0.5, 1, 0], [-0.8, 1.2, 0.8, -0.5, -1, 0], [0.8, 1.2, 0.8, -0.5, -1, 0],
         [-0.8, 1.2, 0.8, -0.5, 1, 0], [0, 1.2, 1.0, -0.3, 0, 1.5], [0, 1.2, 1.0, -0.3, 0, -1.5])


class DemoController:
    """The waypoint plans, each arm's warm-started IK first seeded on the elbow branch that reaches its
    first grasp (a multi-seed solve), since from READY it can settle on one that cannot."""

    def __init__(self, plans):
        from yam_arena import ik
        self.plans = plans
        for side, plan in plans.items():
            w = next((w for w in plan.waypoints if w.finger == FINGER_CLOSED), plan.waypoints[0])
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
