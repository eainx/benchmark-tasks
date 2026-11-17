"""Bakery / parts dispatch: "Stack the full tote on top of the other full tote."

Two full 400 x 300 mm stack-nest bale-arm crates stand on the bench. Tote A (back) has its two wire bails
hanging out, so as it stands a crate set on it would nest down onto its cartons. The worker swings A's bails
in over its rim, one hand per end (a light pinch on the bar: the bar turns in the fingers as it is carried
round its pivots, and is let go past the top so it drops onto the rim's notches), then takes tote B with both
hands by the grip bars over its end hand holes (one finger inside the end wall, one outside), lifts it level
and lowers it onto the two bails.
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
import tote as Tt  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "tote-bail-stack"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = ("tote_a", "bail_l", "bail_r", "tote_b")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = Tt.scene_objects()
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_tote_bail_stack", scene=Scene(objects), embodiment=embodiment,
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
    """Reports the finger pads (for the scorer's clearance). Nothing here moves a body: the bails are real
    hinged rigid bodies colliding with their crate, and tote B is carried only by the fingers' contacts."""

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


REST = {"left": (0.20, 0.24, Tt.TOP + 0.15), "right": (0.20, -0.24, Tt.TOP + 0.15)}
SIDE = {"left": 1, "right": -1}


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def frame(a, close):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - np.dot(c, a) * a
    c = c / np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], axis=1)


FLIP = np.diag([-1.0, -1.0, 1.0])          # the same grip turned half round its approach axis


def _pick(side, pos, Rs):
    """Of the grips Rs (one rotation, or it turned half round), the one the arm reaches best at pos."""
    from yam_arena import ik, station
    best = None
    for R in Rs:
        for sd in SEEDS:
            q, pe, re = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(R), np.asarray(sd, float),
                                 base_pos=station.BASES[side], iters=150)
            e = pe + 0.02 * re
            if best is None or e < best[0]:
                best = (e, R)
    return best[1]


# -- the bails: pinch the bar lightly near its end, carry it round its pivots, let go past the top --------
TH = math.radians(60.0)                    # the hand comes in from outside the crate's end, 60 deg off straight down
GX = -0.120                                # the bar is taken this far along it from the crate's middle (clear of the leg bend)
BAIL_OPEN = -0.007                         # jaws opened only this far round the 6 mm bar (the end wall is 9 mm behind it)
BAIL_PINCH = -0.0018                       # a light pinch (about 6 N a pad): the bar can turn between the pads
ARC = (-90, -70, -45, -20, 0, 25, 50, 70, 90, 110, 130, 150)    # leg angles the bar is carried through (deg)
BITE = 0.022                               # the bar sits this far in from the pad tips (the pads are 45 mm long)


def bail_axes(s):
    a = np.array([0.0, -s * math.sin(TH), -math.cos(TH)])
    c = np.array([0.0, s * math.cos(TH), -math.sin(TH)])
    return a, c


def bar_point(s, phi_deg):
    phi = math.radians(phi_deg)
    return np.array([Tt.A_C[0] + GX, Tt.A_C[1] + s * (Tt.Y_PIV + Tt.L_LEG * math.cos(phi)),
                     Tt.TOP + Tt.Z_PIV + Tt.L_LEG * math.sin(phi)])


# -- tote B: each hand takes the grip bar over an end hand hole, one finger inside the crate and one outside
# (the jaws close across the 18 mm thick end rim band), the hand pointing forward and 40 deg down ----------
PITCH = math.radians(40.0)
A_B = np.array([math.cos(PITCH), 0.0, -math.sin(PITCH)])
Y_GRIP = Tt.L_TOP / 2 - Tt.RIM_W / 2       # the grip bar's middle (the end rim band, 18 mm thick)
Z_GRIP = (Tt.HOLE_Z1 + Tt.H) / 2           # ... and its height over the crate's underside (22 mm tall)
B_OPEN = -0.022                            # jaws opened round the bar: the inner finger goes down inside the end wall
B_PINCH = -0.0050                          # 4 mm into the 18 mm bar a side (about 20 N a pad)
STACK_Z = Tt.H + 2 * Tt.R_WIRE             # B's underside over A's when it stands on the bails


def b_tcp(s, x, under):
    """TCP for the hand at B's end s, with B's middle at x and its underside at height `under` over the bench:
    the 45 mm pads centred on the grip bar over the hand hole."""
    pad = np.array([x, s * Y_GRIP, Tt.TOP + under + Z_GRIP])
    return pad - A_B * (PAD_TIP - 0.0225)


def demo():
    plans = {}
    for side, s in SIDE.items():
        a, c = bail_axes(s)
        g0 = bar_point(s, ARC[0]) - a * (PAD_TIP - BITE)
        Rb = _pick(side, g0, (frame(a, c), frame(a, c) @ FLIP))
        w = [_w(1.0, REST[side], FINGER_OPEN, "look", DOWN),
             _w(2.5, g0 - a * 0.050, BAIL_OPEN, "behind the hanging bail's bar", Rb),
             _w(1.5, g0, BAIL_OPEN, "jaws round the bar", Rb),
             _w(0.8, g0, BAIL_PINCH, "pinch the bar lightly", Rb)]
        for k, ph in enumerate(ARC[1:]):
            w.append(_w(0.6 if ph < 60 else 0.9, bar_point(s, ph) - a * (PAD_TIP - BITE), BAIL_PINCH,
                        "swing the bail up and over", Rb))
        w.append(_w(0.6, bar_point(s, ARC[-1]) - a * (PAD_TIP - BITE), BAIL_PINCH, "past the top: hold it a moment", Rb))
        top = bar_point(s, ARC[-1]) - a * (PAD_TIP - BITE)
        w += [_w(0.9, top, BAIL_OPEN, "let go: it drops onto the rim", Rb),
              _w(0.9, top - a * 0.040, BAIL_OPEN, "hand back", Rb)]
        # tote B
        p_in = b_tcp(s, Tt.B_C[0], 0.0)
        Rf = _pick(side, b_tcp(s, Tt.A_C[0], STACK_Z), (frame(A_B, (0.0, 1.0, 0.0)), frame(A_B, (0.0, 1.0, 0.0)) @ FLIP))
        p_out = p_in + np.array([0.0, 0.0, 0.060])
        over = Tt.H + 2 * Tt.R_WIRE + 0.030
        w += [_w(2.5, p_out, B_OPEN, "over tote B's end grip bar", Rf),
              _w(1.5, p_in, B_OPEN, "fingers either side of the bar", Rf),
              _w(0.8, p_in, B_PINCH, "grip the bar", Rf),
              _w(1.2, b_tcp(s, Tt.B_C[0], 0.020), B_PINCH, "lift B", Rf),
              _w(1.8, b_tcp(s, Tt.B_C[0] + 0.030, 0.165), B_PINCH, "lift B clear of A's rim", Rf),
              _w(2.6, b_tcp(s, Tt.A_C[0], over), B_PINCH, "carry B over A", Rf),
              _w(1.6, b_tcp(s, Tt.A_C[0], STACK_Z + 0.002), B_PINCH, "lower B level onto the bails", Rf),
              _w(0.6, b_tcp(s, Tt.A_C[0], STACK_Z - 0.003), B_PINCH, "set it down", Rf),
              _w(0.8, b_tcp(s, Tt.A_C[0], STACK_Z - 0.003), B_OPEN, "let go", Rf),
              _w(1.5, b_tcp(s, Tt.A_C[0], STACK_Z + 0.110), B_OPEN, "hands straight up off the bar", Rf),
              _w(1.5, b_tcp(s, Tt.A_C[0], STACK_Z + 0.110) + np.array([0.0, s * 0.090, 0.0]), B_OPEN, "hands out, clear of the totes", Rf),
              _w(4.0, b_tcp(s, Tt.A_C[0], STACK_Z + 0.110) + np.array([0.0, s * 0.090, 0.0]), B_OPEN, "hold", Rf)]
        plans[side] = w
    return plans


def plan_poses():
    out = []
    for side, wps in demo().items():
        for w in wps:
            out.append((side, np.asarray(w.pos, float), np.asarray(w.yaw, float), w.label))
    return out


SEEDS = ([0, 1, 1, -0.6, 0, 0], [0.5, 1.5, 1.2, -1, 0, 0], [-0.5, 1.5, 1.2, -1, 0, 0], [0, 2, 2, -1.5, 0, 0],
         [0.8, 1.2, 0.8, -0.5, 1, 0], [-0.8, 1.2, 0.8, -0.5, -1, 0], [0.8, 1.2, 0.8, -0.5, -1, 0],
         [-0.8, 1.2, 0.8, -0.5, 1, 0], [0, 1.2, 1.0, -0.3, 0, 1.5], [0, 1.2, 1.0, -0.3, 0, -1.5])


class DemoController:
    """The waypoint plans, each arm's warm-started IK first seeded on the elbow branch that reaches its
    first grip (a multi-seed solve), since from READY it can settle on one that cannot."""

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
