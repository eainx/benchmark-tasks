"""Desk: finish a Rubik's cube that is one move from solved.

A 3x3 speedcube stands on a speedcubing mat. Its lower two layers are solved; its top layer is a quarter
turn out, so each side shows a top row in the colour of the face next to it. Reading the colours says
which way to turn it: a quarter turn anticlockwise seen from above. The hand takes the top layer, keeps
the cube down on the mat, turns the layer and leaves it square, then moves away.
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
import parameters as B  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "rubiks-cube-last-turn"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 800
OBJECT_NAMES = ("mat", "cube", "top_layer")
OVERVIEW_EYE = (0.55, -0.40, 0.98)
OVERVIEW_TARGET = (0.28, -0.09, 0.78)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.mat_object(), B.base_object(), B.layer_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_rubiks_cube_last_turn", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
GRIP_Z = B.TOP + B.LAYER_Z - 0.003          # fingertips on the top layer's sides, 6 mm above the middle layer
PRE = FINGER_OPEN * 0.80                     # jaws 75 mm apart: wider than the 57 mm cube
TURN = 92.0                                  # a hair past square: the layer springs back a little on release


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, yaw=0.0):
    rot = simkit.rot_z(math.radians(yaw)) @ DOWN
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def demo():
    """Right hand: jaws across the top layer, close on it, press a little down to keep the cube on the mat,
    turn a quarter turn anticlockwise (seen from above), let go and lift away. Left hand stays at rest."""
    cx, cy = B.CX, B.CY
    hi = B.TOP + B.LAYER_Z + B.LAYER_H / 2 + 0.04
    R = [Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
         _tip(1.5, (cx, cy, hi), "over the cube, jaws open across the top layer", PRE),
         _tip(1.2, (cx, cy, GRIP_Z), "down round the top layer", PRE),
         _tip(0.8, (cx, cy, GRIP_Z - 0.001), "close on it, pressing the cube onto the mat")]
    for i in range(1, 9):
        R.append(_tip(0.35, (cx, cy, GRIP_Z - 0.001), "turn the top layer anticlockwise", yaw=TURN * i / 8))
    R += [_tip(0.5, (cx, cy, GRIP_Z - 0.001), "hold it square", yaw=TURN),
          _tip(0.6, (cx, cy, GRIP_Z), "let go", PRE, yaw=TURN),
          _tip(1.0, (cx, cy, hi + 0.03), "lift away", PRE, yaw=TURN),
          Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN),
          Waypoint(2.5, REST["right"], FINGER_OPEN, "done", DOWN)]
    L = [Waypoint(sum(w.duration for w in R), REST["left"], FINGER_OPEN, "rest", DOWN)]
    return {"left": L, "right": R}
