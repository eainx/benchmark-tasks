"""Foundry: "Pour the metal into the mould."

A closed sand mould with its pouring cup, and a crucible of molten bronze standing in a two-man
shank. The crucible is hot: it is lifted, carried and tipped by the shank's two handles, one hand
on each, poured into the cup and set back down.
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
import shank as S  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "crucible-shank-pour"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1400
OBJECT_NAMES = ("crucible", *S.BEADS)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    top_down_overview(embodiment)

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (*(S.mould_object(k) for k in range(len(S.MOULD_YS))), S.crucible_object(), *S.bead_objects())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_crucible_shank_pour", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


OVERVIEW_EYE = (0.29, 0.0, 1.60)      # straight above the bench, image up = away from the arms


def top_down_overview(embodiment):
    """Point the recording-only overview camera straight down on the bench (the model never sees it)."""
    from yam_arena import ik
    cam = embodiment.camera_config.overview_cam
    forward, left, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    cam.offset.pos = tuple(float(v) for v in OVERVIEW_EYE)
    cam.offset.rot = tuple(float(v) for v in ik.matrix_to_quat_xyzw(np.column_stack([forward, left, up])))


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

POUR_TILT = math.radians(88.0)
DRY = 1 - S.WET
HANDLE_FINGER = -(S.HANDLE_W - 0.009) / 2
REST_AXIS = np.array([S.CRU_XY[0], S.CRU_XY[1], S.TOP + S.SHANK_Z])
POUR_DX, POUR_DZ = 0.010, 0.068     # pivot over each cup: high enough that the near foot clears the cup as it tips


def pour_axis(k):
    bx, by = S.basin_xy(k)
    return np.array([bx + POUR_DX, by, S.CUP_TOP + POUR_DZ])
CARRY_Z = S.TOP + 0.135


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def pair(seconds, axis, tilt, finger, label):
    """Both hands on the shank's handles, jaws across the bars; tipping turns both wrists about the
    handle line, toward the arms, so the lip goes down on the near side."""
    r = ry(-tilt)
    out = {}
    for side, sgn in (("left", 1), ("right", -1)):
        grip = np.asarray(axis, float) + np.array([0.0, sgn * S.GRIP_Y, 0.0])
        rot = r @ simkit.rot_z(sgn * math.pi / 2) @ DOWN
        tcp = grip + r @ np.array([0.0, 0.0, PAD_TIP - 0.004])
        out[side] = Waypoint(seconds, tuple(map(float, tcp)), finger, label, rot)
    return out


def demo():
    rest, pour = REST_AXIS, pour_axis(DRY)
    up = lambda p, z: np.array([p[0], p[1], z])  # noqa: E731
    f, ft = HANDLE_FINGER, HANDLE_FINGER - 0.0003
    seq = [
        pair(2.5, rest + [0, 0, 0.06], 0.0, FINGER_OPEN, "over the shank's handles"),
        pair(1.5, rest, 0.0, FINGER_OPEN, "down round the handles"),
        pair(1.0, rest, 0.0, f, "take the shank"),
        pair(2.0, up(rest, CARRY_Z), 0.0, f, "lift the crucible"),
        pair(2.5, up(pour, CARRY_Z), 0.0, f, "carry it over the dry mould's pouring cup"),
        pair(1.5, pour, 0.0, f, "down to the cup"),
        pair(0.8, pour, 0.0, ft, "steady"),
        pair(5.0, pour, POUR_TILT, ft, "tip it, steadily"),
        pair(3.0, pour, POUR_TILT, ft, "let it all run in"),
        pair(2.0, pour, 0.0, ft, "come level"),
        pair(1.5, up(pour, CARRY_Z), 0.0, f, "lift clear of the mould"),
        pair(2.5, up(rest, CARRY_Z), 0.0, f, "back over its place"),
        pair(1.8, rest + [0, 0, 0.002], 0.0, f, "set it down"),
        pair(1.0, rest + [0, 0, 0.002], 0.0, FINGER_OPEN, "let go"),
        pair(1.5, rest + [0, 0, 0.07], 0.0, FINGER_OPEN, "clear"),
        pair(2.0, rest + [0, 0, 0.07], 0.0, FINGER_OPEN, "stand clear"),
    ]
    return {side: [p[side] for p in seq] for side in ("left", "right")}
