"""Kitchen sink: one pump of dish soap onto the sponge.

A 250 ml pump bottle of dish soap stands on the counter with its spout turned toward the back; a sponge
lies against the bottle's right side. The pump head swivels on its stem: the spout has to be swung round
until it overhangs the sponge, then the head pressed down through one full stroke against its spring
(about 6 N at the bottom) and let come back up. A stroke with the spout anywhere else puts the soap on
the counter.
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

NAME = "soap-pump-one-dose"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1000
OBJECT_NAMES = ("bottle", "stem", "head", "sponge")
OVERVIEW_EYE = (0.62, -0.62, 1.10)
OVERVIEW_TARGET = (0.25, -0.10, 0.83)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in (B.bottle_object(), B.stem_object(), B.head_object(), B.sponge_object())]
    cfg = IsaacLabArenaEnvironment(name="yam_soap_pump_aim_dispense", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
SWEEP_R = 0.034                                       # fingertip along the spout from the head's axis
SWEEP_D = 0.013                                       # fingertip centre beside the spout's mid-line
TURN = -90.0                                          # spout from straight back to over the sponge


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, rot=DOWN):
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


class _Arm:
    def __init__(self, side):
        self.side = side
        self.w = [Waypoint(1.5, REST[side], FINGER_OPEN, "look", DOWN)]

    @property
    def t(self):
        return sum(w.duration for w in self.w)

    def add(self, *wps):
        self.w.extend(wps)
        return self

    def until(self, t, label="wait"):
        last = self.w[-1]
        if t > self.t + 1e-6:
            self.w.append(Waypoint(t - self.t, last.pos, last.finger, label, last.yaw))
        return self


def demo():
    """Right hand: a fingertip behind the spout swings it clockwise a quarter turn until it overhangs the
    sponge; then the fingertips press the head's top straight down through the full stroke, hold, and let
    it come back up; the hand lifts clear. (The bottle stands where only the right arm reaches its head.)"""
    R, L = _Arm("right"), _Arm("left")
    zs = B.HEAD_Z + 0.002
    hi = B.HEAD_Z + 0.025

    def at(th, d=SWEEP_D):
        t = math.radians(th)
        return (B.BX + SWEEP_R * math.cos(t) - d * math.sin(t), B.BY + SWEEP_R * math.sin(t) + d * math.cos(t))

    p0 = at(0.0, SWEEP_D + 0.02)
    R.add(_tip(2.5, (p0[0], p0[1], hi), "over beside the spout"),
          _tip(1.2, (p0[0], p0[1], zs), "down beside it"),
          _tip(0.8, (*at(0.0), zs), "against it"))
    for th in range(-10, int(TURN) - 11, -10):
        R.add(_tip(0.4, (*at(float(th)), zs), "swing it round clockwise"))
    pe = at(TURN - 10.0, SWEEP_D + 0.02)
    top = B.HEAD_Z + B.HEAD_H / 2
    R.add(_tip(0.6, (pe[0], pe[1], zs), "off the spout"),
          _tip(0.8, (pe[0], pe[1], hi), "up"),
          _tip(1.2, (B.BX, B.BY, top + 0.02), "over the head"),
          _tip(0.6, (B.BX, B.BY, top + 0.002), "on the head"),
          _tip(1.2, (B.BX, B.BY, top - B.STROKE - 0.012), "press it down the full stroke"),
          _tip(0.5, (B.BX, B.BY, top - B.STROKE - 0.012), "hold"),
          _tip(1.0, (B.BX, B.BY, top + 0.004), "let it come back up"),
          _tip(1.0, (B.BX, B.BY, hi + 0.03), "lift clear"),
          Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN))
    end = max(R.t, L.t) + 2.0
    L.until(end, "done")
    R.until(end, "done")
    return {"left": L.w, "right": R.w}
