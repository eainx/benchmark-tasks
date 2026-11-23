"""Laundry shelf: open a squeeze-and-turn child-resistant cap on a bottle of household cleaner.

The cap only turns anticlockwise while its two red panels are squeezed in together (they lift its locking
lugs out of the ratchet teeth on the neck); let go of them and it locks wherever it is. Half a turn frees
the thread and the cap lifts off. The flat 0.85 kg bottle turns with the cap unless the other hand holds it.
The hand cannot turn half a turn in one go, so it squeezes, turns, lets go, takes a fresh grip -- squeezing
again -- and turns again; then it lifts the cap off and puts it down on the counter.
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

NAME = "safety-cap-squeeze-turn"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1400
OBJECT_NAMES = ("bottle", "cap", "tab_a", "tab_b")
OVERVIEW_EYE = (0.62, -0.55, 1.05)
OVERVIEW_TARGET = (0.27, -0.09, 0.84)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.bottle_object(), B.cap_object(), B.tab_object(1), B.tab_object(-1))
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_safety_cap_squeeze_turn", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """The cap's ratchet and thread, from measured poses only.

    While both panels are pressed in at least 2 mm the lugs clear the ratchet: the cap's hinge on the neck is
    free to turn (only the thread's drag). Otherwise a stiff drive holds the cap at the angle it has (the
    lugs in the teeth). Once it has turned half a turn anticlockwise the thread is free: the hinge is
    switched off and the cap is a loose body (with its panels).
    """

    def __init__(self, env):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.joint = None
        for prim in stage.Traverse():
            if str(prim.GetPath()).endswith("/cap/hinge_joint"):
                self.joint = prim
        if self.joint is None:
            raise RuntimeError("cap hinge not found on the stage")

    def _hold(self, stiffness, target):
        self.joint.GetAttribute("drive:angular:physics:stiffness").Set(float(stiffness))
        self.joint.GetAttribute("drive:angular:physics:targetPosition").Set(float(target))

    def reset(self):
        self.free = False
        self.locked = True
        self.turn = 0.0
        self._hold(B.LOCK_K, 0.0)
        self.joint.GetAttribute("physics:jointEnabled").Set(True)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        cap, bottle = root_poses["cap"], root_poses["bottle"]
        a = B.tab_in(cap, root_poses["tab_a"], 1)
        b = B.tab_in(cap, root_poses["tab_b"], -1)
        squeezed = a >= B.SQUEEZED and b >= B.SQUEEZED
        if not self.free:
            d = B.yaw_deg(cap) - B.yaw_deg(bottle)
            d = (d + 180.0) % 360.0 - 180.0
            # unwrap against the running total
            k = round((self.turn - d) / 360.0)
            self.turn = d + 360.0 * k
            if squeezed and self.locked:
                self.locked = False
                self._hold(0.0, self.turn)
            elif not squeezed and not self.locked:
                self.locked = True
                self._hold(B.LOCK_K, self.turn)
            if self.turn >= B.FREE_DEG:
                self.free = True
                self.joint.GetAttribute("physics:jointEnabled").Set(False)
        return {"squeezed": bool(squeezed), "turn_deg": round(self.turn, 2), "free": bool(self.free),
                "locked": bool(self.locked), "panels_mm": [round(1000 * a, 2), round(1000 * b, 2)]}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, yaw=0.0):
    rot = simkit.rot_z(math.radians(yaw)) @ DOWN
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def demo():
    """Left hand: fingertips press down on the bottle's far end and hold it still on the counter. Right hand:
    squeezes the cap's panels between its jaws and turns it a quarter turn anticlockwise, lets go, takes a
    fresh grip on the panels (squeezing again) and turns another quarter; then lifts the freed cap off and
    stands it on the counter."""
    cx, cy = B.BX, B.NECK_Y
    z_tab = B.TOP + B.CAP_Z - 0.004                   # fingertips on the panels' lower half
    hi = B.TOP + B.CAP_Z + B.CAP_H / 2 + 0.03
    hold = (B.BX, B.BY + B.BODY[1] / 2 - 0.015)      # 150 mm from the cap: the two hands clear each other
    zb = B.TOP + B.BODY[2]
    L = [Waypoint(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
         _tip(1.5, (hold[0], hold[1], zb + 0.04), "over the bottle's far end"),
         _tip(0.8, (hold[0], hold[1], zb + 0.002), "fingertips on it"),
         _tip(0.6, (hold[0], hold[1], zb - 0.004), "press it down: hold it still")]
    t_held = sum(w.duration for w in L)
    R = [Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
         _tip(1.5, (cx, cy, hi + 0.02), "over the cap, jaws across the panels", FINGER_OPEN, yaw=0.0)]
    R.append(Waypoint(max(0.1, t_held - sum(w.duration for w in R)), R[-1].pos, FINGER_OPEN, "wait", R[-1].yaw))
    for k, (y0, y1) in enumerate(((0.0, 90.0), (-90.0, 12.0))):
        R += [_tip(0.8, (cx, cy, hi), "over the panels", FINGER_OPEN, yaw=y0),
              _tip(1.0, (cx, cy, z_tab), "down round the panels", FINGER_OPEN, yaw=y0),
              _tip(0.8, (cx, cy, z_tab), "squeeze the panels in", yaw=y0)]
        for i in range(1, 7):
            R.append(_tip(0.35, (cx, cy, z_tab), "keep squeezing, turn it anticlockwise", yaw=y0 + (y1 - y0) * i / 6))
        if k == 0:
            R += [_tip(0.6, (cx, cy, z_tab), "let go: it locks there", FINGER_OPEN, yaw=y1),
                  _tip(0.6, (cx, cy, hi), "up", FINGER_OPEN, yaw=y1)]
    put = (0.200, -0.200)
    R += [_tip(1.0, (cx, cy, hi + 0.02), "lift the freed cap off", yaw=12.0),
          _tip(1.6, (put[0], put[1], hi + 0.02), "over the counter", yaw=12.0),
          _tip(1.0, (put[0], put[1], B.TOP + B.CAP_H / 2 - 0.004 + 0.002), "stand it down", yaw=12.0),
          _tip(0.6, (put[0], put[1], B.TOP + B.CAP_H / 2 - 0.004 + 0.002), "let go", FINGER_OPEN, yaw=12.0),
          _tip(1.0, (put[0], put[1], hi + 0.05), "lift away", FINGER_OPEN, yaw=12.0),
          Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN)]
    t_free = sum(w.duration for w in R) - 6.7
    L.append(Waypoint(max(0.1, t_free - sum(w.duration for w in L)), L[-1].pos, FINGER_CLOSED, "keep holding", DOWN))
    L += [_tip(1.0, (hold[0], hold[1], zb + 0.08), "lift away"),
          Waypoint(1.5, REST["left"], FINGER_OPEN, "rest", DOWN)]
    end = max(sum(w.duration for w in L), sum(w.duration for w in R)) + 2.0
    for arm, side in ((L, "left"), (R, "right")):
        arm.append(Waypoint(max(0.1, end - sum(w.duration for w in arm)), REST[side], FINGER_OPEN, "done", DOWN))
    return {"left": L, "right": R}
