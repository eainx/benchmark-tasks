"""Woodshop: set a router-table fence 25 mm from the bit, parallel, and lock it.

The router hangs under the bench; its 1/2 in straight bit stands up through the insert plate. The
600 mm MDF fence lies beyond the bit, each end of its base on a T-bolt in a T-track along x, a star knob
on top. With the knobs loose each end slides along its track and the fence turns about the other end,
so the two ends have to be brought in together, square to the tracks, until the face is 25 mm from the
bit's centre; then both knobs are tightened without knocking the fence off its setting.
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

NAME = "router-fence-set-parallel"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = ("fence", "knob_left", "knob_right", "carrier_left", "carrier_right", "slot_right")
# The recorded overview: from in front of the bench, off to the right and above the arms, looking at the
# fence face, the bit and both knobs.
OVERVIEW_EYE = (-0.25, -0.80, 1.45)
OVERVIEW_TARGET = (0.36, 0.0, 0.78)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.carrier_object("left"), B.carrier_object("right"), B.fence_object(), B.slot_object(),
             B.knob_object("left"), B.knob_object("right")) + B.bit_objects()
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_router_fence_set_parallel", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """The star knobs' clamp, from measured poses only.

    A knob turned clockwise 75 degrees or more from finger-loose clamps its end of the fence: that T-bolt
    carrier's slide gets a stiff drive holding the position it has at that moment (the T-bolt head pulled
    up against the track's lips). Backed off below 60 degrees, the end is loose again. Nothing else is set
    by this model: where the fence is comes from the hands' contacts and the slides' drag alone.
    """

    def __init__(self, env):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.joint = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for s in ("left", "right"):
                if path.endswith(f"/carrier_{s}/spring_joint"):
                    self.joint[s] = prim
        if len(self.joint) != 2:
            raise RuntimeError("T-bolt slides not found on the stage")
        self.x0 = {s: B.carrier_pos(s)[0] for s in ("left", "right")}

    def _drive(self, side, stiffness, target):
        prim = self.joint[side]
        prim.GetAttribute("drive:linear:physics:stiffness").Set(float(stiffness))
        prim.GetAttribute("drive:linear:physics:targetPosition").Set(float(target))

    def reset(self):
        self.locked = {"left": False, "right": False}
        for s in ("left", "right"):
            self._drive(s, 0.0, 0.0)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        turn = {s: B.knob_turn_deg(root_poses[f"knob_{s}"]) for s in ("left", "right")}
        for s in ("left", "right"):
            x = float(np.asarray(root_poses[f"carrier_{s}"])[0]) - self.x0[s]
            if not self.locked[s] and turn[s] >= B.LOCK_DEG:
                self.locked[s] = True
                self._drive(s, B.CLAMP_K, x)
            elif self.locked[s] and turn[s] < B.UNLOCK_DEG:
                self.locked[s] = False
                self._drive(s, 0.0, 0.0)
        return {"locked": dict(self.locked), "turn_deg": {s: round(v, 1) for s, v in turn.items()}}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
PUSH_Y = 0.170                                        # inboard of the knobs
PUSH_Z = 0.045                                        # fingertips above the base board, on the face board's back
HAND_FRONT = 0.019                                    # the hand's front (toward -x) ahead of the TCP, measured
CAL = {"left": -0.0010, "right": -0.0003}             # push-stop calibration from a measured run (m): arm lag
TURN_DEG = 100
LEVER_PUSH = 0.016                                    # fingertip centre behind the lever's mid-line


def _tip(seconds, xyz, label, rot=None, finger=FINGER_CLOSED):
    rot = DOWN if rot is None else rot
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
    """Push the far (left) end in until the fence is parallel at the right end's 33 mm; push both ends in
    together to 25 mm; then sweep both clamping levers a quarter turn clockwise, and stand clear."""
    zt = B.TOP + PUSH_Z
    hi = B.TOP + B.F_H + 0.10
    arms = {s: _Arm(s) for s in ("left", "right")}
    L, R = arms["left"], arms["right"]
    sy = {"left": PUSH_Y, "right": -PUSH_Y}
    # where the face is now at each push point (start pose: a straight line through both ends)
    def start_d(y):
        return B.START_R + (B.START_L - B.START_R) * (y + B.END_Y) / (2 * B.END_Y)
    behind = {s: B.BX + start_d(sy[s]) + B.F_T + HAND_FRONT for s in arms}
    for s, a in arms.items():
        a.add(_tip(1.5, (behind[s] + 0.03, sy[s], hi), "over behind the fence"),
              _tip(1.2, (behind[s] + 0.02, sy[s], zt), "down behind its face board"),
              _tip(0.8, (behind[s] + 0.002, sy[s], zt), "up to it"))
    t0 = max(a.t for a in arms.values())
    for a in arms.values():
        a.until(t0)
    # 1. the left end in until the face is parallel at the right end's distance (pivoting about the right T-bolt)
    par = B.BX + B.START_R + B.F_T + HAND_FRONT + CAL["left"]
    L.add(_tip(2.5, (par, PUSH_Y, zt), "push the far end in, parallel"),
          _tip(0.5, (par, PUSH_Y, zt), "settle"))
    R.until(L.t)
    # 2. both in together to 25 mm
    goal = {s: B.BX + B.TARGET + B.F_T + HAND_FRONT + CAL[s] for s in arms}
    n = 6
    for i in range(1, n + 1):
        L.add(_tip(0.6, (par + (goal["left"] - par) * i / n, PUSH_Y, zt), "both ends in together"))
        R.add(_tip(0.6, (par + (goal["right"] - par) * i / n if i > 1 else par, -PUSH_Y, zt), "both ends in together"))
    for a, y in ((L, PUSH_Y), (R, -PUSH_Y)):
        g = goal[a.side]
        a.add(_tip(2.0, (g, y, zt), "hold at 25 mm while the drag settles"),
              _tip(0.8, (g + 0.03, y, zt), "off the fence"),
              _tip(0.8, (g + 0.03, y, hi), "up"))
    # 3. both levers a quarter turn clockwise: a fingertip behind each lever's grip end sweeps it round
    kx = B.BX + B.TARGET + B.KNOB_X                  # lever axis x with the fence at 25 mm, square
    for a, y in ((L, B.END_Y), (R, -B.END_Y)):
        ax = np.array([kx, y])
        zl = B.lever_z()

        def at(th, d=LEVER_PUSH):
            t = math.radians(th)
            return (ax[0] + B.LEVER_R * math.cos(t) - d * math.sin(t), ax[1] + B.LEVER_R * math.sin(t) + d * math.cos(t))

        p0 = at(0.0, LEVER_PUSH + 0.02)
        a.add(_tip(1.2, (p0[0] + 0.02, p0[1], hi), "behind the lever"),
              _tip(0.8, (p0[0], p0[1], zl + 0.04), "over beside its grip end"),
              _tip(0.8, (p0[0], p0[1], zl), "down beside it"),
              _tip(0.6, (*at(0.0), zl), "against it"))
        for th in range(-10, -TURN_DEG - 1, -10):
            a.add(_tip(0.35, (*at(float(th)), zl), "sweep it round clockwise"))
        pe = at(-TURN_DEG, LEVER_PUSH + 0.02)
        a.add(_tip(0.6, (pe[0], pe[1], zl), "off it"),
              _tip(1.0, (pe[0], pe[1], hi), "up"),
              Waypoint(1.5, REST[a.side], FINGER_OPEN, "rest", DOWN))
    end = max(a.t for a in arms.values()) + 2.0
    for a in arms.values():
        a.until(end, "done")
    return {s: a.w for s, a in arms.items()}
