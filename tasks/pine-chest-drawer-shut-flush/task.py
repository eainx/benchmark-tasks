"""Bedroom: shut a wide wood-runner chest drawer that has gone in crooked.

A low solid-pine one-drawer chest stands on the bench, front to the arms. Its 581 mm wide drawer rides
on two beech runners and is steered only by the chest's side panels, 1.5 mm clear each side. It is
130 mm out and cocked about 1 degree, its left end further out. Pushed at one knob, the drawer turns
in its play until its back corner and the opposite front edge bear on the side panels, and then the
harder it is pushed the harder it binds. It shuts only when it is first squared up and then pushed
home evenly, until both ends of its inset front are flush with the chest's face.
"""
from __future__ import annotations

import json
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

NAME = "pine-chest-drawer-shut-flush"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = ("drawer",)
# The recorded overview: from in front of the chest, off to the right and above the arms, so the
# drawer front, both its ends and both hands are in view (the default view looks at the chest's back).
OVERVIEW_EYE = (-0.30, -0.85, 1.55)
OVERVIEW_TARGET = (0.36, 0.0, 0.80)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in (B.chest_object(), B.drawer_object())]
    cfg = IsaacLabArenaEnvironment(name="yam_chest_drawer_shut_square", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
FINGER_HALF = 0.010                                   # closed fingers' half thickness along the jaw axis
ALONG_X = simkit.rot_z(np.pi / 2) @ DOWN              # jaws along x: the closed pair's flat face pushes along x
PUSH = DOWN                                           # jaws closed along y, pointing down: the fingers' sides push
PUSH_Y = 0.120                                        # inboard of the knobs (at 200 mm), clear of them
PUSH_Z = 0.050                                        # fingertips 50 mm above the bench: low on the drawer front
HAND_FRONT = 0.012                                    # the closed fingers' front face ahead of the TCP


def _tip(seconds, xyz, label, rot=None, finger=FINGER_CLOSED):
    """Closed fingertips (the pads' lower ends) at ``xyz``."""
    rot = PUSH if rot is None else rot
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
    """Square the drawer up by its trailing (left) end, then push both ends home together.

    The hands push the drawer front low down (50 mm above the bench), inboard of the knobs, with the
    jaws closed and pointing down: the hand then stays below the chest's top rail, so only the drawer's
    own stop blocks end the push. A file ONE_KNOB in the task folder instead pushes the right end alone,
    all the way: the binding check."""
    start = B.start_pose()
    y_push = {"left": PUSH_Y, "right": -PUSH_Y}
    face = {s: B.front_point(start, y) for s, y in y_push.items()}
    zt = B.TOP + PUSH_Z
    hi = B.TOP + B.CH
    shut_x = B.FX - HAND_FRONT + 0.004               # commanded 4 mm past flush: the stop blocks end it
    arms = {s: _Arm(s) for s in ("left", "right")}
    for s, arm in arms.items():
        f = face[s]
        arm.add(_tip(1.5, (f[0] - 0.06, f[1], hi), "over in front of the drawer"),
                _tip(1.2, (f[0] - 0.06, f[1], zt), "down in front of it"),
                _tip(1.0, (f[0] - HAND_FRONT - 0.004, f[1], zt), "up to its front"))
    t0 = max(a.t for a in arms.values())
    for a in arms.values():
        a.until(t0)
    L, R = arms["left"], arms["right"]
    fl, fr = face["left"], face["right"]
    if (_here / "ONE_KNOB").exists():
        R.add(_tip(1.0, (fr[0] - HAND_FRONT + 0.001, fr[1], zt), "on the right end"),
              _tip(6.0, (shut_x, fr[1], zt), "push the right end home alone"),
              _tip(1.5, (shut_x, fr[1], zt), "keep pushing"),
              _tip(1.0, (shut_x - 0.08, fr[1], zt), "back off"),
              _tip(1.0, (shut_x - 0.08, fr[1], hi), "up"))
        L.add(_tip(1.0, (fl[0] - 0.08, fl[1], hi), "stand off"))
    else:
        lead = fr[0] - fl[0]                             # how much further in the right end is
        R.add(_tip(1.0, (fr[0] - HAND_FRONT + 0.001, fr[1], zt), "on the right end"))
        L.add(_tip(1.0, (fl[0] - HAND_FRONT + 0.001, fl[1], zt), "on the left end"),
              _tip(1.5, (fl[0] + lead - HAND_FRONT + 0.001, fl[1], zt), "push the trailing end level"))
        R.until(L.t)
        n = 8
        x0l, x0r = fl[0] + lead - HAND_FRONT + 0.001, fr[0] - HAND_FRONT + 0.001
        for i in range(1, n + 1):
            f = i / n
            L.add(_tip(0.6, (x0l + f * (shut_x - x0l), fl[1] + f * (PUSH_Y - fl[1]), zt), "push home evenly"))
            R.add(_tip(0.6, (x0r + f * (shut_x - x0r), fr[1] + f * (-PUSH_Y - fr[1]), zt), "push home evenly"))
        for a, y in ((L, PUSH_Y), (R, -PUSH_Y)):
            a.add(_tip(1.0, (shut_x, y, zt), "seat it"),
                  _tip(1.0, (shut_x - 0.08, y, zt), "back off"),
                  _tip(1.0, (shut_x - 0.08, y, hi), "up clear"),
                  Waypoint(1.5, REST[a.side], FINGER_OPEN, "rest", DOWN))
    end = max(a.t for a in arms.values()) + 2.0
    for a in arms.values():
        a.until(end, "done")
    return {s: a.w for s, a in arms.items()}
