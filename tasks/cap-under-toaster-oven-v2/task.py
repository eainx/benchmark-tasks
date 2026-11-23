"""Kitchen: fish a bottle cap out from under a toaster oven with a wooden spoon's handle.

A plastic soda-bottle cap has rolled under a countertop toaster oven. The oven stands 22 mm clear of
the bench on four rubber feet set well in from its sides, so the cap cannot be seen from above and no
finger fits under. The 6 kg oven slides if it is shoved. A wooden spoon lies on the bench to the right:
its handle, slid under and laid behind the cap between the feet, rakes the cap out.
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

NAME = "cap-under-toaster-oven-v2"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1200
OBJECT_NAMES = ("oven", "cap", "spoon")
# The recorded overview: low, in front and to the right, looking along the bench into the gap under the oven.
OVERVIEW_EYE = (0.02, -0.78, 0.98)
OVERVIEW_TARGET = (0.38, -0.15, 0.76)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in (B.oven_object(), B.cap_object(), B.spoon_object())]
    cfg = IsaacLabArenaEnvironment(name="yam_cap_under_toaster_oven", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
ALONG_X = simkit.rot_z(np.pi / 2) @ DOWN              # jaws closing along x, across the handle
GRIP_Z = 0.004                                        # fingertips just above the bench round the 9 mm handle


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, rot=ALONG_X):
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
    """Right hand: take the spoon by its handle near the bowl, slide it along to behind the cap's line,
    push the handle in under the oven's right side between the feet until its tip is past the cap, draw it
    toward the front so the handle rakes the cap out, then draw the spoon clear, lay it down and let go."""
    R, L = _Arm("right"), _Arm("left")
    gy = B.SPOON_TIP_Y - B.GRIP_FROM_TIP                # grip point on the handle (y)
    z = B.TOP + GRIP_Z
    hi = B.TOP + 0.10
    rake_x = B.CAP0[0] + B.CAP_R + B.HANDLE[0] / 2 + 0.016     # handle line behind the cap, clear of it
    tip_in = B.CAP0[1] + 0.040                                # handle tip 40 mm past the cap's centre
    grip_in = tip_in - B.GRIP_FROM_TIP
    out_x = B.FRONT - 0.060
    R.add(_tip(1.5, (B.SPOON_X, gy, hi), "over the spoon's handle", FINGER_OPEN),
          _tip(1.0, (B.SPOON_X, gy, z), "down round it", FINGER_OPEN),
          _tip(0.6, (B.SPOON_X, gy, z), "take it"),
          _tip(2.0, (rake_x, gy, z), "slide it along, tip outside the oven, to behind the cap's line"),
          _tip(2.0, (rake_x, grip_in, z), "push the handle in under the oven past the cap"),
          _tip(0.5, (rake_x, grip_in, z), "settle"))
    n = 6
    for i in range(1, n + 1):
        R.add(_tip(0.7, (rake_x + (out_x - rake_x) * i / n, grip_in, z), "draw it toward the front: rake the cap out"))
    R.add(_tip(1.2, (out_x, grip_in - 0.06, z), "draw the spoon clear"),
          _tip(0.5, (out_x, grip_in - 0.06, z), "lay it down", FINGER_OPEN),
          _tip(1.0, (out_x, grip_in - 0.06, hi), "lift off", FINGER_OPEN),
          Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN))
    end = R.t + 2.0
    L.until(end, "done")
    R.until(end, "done")
    return {"left": L.w, "right": R.w}
