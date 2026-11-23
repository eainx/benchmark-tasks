"""Kitchen: close a large box of kitchen matches.

The box lies flat on the counter, its tray pulled 50 mm out of the sleeve's right end. The sleeve is light
(17 g): pushed on its own, the tray just drags the sleeve along with it. One hand pins the sleeve down while
the other pushes the tray home, square, until its end is flush with the sleeve's end -- not past it, or it
comes out of the other end.
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

NAME = "matchbox-close-pinned"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 800
OBJECT_NAMES = ("sleeve", "tray")
OVERVIEW_EYE = (0.05, -0.45, 0.98)
OVERVIEW_TARGET = (0.30, -0.06, 0.76)

DOWN = simkit.patch_demo()
ALONG_X = simkit.rot_z(np.pi / 2) @ DOWN


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in (B.sleeve_object(), B.tray_object())]
    cfg = IsaacLabArenaEnvironment(name="yam_matchbox_close_pinned", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
FACE_Y = 0.010                                        # first guess: the fingers' leading face ahead of the tip point (+y)
_FIX = {"at": None, "from": None, "to": None, "goal_end": None}


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, rot=DOWN):
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def _rot_x(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


PUSH = _rot_x(0.5) @ ALONG_X                          # jaws along x (narrow across y), fingertips leading toward +y:
                                                      # the hand stays clear of the sleeve's end as the tray goes in


def demo():
    """Left hand: fingertips down on the sleeve's top near its left end, pinning it. Right hand: fingertips
    against the tray's right end, low, push it home along the sleeve until flush, then both lift away."""
    top = B.TOP + B.SL[2]
    pin = (B.MX, B.MY + 0.030)
    tray_end = B.MY - B.OUT - B.TL / 2                  # the tray's right end, start
    flush_end = B.MY - B.SL[1] / 2                      # the sleeve's right end
    z = B.TOP + 0.013
    L = [Waypoint(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
         _tip(1.5, (pin[0], pin[1], top + 0.03), "over the sleeve's left end", rot=ALONG_X),
         _tip(0.8, (pin[0], pin[1], top + 0.002), "fingertips on it", rot=ALONG_X),
         _tip(0.6, (pin[0], pin[1], top - 0.0015), "pin it down", rot=ALONG_X)]
    t_pin = sum(w.duration for w in L)
    R = [Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
         _tip(1.5, (B.MX, tray_end - FACE_Y - 0.03, z + 0.04), "over beyond the tray's end", rot=PUSH),
         _tip(1.0, (B.MX, tray_end - FACE_Y - 0.03, z), "down beside it", rot=PUSH)]
    R.append(Waypoint(max(0.1, t_pin - sum(w.duration for w in R)), R[-1].pos, FINGER_CLOSED, "wait for the pin", PUSH))
    mid = tray_end + 0.020
    R += [_tip(0.8, (B.MX, tray_end - FACE_Y - 0.002, z), "up to the tray's end", rot=PUSH),
          _tip(1.6, (B.MX, mid - FACE_Y, z), "push it in part way", rot=PUSH),
          _tip(0.6, (B.MX, mid - FACE_Y, z), "hold: check where its end is", FINGER_CLOSED + 1e-6, rot=PUSH)]
    _FIX["at"] = sum(w.duration for w in R) - 0.2
    _FIX["from"] = len(R)
    goal = flush_end - FACE_Y
    n = 4
    for i in range(1, n + 1):
        R.append(_tip(0.6, (B.MX, mid - FACE_Y + (goal - mid + FACE_Y) * i / n, z), "push it home, square", rot=PUSH))
    R += [_tip(0.6, (B.MX, goal, z), "flush: stop", rot=PUSH)]
    _FIX["to"] = len(R) - 1
    R += [_tip(0.8, (B.MX, goal - 0.04, z), "back off", rot=PUSH),
          _tip(0.8, (B.MX, goal - 0.04, top + 0.06), "up clear", rot=PUSH),
          Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN)]
    t_done = sum(w.duration for w in R) - 3.1
    L.append(Waypoint(max(0.1, t_done - sum(w.duration for w in L)), L[-1].pos, FINGER_CLOSED, "keep it pinned", ALONG_X))
    L += [_tip(0.8, (pin[0], pin[1], top + 0.06), "lift off", rot=ALONG_X),
          Waypoint(1.5, REST["left"], FINGER_OPEN, "rest", DOWN)]
    end = max(sum(w.duration for w in L), sum(w.duration for w in R)) + 2.0
    for arm, side in ((L, "left"), (R, "right")):
        arm.append(Waypoint(max(0.1, end - sum(w.duration for w in arm)), REST[side], FINGER_OPEN, "done", DOWN))
    return {"left": L, "right": R}


class _Controller:
    """Scripted demo only (reads ground truth, as scripted demos may): while the hand holds the half-pushed
    tray still, measure how far the tray's end is from where the plan assumed, and shift the rest of the
    push so the end stops flush with the sleeve's."""

    def __init__(self, env, plans):
        self.env, self.plans, self.done = env, plans, False

    def action(self, clock):
        if not self.done and clock >= _FIX["at"]:
            self.done = True
            base = self.env.unwrapped
            def pose(n):
                d = base.scene[n].data
                return np.r_[(d.root_pos_w[0] - base.scene.env_origins[0]).cpu().numpy(), d.root_quat_w[0].cpu().numpy()]
            out = B.tray_out(pose("sleeve"), pose("tray"))
            planned_out = B.OUT - 0.020
            shift = 0.3 * (out - planned_out)            # part of it only: the lag under load shrinks as the tray stops
            plan = self.plans["right"]
            for i in range(_FIX["from"], _FIX["to"] + 1):
                wp = plan.waypoints[i]
                wp.pos = tuple(float(v) for v in np.asarray(wp.pos) + np.array([0.0, shift, 0.0]))
            plan._simkit_moves = None
            print(f"[demo] tray end {1000 * out:.1f} mm out at the check (plan {1000 * planned_out:.1f}): push shifted {1000 * shift:.1f} mm", flush=True)
        return np.concatenate([self.plans[s].action7(clock)[0] for s in ("left", "right")])


def make_demo_controller(env, plans):
    return _Controller(env, plans)
