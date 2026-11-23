"""Desk: pack a small desktop robot back into its retail box.

The open box base stands on the desk with its die-cut foam insert; the robot, the top foam pad and the
box's lid lie beside it. The robot goes down into its foam pocket the right way up, the pad goes on top of
it, and the telescoping lid slides down over the base until it rests on the pad.
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

NAME = "robot-pack-foam-box"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1400
OBJECT_NAMES = ("box_base", "robot", "top_pad", "lid")
OVERVIEW_EYE = (0.75, -0.55, 1.10)
OVERVIEW_TARGET = (0.27, -0.02, 0.78)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.base_object(), B.robot_object(), B.pad_object(), B.lid_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_robot_pack_foam_box", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

PAD_DRAG = 0.010                             # letting go drags the light pad about 10 mm toward +y: aim short
REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}


def _open_for(width, margin=0.012):
    """Finger value with the pads `margin` wider than `width` in all."""
    return max(FINGER_OPEN, FINGER_OPEN * (width + margin) / 0.094)


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, yaw=0.0):
    rot = simkit.rot_z(math.radians(yaw)) @ DOWN
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


_FIXES = []


def _carry(arm, start, goal, tip_dz_pick, z_pick_root, z_place_root, width, name, lift=0.10, side="right",
           part=None, offset=False):
    """Pick a part by its sides (jaws across y) and set it down at goal: tip_dz_pick is the fingertips'
    height relative to the part's root; z_*_root are the root's heights above the bench."""
    pre = FINGER_OPEN
    let_go = max(FINGER_OPEN, FINGER_OPEN * (width + 0.020) / 0.094)   # open only 20 mm wider: a light part is not dragged
    sx, sy = start
    if offset:
        # the +y finger closes first and slides a light part toward -y: come down with the -y finger just
        # outside the part's -y side, so the part is pushed against it rather than past it
        sy = sy + (0.047 - width / 2 - 0.002)
    gx, gy = goal
    zp = B.TOP + z_pick_root + tip_dz_pick
    zg = B.TOP + z_place_root + tip_dz_pick
    arm += [_tip(1.4, (sx, sy, zp + 0.06), f"over the {name}", pre),
            _tip(1.0, (sx, sy, zp), f"down round the {name}", pre),
            _tip(0.7, (sx, sy, zp), f"take the {name}"),
            _tip(1.0, (sx, sy, B.TOP + lift), "lift it"),
            _tip(1.6, (gx, gy, B.TOP + lift), "over the box"),
            _tip(1.0, (gx, gy, B.TOP + lift), "hold still: check where it sits in the hand")]
    _FIXES.append({"side": side, "part": part, "goal": (gx, gy), "at": sum(w.duration for w in arm) - 0.1,
                   "from": len(arm), "to": len(arm) + 5})
    arm += [_tip(1.0, (gx, gy, B.TOP + lift), "hold still: check again")]
    _FIXES.append({"side": side, "part": part, "goal": (gx, gy), "at": sum(w.duration for w in arm) - 0.1,
                   "from": len(arm), "to": len(arm) + 6})
    arm += [_tip(1.2, (gx, gy, zg + 0.025), "down to just above its place"),
            _tip(0.8, (gx, gy, zg + 0.025), "hold still: check once more")]
    _FIXES.append({"side": side, "part": part, "goal": (gx, gy), "at": sum(w.duration for w in arm) - 0.1,
                   "from": len(arm), "to": len(arm) + 4})
    arm += [_tip(1.0, (gx, gy, zg + 0.012), "down over its place"),
            _tip(0.8, (gx, gy, zg), "set it in"),
            _tip(0.6, (gx, gy, zg), "let go", let_go),
            _tip(0.8, (gx, gy, B.TOP + lift + 0.03), "lift away", let_go)]


def demo():
    """Right hand: the robot into its pocket, the top pad onto it, then the lid down over the base until it
    rests on the pad. Left hand stays at rest."""
    _FIXES.clear()
    R = [Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN)]
    _carry(R, B.BOT0, (B.BX, B.BY), 0.012, B.BOT[2] / 2, B.BOT_SEATED + 0.002, B.BOT[1], "robot",
           side="right", part="robot")
    _carry(R, B.PAD0, (B.BX, B.BY - PAD_DRAG), 0.003, B.PAD[2] / 2, B.PAD_SEATED - 0.004, B.PAD[1], "top pad",
           side="right", part="top_pad", offset=True)
    _carry(R, B.LID0, (B.BX, B.BY), B.LID_TOP_T / 2 - 0.004, B.SKIRT + B.LID_TOP_T / 2, B.LID_SEATED - 0.004,
           B.LID_OUT[1], "lid", lift=0.15, side="right", part="lid", offset=True)
    R += [Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN)]
    L = [Waypoint(sum(w.duration for w in R), REST["left"], FINGER_OPEN, "rest", DOWN)]
    end = max(sum(w.duration for w in L), sum(w.duration for w in R)) + 2.5
    for arm, side in ((L, "left"), (R, "right")):
        arm.append(Waypoint(max(0.1, end - sum(w.duration for w in arm)), REST[side], FINGER_OPEN, "done", DOWN))
    return {"left": L, "right": R}


class _Controller:
    """Scripted demo only (reads ground truth, as scripted demos may): while a hand holds a part still over
    the box, measure where the part sits and shift the rest of the set-down so the part, not the hand,
    lands on its place."""

    def __init__(self, env, plans):
        self.env, self.plans, self.done = env, plans, set()

    def action(self, clock):
        for k, f in enumerate(_FIXES):
            if k in self.done or clock < f["at"]:
                continue
            self.done.add(k)
            base = self.env.unwrapped
            def pose(n):
                d = base.scene[n].data
                return np.r_[(d.root_pos_w[0] - base.scene.env_origins[0]).cpu().numpy(), d.root_quat_w[0].cpu().numpy()]
            part, box = pose(f["part"]), pose("box_base")
            pos = part[:3].astype(float)
            dyaw = (B.yaw_deg(part) - B.yaw_deg(box) + 90.0) % 180.0 - 90.0     # square to the box, either way round
            turn = simkit.rot_z(math.radians(-dyaw))
            plan = self.plans[f["side"]]
            hold = np.asarray(plan.waypoints[f["from"] - 1].pos, float)     # the hand while it holds still
            rel = pos - hold
            d = np.array([pos[0] - f["goal"][0], pos[1] - f["goal"][1], 0.0])
            shift = -d
            for i in range(f["from"], f["to"] + 1):
                wp = plan.waypoints[i]
                p = np.asarray(wp.pos, float)
                c = p + rel                                                  # the part's centre under this waypoint
                wp.pos = tuple(float(v) for v in (c + turn @ (p - c) - d))
                wp.yaw = turn @ np.asarray(wp.yaw, float)
            plan._simkit_moves = None
            print(f"[demo] {f['part']} {1000 * shift[0]:.1f}, {1000 * shift[1]:.1f} mm and {-dyaw:.1f} deg off its place: set-down corrected", flush=True)
        return np.concatenate([self.plans[s].action7(clock)[0] for s in ("left", "right")])


def make_demo_controller(env, plans):
    return _Controller(env, plans)
