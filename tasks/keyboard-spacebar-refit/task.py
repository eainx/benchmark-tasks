"""Desk: refit a mechanical keyboard's spacebar onto its switch and both stabilizer stems.

The 6.25u spacebar has been pulled off for cleaning and lies keycap-up beside the keyboard. Its three
sockets -- the switch's in the middle and the stabilizers' 50 mm either side -- have to come down over
their three stems together, the cap level, and be pressed home; the two stabilizer stems ride on one wire,
so a press at one end cocks them and the far socket misses. Seated, the key must still spring back up.
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

NAME = "keyboard-spacebar-refit"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1000
OBJECT_NAMES = ("spacebar", "switch_stem", "stab_bar")
OVERVIEW_EYE = (0.62, -0.55, 1.02)
OVERVIEW_TARGET = (0.30, -0.10, 0.77)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.keyboard_object(), B.switch_stem(), B.stab_bar(), B.cap_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_keyboard_spacebar_refit", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
ALONG_X = simkit.rot_z(np.pi / 2) @ DOWN              # jaws closing along x, across the cap's 18.5 mm depth
GRIP_TIP = 0.0020                                     # fingertips 2 mm above the cap's lower edge (the pads only bite near their tips)
                                                      # fingertips' downward arc as the jaws open over the plate
CAP_FROM_TCP = (B.SB_H - B.TOP_T / 2) - GRIP_TIP - PAD_TIP   # cap root above the TCP while held
RELEASE = 0.20 * FINGER_OPEN                          # jaws opened only a few mm past the walls: their tips arc down as they open
HELD = FINGER_CLOSED + 1e-6                           # a separate move: the dwell where the grip is measured
_SHIFTS = []                                          # [(time, first waypoint, last waypoint)] to correct


def _tcp(seconds, tcp, label, finger=FINGER_CLOSED, rot=ALONG_X):
    return Waypoint(seconds, tuple(map(float, tcp)), finger, label, np.array(rot, float))


GRIP_Y = 0.0                                           # the right hand takes the cap at its middle (balanced)
PRESS_Y = 0.054                                       # both hands press near the ends, 108 mm apart: the hands clear each other


PRE_GRIP = 0.60 * FINGER_OPEN                         # 12 mm each side: the tilted front fingertip stays off the desk
GRIP_TILT = 0.0                                      # fingertips leaning back (+x) 24 degrees: the back finger
GRIP_ROT = simkit.rot_y(-GRIP_TILT) @ ALONG_X         # pinches the back wall near its top, the front finger the
                                                      # front wall near its bottom, and both fingers' bodies lean
                                                      # forward, clear of the row of keys behind the spacebar
GRIP_MID = -0.0010                                          # the fingertips' midpoint 5.6 mm above the cap's lower edge:
                                                      # the front tip 2 mm up the front wall, the back tip near the top
CAP_OFF = -np.array([0.0, GRIP_Y, GRIP_MID]) + GRIP_ROT @ np.array([0.0, 0.0, PAD_TIP])   # cap root - TCP


def _cap_at(seconds, root, label, finger=FINGER_CLOSED):
    """The right hand placed so that the held cap's root is at ``root``."""
    r = np.asarray(root, float)
    return _tcp(seconds, r - CAP_OFF, label, finger, rot=GRIP_ROT)


def _tip(seconds, xyz, label, finger=FINGER_CLOSED, rot=DOWN):
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def _rot_x(a):
    c, s_ = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s_], [0.0, s_, c]])


def demo():
    """Right hand: take the spacebar across its depth at the middle, lift it, carry it level over its
    place (stopping twice while the grip is checked: the cap squared and centred over the stems), lower it
    until the fingertips are just above the neighbouring keys and let it drop onto the stems -- its skirt
    settles round the switch housing. Then both hands' fingertips press its two ends down together until it
    seats on all three stems, and lift off so the key springs back up."""
    w = [Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN)]
    cx, cy = B.CAP0
    root0 = np.array([cx, cy, B.TOP + B.SB_H - B.TOP_T / 2])
    hover = np.array([B.KX, B.KY, B.TOP + B.TIP_Z + 0.030 + B.TOP_T / 2 + B.BOSS_H])
    near = np.array([B.KX, B.KY, B.TOP + B.TIP_Z + 0.008 + B.TOP_T / 2 + B.BOSS_H])
    rest = np.array([B.KX, B.KY, B.TOP + B.TIP_Z - 0.0015 + B.TOP_T / 2 + B.BOSS_H])
    low = np.array([B.KX, B.KY, B.TOP + B.TIP_Z + 0.001 + B.TOP_T / 2 + B.BOSS_H])   # sockets 4 mm over the cross tips: as low as the hand goes before the back finger meets the row behind
    w += [_cap_at(1.5, root0 + [0, 0, 0.04], "over the spacebar", PRE_GRIP),
          _cap_at(1.0, root0, "down round it, jaws just wider than the cap", PRE_GRIP),
          _cap_at(0.6, root0, "take it"),
          _cap_at(1.0, root0 + [0, 0, 0.04], "lift it"),
          _cap_at(1.6, hover, "carry it over its place, level"),
          _cap_at(0.8, hover, "hold still: check the grip", HELD)]
    _SHIFTS.clear()
    first = len(w)
    w += [_cap_at(0.6, hover, "over its place"),
          _cap_at(1.2, near, "down to just above the crosses"),
          _cap_at(0.8, near, "hold still: check again", HELD + 1e-6)]
    second = len(w)
    w += [_cap_at(0.4, near, "squared over the stems"),
          _cap_at(1.2, low, "down to 1 mm over the cross tips"),
          _cap_at(0.8, low, "hold still: check once more", HELD + 2e-6)]
    third = len(w)
    w += [_cap_at(0.4, low, "squared over the crosses"),
          _cap_at(2.0, rest, "slowly down: the crosses 1.5 mm into their sockets"),
          _cap_at(1.0, rest, "let go: it sits on the crosses", FINGER_OPEN),
          _tcp(1.0, np.zeros(3), "lift straight off", FINGER_OPEN, rot=GRIP_ROT)]
    last = len(w) - 1
    _SHIFTS.append((sum(x.duration for x in w[:first]) - 0.3, first, last, hover))
    _SHIFTS.append((sum(x.duration for x in w[:second]) - 0.3, second, last, near))
    _SHIFTS.append((sum(x.duration for x in w[:third]) - 0.3, third, last, low))
    w[-1].pos = tuple(np.asarray(w[-2].pos) + np.array([0.0, 0.0, 0.03]))
    top = B.TOP + B.SEATED_TOP
    t_press = sum(x.duration for x in w)
    ends = {"right": -PRESS_Y, "left": PRESS_Y}
    left = [Waypoint(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
            Waypoint(max(0.1, t_press - 1.5), REST["left"], FINGER_OPEN, "wait", DOWN)]
    for arm, side in ((w, "right"), (left, "left")):
        y = B.KY + ends[side]
        lean = _rot_x(0.5 if side == "right" else -0.5) @ ALONG_X   # fingertips leaning in toward the middle,
                                                                     # the two hands' bodies leaning apart
        arm += [_tip(1.2, (B.KX, y, top + 0.02), "over this end of the spacebar", rot=lean),
                _tip(0.6, (B.KX, y, top + 0.004), "fingertips on it", rot=lean),
                _tip(1.4, (B.KX, y, top - 0.0040), "both ends down together: seat it", rot=lean),
                _tip(0.6, (B.KX, y, top - 0.0040), "hold", rot=lean),
                _tip(3.0, (B.KX, y, top + 0.001), "ease off slowly: let the key come up under the fingertips", rot=lean),
                _tip(0.8, (B.KX, y, top + 0.03), "lift off", rot=lean),
                Waypoint(1.5, REST[side], FINGER_OPEN, "rest", DOWN)]
    end = max(sum(x.duration for x in w), sum(x.duration for x in left)) + 2.0
    for arm, side in ((w, "right"), (left, "left")):
        arm.append(Waypoint(max(0.1, end - sum(x.duration for x in arm)), REST[side], FINGER_OPEN, "done", DOWN))
    return {"left": left, "right": w}


class _Controller:
    """Scripted demo only (reads ground truth, as scripted demos may): at two still moments while the cap
    is held over the stems, measure where it actually sits relative to its goal (position and turn) and
    shift and turn the following waypoints so the cap, not the hand, arrives over the stems."""

    def __init__(self, env, plans):
        self.env, self.plans, self.done = env, plans, set()

    def _correct(self, k, first, last, goal_cap):
        from yam_arena import ik, station
        import simkit as sk
        base = self.env.unwrapped
        obj = base.scene["spacebar"].data
        cap = (obj.root_pos_w[0] - base.scene.env_origins[0]).cpu().numpy().astype(float)
        qx, qy, qz, qw = (float(v) for v in obj.root_quat_w[0].cpu().numpy())   # (x, y, z, w) in this Isaac Lab
        rc = B.quat_matrix((qx, qy, qz, qw))
        if rc[0, 0] < 0:                                  # the cap is symmetric end for end
            rc = rc @ sk.rot_z(np.pi)
        turn = rc.T                                       # the rotation that brings the cap back level and square
        yaw = float(np.degrees(np.arctan2(rc[1, 0], rc[0, 0])))
        tilt = float(np.degrees(np.arccos(np.clip(rc[2, 2], -1, 1))))
        plan = self.plans["right"]
        d = cap - np.asarray(goal_cap, float)                           # where the cap is against where it should be
        if np.linalg.norm(d) > 0.015:
            print(f"[demo] cap not in the hand (offset {np.round(d * 1000, 1)} mm): no correction", flush=True)
            return
        for i in range(first, last + 1):
            wp = plan.waypoints[i]
            p = np.asarray(wp.pos, float)
            # turn the hand about the cap's centre (held CAP_FROM_TCP above the TCP), then shift by -d
            centre = p + CAP_OFF
            wp.pos = tuple(float(v) for v in (centre + turn @ (p - centre) - d))
            wp.yaw = turn @ np.asarray(wp.yaw, float)
        plan._simkit_moves = None
        print(f"[demo] correction {k}: offset {np.round(d * 1000, 2)} mm, yaw {yaw:.2f} deg, tilt {tilt:.2f} deg", flush=True)

    def action(self, clock):
        for k, (at, first, last, goal) in enumerate(_SHIFTS):
            if k not in self.done and clock >= at:
                self.done.add(k)
                self._correct(k, first, last, goal)
        return np.concatenate([self.plans[s].action7(clock)[0] for s in ("left", "right")])


def make_demo_controller(env, plans):
    return _Controller(env, plans)
