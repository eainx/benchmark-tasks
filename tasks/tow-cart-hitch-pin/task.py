"""Material handling: "Hitch the cart behind to the cart in front."

Two tow carts stand nose to tail on their casters (cropped sections, real size). The lead cart's rear has
a clevis jaw (two 12 mm steel plates, 45 mm apart) with a 3/4 in drop pin seated through it; the trailing
cart's 32 mm drawbar, on a vertical pivot, lies swung 22 degrees off line with its forged eye outside the
jaw. Coupling them is a two-hand job: one hand draws the pin up by its T-handle and keeps it up (it drops
back under its own weight), the other swings the drawbar's eye into the jaw and holds it exactly under
the pin; then the pin is let go so it falls through the eye and the lower plate. Only a centred eye lets
the pin pass; anywhere else the pin comes down on the eye and the carts stay uncoupled.
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
import hitch as H  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "tow-cart-hitch-pin"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
SCENE_ONLY = False
MAX_STEPS = 80 if SCENE_ONLY else 1200
OBJECT_NAMES = ("drawbar", "pin")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (H.lead_cart_object(), H.trail_cart_object(), H.drawbar_object(), H.pin_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_tow_cart_hitch_pin", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------------------
# Right hand: pitched 45 degrees forward, closes on the pin's T-handle crossbar, draws the pin up 78 mm and
# keeps it there; once the eye is held under it, opens: the pin drops through the eye.
# Left hand: straight down on the drawbar's red grip sleeve (fingers across the bar), swings the bar on its
# pivot until the eye sits under the pin, holds it there until the pin is through, then lets go.


def frame(a, close):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - (c @ a) * a
    c = c / np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


def front(p_deg):
    """Coming in from the arms' side (+x), pitched down p_deg, fingers closing along y."""
    p = math.radians(p_deg)
    return frame((math.cos(p), 0.0, -math.sin(p)), (0.0, -1.0, 0.0))


def across_bar(theta_deg):
    """Straight down, fingers closing across the drawbar at swing angle theta."""
    t = math.radians(theta_deg)
    return frame((0.0, 0.0, -1.0), (math.cos(t), math.sin(t), 0.0))


F_BAR = -(0.032 - 0.005) / 2          # the 32 mm bar squeezed ~5 mm
F_BAR_NEAR = -0.032
F_HANDLE = -(0.015 - 0.006) / 2       # the 15 mm crossbar squeezed
F_HANDLE_NEAR = -0.030
PIN_P = 45.0
LIFT = 0.078
GRIP_R = 0.300                        # the left hand's grip on the bar, from the pivot (the red sleeve)
REST = {"left": (0.18, 0.20, H.TOP + 0.18), "right": (0.18, -0.20, H.TOP + 0.18)}
ALIGN_DEG = 0.0                       # the bar on line: the eye's centre on the pin's axis
PIN_AIM = np.array([-0.004, 0.0, 0.0])  # aim the pad centre a little short of the crossbar's middle


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def demo():
    if SCENE_ONLY:
        return {side: [_w(1.5, REST[side], FINGER_OPEN, "look", DOWN), _w(MAX_STEPS / HZ, REST[side], FINGER_OPEN, "hold", DOWN)]
                for side in ("left", "right")}
    up = np.array([0.0, 0.0, 1.0])
    handle = np.array([H.HX, H.HY, H.HANDLE_Z]) + PIN_AIM
    back = np.array([-0.05, 0.0, 0.05])
    right = [
        _w(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
        _w(2.0, handle + back, F_HANDLE_NEAR, "behind and over the pin's T-handle", front(PIN_P)),
        _w(1.2, handle, F_HANDLE_NEAR, "round the crossbar", front(PIN_P)),
        _w(0.8, handle, F_HANDLE, "grip the T-handle", front(PIN_P)),
        _w(1.6, handle + LIFT * up, F_HANDLE, "draw the pin up out of the jaw", front(PIN_P)),
    ]
    t_up = sum(w.duration for w in right)
    th0 = H.START_DEG
    g0 = H.bar_point(th0, GRIP_R)
    left = [
        _w(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
        _w(2.0, g0 + 0.07 * up, F_BAR_NEAR, "over the drawbar's grip sleeve", across_bar(th0)),
        _w(1.2, g0, F_BAR_NEAR, "down round the bar", across_bar(th0)),
        _w(0.8, g0, F_BAR, "grip the bar", across_bar(th0)),
    ]
    t = sum(w.duration for w in left)
    left.append(_w(max(0.5, t_up + 0.3 - t), g0, F_BAR, "wait for the pin to be up", across_bar(th0)))
    angles = [-17.0, -12.0, -7.0, -3.0, -1.0, ALIGN_DEG]
    durs = [0.9, 0.9, 0.9, 0.9, 0.8, 0.8]
    for a, d in zip(angles, durs):
        left.append(_w(d, H.bar_point(a, GRIP_R), F_BAR, "swing the eye into the jaw under the pin", across_bar(a)))
    left.append(_w(1.0, H.bar_point(ALIGN_DEG, GRIP_R), F_BAR, "hold the eye under the pin", across_bar(ALIGN_DEG)))
    t_aligned = sum(w.duration for w in left)
    t = sum(w.duration for w in right)
    right += [
        _w(max(0.5, t_aligned - t), handle + LIFT * up, F_HANDLE, "keep the pin up", front(PIN_P)),
        _w(0.6, handle + LIFT * up, F_HANDLE_NEAR, "let the pin drop", front(PIN_P)),
        _w(1.0, handle + LIFT * up + 0.02 * up, F_HANDLE_NEAR, "fingers up off the handle", front(PIN_P)),
        _w(1.2, handle + LIFT * up + back + 0.04 * up, FINGER_OPEN, "back away", front(PIN_P)),
        _w(2.0, REST["right"], FINGER_OPEN, "stand clear", DOWN),
        _w(3.0, REST["right"], FINGER_OPEN, "done", DOWN),
    ]
    gA = H.bar_point(ALIGN_DEG, GRIP_R)
    t_drop = sum(w.duration for w in right[:-4])
    t = sum(w.duration for w in left)
    left += [
        _w(max(0.5, t_drop + 0.5 - t), gA, F_BAR, "hold the bar while the pin goes through", across_bar(ALIGN_DEG)),
        _w(0.6, gA, F_BAR_NEAR, "let go of the bar", across_bar(ALIGN_DEG)),
        _w(1.0, gA + 0.08 * up, F_BAR_NEAR, "up off the bar", across_bar(ALIGN_DEG)),
        _w(2.0, REST["left"], FINGER_OPEN, "stand clear", DOWN),
        _w(3.0, REST["left"], FINGER_OPEN, "done", DOWN),
    ]
    return {"left": left, "right": right}
