"""Mailroom: "Sort the letters into their room boxes."

A desktop wooden mail sorter with six labelled pigeonholes (rooms 101-302) stands on the desk; beside it
a wire step organizer holds four incoming #10 envelopes upright, each addressed to a room in large type.
Each letter is taken out by its top edge, its room read, turned flat and slid fully into the pigeonhole
labelled with that room.
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
import mail as T  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "mail-sort-pigeonholes"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
SCENE_ONLY = False                     # True: static render for scene approval (arms rest, a short settle run)
MAX_STEPS = 80 if SCENE_ONLY else 2400
ENVELOPES = T.ENV_NAMES
OBJECT_NAMES = ("organizer",) + ENVELOPES

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (T.sorter_object(), T.organizer_object()) + tuple(T.envelope_object(i) for i in range(len(T.LETTERS)))
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_mail_sort_pigeonholes", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------------------
# One hand (the right). For each letter, front first: grip it by its top edge straight down (jaws closing
# across it, along x), lift it 12 mm clear of the end rail and slide it out sideways along its section
# past the organizer's open end; turn the grip about the vertical (the letter now lies along x, hanging);
# carry it along to x 0.46, clear of the organizer, and tip it flat there at y -0.12 (tool pointing +y,
# the letter ahead of the fingers; the flat grip's forearm reaches back, so never over the organizer);
# carry it along the sorter's
# front to its pigeonhole, slide it 70 mm in, open so it drops onto the compartment floor, back straight
# out, and come back the same way with the grip turned back. Every pose on this path solves by inverse
# kinematics to under 1 mm and 1 degree (README). The left hand is not needed and stays at rest.

REST = {"left": (0.08, 0.30, T.TOP + 0.20), "right": (0.20, -0.18, T.TOP + 0.18)}
FO = FINGER_OPEN
F_NARROW = -0.011                      # 22 mm apart: between the sections' neighbours
F_PINCH = 0.0                          # the 4 mm letter squeezed hard: jaws commanded shut
F_RELEASE = -0.008                     # jaws 16 mm apart: the letter drops free
GRIP_DEPTH = 0.021                     # pad tips this far below the letter's top edge
PAD = 0.013                            # pad tips past the tool centre
LIFT = 0.012                           # lifted this much before sliding out (clears the 6 mm end rail)
OVER = 0.035                           # tool this far above a letter's top edge before it comes down
Y_OUT = -0.165                         # slid out this far (its back edge clear of the organizer's end)
Y_TIP = -0.12                          # tipped flat here, in front of the sorter
Y_IN = 0.070                           # slid this far in (the opening is at y 0)
Y_BACK = -0.14
X_TIP = 0.45                           # letters are tipped flat out here, clear of the organizer (the flat grip's
                                       # forearm reaches back over whatever lies behind the hand)
Y_STN = -0.14                          # where the grip is tipped flat and back
Z_TIP = T.TOP + 0.11                   # the hanging letter's lower edge then clears the desk by about 13 mm


def frame(a, close):
    """Tool rotation: approach ``a``, fingers closing along ``close``."""
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - (c @ a) * a
    c = c / np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


GRIP_X = frame((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))      # straight down, jaws closing along x (across the letter)
GRIP_Y = frame((0.0, 0.0, -1.0), (0.0, -1.0, 0.0))     # straight down, jaws closing along -y
FLAT = frame((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))         # pointing +y, jaws closing vertically


def _w(t, p, f, label, rot):
    return Waypoint(t, np.asarray(p, float), f, label, rot)


def demo():
    if SCENE_ONLY:
        return {side: [_w(1.5, REST[side], FO, "look", DOWN),
                       _w(MAX_STEPS / HZ, REST[side], FO, "hold", DOWN)] for side in ("left", "right")}
    # Each waypoint is its own straight move: a finger target nudged by a micrometre ends a spline run.
    k = [0]

    def f(v):
        k[0] += 1
        return v - 1e-6 * (k[0] % 2)

    right = [_w(1.5, REST["right"], FO, "look", DOWN)]
    for i, (room, _, _) in enumerate(T.LETTERS):
        x, y, zc = T.envelope_pos(i)
        top = zc + T.ENV[2] / 2
        zg = top - GRIP_DEPTH + PAD
        zs = zg + LIFT
        r, c = T.slot_of(room)
        o = T.slot_centre(r, c)
        zi = o[2] + 0.026 + (0.015 if r == 2 else 0.0)   # held over the floor: room for the arm's sag (more at the top row)
        xs = o[0] + 0.004                                 # the letter rides 4 mm to -x of the tool centre: centre it in the opening
        zt = zs + 0.02                                    # tool height while turning and tipping
        if i == 0:
            right += [_w(1.5, np.array([x, Y_OUT, top + OVER]), f(F_NARROW), f"in line with letter {room}", GRIP_X)]
        right += [
            _w(2.0, np.array([x, y, top + OVER]), f(F_NARROW), f"over letter {room}", GRIP_X),
            _w(1.2, np.array([x, y, zg]), f(F_NARROW), f"down round letter {room}", GRIP_X),
            _w(0.6, np.array([x, y, zg]), F_PINCH, f"pinch letter {room} by its top edge", GRIP_X),
            _w(0.8, np.array([x, y, zs]), f(F_PINCH), "lift it clear of the end rail", GRIP_X),
            _w(2.5, np.array([x, Y_OUT, zs]), f(F_PINCH), "slide it out sideways along its section", GRIP_X),
            _w(1.5, np.array([x, Y_OUT, zt]), f(F_PINCH), "turn the grip: the letter now lies along x", GRIP_Y),
            _w(1.5, np.array([X_TIP, Y_OUT, Z_TIP]), f(F_PINCH), "along, clear of the organizer", GRIP_Y),
            _w(1.0, np.array([X_TIP, Y_STN, Z_TIP]), f(F_PINCH), "in front of the sorter", GRIP_Y),
            _w(2.0, np.array([X_TIP, Y_STN, Z_TIP]), f(F_PINCH), "tip it flat", FLAT),
            _w(2.0, np.array([xs, Y_TIP, zi]), f(F_PINCH), f"along to pigeonhole {room}", FLAT),
            _w(2.0, np.array([xs, Y_IN, zi]), f(F_PINCH), f"slide it all the way into {room}", FLAT),
            _w(0.6, np.array([xs, Y_IN, zi]), F_RELEASE, "let go", FLAT),
            _w(0.5, np.array([xs, Y_IN, zi]), f(F_RELEASE), "let it settle on the floor", FLAT),
            _w(1.5, np.array([xs, Y_BACK, zi]), f(F_RELEASE), "back straight out", FLAT),
        ]
        if i + 1 < len(T.LETTERS):
            xn, yn, zcn = T.envelope_pos(i + 1)
            topn = zcn + T.ENV[2] / 2
            ztn = topn - GRIP_DEPTH + PAD + LIFT + 0.02
            right += [
                _w(2.0, np.array([X_TIP, Y_STN, Z_TIP]), f(F_NARROW), "back along the front", FLAT),
                _w(2.0, np.array([X_TIP, Y_STN, Z_TIP]), f(F_NARROW), "tip the grip back down", GRIP_Y),
                _w(1.0, np.array([X_TIP, Y_OUT, Z_TIP]), f(F_NARROW), "out", GRIP_Y),
                _w(1.5, np.array([xn, Y_OUT, ztn]), f(F_NARROW), "to the organizer's end", GRIP_Y),
                _w(1.5, np.array([xn, Y_OUT, topn + OVER]), f(F_NARROW), "turn the grip across the next letter", GRIP_X),
            ]
    done = np.array([X_TIP, Y_TIP, Z_TIP])
    right += [
        _w(2.0, done, f(F_NARROW), "away from the sorter", FLAT),
        _w(1.0, done, FO, "open and stand clear", FLAT),
        _w(4.0, done, f(FO), "done", FLAT),
    ]
    left = [_w(1.5, REST["left"], FO, "look", DOWN),
            _w(sum(w.duration for w in right), REST["left"], FO, "stand by", DOWN)]
    return {"left": left, "right": right}
