"""Food prep: "Seal the bag."

A filled 1-quart zip-top freezer bag of chopped vegetables lies on a butcher block, its press-to-seal
track open and standing out past the block's near edge. One hand holds the bag down at a corner while
the other pinches the track between its fingers at that corner and slides along to the other corner,
pressing the rib into its groove the whole way; then both hands let go and stand clear.
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
import bag as B  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "zip-top-bag-seal"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = ("bag", *(f"coin{i + 1}" for i in range(6)))

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (B.board_object(), B.scraps_object(), B.bowl_object(), B.bag_box_object(), B.bag_object(),
             *(B.coin_object(i) for i in range(len(B.COINS))))
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_zip_top_bag_seal", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


# -- the track's closing rule ------------------------------------------------------------------------
GAP_MAX = B.CLOSED_T + 0.0010        # pads squeezed to within 1 mm of the pressed-home track's 2.6 mm
                                     # (the open track stands 6.2 mm: rib 3 mm clear above its groove)
SQUARE_DEG = 30.0                    # the pinch across the track: fingers' closing line within this of the films' normal
PAD_DEPTH = (-0.008, 0.013)          # the track between the pads: from 8 mm behind the TCP to the pad tips
PAD_HALF_W = 0.009                   # cells under the pads' width (18 mm pads)
BETWEEN = 0.008                      # the track on the pads' closing line (the clamped track sits 5-6 mm off the
                                     # tool centre's line in this gripper model)

SHARED = {}


def _arr(x):
    x = getattr(x, "torch", x)
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class Dynamics:
    """The track closing, from measured poses only (a disclosed stand-in for the rib snapping into its
    groove).

    A cell of the track closes where a hand's pads are squeezed shut across it: fingers closed to within
    1 mm of the pressed-home track's thickness (2.6 mm; open, the profiles stand 6.2 mm), the track between the pads (between their tips
    and 8 mm behind the tool centre, within 8 mm of the tool centre's closing line, under their 18 mm width) and the
    fingers' closing line within 30 degrees of the films' normal, so the rib is pressed square onto its
    groove. A closed cell stays closed and reads green. The pads touch the groove profile itself (the
    rib above it is drawn only), so sliding a clamped pinch along the track pulls on the bag by friction.
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        self.stage = omni.usd.get_context().get_stage()
        index = {i: (name, k) for name, ids in B.TRACK_PIECES.items() for k, i in enumerate(ids)}
        self.prims = {}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if "/bag/piece" in path:
                tail = path.rsplit("/bag/piece", 1)[1]
                if tail.isdigit() and int(tail) in index:
                    self.prims[index[int(tail)]] = prim
        if len(self.prims) != 3 * B.N_CELLS:
            raise RuntimeError(f"bag track pieces not found on the stage ({len(self.prims)})")

    def _show(self, k, closed):
        from pxr import UsdGeom
        for name, vis in (("groove", not closed), ("rib", not closed), ("sealed", closed)):
            im = UsdGeom.Imageable(self.prims[(name, k)])
            im.MakeVisible() if vis else im.MakeInvisible()

    def reset(self):
        self.closed = [False] * B.N_CELLS
        for k in range(B.N_CELLS):
            self._show(k, False)

    def before_step(self, step, action):
        pass

    def _hand(self, side, robot):
        art = self.base.scene[f"{side}_robot"]
        names = list(art.body_names)
        i = names.index("gripper")
        w, x, y, z = (float(v) for v in _arr(art.data.body_quat_w)[0, i])      # Isaac Lab: w, x, y, z
        rg = B.quat_matrix((x, y, z, w))
        tcp = np.asarray(robot[side]["tcp"], float)[:3]
        axis, approach = rg[:, 0], rg[:, 2]               # the fingers close along the tool's x; it points along z
        gap = -float(sum(robot[side]["fingers"]))
        return tcp, approach, axis, gap

    def after_step(self, step, objects, root_poses, robot):
        bag = np.asarray(root_poses["bag"], float)
        r = B.quat_matrix(bag[3:7])
        normal = r[:, 2]
        pinch = {}
        for side in ("left", "right"):
            tcp, approach, axis, gap = self._hand(side, robot)
            square = abs(float(axis @ normal)) >= math.cos(math.radians(SQUARE_DEG))
            width = np.cross(approach, axis)
            on = []
            if gap <= GAP_MAX and square:
                for k in range(B.N_CELLS):
                    p = bag[:3] + r @ np.array([B.TRACK_X - B.ROOT_X, B.cell_y(k) - B.BAG_Y, B.GROOVE_Z - B.ROOT_Z])
                    d = p - tcp
                    if (PAD_DEPTH[0] <= d @ approach <= PAD_DEPTH[1] and abs(d @ axis) <= BETWEEN
                            and abs(d @ width) <= PAD_HALF_W):
                        on.append(k)
            for k in on:
                if not self.closed[k]:
                    self.closed[k] = True
                    self._show(k, True)
            pinch[side] = {"gap_mm": round(1000 * gap, 2), "square": bool(square), "cells": on,
                           "axis": [round(float(v), 3) for v in axis], "approach": [round(float(v), 3) for v in approach]}
        SHARED["bag"] = bag
        return {"closed": int(sum(self.closed)), "closed_cells": "".join("1" if c else "0" for c in self.closed),
                "pinch": pinch}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------------------
# The left hand comes in from the left side and pinches the mouth's left corner, track and flanges
# together, and holds it there; the right hand pinches the track beside it and slides along it to the
# right corner, pulling along the track against the held corner. Then the left lets go and the right
# comes back to press home the bit of track the left hand was holding.

REST = {"left": (0.16, 0.24, B.TOP + 0.16), "right": (0.16, -0.24, B.TOP + 0.16)}
PITCH = math.radians(25.0)           # each hand comes at the track from above as well
A_R = np.array([math.cos(PITCH), 0.0, -math.sin(PITCH)])               # right hand: in across the mouth (+x)
ROT_R = np.column_stack([np.array([math.sin(PITCH), 0.0, math.cos(PITCH)]), np.array([0.0, -1.0, 0.0]), A_R])
A_L = np.array([0.0, -math.cos(PITCH), -math.sin(PITCH)])              # left hand: in from the left side (-y)
ROT_L = np.column_stack([np.array([0.0, -math.sin(PITCH), math.cos(PITCH)]), np.array([-1.0, 0.0, 0.0]), A_L])
F_PINCH = -(B.PROFILE_T - 0.0008) / 2          # squeeze the 1.6 mm profile: commanded 0.8 mm apart
F_HOLD = FINGER_CLOSED                         # the holding hand pinches its corner hard
F_HALF = -0.012                                # pads 24 mm apart coming in round the track
CORNER_L = np.array([B.TRACK_X + 0.002, B.BAG_Y + B.BAG_W / 2 - 0.0075, B.GROOVE_Z])   # the left hand's pinch
Y0, Y1 = B.BAG_Y + 0.060, B.BAG_Y - 0.086      # the right hand's slide, from beside the held corner
Y_LAST = B.BAG_Y + 0.073                       # then the held corner, pressed home
AHEAD = 0.004                                  # the track this far ahead of the TCP


def _at(p, a, back=0.0, up=0.0):
    return tuple(map(float, np.asarray(p, float) - a * (AHEAD + back) + np.array([0.0, 0.0, up])))


def _r(y, back=0.0, up=0.0):
    return _at((B.TRACK_X, y, B.GROOVE_Z), A_R, back, up)


SLIDE_S = 8.0


def demo():
    right = [
        Waypoint(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
        Waypoint(2.5, _r(Y0, back=0.060, up=0.030), F_HALF, "in front of the mouth", ROT_R),
        Waypoint(1.5, _r(Y0, back=0.030), F_HALF, "level with the track", ROT_R),
        Waypoint(1.5, _r(Y0), F_HALF, "round the track beside the held corner", ROT_R),
        Waypoint(1.0, _r(Y0), F_PINCH, "pinch it", ROT_R),
    ]
    n = 8
    for i in range(1, n + 1):
        right.append(Waypoint(SLIDE_S / n, _r(Y0 + (Y1 - Y0) * i / n), F_PINCH, "slide along the track", ROT_R))
    right += [
        Waypoint(0.5, _r(Y1 + 0.004), F_HALF, "ease the pinch", ROT_R),
        Waypoint(0.8, _r(Y1 - 0.004), F_HALF, "round the far corner", ROT_R),
        Waypoint(1.0, _r(Y1 - 0.004), F_PINCH, "press the far corner home", ROT_R),
        Waypoint(0.8, _r(Y1), F_HALF, "let go of the track", ROT_R),
        Waypoint(1.2, _r(Y1, back=0.040, up=0.020), F_HALF, "draw back off the mouth", ROT_R),
    ]
    t_back = sum(w.duration for w in right)
    right += [
        Waypoint(3.0, _r(Y_LAST, back=0.040, up=0.020), F_HALF, "across to the held corner", ROT_R),
        Waypoint(1.5, _r(Y_LAST, back=0.030), F_HALF, "level with the track", ROT_R),
        Waypoint(1.2, _r(Y_LAST), F_HALF, "round the track at the corner", ROT_R),
        Waypoint(1.0, _r(Y_LAST), F_PINCH, "press the last of it home", ROT_R),
        Waypoint(0.8, _r(Y_LAST), F_HALF, "let go", ROT_R),
        Waypoint(1.2, _r(Y_LAST, back=0.050, up=0.020), F_HALF, "draw back", ROT_R),
        Waypoint(2.0, REST["right"], FINGER_OPEN, "stand clear", DOWN),
        Waypoint(2.5, REST["right"], FINGER_OPEN, "done", DOWN),
    ]
    left = [
        Waypoint(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
        Waypoint(2.5, _at(CORNER_L, A_L, back=0.060, up=0.030), F_HALF, "beside the mouth's left corner", ROT_L),
        Waypoint(1.5, _at(CORNER_L, A_L, back=0.030), F_HALF, "level with the track", ROT_L),
        Waypoint(1.2, _at(CORNER_L, A_L), F_HALF, "round the corner", ROT_L),
        Waypoint(0.8, _at(CORNER_L, A_L), F_HOLD, "pinch the corner and hold it", ROT_L),
    ]
    t = sum(w.duration for w in left)
    left += [
        Waypoint(max(0.5, t_back - t), _at(CORNER_L, A_L), F_HOLD, "hold the corner", ROT_L),
        Waypoint(0.6, _at(CORNER_L, A_L), F_HALF, "let go of the corner", ROT_L),
        Waypoint(1.2, _at(CORNER_L, A_L, back=0.060, up=0.030), F_HALF, "draw back off the bag", ROT_L),
        Waypoint(1.5, REST["left"], FINGER_OPEN, "stand clear", DOWN),
        Waypoint(12.0, REST["left"], FINGER_OPEN, "done", DOWN),
    ]
    return {"left": left, "right": right}
