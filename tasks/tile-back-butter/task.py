"""Tiling: "Replace the missing tile on the step."

A raised step tiled with 75 x 150 mm ceramic tiles, spacers still in the joints; one front tile has
been chipped out and its old adhesive scraped off. The new tile stands on edge leaning against a stack
of spares on a cardboard sheet on the floor, and a tub of mixed adhesive stands by with a notched trowel
across its rim. A gap boxed in by tiles cannot be troweled, so the adhesive goes on the
tile's back: one hand holds the tile still, the other combs adhesive over its back; then it is turned
over, set in the gap level with its neighbours and pressed down to bed it.
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
import tile as T  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "tile-back-butter"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 2000
OBJECT_NAMES = ("step", "tile", "spares", "tub", "trowel")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (T.step_object(), T.tile_object(), T.spares_object(), T.tub_object(), T.trowel_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_tile_back_butter", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                           # measured state for the scripted demo's controller
LOAD_CELLS = 84                       # cells one trowel load covers (a full back is 72)
PRESS_S = 0.5                         # a fingertip pressing on the set tile this long beds it


class Dynamics:
    """Tile adhesive, from measured poses only (a disclosed stand-in for a paste).

    The tile's back is 72 cells of 12.5 mm. A cell is buttered where the trowel's notched edge passes
    over it with the blade 25-75 degrees to the back, the edge within 6.5 mm of the buttered layer's
    top, while the trowel still carries adhesive; a load covers 84 cells and the edge dug into the tub's adhesive
    reloads it. Buttered cells are drawn as combed ridges. A hidden slab inset 25 mm from the tile's
    edges stands for the layer's body: the trowel rides on it and a set tile bears on it, so a
    buttered tile sits level in the gap. The tile is bedded once a closed fingertip has pressed on its
    face for 0.5 s with the tile set in the gap (fingers shut, pointing down onto it).
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        self.cells = [None] * T.CELLS
        self.slab = None
        self.load_prim = None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for k in range(T.CELLS):
                if path.endswith(f"/tile/piece{T.CELL_PIECE0 + k}"):
                    self.cells[k] = prim
            if path.endswith(f"/tile/piece{T.BED_PIECE}"):
                self.slab = prim
            if path.endswith(f"/trowel/piece{T.LOAD_PIECE}"):
                self.load_prim = prim
        if any(c is None for c in self.cells) or self.slab is None or self.load_prim is None:
            raise RuntimeError("tile cells, bed slab or trowel load not found on the stage")

    def _vis(self, prim, on):
        from pxr import UsdGeom
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if on else img.MakeInvisible()

    def reset(self):
        for prim in self.cells:
            self._vis(prim, False)
        self._vis(self.slab, False)
        self._vis(self.load_prim, True)
        self.covered = [False] * T.CELLS
        self.load = LOAD_CELLS
        self.press = 0
        self.bedded = False
        self._stand_tile()

    def _stand_tile(self):
        """Stand the new tile on its long edge, leaning on the spare stack (a spawned body can only be yawed)."""
        import torch
        obj = self.base.scene["tile"]
        dev = self.base.device
        origin = self.base.scene.env_origins[0].cpu().numpy()
        c, r = T.lean_pose()
        pose = torch.tensor([[*(c + origin), *T.matrix_quat_xyzw(r)]], dtype=torch.float32, device=dev)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=dev))

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        tile = np.asarray(root_poses["tile"], float)
        trowel = np.asarray(root_poses["trowel"], float)
        if T.in_heap(trowel) and self.load < LOAD_CELLS:
            self.load = LOAD_CELLS
            self._vis(self.load_prim, True)
        if self.load > 0:
            for k in T.combed_cells(tile, trowel):
                if not self.covered[k] and self.load > 0:
                    self.covered[k] = True
                    self.load -= 1
                    self._vis(self.cells[k], True)
            if self.load == 0:
                self._vis(self.load_prim, False)
        off, yaw, lip, face_up = T.tile_set(tile)
        in_gap = face_up and off < 0.004 and yaw < 3.0 and lip < 0.002
        pressing = False
        if in_gap:
            r = T.quat_matrix(tile[3:7])
            from yam_arena import ik, station
            for side in ("left", "right"):
                tcp, rt = ik.tcp_pose(np.asarray(robot[side]["joints"], float), station.BASES[side])
                tip = np.asarray(tcp, float) + np.asarray(rt, float)[:, 2] * PAD_TIP
                q = r.T @ (tip - tile[:3])
                closed = float(np.mean(robot[side]["fingers"])) > -0.003 and float(np.asarray(rt)[2, 2]) < -0.5   # shut, pointing down
                if closed and abs(q[0]) < T.TILE[0] / 2 and abs(q[1]) < T.TILE[1] / 2 and -0.004 < -q[2] - T.TILE[2] / 2 < 0.004:
                    pressing = True
        self.press = self.press + 1 if pressing else 0
        if self.press >= PRESS_S * HZ:
            self.bedded = True
        n = sum(self.covered)
        a, b, _ = T.trowel_edge(trowel)
        SHARED["tile"] = tile
        SHARED["edge_mid"] = (a + b) / 2
        SHARED["left_tcp"] = np.asarray(robot["left"]["tcp"], float)[:3]
        return {"covered_cells": n, "cells": T.CELLS, "coverage": round(n / T.CELLS, 3), "load_cells": self.load,
                "bedded": bool(self.bedded), "pressing": bool(pressing)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo --------------------------------------------------------------------------------------------------
REST = {"left": (0.16, 0.24, T.TOP + 0.20), "right": (0.12, -0.22, T.TOP + 0.20)}   # the right one clear of the trowel on the tub
PIN = 0.061                           # the left hand takes the tile's long edge this far from its middle
GRIP_TILE = 0.0                       # fingers commanded shut on the 8 mm tile: a firm squeeze
GRIP_HANDLE = -0.010                  # fingers squeezing the 28 mm handle
PINCH = -0.005                       # TCP this far inside the tile's edge: pad tips 18 mm onto it


def _grip(jaw, app):
    jaw = np.asarray(jaw, float)
    app = np.asarray(app, float)
    return np.column_stack([jaw, np.cross(app, jaw), app])


SIDE_UP = _grip((0, 0, -1), (1, 0, 0))         # holding the tile by its near edge, back up
SIDE_DN = _grip((0, 0, 1), (1, 0, 0))          # the same hand rolled half a turn: the tile turned over, back down
SIDE_MID = _grip((0, -1, 0), (1, 0, 0))        # half way over
PRESS = simkit.rot_y(math.radians(60.0)) @ SIDE_DN   # closed fingertips pitched down onto the set tile
BUT_C = np.array([0.2375, -0.050, T.TOP + 0.180])   # the tile's centre while it is buttered
EDGE_Z = BUT_C[2] + T.TILE[2] / 2 + T.BUTTER + 0.0005


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


TROWEL_GRIP = rot_x(math.radians(45.0)) @ DOWN   # the blade tilted 45 degrees, its notched edge low on the right
RT0 = simkit.rot_z(T.TROWEL_YAW)
HL = np.array([-0.004, 0.0, T.HANDLE_Z])
EDGE_L = np.array([T.BLADE[0] / 2, 0.0, 0.0])
TROWEL_TCP0 = T.TROWEL_START + RT0 @ HL


def _comb_tcp(edge_mid):
    """TCP holding the trowel so its notched edge's middle is at ``edge_mid``."""
    rt = TROWEL_GRIP @ DOWN.T @ RT0
    return np.asarray(edge_mid, float) - rt @ EDGE_L + rt @ HL


LEAN_GRIP = _grip((-1, 0, 0), (0, 0, -1))      # straight down over the leaning tile's top edge, jaws across it


def _lean_grip():
    """TCP and grip taking the leaning tile by its top edge, fingers straight down: the tile swings upright
    between them as it is lifted, so it hangs plumb in the hand."""
    c, r = T.lean_pose()
    top = c + r @ np.array([-T.TILE[0] / 2, PIN, 0.0])
    d = 0.010                                      # the pads' middle, this far down the tile
    tcp = top + np.array([-d * math.sin(math.radians(T.LEAN_DEG)), 0.0, PINCH])
    return tcp, LEAN_GRIP, np.array([0.0, 0.0, 1.0])


R_HANG = simkit.rot_y(math.pi / 2)                 # the tile hanging plumb: local x down, back toward +x


def _held_tcp(centre, grip):
    """TCP holding the tile (taken by its edge at PIN) with its centre at ``centre``."""
    rt = grip @ LEAN_GRIP.T @ R_HANG                 # the tile's rotation in a hand turned to ``grip``
    return np.asarray(centre, float) + rt @ np.array([-(T.TILE[0] / 2 + PINCH), PIN, 0.0])


PRESS_X = 0.312


def _tip_tcp(tip):
    """TCP putting the closed fingertips at ``tip`` with the pitched press grip."""
    return np.asarray(tip, float) - PRESS[:, 2] * PAD_TIP


def W(sec, pos, finger, label, rot):
    return Waypoint(sec, tuple(map(float, pos)), finger, label, rot)


def demo():
    FO, FC = FINGER_OPEN, FINGER_CLOSED
    tcp_lean, g_lean, up = _lean_grip()
    but = _held_tcp(BUT_C, SIDE_UP)
    gap_tcp = _held_tcp(T.GAP, SIDE_DN)
    left = [
        W(1.5, REST["left"], FO, "look", DOWN),
        W(2.0, tcp_lean + 0.05 * up + np.array([-0.02, 0.0, 0.0]), -0.020, "over the leaning tile's top edge", g_lean),
        W(1.2, tcp_lean, -0.020, "fingers either side of its top edge", g_lean),
        W(0.6, tcp_lean, GRIP_TILE, "take it", g_lean),
        W(1.4, tcp_lean + np.array([0.0, 0.0, 0.05]), GRIP_TILE, "lift it", g_lean),
        W(1.0, tcp_lean + np.array([-0.02, 0.0, 0.12]), GRIP_TILE, "lift it clear of the stack", g_lean),
        W(2.6, but + np.array([-0.01, 0.0, -0.01]), GRIP_TILE, "turn it back up", SIDE_UP),
        W(1.0, but, GRIP_TILE, "hold it out for buttering", SIDE_UP),
    ]
    t_ready = sum(w.duration for w in left)
    take = TROWEL_TCP0
    right = [
        W(max(1.5, t_ready - 6.0), REST["right"], FO, "look", DOWN),
        W(1.5, take + np.array([0.0, 0.0, 0.03]), FO, "over the trowel's handle", DOWN),
        W(1.0, take, FO, "round the handle", DOWN),
        W(0.6, take, GRIP_HANDLE, "take the trowel", DOWN),
        W(0.8, take + np.array([0.0, 0.0, 0.03]), GRIP_HANDLE, "lift it off the tub", DOWN),
    ]
    t_r = sum(w.duration for w in right)
    y0, y1 = BUT_C[1] - T.TILE[1] / 2 - 0.012, BUT_C[1] + 0.016
    x1 = BUT_C[0]
    right += [
        W(max(1.5, t_ready + 0.3 - t_r), _comb_tcp((x1, y0, EDGE_Z + 0.015)), GRIP_HANDLE, "blade over the tile's right end", TROWEL_GRIP),
        W(0.8, _comb_tcp((x1, y0, EDGE_Z)), GRIP_HANDLE, "notched edge down on it", TROWEL_GRIP),
    ]
    for k in range(1, 5):
        right.append(W(0.8, _comb_tcp((x1, y0 + (y1 - y0) * k / 4, EDGE_Z)), GRIP_HANDLE, "comb the adhesive on", TROWEL_GRIP))
    x2 = BUT_C[0] - T.TILE[0] / 2 + 0.024 + T.BLADE[1] / 2      # the blade's near end clear of the fingertips
    y2, y3 = BUT_C[1] - 0.010, BUT_C[1] + T.TILE[1] / 2 + 0.008
    right += [
        W(0.5, _comb_tcp((x1, y1, EDGE_Z + 0.02)), GRIP_HANDLE, "lift the edge", TROWEL_GRIP),
        W(1.0, _comb_tcp((x2, y2, EDGE_Z + 0.02)), GRIP_HANDLE, "over the last strip, clear of the fingers", TROWEL_GRIP),
        W(0.6, _comb_tcp((x2, y2, EDGE_Z)), GRIP_HANDLE, "notched edge down on it", TROWEL_GRIP),
        W(0.8, _comb_tcp((x2, (y2 + y3) / 2, EDGE_Z)), GRIP_HANDLE, "comb the adhesive on", TROWEL_GRIP),
        W(0.8, _comb_tcp((x2, y3, EDGE_Z)), GRIP_HANDLE, "comb the adhesive on", TROWEL_GRIP),
        W(0.6, _comb_tcp((x2, y3, EDGE_Z + 0.04)), GRIP_HANDLE, "lift the trowel off", TROWEL_GRIP),
        W(2.0, take + np.array([0.0, 0.0, 0.03]), GRIP_HANDLE, "back over the tub", DOWN),
        W(1.0, take + np.array([0.0, 0.0, 0.003]), GRIP_HANDLE, "lay it across the rim", DOWN),
        W(0.6, take + np.array([0.0, 0.0, 0.003]), FO, "let go", DOWN),
        W(0.8, take + np.array([0.0, 0.0, 0.03]), FO, "lift off", DOWN),
        W(1.5, REST["right"], FO, "stand clear", DOWN),
    ]
    t_done = sum(w.duration for w in right[:-4])          # the trowel is off the tile and on its way back
    hold = max(0.5, t_done - t_ready)
    left[-1] = W(1.0 + hold, but, GRIP_TILE, "hold it out for buttering", SIDE_UP)
    gz = T.GAP[2]
    left += [
        W(1.2, _held_tcp(BUT_C + np.array([0.0, 0.0, 0.05]), SIDE_UP), GRIP_TILE, "lift it clear", SIDE_UP),
        W(1.2, np.array([0.22, 0.04, T.TOP + 0.25]), GRIP_TILE, "turn it over", SIDE_MID),
        W(1.2, np.array([0.24, 0.045, T.TOP + 0.25]), GRIP_TILE, "turn it over", SIDE_DN),
        W(1.8, gap_tcp + np.array([0.0, 0.0, 0.03]), GRIP_TILE, "over the gap", SIDE_DN),
        W(2.0, gap_tcp + np.array([0.0, 0.0, 0.001]), GRIP_TILE, "lower it into the gap", SIDE_DN),
        W(0.6, gap_tcp + np.array([0.0, 0.0, 0.001]), -0.010, "let it go", SIDE_DN),
        W(0.8, gap_tcp + np.array([-0.035, 0.0, 0.004]), -0.010, "fingers out", SIDE_DN),
        W(0.6, gap_tcp + np.array([-0.035, 0.0, 0.004]), FC, "close the hand", SIDE_DN),
        W(1.2, _tip_tcp((PRESS_X, T.GAP[1], T.FACE_Z + 0.02)), FC, "over the new tile", PRESS),
        W(0.8, _tip_tcp((PRESS_X, T.GAP[1], T.FACE_Z - 0.0015)), FC, "press it down", PRESS),
        W(1.2, _tip_tcp((PRESS_X, T.GAP[1], T.FACE_Z - 0.0015)), FC, "press it down", PRESS),
        W(0.8, _tip_tcp((PRESS_X, T.GAP[1], T.FACE_Z + 0.04)), FC, "lift off", PRESS),
        W(1.5, np.array([0.19, 0.12, T.TOP + 0.21]), FO, "stand clear", SIDE_DN),
    ]
    tl, tr = sum(w.duration for w in left), sum(w.duration for w in right)
    if tl < tr:
        left.append(W(tr - tl + 1.5, left[-1].pos, FO, "done", left[-1].yaw))
    else:
        right.append(W(tl - tr + 1.5, REST["right"], FO, "done", DOWN))
    return {"left": left, "right": right}


SET_LABELS = ("over the gap", "lower it into the gap")
COMB_LABELS = ("notched edge down on it", "comb the adhesive on")


class DemoController:
    """The scripted plans, steered by measured state as a scripted demo may: the trowel's notched edge
    is kept on the buttered layer's top over the held tile, and the held tile is lined up over the gap."""

    def __init__(self, plans):
        self.plans = plans
        self.ez = 0.0
        self.gap = np.zeros(3)
        self.hz = 0.0                                   # left hand: holding the tile up against the trowel
        self.yaw = 0.0                                  # left hand: squaring the held tile to the gap

    def action(self, t):
        from yam_arena import ik
        out = []
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, label = plan.target(t)
            pos = np.array(pos, float)
            if side == "right" and label in COMB_LABELS and "edge_mid" in SHARED:
                tile = np.asarray(SHARED["tile"], float)
                want = tile[2] + T.TILE[2] / 2 + T.BUTTER + 0.0005
                self.ez = float(np.clip(self.ez + 0.2 * (want - SHARED["edge_mid"][2]), -0.004, 0.012))
            if side == "right" and label in COMB_LABELS + ("lift the edge", "over the last strip, clear of the fingers"):
                pos[2] += self.ez
            if side == "left" and label == "hold it out for buttering" and "tile" in SHARED:
                tile = np.asarray(SHARED["tile"], float)
                self.hz = float(np.clip(self.hz + 0.15 * (BUT_C[2] - tile[2]), -0.01, 0.04))
            if side == "left" and label in ("hold it out for buttering", "lift it clear"):
                pos[2] += self.hz
            if side == "left" and label in SET_LABELS and "tile" in SHARED:
                tile = np.asarray(SHARED["tile"], float)
                rt = T.quat_matrix(tile[3:7])
                ang = math.atan2(-rt[0, 1], -rt[1, 1])        # the tile's long axis against the gap's (turned over: -y)
                self.yaw = float(np.clip(self.yaw + 0.15 * ang, -0.35, 0.35))
                err = np.array([T.GAP[0] - tile[0], T.GAP[1] - tile[1], 0.0])
                self.gap = np.clip(self.gap + 0.2 * err, -0.03, 0.03)
            if side == "left" and label in SET_LABELS + ("let it go", "fingers out"):
                pos += self.gap
                rot = simkit.rot_z(self.yaw) @ rot
            q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), plan.q, base_pos=plan.base_pos, iters=60)
            plan.q = q
            grip = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            out.append(np.r_[q, grip])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
