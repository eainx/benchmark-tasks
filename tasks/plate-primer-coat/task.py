"""Fabrication shop: "Give the steel plate a coat of primer."

A bare steel plate, a tray of red-oxide primer and a block brush. Nothing names the brush or the
tray: the brush has to be dipped before it puts anything on the plate, one dip covers only about
half of it, and the coat goes on where the loaded bristles pass over the plate.
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
import paint as P  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "plate-primer-coat"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = ("brush", *P.DROP_NAMES)

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    top_down_overview(embodiment)

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (P.plate_object(), P.tray_object(), P.brush_object(), *P.drop_objects())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_plate_primer_coat", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """The primer, from the brush's measured pose only. Bristle tips below the primer's surface in the
    tray load the brush with ``LOAD`` patches of coat; loaded tips on the plate's top put coat on each
    10 mm patch they pass over, one unit of load per new patch. A painted patch is shown (its drawn
    tile made visible) and stays painted. A freshly dipped brush is dripping until its tips are drawn
    across the tray's far rim: carried un-wiped outside the tray it lets fall a drop of primer (a small
    rigid sphere released at the tips) every ``DRIP_EVERY`` steps. Drops landing outside the plate and
    tray are counted as drips. Nothing else is moved: a disclosed stand-in for wet paint."""

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        self.tiles = [None] * P.TILES
        self.wet = None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for k in range(P.TILES):
                if path.endswith(f"/plate/piece{k}"):
                    self.tiles[k] = prim
            if path.endswith(f"/brush/piece{P.WET_PIECE}"):
                self.wet = prim
        if any(t is None for t in self.tiles) or self.wet is None:
            raise RuntimeError("primer tiles or brush tips not found on the stage")

    def _show(self, prim, on):
        from pxr import UsdGeom
        img = UsdGeom.Imageable(prim)
        img.MakeVisible() if on else img.MakeInvisible()

    def _write(self, name, pos):
        import torch
        obj = self.base.scene[name]
        origin = self.base.scene.env_origins[0].cpu().numpy()
        pose = torch.tensor([[*(np.asarray(pos) + origin), 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=self.base.device)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=self.base.device))

    def reset(self):
        self.dripping = False
        self.free = list(P.DROP_NAMES)
        self.falling = {}
        self.drips = 0
        self.smear_steps = 0
        self.landed = []
        for i, n in enumerate(P.DROP_NAMES):
            self._write(n, P.parking(i))
        self.load = 0
        self.dips = 0
        self.dipped = False
        self.painted = set()
        for t in self.tiles:
            self._show(t, False)
        self._show(self.wet, False)

    def before_step(self, step, action):
        pass

    def _drip(self, step, pose, objects, root_poses):
        tips, _, _ = P.bristle_tips(pose)
        if P.in_primer(pose):
            self.dripping = True
        elif P.on_rim(pose):
            self.dripping = False
        elif self.dripping and not P.over_tray(tips) and step % P.DRIP_EVERY == 0 and self.free:
            n = self.free.pop(0)
            self._write(n, tips - np.array([0.0, 0.0, P.DROP_R + 0.002]))
            self.falling[n] = step
        for n, t0 in list(self.falling.items()):
            p = np.asarray(root_poses[n], float)[:3]
            if step - t0 > 3 and float(np.linalg.norm(objects[n].lin_vel)) < 0.02:
                del self.falling[n]
                if not (P.on_plate(p) or P.over_tray(p)):
                    self.drips += 1        # stays where it fell, on the bench
                self.landed.append(n)

    def after_step(self, step, objects, root_poses, robot):
        pose = np.asarray(root_poses["brush"], float)
        self._drip(step, pose, objects, root_poses)
        if self.load > 0 and P.tips_on_bench(pose):
            self.smear_steps += 1                 # a loaded brush wiped on the bench
        if P.in_primer(pose):
            if not self.dipped:
                self.dips += 1
            self.dipped = True
            self.load = P.LOAD
        else:
            self.dipped = False
        if self.load > 0:
            for k in P.tiles_under(pose):
                if k not in self.painted and self.load > 0:
                    self.painted.add(k)
                    self.load -= 1
                    self._show(self.tiles[k], True)
        self._show(self.wet, self.load > 0)
        return {"painted": sorted(self.painted), "coverage": len(self.painted) / P.TILES, "load": int(self.load),
                "dips": int(self.dips), "dripping": bool(self.dripping), "drips": int(self.drips),
                "bench_smear_steps": int(self.smear_steps)}


def make_dynamics(env):
    return Dynamics(env)


OVERVIEW_EYE = (0.29, 0.08, 1.58)      # straight above the bench, image up = away from the arms


def top_down_overview(embodiment):
    """Point the recording-only overview camera straight down on the bench (the model never sees it)."""
    from yam_arena import ik
    cam = embodiment.camera_config.overview_cam
    forward, left, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    cam.offset.pos = tuple(float(v) for v in OVERVIEW_EYE)
    cam.offset.rot = tuple(float(v) for v in ik.matrix_to_quat_xyzw(np.column_stack([forward, left, up])))


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

HANDLE_FINGER = -(2 * P.HANDLE_R - 0.009) / 2
FAR_X = P.PLATE_XY[0] + P.PLATE[0] / 2 + P.BRISTLE_BLOCK[0] / 2 + 0.003   # bristles just past the far edge
NEAR_X = P.PLATE_XY[0] - P.PLATE[0] / 2 + P.BRISTLE_BLOCK[0] / 2 - 0.004  # last patches covered
STROKE_Z = P.PLATE_TOP + 0.0006
DIP_Z = P.TOP + P.TRAY_WALL + 0.003


def _hand(seconds, tips_xyz, finger, label):
    """Down grip on the handle, jaws across it, given where the bristle tips' centre should be."""
    x, y, z = tips_xyz
    tip = (x + P.GRIP_X, y, z + P.HANDLE_Z - 0.004)
    return Waypoint(seconds, (tip[0], tip[1], tip[2] + PAD_TIP), finger, label, DOWN)


def demo():
    bx, by = P.BRUSH_XY
    tx, ty = P.TRAY_XY
    f = HANDLE_FINGER
    left = [
        Waypoint(2.0, (0.22, 0.22, P.TOP + 0.18), FINGER_OPEN, "look at the plate", DOWN),
        _hand(2.0, (bx, by, P.TOP + 0.06), FINGER_OPEN, "over the brush"),
        _hand(1.5, (bx, by, P.TOP), FINGER_OPEN, "down round the handle"),
        _hand(1.0, (bx, by, P.TOP), f, "take the brush"),
        _hand(1.5, (bx, by, P.TOP + 0.06), f, "pick it up"),
    ]
    for k in range(2):
        left += [
            _hand(1.8 if k == 0 else 1.5, (tx, ty, P.TOP + 0.05), f, "over the primer"),
            _hand(1.2, (tx, ty, DIP_Z), f, "dip the brush"),
            _hand(0.8, (tx, ty, DIP_Z), f, "let it soak up"),
            _hand(1.0, (tx, ty, P.TOP + 0.05), f, "lift it out"),
            _hand(1.0, (P.WIPE_WALL_X - 0.012, ty, P.RIM_TOP + 0.012), f, "over the far rim"),
            _hand(0.8, (P.WIPE_WALL_X - 0.004, ty, P.RIM_TOP - 0.001), f, "tips on the rim"),
            _hand(1.0, (P.WIPE_WALL_X + 0.006, ty, P.RIM_TOP - 0.001), f, "wipe the excess off on it"),
            _hand(0.6, (P.WIPE_WALL_X + 0.006, ty, P.RIM_TOP + 0.03), f, "lift off the rim"),
            _hand(1.8, (FAR_X, 0.0, STROKE_Z + 0.03), f, "over the far edge of the plate"),
            _hand(1.0, (FAR_X, 0.0, STROKE_Z), f, "bristles down"),
            _hand(3.5, (NEAR_X, 0.0, STROKE_Z), f, "draw the brush along the plate"),
            _hand(0.8, (NEAR_X, 0.0, STROKE_Z + 0.03), f, "lift"),
        ]
    left += [
        _hand(2.2, (tx, ty, P.TOP + 0.05), f, "back over the tray"),
        _hand(1.5, (tx, ty, P.TOP + P.TRAY_WALL + 0.002), f, "rest the wet brush in the tray"),
        _hand(1.0, (tx, ty, P.TOP + P.TRAY_WALL + 0.002), FINGER_OPEN, "let go"),
        _hand(1.5, (tx, ty, P.TOP + 0.08), FINGER_OPEN, "clear"),
        _hand(2.0, (tx, ty, P.TOP + 0.08), FINGER_OPEN, "stand clear"),
    ]
    return {"left": left}
