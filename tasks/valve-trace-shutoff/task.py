"""Plant maintenance: "Stop the coolant spraying on the control box."

A coolant header with three ball valves; their branch lines cross over and under one another on the
way to the machine trough, so each line ends on a different side from its valve. One line's joint is
cracked and sprays onto an electrical control box. The model has to trace that line back to its own
valve and shut only that one -- the other two lines feed running machines.
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
import valves as V  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "valve-trace-shutoff"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 600
OBJECT_NAMES = (*V.NAMES, *V.DROP_NAMES)

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
    specs = (V.piping_object(), *(V.lever_object(k) for k in range(3)), *V.drop_objects())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_valve_trace_shutoff", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """The coolant, from the levers' measured angles. While a valve passes coolant its nozzle drips into
    the trough (one drop every 3 control steps); while valve A passes coolant its cracked joint sprays a
    fan of drops (two a step) toward the control panel; spray that lands on or at the foot of the panel
    shows as a wet face and a growing puddle (drawn). Drops are released with a fixed velocity and then
    fly as ordinary rigid spheres; when one lands or stops it is judged (did it end on the control box?)
    and parked under the bench for reuse. A disclosed stand-in for pipe flow; no drop is moved after release."""

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        self.rng = np.random.default_rng(0)
        V.piping_object()                       # fills V.TOGGLES
        stage = omni.usd.get_context().get_stage()
        self.drawn = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for key, idx in V.TOGGLES.items():
                if path.endswith(f"/piping/piece{idx}"):
                    self.drawn[key] = prim
        if set(self.drawn) != set(V.TOGGLES):
            raise RuntimeError("drawn coolant pieces not found")
        self.shown = {}

    def _show(self, key, on):
        if self.shown.get(key) == on:
            return
        from pxr import UsdGeom
        img = UsdGeom.Imageable(self.drawn[key])
        img.MakeVisible() if on else img.MakeInvisible()
        self.shown[key] = on

    def _write(self, name, pos, vel):
        import torch
        obj = self.base.scene[name]
        dev = self.base.device
        origin = self.base.scene.env_origins[0].cpu().numpy()
        pose = torch.tensor([[*(np.asarray(pos) + origin), 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=dev)
        v = torch.tensor([[*vel, 0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
        obj.write_root_pose_to_sim_index(root_pose=pose)
        obj.write_root_velocity_to_sim_index(root_velocity=v)

    def reset(self):
        self.free = list(V.DROP_NAMES)
        self.flight = {}            # name -> (emit step, from the leak?)
        self.box_hits = 0
        self.deg = [0.0, 0.0, 0.0]
        self.shown = {}
        for key in self.drawn:
            self._show(key, key.startswith("stream"))
        for i, n in enumerate(V.DROP_NAMES):
            self._write(n, V.parking(i), (0.0, 0.0, 0.0))

    def _emit(self, step, pos, vel, leak):
        if not self.free:
            return
        n = self.free.pop(0)
        self._write(n, pos, vel)
        self.flight[n] = (step, leak)

    def before_step(self, step, action):
        if step <= 0:
            return
        for k in range(3):
            if V.flowing(self.deg[k]) and step % 3 == k:
                self._emit(step, (V.OUT_X, V.OUT_Y[k], V.LINE_Z[k] - 0.030), (0.0, 0.0, -0.3), False)
        if V.flowing(self.deg[V.LEAK_LINE]):
            # the crack sprays a fan: two drops a step, spread across and up, toward the panel
            lx, ly = V.LEAK_AT
            for _ in range(2):
                j = self.rng.uniform(-1.0, 1.0, 3)
                self._emit(step, (lx + 0.004 * j[0], ly - V.PIPE_R * 1.6, V.LINE_Z[V.LEAK_LINE] + 0.003 * j[1]),
                           (0.35 * j[0], -0.95 + 0.15 * j[2], 0.55 + 0.25 * j[1]), True)

    def after_step(self, step, objects, root_poses, robot):
        self.deg = [V.lever_deg(np.asarray(root_poses[n])[3:7]) for n in V.NAMES]
        for k in range(3):
            self._show(f"stream{k}", V.flowing(self.deg[k]))
        self._show("sheen", self.box_hits > 0)
        for j, n in enumerate(V.PUDDLE_AT):
            self._show(f"puddle{j}", self.box_hits >= n)
        done = []
        for name, (t0, leak) in self.flight.items():
            pos = np.asarray(root_poses[name], float)[:3]
            speed = float(np.linalg.norm(objects[name].lin_vel))
            age = step - t0
            if age > 3 and (pos[2] < V.TOP + 0.006 or speed < 0.05 or age > 30):
                done.append(name)
                if leak and V.on_box(pos):
                    self.box_hits += 1
        for name in done:
            del self.flight[name]
            self._write(name, V.parking(V.DROP_NAMES.index(name)), (0.0, 0.0, 0.0))
            self.free.append(name)
        return {"lever_deg": [round(d, 1) for d in self.deg], "flowing": [V.flowing(d) for d in self.deg],
                "leak_spraying": bool(V.flowing(self.deg[V.LEAK_LINE])), "box_hits": int(self.box_hits)}


def make_dynamics(env):
    return Dynamics(env)


OVERVIEW_EYE = (0.34, 0.03, 1.70)      # straight above the bench, image up = away from the arms


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

REST = {"left": (0.12, 0.22, V.TOP + 0.20), "right": (0.12, -0.22, V.TOP + 0.20)}
PIN_FINGER = -(2 * V.PIN[0] - 0.008) / 2


def _pin(seconds, k, deg, finger, label, lift=0.0):
    """Down grip round the lever's pin; the pin turns in the jaws, so the wrist keeps its yaw."""
    p = V.grip_point(k, deg)
    return Waypoint(seconds, (p[0], p[1], p[2] + PAD_TIP + lift), finger, label, DOWN)


def demo():
    k = V.LEAK_LINE
    left = [Waypoint(3.0, REST["left"], FINGER_OPEN, "trace the spraying line back to its valve", DOWN),
            _pin(2.0, k, 0.0, FINGER_OPEN, "over valve A's lever", lift=0.05),
            _pin(1.2, k, 0.0, FINGER_OPEN, "down round its pin"),
            _pin(0.8, k, 0.0, PIN_FINGER, "take it")]
    for j in range(1, 11):
        left.append(_pin(0.45, k, (V.CLOSED_DEG + 5.0) * j / 10, PIN_FINGER, "swing it a quarter turn shut, onto its stop"))
    left += [_pin(1.5, k, V.CLOSED_DEG + 5.0, PIN_FINGER, "hold it against the stop"),
             _pin(0.6, k, V.CLOSED_DEG, FINGER_OPEN, "let go"),
             _pin(0.8, k, V.CLOSED_DEG, FINGER_OPEN, "lift off", lift=0.05),
             Waypoint(2.0, REST["left"], FINGER_OPEN, "stand clear", DOWN),
             Waypoint(5.0, REST["left"], FINGER_OPEN, "the other machines still running", DOWN)]
    return {"left": left}
