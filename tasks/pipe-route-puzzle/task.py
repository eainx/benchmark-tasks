"""Plant maintenance: "Get the coolant through to the machine."

A panel of twelve rotary pipe fittings between the coolant supply (left) and the line out to the
machine (right). Coolant fills every fitting it can reach; the model has to read the panel, work
out which fittings to turn and by how much so that one tight line joins supply to outlet, and turn
them by their hub knobs.
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
import pipes as P  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "pipe-route-puzzle"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = P.NAMES

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
    specs = (P.board_object(), *(P.fitting_object(rc) for rc in P.CELLS))
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_pipe_route_puzzle", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


OVERVIEW_EYE = (0.30, 0.0, 1.62)      # straight above the bench, image up = away from the arms


def top_down_overview(embodiment):
    """Point the recording-only overview camera straight down on the bench (the model never sees it)."""
    from yam_arena import ik
    cam = embodiment.camera_config.overview_cam
    forward, left, up = np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
    cam.offset.pos = tuple(float(v) for v in OVERVIEW_EYE)
    cam.offset.rot = tuple(float(v) for v in ik.matrix_to_quat_xyzw(np.column_stack([forward, left, up])))


class Dynamics:
    """Coolant, from the fittings' measured angles only: a fitting left within 15 degrees of a quarter
    turn joins its neighbours where their ends meet; coolant fills everything connected to the
    supply and is drawn by showing each wet fitting's blue pipe. Nothing is moved."""

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        paths = {}
        for prim in stage.Traverse():
            paths[str(prim.GetPath())] = prim
        self.wet = {}
        for rc in P.CELLS:
            self.wet[rc] = [p for k in P.wet_piece_range(rc) for path, p in paths.items()
                            if path.endswith(f"/{P.name(rc)}/piece{k}")]
            if len(self.wet[rc]) != len(P.wet_piece_range(rc)):
                raise RuntimeError(f"wet pipe of {P.name(rc)} not found")
        self.outlet = [p for path, p in paths.items() if path.endswith(f"/panel/piece{P.OUTLET_WET_PIECE}")]
        if len(self.outlet) != 1:
            raise RuntimeError("outlet pipe not found")
        self.shown = {}

    def _show(self, key, prims, on):
        if self.shown.get(key) == on:
            return
        from pxr import UsdGeom
        for prim in prims:
            img = UsdGeom.Imageable(prim)
            img.MakeVisible() if on else img.MakeInvisible()
        self.shown[key] = on

    def reset(self):
        self.shown = {}
        for rc in P.CELLS:
            self._show(rc, self.wet[rc], False)
        self._show("outlet", self.outlet, False)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        turns = {rc: P.quarter_turns(P.yaw_of(np.asarray(root_poses[P.name(rc)])[3:7])) for rc in P.CELLS}
        wet, out, leaks = P.flow(turns)
        for rc in P.CELLS:
            self._show(rc, self.wet[rc], rc in wet)
        self._show("outlet", self.outlet, out)
        return {"turns": {P.name(rc): t for rc, t in turns.items()}, "wet": sorted(P.name(rc) for rc in wet),
                "through": bool(out), "open_ends": int(leaks)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

KNOB_FINGER = -(P.KNOB[0] - 0.008) / 2
APPROACH_FINGER = -0.025                  # half open: the jaws stay clear of the neighbouring knobs
GRIP_Z = P.DISC_Z + P.KNOB_Z + P.KNOB[2] / 2 - 0.012     # pad tips 12 mm below the handle's top

# the turns the demo makes: cell -> quarter turns (+ is counter-clockwise seen from above)
TURNS = {"left": [((1, 0), 1), ((1, 1), 2), ((2, 1), 1)],
         "right": [((2, 3), 1), ((1, 3), -1)]}
REST = {"left": (0.12, 0.22, P.TOP + 0.20), "right": (0.12, -0.22, P.TOP + 0.20)}   # back by the bases, off the panel


def _knob(seconds, rc, yaw, finger, label, lift=0.0):
    x, y = P.cell_xy(rc)
    return Waypoint(seconds, (x, y, GRIP_Z + PAD_TIP + lift), finger, label, simkit.rot_z(yaw) @ DOWN)


def wrist_start(turns_now, quarters):
    """Wrist yaw that puts the jaws across the T-handle (which lies along the fitting's local y, so the
    jaws close along the fitting's local x... i.e. a wrist yaw of the fitting's yaw + 90 deg, modulo a half
    turn), chosen so the turn stays within +-90 deg of the wrist's straight-down pose."""
    a = math.radians(90 * turns_now + 90)
    a = (a + math.pi / 2) % math.pi - math.pi / 2           # into [-90, 90)
    if quarters > 0 and a > 0:
        a -= math.pi
    if quarters < 0 and a < 0:
        a += math.pi
    if quarters < 0 and abs(a + math.pi / 2) < 1e-6:
        a = math.pi / 2
    return a


def turn(rc, quarters, turns_now):
    """Take the T-handle, turn the wrist a quarter turn at a time, re-taking it between quarters."""
    out = []
    step = 1 if quarters > 0 else -1
    for _ in range(abs(quarters)):
        a0 = wrist_start(turns_now, step)
        a1 = a0 + step * math.pi / 2
        out += _turn_once(rc, a0, a1, step)
        turns_now = (turns_now + step) % 4
    return out


def _turn_once(rc, a0, a1, step):
    return [
        _knob(2.0, rc, a0, APPROACH_FINGER, f"over fitting {rc}", lift=0.05),
        _knob(1.2, rc, a0, APPROACH_FINGER, "down round its knob"),
        _knob(0.8, rc, a0, KNOB_FINGER, "take the knob"),
        _knob(1.6, rc, a1, KNOB_FINGER, f"turn it {90 * step:+d} degrees"),
        _knob(0.6, rc, a1, APPROACH_FINGER, "let go"),
        _knob(0.8, rc, a1, APPROACH_FINGER, "lift off", lift=0.05),
    ]


def demo():
    plan = {}
    for side in ("left", "right"):
        wps = [Waypoint(2.0, REST[side], FINGER_OPEN, "look at the panel", DOWN)]
        for rc, q in TURNS[side]:
            wps += turn(rc, q, P.LAYOUT[rc][1])
        wps.append(Waypoint(2.0, REST[side], FINGER_OPEN, "stand clear", DOWN))
        plan[side] = wps
    total = max(sum(w.duration for w in v) for v in plan.values())
    for side, wps in plan.items():
        wps.append(Waypoint(total - sum(w.duration for w in wps) + 2.5, REST[side], FINGER_OPEN, "watch the flow", DOWN))
    return plan
