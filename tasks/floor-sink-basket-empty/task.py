"""Commercial kitchen close-down: "Empty the floor sink's basket into the bus tub."

A 12 x 12 x 6 in floor sink sits flush in a quarry-tile floor (a tiled platform on the bench). A shallow perforated
basket hangs on the sink's seat lugs 115 mm down, its wire bail folded flat to the right; there is no rim to take, so it
comes out only by the bail. The right hand grips the bail's sleeve, swings the bail up and lifts the basket out level,
then holds it over the bus tub; the left hand slides its flat fingers under the basket's left bottom edge from behind and
lifts it, turning the basket over on its bail until the scraps slide out into the tub.
(drain.py still holds a hinged-grate variant, WITH_GRATE = False: its opening stalled and it is not in the scene.)
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
import drain as D  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_OPEN, FINGER_CLOSED  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "floor-sink-basket-empty"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MODE = "demo"           # "scene": arms rest (static render); "demo": full demo
MAX_STEPS = {"scene": 80, "demo": 1200}[MODE]
OBJECT_NAMES = (*(("grate",) if D.WITH_GRATE else ()), "bucket", "bail", *(d[0] for d in D.DEBRIS), "tub")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    objects += [yscene.task_object(s) for s in D.scene_objects()]
    cfg = IsaacLabArenaEnvironment(name="yam_floor_sink_basket_empty", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """Nothing is applied: every part moves only by gravity, its joints (the grate's hinge, the bail's pivots) and
    contacts. Reports the grate's angle and the basket's height for the log."""

    def __init__(self, env):
        self.base = env.unwrapped

    def reset(self):
        pass

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        from scoring import tilt_deg
        b = np.asarray(root_poses["bucket"], float)
        g = root_poses.get("grate") if hasattr(root_poses, "get") else None
        return {"grate_angle_deg": round(tilt_deg(np.asarray(g, float)), 1) if g is not None else None, "basket_xyz": [round(float(v), 4) for v in b[:3]],
                "basket_tilt_deg": round(tilt_deg(b), 1)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


REST = {"left": (0.15, 0.20, D.TOP + 0.30), "right": (0.15, -0.20, D.TOP + 0.30)}
REST_P = 0.9                    # rest pose pitched forward (a straight-down hand cannot rise this high)


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def demo():
    if MODE == "scene":
        return {side: [_w(1.5, REST[side], FINGER_OPEN, "look", _ry(-REST_P) @ DOWN), _w(MAX_STEPS / HZ, REST[side], FINGER_OPEN, "hold", _ry(-REST_P) @ DOWN)]
                for side in ("left", "right")}
    return full_demo()


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


pit = lambda p: _ry(-p) @ DOWN                         # pitched forward p (pi/2: pointing +x), jaws closing along y
rol = lambda p: _ry(-p) @ _rz(np.pi / 2) @ DOWN        # pitched forward p, jaws closing in the x-z plane (a bar along y)
SIDE = _rx(np.pi / 2) @ DOWN                           # right hand pointing +y (along the grate's near edge), jaws along z
side = lambda phi: _ry(phi - np.pi / 2) @ SIDE         # ... rolled so the jaws close across the grate's flange at tilt phi
HALF = FINGER_OPEN * 0.6
rolL = lambda p: _ry(-p) @ _rz(-np.pi / 2) @ DOWN     # as rol, wrist turned the other way (left arm's reach)


class _Arm:
    def __init__(self):
        self.w, self.t, self.last = [], 0.0, None

    def go(self, dt, p, f, label, rot):
        self.w.append(_w(dt, p, f, label, rot))
        self.t += dt
        self.last = (np.asarray(p, float), f, rot)

    def hold(self, until, label="wait"):
        if until > self.t + 1e-6:
            p, f, rot = self.last
            self.go(until - self.t, p, f, label, rot)


def _sync(*arms):
    t = max(a.t for a in arms)
    for a in arms:
        a.hold(t)


# grate on the sink's far half: it turns about the top of the recess's far wall
PIV = np.array([D.G_X[1] + 0.002, 0.0, D.FLOOR])
LIP_REL = np.array([D.G_X[0] + 0.0014 - PIV[0], 0.0, -0.011])     # mid-flange of the near edge, 1 mm above its bottom
Y_LIP = D.SY - 0.090


def lip(phi):
    """TCP on the near edge's turned-down flange with the grate turned phi (rad) up about its far edge."""
    p = PIV + _ry(phi) @ LIP_REL
    return np.array([p[0], Y_LIP, p[2]])


def normal(phi):
    return _ry(phi) @ np.array([0.0, 0.0, 1.0])


def full_demo():
    L, R = _Arm(), _Arm()
    up = np.array([0.0, 0.0, 1.0])
    L.go(1.5, REST["left"], FINGER_OPEN, "look", pit(REST_P))
    R.go(1.5, REST["right"], FINGER_OPEN, "look", pit(REST_P))
    LEAD = 8.0                                              # deg
    PG = 0.30                                               # grip the strip with the hand pitched 17 deg forward
    if D.WITH_GRATE:
        # ---- 1. right: jaws round the strip at the grate's free edge (one finger down the hand-hold slot), swing it open
        R.go(2.0, strip(0) + 0.06 * up, HALF, "over the hand-hold slot", rol(PG))
        R.go(1.5, strip(0), HALF, "a finger down the slot, one past the edge", rol(PG))
        R.go(0.8, strip(0), FINGER_CLOSED, "grip the strip", rol(PG))
        # the arm's position servo sags under the grate's weight, so the hand leads the grate along its arc a little
        for deg, dt in ((12, 1.5), (25, 1.5), (40, 1.8), (55, 1.8), (70, 1.8), (82, 1.8), (93, 1.6)):
            R.go(dt, strip(np.radians(deg + LEAD)), FINGER_CLOSED, f"swing the grate open {deg} deg", rol(min(PG + np.radians(deg), 1.3)))
        R.go(0.8, strip(np.radians(93)), FINGER_OPEN, "let it rest on its stop", rol(1.3))
        R.go(1.2, strip(np.radians(93)) - 0.05 * _ry(-1.3) @ np.array([0, 0, -1.0]), FINGER_OPEN, "back off", rol(1.3))
        R.go(1.5, np.array([0.14, -0.10, 1.08]), FINGER_OPEN, "hand up and back", rol(1.3))
        R.go(1.5, np.array([0.18, -0.15, 1.06]), FINGER_OPEN, "hand round", pit(1.3))
        R.go(2.0, REST["right"], FINGER_OPEN, "rest", pit(REST_P))
    # ---- 2. right: bail up, basket out, over the bus tub
    R.go(2.0, bar(np.radians(D.BAIL_DOWN_DEG)) + 0.07 * up, HALF, "over the folded bail", DOWN)
    R.go(1.5, bar(np.radians(D.BAIL_DOWN_DEG)), HALF, "jaws round the bail's sleeve", DOWN)
    R.go(0.8, bar(np.radians(D.BAIL_DOWN_DEG)), FINGER_CLOSED, "grip the bail", DOWN)
    R.go(1.2, bar(np.radians(45)), FINGER_CLOSED, "swing the bail up", DOWN)
    R.go(1.2, bar(0.0), FINGER_CLOSED, "bail upright", pit(0.3))
    lift = bar(0.0) + np.array([0, 0, 0.215])          # 90 mm over what is needed: the arm sags under the basket
    R.go(4.5, lift, FINGER_CLOSED, "lift the basket out slowly, level", pit(1.1))
    R.go(1.5, np.array([0.31, -0.08, OVER[2]]), FINGER_CLOSED, "carry it toward the tub", pit(1.1))
    R.go(2.0, OVER, FINGER_CLOSED, "carry it over the bus tub", pit(1.1))
    R.hold(R.t + 1.5, "let it stop swinging")
    FORK = rolL(np.pi / 2)                                  # hand pointing +x, fingers closed flat (jaws along z): a fork
    # ---- 3. left: lift the basket's left bottom edge, tipping it over on its bail toward -y; scraps fall out
    L.hold(R.t - 1.5)                                       # the basket is over the tub by now
    L.go(3.0, tip(0.0) + np.array([-0.17, 0.0, -0.028]), FINGER_CLOSED, "behind the basket, below its left edge", FORK)
    L.go(2.0, tip(0.0) + np.array([0.0, 0.0, -0.028]), FINGER_CLOSED, "fingers slid in under its bottom", FORK)
    _sync(L, R)
    L.go(1.2, tip(0.0), FINGER_CLOSED, "up against its bottom", FORK)
    for deg in (25, 50, 75, 95, 115, 130):
        L.go(1.1, tip(np.radians(deg)), FINGER_CLOSED, f"tip it {deg} deg", FORK)
    L.hold(L.t + 2.0, "scraps slide out into the tub")
    for deg in (100, 65, 30, 0):
        L.go(1.2, tip(np.radians(deg), -0.02), FINGER_CLOSED, f"let it back {deg} deg", FORK)
    L.go(1.5, tip(0.0) + np.array([-0.17, 0.0, -0.028]), FINGER_CLOSED, "fingers back out", FORK)
    L.go(2.0, REST["left"], FINGER_OPEN, "rest", pit(REST_P))
    _sync(L, R)
    L.hold(L.t + 3.0, "settle")
    R.hold(R.t + 3.0, "hold the empty basket over the tub")
    return {"left": L.w, "right": R.w}


# grate: hinged at its near edge about y; opening turns its free (+x) edge up and over toward the robot
HINGE_P = np.array([D.HINGE[0], D.SY, D.HINGE[1]])
STRIP_REL = np.array([D.G_X[1] - D.LIFT_SLOT[2] / 2 - D.HINGE[0], 0.0, -0.011])   # TCP 6 mm under the bar: a deep grip


def strip(phi):
    return HINGE_P + _ry(-phi) @ STRIP_REL


def bar(a):
    """The bail's sleeve with the bail turned a (rad) from upright toward -y about its pivots."""
    return np.array([D.SX, D.SY - D.BAIL_RISE * np.sin(a), D.BAIL_PIVOT_Z + D.BAIL_RISE * np.cos(a)])


OVER = np.array([0.300, -0.200, 1.150])                   # the bail's sleeve over the tub
GRIP_TO_PIV = D.BAIL_RISE
SAG = np.array([0.0, -0.022, -0.030])                    # where the loaded right hand actually holds the bail (measured sag)
EDGE_REL = np.array([0.0, D.B_HALF, -(D.BAIL_PIVOT_Z - (D.B_C[2] - D.B_H / 2))])   # basket's left bottom edge from its pivots


def tip_point(th):
    """Where the basket's left bottom edge is with the basket turned th about x (left side up): first about its bail
    pivots until the bail meets its stop, then with the bail about the grip."""
    G = OVER + SAG
    P = G - np.array([0.0, 0.0, GRIP_TO_PIV])
    th1 = min(th, np.radians(D.BAIL_STOP_DEG))
    c = P + _rx(th1) @ EDGE_REL
    if th > th1:
        c = G + _rx(th - th1) @ (c - G)
    return c


def tip(th, dz=-0.010):
    c = tip_point(th)
    return np.array([D.SX + 0.03, c[1] - 0.035, c[2] + dz])
