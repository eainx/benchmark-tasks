"""Fast food back room: "Connect the new syrup box."

Two 5-gallon bag-in-box cola syrup cartons stand on a wire shelf, knock-out faces toward the arms. The empty
one's spout still wears the syrup line's quick-disconnect connector; the new one's knock-out is pushed in and
its spout pulled out through it. The connector is turned a quarter turn counter-clockwise (seen from the
front) and drawn off the empty spout; the new spout is held by its flange (it rides on the slack bag and gives
way when pushed), the connector pushed onto it and turned a quarter turn clockwise to lock.
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
import syrup as S  # noqa: E402
from scoring import TrialScorer, roll_deg  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "syrup-box-connect"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
SCENE_ONLY = False
MAX_STEPS = 80 if SCENE_ONLY else 1200
OBJECT_NAMES = ("new_spout", "connector")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (S.rack_object(), S.board_object(), S.carton_object("new_carton", S.NEW_Y, False),
             S.carton_object("old_carton", S.OLD_Y, True), S.new_spout_object(), S.connector_object(), S.line_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_syrup_box_connect", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


# -- what rigid bodies cannot do by themselves -------------------------------------------------------------
GRIP_MM = 25.0                        # a closed hand's pad tips this close to the collar's centre: gripping it
OFF_DEG = -85.0                       # turned this far counter-clockwise on the empty spout: its lugs are free
ENGAGE_RADIAL = 0.003                 # pushed onto the new spout: off its axis by at most this ...
ENGAGE_AXIAL = 0.004                  # ... and within this of fully on ...
ENGAGE_AXIS_DEG = 8.0                 # ... square within this
LOCK_DEG = 85.0                       # turned this far clockwise on the new spout: locked
DETENT = 0.5                          # N m s / deg: the collar's detent while nobody turns it
TURNING = 0.004                       # ... while a hand grips and turns it


def collar_centre(pose):
    p = np.asarray(pose, float)
    return p[:3] + S.quat_x_axis(p[3:7]) * S.COLLAR_X


def pad_tips(robot):
    from yam_arena import ik
    from yam_arena.station import BASES
    out = {}
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, np.asarray(BASES[side], float))
        closed = abs(float(robot[side]["fingers"][0])) < 0.75 * abs(FINGER_OPEN)
        out[side] = (pos + R[:, 2] * PAD_TIP, closed)
    return out


class Dynamics:
    """Disclosed stand-ins, from measured poses only; no position targets anywhere, so nothing snaps.

    * Connector on the empty spout: a hinge about the spout's axis (-95..+5 deg) held by a stiff damping drive
      (the collar's detent); a closed hand gripping the collar (pad tips within 25 mm of its centre) lowers the
      damping so the hand can turn it. Turned 85 deg counter-clockwise the lugs are free: the hinge is
      switched off and the connector comes off in the hand.
    * Connector onto the new spout: a hinge to the spout, made at spawn and switched on when the connector is
      pushed on square (within 3 mm of the spout's axis, 4 mm of fully on, 8 deg). The spout itself rides on a
      sprung slide (the slack bag), so unless held it is pushed back into the carton.
    * Locked: turned 85 deg clockwise on the new spout, the hinge's damping is made very large.
    * The syrup line is redrawn every step from the pump to the connector's barb.
    """

    def __init__(self, env):
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        self.j = {}
        self.line = None
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if path.endswith("/connector/hinge_joint"):
                self.j["old"] = prim
            elif path.endswith("/connector/hinge2_joint"):
                self.j["new"] = prim
            elif path.endswith("/syrup_line/line"):
                self.line = prim
        if len(self.j) != 2 or self.line is None:
            raise RuntimeError(f"joints/line not found: {sorted(self.j)} {self.line}")
        self.origin = np.asarray(S.connector_pos(S.OLD_Y), float)

    def _on(self, key, on):
        self.j[key].GetAttribute("physics:jointEnabled").Set(bool(on))

    def _damp(self, key, d):
        self.j[key].GetAttribute("drive:angular:physics:damping").Set(float(d))

    def reset(self):
        self.on_old = True
        self.engaged = False
        self.locked = False
        self.roll0 = 0.0
        self.roll_new0 = None
        self.prev = None
        self.unwrapped = 0.0
        self._on("old", True)
        self._on("new", False)
        self._damp("old", DETENT)
        self._damp("new", TURNING)

    def before_step(self, step, action):
        pass

    def _roll(self, pose):
        r = roll_deg(pose)
        if self.prev is None:
            self.unwrapped = r
        else:
            d = (r - self.prev + 180.0) % 360.0 - 180.0
            self.unwrapped += d
        self.prev = r
        return self.unwrapped

    def _redraw(self, pose):
        from pxr import Gf, Vt
        p, t = S.line_mesh(pose, self.origin)
        n = np.zeros_like(p)
        fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        for k in range(3):
            np.add.at(n, t[:, k], fn)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        self.line.GetAttribute("points").Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in p]))
        self.line.GetAttribute("normals").Set(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in n]))

    def after_step(self, step, objects, poses, robot):
        conn = np.asarray(poses["connector"], float)
        spout = np.asarray(poses["new_spout"], float)
        roll = self._roll(conn)
        tips = pad_tips(robot)
        cc = collar_centre(conn)
        gripped = any(closed and float(np.linalg.norm(tip - cc)) < GRIP_MM / 1000 for tip, closed in tips.values())
        if self.on_old:
            self._damp("old", TURNING if gripped else DETENT)
            if roll - self.roll0 <= OFF_DEG:
                self._on("old", False)
                self.on_old = False
        elif not self.engaged:
            seat = spout[:3] + (S.connector_pos(S.NEW_Y) - S.spout_axis(S.NEW_Y))
            rel = conn[:3] - seat
            ax = S.quat_x_axis(conn[3:7])
            tilt = math.degrees(math.acos(float(np.clip(ax[0], -1.0, 1.0))))
            if math.hypot(rel[1], rel[2]) <= ENGAGE_RADIAL and abs(rel[0]) <= ENGAGE_AXIAL and tilt <= ENGAGE_AXIS_DEG:
                self._on("new", True)
                self.engaged = True
                self.roll_new0 = roll
        elif not self.locked:
            self._damp("new", TURNING if gripped else 0.05)
            if roll - self.roll_new0 >= LOCK_DEG:
                self._damp("new", 1e4)
                self.locked = True
        self._redraw(conn)
        return {"on_old": self.on_old, "engaged": self.engaged, "locked": self.locked,
                "turn_old_deg": round(roll - self.roll0, 1) if self.on_old else None,
                "turn_new_deg": round(roll - self.roll_new0, 1) if self.roll_new0 is not None else None,
                "gripped": bool(gripped), "spout_in_mm": round(1000 * float(spout[0] - S.spout_axis(S.NEW_Y)[0]), 1)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- scripted demo -----------------------------------------------------------------------------------------
# Left hand: the connector. Grips its collar level from the front (fingers across, roll -90), turns it a
# quarter turn counter-clockwise (roll -90 -> 0), draws it off, carries it to the new spout, pushes it on
# square and turns it a quarter turn clockwise (roll 0 -> -90), lets go and backs off.
# Right hand: holds the new spout's flange from above (fingers closing front-to-back across it) while the
# connector goes on, then lets go.
F_SPAN = -0.012                       # jaws opened 24 mm: either side of the 4 mm flange, clear of the carton face
SAG = 0.004                           # the connector hangs ~4 mm low in the level grip: aim that much higher
REST = {"left": (0.14, 0.20, S.TOP + 0.24), "right": (0.14, -0.20, S.TOP + 0.24)}
X = np.array([1.0, 0.0, 0.0])


def frame(a, close):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    c = np.asarray(close, float)
    c = c - np.dot(c, a) * a
    c = c / np.linalg.norm(c)
    return np.stack([c, np.cross(a, c), a], axis=1)


def level(roll):
    r = math.radians(roll)
    return frame(X, (0.0, math.sin(r), math.cos(r)))


def _w(t, p, f, label, rot):
    return Waypoint(t, tuple(float(v) for v in p), f, label, rot)


def demo():
    if SCENE_ONLY:
        return {side: [_w(1.5, REST[side], FINGER_OPEN, "look", DOWN), _w(MAX_STEPS / HZ, REST[side], FINGER_OPEN, "hold", DOWN)]
                for side in ("left", "right")}
    old = S.connector_pos(S.OLD_Y) + X * S.COLLAR_X - X * PAD_TIP          # TCP for the collar on the empty spout
    new = S.connector_pos(S.NEW_Y) + X * S.COLLAR_X - X * PAD_TIP + np.array([0.0, 0.0, SAG])   # ... on the new spout
    up = np.array([0.0, 0.0, 1.0])
    left = [
        _w(1.5, REST["left"], FINGER_OPEN, "look", DOWN),
        _w(2.5, old - X * 0.07, FINGER_OPEN, "in front of the connector", level(-90)),
        _w(1.5, old, FINGER_OPEN, "round its collar", level(-90)),
        _w(0.8, old, FINGER_CLOSED, "grip the collar", level(-90)),
        _w(1.2, old, FINGER_CLOSED, "turn it", level(-45)),
        _w(1.2, old, FINGER_CLOSED, "quarter turn counter-clockwise", level(0)),
        _w(1.5, old - X * 0.07, FINGER_CLOSED, "draw it off the spout", level(0)),
        _w(2.0, old - X * 0.09 + up * 0.04 + np.array([0.0, -0.165, 0.0]), FINGER_CLOSED, "carry it over", level(0)),
        _w(2.0, new - X * 0.05, FINGER_CLOSED, "line it up with the new spout", level(0)),
        _w(1.0, new - X * 0.05, FINGER_CLOSED, "steady", level(0)),
        _w(2.0, new + X * 0.002, FINGER_CLOSED, "push it on", level(0)),
        _w(1.2, new + X * 0.002, FINGER_CLOSED, "turn it", level(-45)),
        _w(1.2, new + X * 0.002, FINGER_CLOSED, "quarter turn clockwise: locked", level(-90)),
        _w(0.8, new, FINGER_OPEN, "let go", level(-90)),
        _w(1.5, new - X * 0.08, FINGER_OPEN, "back off", level(-90)),
        _w(2.0, REST["left"], FINGER_OPEN, "rest", DOWN),
        _w(3.0, REST["left"], FINGER_OPEN, "hold", DOWN),
    ]
    # right hand: from the right side, 15 degrees forward of square, its fingers closing front-to-back across
    # the flange's near edge (pad tips 12 mm in over the 23 mm-radius flange)
    a_r = np.array([0.26, 0.97, 0.0])
    a_r = a_r / np.linalg.norm(a_r)
    side = frame(a_r, (1.0, 0.0, 0.0))
    tip = S.spout_axis(S.NEW_Y) + X * 0.002 + np.array([0.0, -(S.FLANGE_R - 0.012), 0.0])
    grip_r = tip - a_r * PAD_TIP
    t_hold = 1.5 + 2.5 + 1.5 + 0.8 + 1.2 + 1.2 + 1.5 + 2.0 - 5.0
    right = [
        _w(1.5, REST["right"], FINGER_OPEN, "look", DOWN),
        _w(t_hold, REST["right"], FINGER_OPEN, "wait", DOWN),
        _w(2.5, grip_r - a_r * 0.035 + up * 0.02, F_SPAN, "beside the new spout's flange", side),
        _w(1.5, grip_r, F_SPAN, "fingers either side of the flange", side),
        _w(0.8, grip_r, FINGER_CLOSED, "hold the flange", side),
        _w(9.0, grip_r, FINGER_CLOSED, "hold it while the connector goes on", side),
        _w(0.8, grip_r, F_SPAN, "let go", side),
        _w(1.2, grip_r - a_r * 0.05 + up * 0.03, FINGER_OPEN, "draw the hand away", side),
        _w(2.0, REST["right"], FINGER_OPEN, "rest", DOWN),
        _w(3.0, REST["right"], FINGER_OPEN, "hold", DOWN),
    ]
    return {"left": left, "right": right}
