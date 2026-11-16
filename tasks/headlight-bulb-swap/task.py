"""Car repair: "Replace the headlight bulb."

A halogen headlamp unit, off the car, stands upright on a folded shop towel, as it sits in the car, with its back toward the right arm. In its
low-beam port sits the old H11 bulb, locked by its quarter-turn bayonet, with the lamp's pigtail
connector latched onto the bulb's L-shaped socket. The new H11 bulb lies in its opened blister pack on
the bench. The connector is unlatched and pulled off, the old
bulb turned a quarter turn counter-clockwise and drawn out, the new one put in by its base and turned
a quarter turn clockwise to lock, and the connector pushed back on until it latches -- without touching
the new bulb's glass.
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
import headlamp as H  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.constants import PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "headlight-bulb-swap"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1800
OBJECT_NAMES = ("old_bulb", "new_bulb", "connector")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (H.towel_object(), H.housing_object(), H.blister_object(), H.new_bulb_object(), H.old_bulb_object(),
             H.connector_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_headlight_bulb_swap", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                                          # measured state the scripted demo steers by

PAD_T = 0.008                                        # finger pad box: thickness out from its face (m) ...
PAD_W = 0.024                                        # ... width across the jaw ...
PAD_L = 0.045                                        # ... and length back from the tip (a generous fingertip box)


def _bases():
    """The arm bases the runner's rig uses (station.BASES: y = +-0.305 m)."""
    from yam_arena import station
    return {side: np.asarray(station.BASES[side], float) for side in ("left", "right")}
TAB_MM = 8.0                                         # a closed hand's pad tips this close to the latch tab: pressing it
TAB_STEPS = 6                                        # ... for 0.3 s: the latch lets go
GRIP_MM = 20.0                                       # a hand's pad tips this close to the old bulb's grip, closed on it
ENGAGE_RADIAL = 0.003                                # new bulb into the port: grip centre off the port axis (m)
ENGAGE_AXIAL = 0.004                                 # ... off its seat along the axis (m)
ENGAGE_AXIS_DEG = 8.0                                # ... its axis off the port's
ENGAGE_LUG_DEG = 15.0                                # ... its socket off straight down (lugs in the notches)
PLUG_MM = 4.0                                        # connector pushed this close to its seat on the new bulb
PLUG_DEG = 10.0


def hand_points(robot):
    """Per hand: the pad tips' centre, the jaw half-opening and both finger pads as oriented boxes
    (centre, axes, half-sizes), from the measured joints and finger travel through the URDF's forward
    kinematics at the runner's arm bases (the jaws close along the gripper's local x)."""
    from yam_arena import ik
    bases = _bases()
    out = {}
    for side in ("left", "right"):
        q = np.asarray(robot[side]["joints"], float)
        pos, R = ik.tcp_pose(q, bases[side])
        half = abs(float(robot[side]["fingers"][0]))
        tip = pos + R[:, 2] * PAD_TIP
        boxes = []
        for sgn in (-1.0, 1.0):
            c = tip + R[:, 0] * sgn * (half + PAD_T / 2) - R[:, 2] * (PAD_L / 2)
            boxes.append((c, R.copy(), np.array([PAD_T / 2, PAD_W / 2, PAD_L / 2])))
        out[side] = (tip, half, boxes)
    return out


def box_distance(points, box):
    """Distance from each point to an oriented box (0 inside)."""
    c, R, h = box
    local = (np.asarray(points, float) - c) @ R
    d = np.maximum(np.abs(local) - h, 0.0)
    return np.linalg.norm(d, axis=1)


def segment_box_distance(a, b, box, n=81):
    """Distance from segment a-b to an oriented box, the segment sampled every ~1/80 of its length."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return float(box_distance(a + t * (b - a), box).min())


class Dynamics:
    """What rigid bodies cannot do by themselves, from measured poses only (disclosed stand-ins).

    * Connector latch: the connector is held on the old bulb's socket (a fixed joint) until its latch
      tab is pressed: a closed hand's pad tips within 8 mm of the tab's top for 0.3 s. Then it is free.
    * Old bulb's bayonet: a hinge about the port axis, -90..0 deg, held by a stiff damping drive (the
      bayonet's detent) until the connector is off (10 mm from its seat) and a closed hand holds the
      bulb's grip; then it turns against the O-ring's light drag. Turned a quarter turn (-88 deg) its
      lugs are at the notches: the hinge is switched off and the bulb draws out.
    * New bulb's bayonet: a hinge made at spawn, switched on when the bulb is put to its seat square
      (grip centre within 3 mm of the port axis and 4 mm of the seat, axis within 8 deg, socket within
      15 deg of straight down: lugs in the notches). Turned +88 deg it is locked: its detent (the
      hinge's lower stop raised to 87.5 deg, the stop at 90.5 the other way) and the stiff damping
      drive. No position targets: nothing snaps.
    * Connector on the new bulb: once the bulb is locked, the connector pushed to within 4 mm and
      10 deg of its seat latches (a joint to the bulb is switched on).
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        self.j = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for key, end in (("old", "/old_bulb/hinge_joint"), ("new", "/new_bulb/hinge_joint"),
                             ("anchor", "/connector/anchor_joint"), ("plug", "/connector/hinge_joint")):
                if path.endswith(end):
                    self.j[key] = prim
        if len(self.j) != 4:
            raise RuntimeError(f"joints not found on the stage: {sorted(self.j)}")

    def _on(self, key, on):
        self.j[key].GetAttribute("physics:jointEnabled").Set(bool(on))

    def _damp(self, key, d):
        self.j[key].GetAttribute("drive:angular:physics:damping").Set(float(d))

    def _lower(self, key, deg):
        self.j[key].GetAttribute("physics:lowerLimit").Set(float(deg))

    def reset(self):
        self.free = False            # the connector is off its latch
        self.tab_n = 0
        self.old_in = True
        self.old_turnable = False
        self.new_on = False
        self.locked = False
        self.plugged = False
        self._on("old", True)
        self._damp("old", H.DETENT_D)
        self._lower("old", -0.5)
        self._on("anchor", True)
        self._on("new", False)
        self._damp("new", H.TURN_D)
        self._lower("new", -0.5)
        self._on("plug", False)

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        old = np.asarray(root_poses["old_bulb"], float)
        new = np.asarray(root_poses["new_bulb"], float)
        conn = np.asarray(root_poses["connector"], float)
        hands = hand_points(robot)
        seat = np.asarray(H.bulb_seated_pos(), float)
        cseat = np.asarray(H.connector_pos(), float)
        Rc, Ro, Rn = H.mat(conn), H.mat(old), H.mat(new)
        # -- the connector's latch
        tab = conn[:3] + Rc @ np.array([0.002, 0.0, H.CONN[2] / 2 + 0.003])
        pressed = any(h[1] < 0.004 and float(np.linalg.norm(h[0] - tab)) < TAB_MM / 1000 for h in hands.values())
        self.tab_n = self.tab_n + 1 if pressed else 0
        if not self.free and self.tab_n >= TAB_STEPS:
            self._on("anchor", False)
            self.free = True
        conn_off_old = self.free and float(np.linalg.norm(conn[:3] - cseat)) > 0.010
        # -- the old bulb's bayonet
        sd = Ro @ np.array([-1.0, 0.0, 0.0])                                  # its socket's direction
        old_deg = math.degrees(math.atan2(sd[2], -sd[0]))
        if self.old_in:
            held = any(h[1] < 0.016 and float(np.linalg.norm(h[0] - old[:3])) < GRIP_MM / 1000 for h in hands.values())
            turnable = conn_off_old and held
            if turnable != self.old_turnable:
                self._damp("old", H.TURN_D if turnable else H.DETENT_D)
                self._lower("old", -(H.LOCK_DEG + 0.5) if turnable else min(old_deg, 0.0) - 0.5)
                self.old_turnable = turnable
            if old_deg <= -(H.LOCK_DEG - 2.0):
                self._on("old", False)
                self.old_in = False
        # -- the new bulb's bayonet
        sn = Rn @ np.array([0.0, 0.0, -1.0])
        new_deg = math.degrees(math.atan2(-sn[0], -sn[2]))
        rel = new[:3] - seat
        radial = math.hypot(rel[0], rel[2])
        axis = Rn @ np.array([0.0, 1.0, 0.0])
        axis_deg = math.degrees(math.acos(float(np.clip(axis[1], -1, 1))))
        lug_deg = math.degrees(math.acos(float(np.clip(-sn[2], -1, 1))))
        if not self.new_on and not self.old_in and radial < ENGAGE_RADIAL and abs(rel[1]) < ENGAGE_AXIAL \
                and axis_deg < ENGAGE_AXIS_DEG and lug_deg < ENGAGE_LUG_DEG:
            self._on("new", True)
            self.new_on = True
        if self.new_on and not self.locked and new_deg >= H.LOCK_DEG - 2.0:
            self._damp("new", H.DETENT_D)
            self._lower("new", min(new_deg, H.LOCK_DEG - 2.0) - 0.5)       # the detent: it cannot turn back
            self.locked = True
        # -- the connector back on, onto the new bulb
        crel = float(np.linalg.norm(conn[:3] - cseat))
        ang = math.degrees(math.acos(float(np.clip((np.trace(Rc) - 1) / 2, -1, 1))))
        if self.locked and self.free and not self.plugged and crel < PLUG_MM / 1000 and ang < PLUG_DEG:
            self._on("plug", True)
            self.plugged = True
        SHARED.update(new=new[:3].copy(), conn=conn[:3].copy(), old=old[:3].copy(), new_on=self.new_on,
                      locked=self.locked, free=self.free, old_in=self.old_in, plugged=self.plugged)
        return {"pad_boxes": [[list(map(float, c)), [list(map(float, r)) for r in R], list(map(float, hh))]
                              for h in hands.values() for c, R, hh in h[2]],
                "tcp_tips": [list(map(float, h[0])) for h in hands.values()],
                "connector_free": self.free, "old_in_port": self.old_in, "old_turnable": self.old_turnable,
                "old_deg": round(old_deg, 1), "new_deg": round(new_deg, 1), "new_engaged": self.new_on,
                "new_locked": self.locked, "connector_on_new": self.plugged,
                "new_radial_mm": round(1000 * radial, 1), "new_axial_mm": round(1000 * rel[1], 1),
                "new_axis_deg": round(axis_deg, 1), "new_lug_deg": round(lug_deg, 1),
                "conn_mm": round(1000 * crel, 1), "conn_deg": round(ang, 1),
                "tab_mm": round(1000 * min(float(np.linalg.norm(h[0] - tab)) for h in hands.values()), 1)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------

def _axis_rot(v, a):
    v = np.asarray(v, float) / np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def _frame(approach, up):
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    up = np.asarray(up, float)
    if abs(float(z @ up)) > 0.95:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def grip(approach, roll_deg, up=(0.0, 0.0, 1.0)):
    """TCP rotation: approach axis, turned ``roll_deg`` about it (jaws close along the local x)."""
    return _axis_rot(approach, math.radians(roll_deg)) @ _frame(approach, up)


def _tcp(tip, rot, label, seconds, finger):
    tip = np.asarray(tip, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


REST = {"left": (0.16, 0.24, H.TOP + 0.20), "right": (0.16, -0.24, H.TOP + 0.20)}
Y = np.array([0.0, 1.0, 0.0])
A_L = np.array([1.0, 0.0, 0.0])                     # the left hand takes the connector level, from behind its leads
CONN_GRIP = -0.0075                                  # jaws on the connector's 18 mm sides
BULB_GRIP = -0.0090                                  # jaws on the bulb's 24 mm grip
PICK_OPEN = -0.0220                                  # jaws 44 mm apart: inside the blister tray's walls
PLAN_T = {}


def _seq(items):
    marks, out, t = {}, [], 0.0
    for w in items:
        if isinstance(w, str):
            marks[w] = t
        else:
            out.append(w)
            t += w.duration
    return out, marks, t


def demo():
    C = np.asarray(H.connector_pos(), float)
    B = np.asarray(H.bulb_seated_pos(), float)
    NB = np.asarray(H.new_bulb_pos(), float)
    tab = C + np.array([0.002, 0.0, H.CONN[2] / 2 + 0.0028 - 0.0015])     # pressing the tab, 1.5 mm in
    down_x = grip((0, 0, -1), 90, up=(1, 0, 0))                           # top-down, jaws along x
    down_y = grip((0, 0, -1), 180, up=(1, 0, 0))                          # top-down, jaws along y: fits between the left's
    g_conn = grip(A_L, 180)                                               # jaws along y (wrist mid-range)
    c_tip = C + np.array([0.006, 0.0, 0.0])                               # pad tips near its front end
    g_old0, g_old1 = grip(Y, 270), grip(Y, 180)                           # jaws vertical, then level: -90 deg about +y
    g_new0, g_new1 = grip(Y, 180), grip(Y, 285)                           # +90 deg about +y locks it (the stop takes the rest)
    b_tip = B + Y * 0.006
    ap_drop = np.array([0.0, 0.87, -0.5])
    g_drop = grip(ap_drop, 180)
    drop = np.array([0.400, -0.195, H.TOP + 0.060])
    n_tip = NB + np.array([0.0, -0.004, -0.004])
    pre_ins = np.array([B[0], B[1] - 0.045, B[2]])
    right = [
        Waypoint(1.5, REST["right"], FINGER_CLOSED, "look", DOWN),
        _tcp(tab + np.array([0, 0, 0.05]), down_y, "over the latch tab", 2.0, FINGER_CLOSED),
        _tcp(tab + np.array([0, 0, 0.02]), down_y, "over the latch tab", 1.6, FINGER_CLOSED),
        _tcp(tab, down_y, "press the tab", 1.4, FINGER_CLOSED),
        "MARK_PRESSED",
        _tcp(tab, down_y, "hold the tab down", 2.2, FINGER_CLOSED),
        _tcp(tab + np.array([0, 0, 0.04]), down_y, "off the tab", 1.0, FINGER_CLOSED),
        _tcp(b_tip - Y * 0.06, g_old0, "behind the old bulb", 2.2, FINGER_OPEN),
        _tcp(b_tip, g_old0, "round its grip", 1.2, FINGER_OPEN),
        _tcp(b_tip, g_old0, "grip the old bulb", 0.6, BULB_GRIP),
        _tcp(b_tip, grip(Y, 225), "turn it a quarter turn back", 1.2, BULB_GRIP),
        _tcp(b_tip, g_old1, "turn it a quarter turn back", 1.2, BULB_GRIP),
        _tcp(b_tip, g_old1, "turn it a quarter turn back", 0.4, BULB_GRIP),
        _tcp(b_tip - Y * 0.030, g_old1, "draw it out", 1.2, BULB_GRIP),
        _tcp(b_tip - Y * 0.050 + np.array([0, 0, 0.02]), grip(np.array([0.0, 0.94, -0.34]), 180), "draw it out", 1.0,
             BULB_GRIP),
        _tcp(drop, g_drop, "set it down on the bench", 2.0, BULB_GRIP),
        _tcp(drop, g_drop, "let it go", 0.5, FINGER_OPEN),
        _tcp(drop - ap_drop * 0.05, g_drop, "back off", 0.8, FINGER_OPEN),
        _tcp(n_tip + np.array([0, 0, 0.07]), down_x, "over the new bulb", 2.0, PICK_OPEN),
        _tcp(n_tip + np.array([0, 0, 0.01]), down_x, "down round its grip", 1.2, PICK_OPEN),
        _tcp(n_tip, down_x, "down round its grip", 0.6, PICK_OPEN),
        _tcp(n_tip, down_x, "take it by its base", 0.6, BULB_GRIP),
        _tcp(n_tip + np.array([0, 0, 0.04]), down_x, "lift it out of the pack", 1.0, BULB_GRIP),
        _tcp(pre_ins + (n_tip - NB) + np.array([0, 0, 0.04]), down_x, "to the port", 2.4, BULB_GRIP),
        _tcp(pre_ins + (n_tip - NB), down_x, "line it up with the port", 1.2, BULB_GRIP),
        "MARK_INSERT",
        _tcp(B + (n_tip - NB) - Y * 0.004, down_x, "into the port", 2.0, BULB_GRIP),
        _tcp(B + (n_tip - NB), down_x, "into the port", 0.8, BULB_GRIP),
        _tcp(B + (n_tip - NB), down_x, "let go of it", 0.5, FINGER_OPEN),
        _tcp(B + (n_tip - NB) + np.array([0, 0, 0.045]), down_x, "up off it", 1.0, FINGER_OPEN),
        _tcp(b_tip - Y * 0.07 + np.array([0, 0, 0.02]), g_new0, "behind the new bulb", 2.2, FINGER_OPEN),
        _tcp(b_tip - Y * 0.05, g_new0, "behind the new bulb", 0.8, FINGER_OPEN),
        _tcp(b_tip, g_new0, "round its grip", 1.2, FINGER_OPEN),
        _tcp(b_tip, g_new0, "grip the new bulb", 0.6, BULB_GRIP),
        _tcp(b_tip, grip(Y, 232), "turn it a quarter turn to lock", 1.2, BULB_GRIP),
        _tcp(b_tip, g_new1, "turn it a quarter turn to lock", 1.2, BULB_GRIP),
        _tcp(b_tip, g_new1, "turn it a quarter turn to lock", 0.4, BULB_GRIP),
        _tcp(b_tip, g_new1, "let go of it", 0.6, FINGER_OPEN),
        _tcp(b_tip - Y * 0.07, g_new1, "back off", 1.2, FINGER_OPEN),
        "MARK_RIGHT_DONE",
        Waypoint(2.0, REST["right"], FINGER_OPEN, "rest", DOWN),
    ]
    right, mr, tr = _seq(right)
    t_press = mr["MARK_PRESSED"]
    t_done = mr["MARK_RIGHT_DONE"]
    park = np.array([0.230, -0.125, H.TOP + 0.150])
    left = [
        Waypoint(1.0, REST["left"], FINGER_OPEN, "look", DOWN),
        _tcp(c_tip - A_L * 0.06, g_conn, "behind the connector", 2.0, FINGER_OPEN),
        _tcp(c_tip, g_conn, "round the connector", 1.2, FINGER_OPEN),
        _tcp(c_tip, g_conn, "take the connector", 0.6, CONN_GRIP),
    ]
    _, _, tl = _seq(left)
    left += [_tcp(c_tip, g_conn, "hold it while the tab is pressed", max(0.2, t_press + 1.6 - tl), CONN_GRIP),
             _tcp(c_tip + np.array([-0.035, 0.0, 0.0]), g_conn, "pull it off the socket", 1.4, CONN_GRIP),
             _tcp(park, g_conn, "hold it aside", 1.6, CONN_GRIP)]
    _, _, tl = _seq(left)
    pre_plug = c_tip + np.array([-0.030, 0.0, 0.0])
    left += [_tcp(park, g_conn, "hold it aside", max(0.2, t_done - 1.0 - tl), CONN_GRIP),
             _tcp(pre_plug, g_conn, "line it up with the new socket", 2.0, CONN_GRIP),
             "MARK_PLUG",
             _tcp(c_tip + np.array([0.0015, 0.0, 0.0]), g_conn, "push it on till it latches", 2.0, CONN_GRIP),
             _tcp(c_tip + np.array([0.0015, 0.0, 0.0]), g_conn, "push it on till it latches", 0.6, CONN_GRIP),
             _tcp(c_tip, g_conn, "let go of it", 0.6, FINGER_OPEN),
             _tcp(c_tip - A_L * 0.07, g_conn, "back off", 1.2, FINGER_OPEN),
             Waypoint(2.0, REST["left"], FINGER_OPEN, "rest", DOWN)]
    left, ml, _ = _seq(left)
    PLAN_T.update(mr)
    PLAN_T.update(ml)
    return {"left": left, "right": right}


ALIGN = ("line it up with the port", "into the port")
PLUG = ("line it up with the new socket", "push it on till it latches")


class DemoController:
    """Follows the waypoint plans by IK each step, steering the last centimetres by measured poses: the new
    bulb onto the port's axis and the connector onto its seat (what a person does by eye)."""

    def __init__(self, plans):
        self.plans = plans
        self.corr = {"right": np.zeros(3), "left": np.zeros(3)}

    def action(self, t):
        from yam_arena import ik
        out = []
        seat = np.asarray(H.bulb_seated_pos(), float)
        cseat = np.asarray(H.connector_pos(), float)
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, label = plan.target(t)
            pos = np.array(pos, float)
            if side == "right" and label in ALIGN and "new" in SHARED and not SHARED.get("new_on"):
                err = seat - np.asarray(SHARED["new"], float)
                err[1] = 0.0                                     # along the axis the plan pushes it in
                self.corr["right"] = np.clip(self.corr["right"] + 0.06 * err, -0.015, 0.015)
                pos = pos + self.corr["right"]
            elif side == "right" and label in ALIGN:
                pos = pos + self.corr["right"]
            if side == "left" and label in PLUG and "conn" in SHARED and not SHARED.get("plugged"):
                err = cseat - np.asarray(SHARED["conn"], float)
                if label == PLUG[0]:
                    err[0] = 0.0
                self.corr["left"] = np.clip(self.corr["left"] + 0.06 * err, -0.015, 0.015)
                pos = pos + self.corr["left"]
            q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), plan.q, base_pos=plan.base_pos, iters=200)
            SHARED[side + "_ik"] = (round(1000 * pe, 1), round(re, 3), label)
            plan.q = q
            g = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            out.append(np.r_[q, g])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
