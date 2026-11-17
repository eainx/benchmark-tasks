"""Car repair: "Replace the headlight bulb."

A halogen headlamp is bolted, as installed, into a real-size section of a car's front engine bay built
on the bench: the radiator support and radiator on its inboard side, the fender apron on its outboard
side and the air-filter box beside its back, which limit how the hands can reach it. In its low-beam
port sits the old H11 bulb, locked by its quarter-turn bayonet, with the lamp's pigtail connector
latched onto the bulb's L-shaped socket; the connector's two leads (seven colliding articulated links)
run to a harness clip on the lamp. The new H11 bulb lies in its opened blister pack on the bench. One
hand holds the connector while the other pushes its sprung latch tab down; the connector is pulled off
and held aside on its leads, the old bulb is turned a quarter turn back and drawn out, the new one is
put in by its base and turned a quarter turn to lock, and the connector is pushed back on until it
latches -- without touching the new bulb's glass.
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
OBJECT_NAMES = ("old_bulb", "new_bulb", "connector", "tab", *(f"lead{i}" for i in range(7)))

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (H.bay_object(), H.housing_object(), H.blister_object(), H.new_bulb_object(), H.old_bulb_object(),
             H.connector_object(), H.tab_object(), *H.lead_objects())
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
TAB_STEPS = 4                                        # the latch tab held pressed past its release point for 0.2 s: it lets go
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
      tab -- a separate sprung key (3.5 kN/m, 2 mm travel) on the connector -- is physically pushed down
      1.2 mm or more (about 4 N) for 0.2 s. Then it is free, hanging on its leads.
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
        tab = np.asarray(root_poses["tab"], float)
        rest = H.CONN[2] / 2 + H.TAB[2] / 2 + 0.0002
        depressed = rest - float((Rc.T @ (tab[:3] - conn[:3]))[2])           # how far the sprung tab is pushed down
        pressed = depressed >= H.TAB_RELEASE
        self.tab_n = self.tab_n + 1 if pressed else 0
        if not self.free and self.tab_n >= TAB_STEPS:
            self._on("anchor", False)
            self.free = True
        conn_off_old = self.free and float(np.linalg.norm(conn[:3] - cseat)) > 0.010
        # -- the old bulb's bayonet
        sd = Ro @ np.array([0.0, -1.0, 0.0])                                  # its socket's direction (-y when locked)
        old_deg = math.degrees(math.atan2(-sd[2], -sd[1]))                    # its turn about +x (-90: unlocked)
        if self.old_in:
            held = any(h[1] < 0.016 and float(np.linalg.norm(h[0] - old[:3])) < GRIP_MM / 1000 for h in hands.values())
            turnable = conn_off_old and held
            if turnable != self.old_turnable:
                self._damp("old", H.TURN_D if turnable else H.DETENT_D)
                self._lower("old", -(H.LOCK_DEG + 0.5) if turnable else min(old_deg, 0.0) - 0.5)
                self.old_turnable = turnable
            if old_deg <= -(H.LOCK_DEG - 2.0) and conn_off_old:          # its socket clear of the connector
                self._on("old", False)
                self.old_in = False
        # -- the new bulb's bayonet
        sn = Rn @ np.array([0.0, 0.0, 1.0])                                   # its socket (up as it goes in)
        new_deg = math.degrees(math.atan2(-sn[1], sn[2]))                     # its turn about +x (+90: locked)
        rel = new[:3] - seat
        radial = math.hypot(rel[1], rel[2])
        axis = Rn @ np.array([1.0, 0.0, 0.0])
        axis_deg = math.degrees(math.acos(float(np.clip(axis[0], -1, 1))))
        lug_deg = math.degrees(math.acos(float(np.clip(sn[2], -1, 1))))
        if not self.new_on and not self.old_in and radial < ENGAGE_RADIAL and abs(rel[0]) < ENGAGE_AXIAL \
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
                "new_radial_mm": round(1000 * radial, 1), "new_axial_mm": round(1000 * rel[0], 1),
                "new_axis_deg": round(axis_deg, 1), "new_lug_deg": round(lug_deg, 1),
                "conn_mm": round(1000 * crel, 1), "conn_deg": round(ang, 1),
                "tab_pressed_mm": round(1000 * depressed, 2)}


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


REST = {"left": (0.16, 0.24, H.TOP + 0.26), "right": (0.16, -0.24, H.TOP + 0.24)}
X = np.array([1.0, 0.0, 0.0])
CONN_PINCH = -0.0100                                 # jaws pinching the connector top to bottom: its tab pressed down
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
    """The right hand pinches the connector (its tab pressed down) and pulls it off outboard, holding it
    aside on its leads; the left turns the old bulb out, sets it on the bench, takes the new one from the
    pack on the air box by its base, puts it in square and turns it home; the right pushes the connector
    back on till it latches; both stand clear."""
    C = np.asarray(H.connector_pos(), float)
    B = np.asarray(H.bulb_seated_pos(), float)
    NB = np.asarray(H.new_bulb_pos(), float)
    g_pinch = grip(X, 90)                                                 # jaws vertical: top pad on the tab
    c_tip = C + np.array([0.006, 0.0, 0.0])
    park = np.array([0.270, -0.265, H.TOP + 0.115])                       # well outboard: clear of the left hand at the port
    g_old0, g_old1 = grip(X, -90), grip(X, 180)                           # -90 deg about +x unlocks it
    g_new0, g_new1 = grip(X, 180), grip(X, 270)                           # +90 deg about +x locks it
    b_tip = B + X * 0.004
    n_tip = NB + X * 0.004
    ap_drop = np.array([0.87, 0.0, -0.5])
    drop = np.array([0.215, -0.020, H.TOP + 0.055])
    high = np.array([0.16, 0.0, H.TOP + 0.26])
    right = [
        Waypoint(1.0, REST["right"], FINGER_OPEN, "look", DOWN),
        _tcp(c_tip - X * 0.07, g_pinch, "behind the connector", 2.2, FINGER_OPEN),
        _tcp(c_tip, g_pinch, "round the connector", 1.4, FINGER_OPEN),
        _tcp(c_tip, g_pinch, "pinch it, pressing the tab", 0.8, CONN_PINCH),
        _tcp(c_tip, g_pinch, "pinch it, pressing the tab", 1.0, CONN_PINCH),
        _tcp(c_tip - np.array([0.0, 0.035, 0.0]), g_pinch, "pull it off the socket", 1.4, CONN_PINCH),
        _tcp(park, g_pinch, "hold it aside on its leads", 1.8, CONN_PINCH),
        "MARK_PARKED",
    ]
    right, mr, tr = _seq(right)
    left = [
        Waypoint(1.0, REST["left"], FINGER_OPEN, "look", DOWN),
        Waypoint(1.5, tuple(high), FINGER_OPEN, "over the air box", DOWN),
    ]
    _, _, tl = _seq(left)
    left += [
        _tcp(b_tip - X * 0.07, g_old0, "behind the old bulb", max(2.0, mr["MARK_PARKED"] - tl), FINGER_OPEN),
        _tcp(b_tip, g_old0, "round its grip", 1.2, FINGER_OPEN),
        _tcp(b_tip, g_old0, "grip the old bulb", 0.6, BULB_GRIP),
        _tcp(b_tip, grip(X, -135), "turn it a quarter turn back", 1.2, BULB_GRIP),
        _tcp(b_tip, g_old1, "turn it a quarter turn back", 1.2, BULB_GRIP),
        _tcp(b_tip, g_old1, "turn it a quarter turn back", 0.4, BULB_GRIP),
        _tcp(b_tip - X * 0.040, g_old1, "draw it out", 1.2, BULB_GRIP),
        _tcp(drop + np.array([0, 0, 0.05]), grip(ap_drop, 180), "to the bench", 1.8, BULB_GRIP),
        _tcp(drop, grip(ap_drop, 180), "set it down", 1.0, BULB_GRIP),
        _tcp(drop, grip(ap_drop, 180), "let it go", 0.5, FINGER_OPEN),
        _tcp(drop - ap_drop * 0.05, grip(ap_drop, 180), "back off", 0.8, FINGER_OPEN),
        _tcp(n_tip - X * 0.08 + np.array([0, 0, 0.03]), g_new0, "behind the new bulb", 2.0, PICK_OPEN),
        _tcp(n_tip - X * 0.06, g_new0, "behind the new bulb", 1.0, PICK_OPEN),
        _tcp(n_tip, g_new0, "round its base", 1.2, PICK_OPEN),
        _tcp(n_tip, g_new0, "take it by its base", 0.6, BULB_GRIP),
        _tcp(n_tip + np.array([0, 0, 0.012]), g_new0, "lift it out of the pack", 0.8, BULB_GRIP),
        _tcp(n_tip - X * 0.07 + np.array([0, 0, 0.02]), g_new0, "draw it back", 1.2, BULB_GRIP),
        _tcp(b_tip - X * 0.06, g_new0, "to the port", 2.4, BULB_GRIP),
        _tcp(b_tip - X * 0.035, g_new0, "line it up with the port", 1.2, BULB_GRIP),
        "MARK_INSERT",
        _tcp(b_tip - X * 0.004, g_new0, "into the port", 1.8, BULB_GRIP),
        _tcp(b_tip, g_new0, "into the port", 0.8, BULB_GRIP),
        _tcp(b_tip, grip(X, 225), "turn it a quarter turn to lock", 1.2, BULB_GRIP),
        _tcp(b_tip, grip(X, 285), "turn it a quarter turn to lock", 1.2, BULB_GRIP),
        _tcp(b_tip, grip(X, 285), "turn it a quarter turn to lock", 0.4, BULB_GRIP),
        _tcp(b_tip, grip(X, 285), "let go of it", 0.6, FINGER_OPEN),
        _tcp(b_tip - X * 0.08, grip(X, 285), "back off", 1.2, FINGER_OPEN),
        "MARK_LEFT_DONE",
        Waypoint(1.5, tuple(high), FINGER_OPEN, "over the air box", DOWN),
        Waypoint(1.5, REST["left"], FINGER_OPEN, "rest", DOWN),
    ]
    left, ml, _ = _seq(left)
    _, _, tr = _seq(right)
    pre_plug = c_tip - np.array([0.0, 0.030, 0.0])
    right += [_tcp(park, g_pinch, "hold it aside on its leads", max(0.2, ml["MARK_LEFT_DONE"] - tr), CONN_PINCH),
              _tcp(pre_plug, g_pinch, "line it up with the new socket", 2.0, CONN_PINCH)]
    right, mr2, _ = _seq(right + ["MARK_PLUG",
                                  _tcp(c_tip + np.array([0.0, 0.0015, 0.0]), g_pinch, "push it on till it latches", 2.0, CONN_PINCH),
                                  _tcp(c_tip + np.array([0.0, 0.0015, 0.0]), g_pinch, "push it on till it latches", 0.6, CONN_PINCH),
                                  _tcp(c_tip, g_pinch, "let go of it", 0.6, FINGER_OPEN),
                                  _tcp(c_tip - X * 0.08, g_pinch, "back off", 1.2, FINGER_OPEN),
                                  Waypoint(1.5, REST["right"], FINGER_OPEN, "rest", DOWN)])
    PLAN_T.update(mr2)
    PLAN_T.update(ml)
    return {"left": left, "right": right}


ALIGN = ("line it up with the port", "into the port")
PORT_BRANCH = ("behind the new bulb", "round its base", "take it by its base", "lift it out of the pack", "draw it back",
               "to the port", "line it up with the port", "into the port")
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
            if side == "left" and label in ALIGN and "new" in SHARED and not SHARED.get("new_on"):
                err = seat - np.asarray(SHARED["new"], float)
                err[0] = 0.0                                     # along the axis the plan pushes it in
                self.corr["left"] = np.clip(self.corr["left"] + 0.06 * err, -0.015, 0.015)
                pos = pos + self.corr["left"]
            elif side == "left" and label in ALIGN:
                pos = pos + self.corr["left"]
            if side == "right" and label in PLUG and "conn" in SHARED and not SHARED.get("plugged"):
                err = cseat - np.asarray(SHARED["conn"], float)
                if label == PLUG[0]:
                    err[1] = 0.0
                self.corr["right"] = np.clip(self.corr["right"] + 0.06 * err, -0.015, 0.015)
                pos = pos + self.corr["right"]
            seed = plan.q
            if side == "left" and label == "draw it out":
                SHARED["left_q_port"] = np.array(plan.q, float)       # the arm's pose at the port, bulb in hand
            if side == "left" and label in PORT_BRANCH and "left_q_port" in SHARED:
                # keep the wrist on the branch it had at the port (the IK otherwise swings it outboard,
                # into the other arm): solve from that pose, then blend toward the current one
                seed = 0.5 * np.asarray(plan.q, float) + 0.5 * SHARED["left_q_port"]
            q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=plan.base_pos, iters=200)
            SHARED[side + "_ik"] = (round(1000 * pe, 1), round(re, 3), label)
            plan.q = q
            g = float(np.clip(finger / FINGER_OPEN, 0.0, 1.0))
            out.append(np.r_[q, g])
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
