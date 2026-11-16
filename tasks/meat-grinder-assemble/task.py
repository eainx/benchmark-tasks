"""Food preparation: "Assemble the meat grinder."

A washed #10 hand mincer stands on its suction foot, barrel level, mouth toward the arms. Its parts
lie on a tea towel beside it: auger, four-wing cross knife, 4.5 mm plate and retaining ring. They go
back in order: the auger into the barrel until its flights meet the back wall, the knife onto the
auger's square with its flat cutting face out, the plate over the auger's pin with its rim notch on the
barrel's key, and the ring screwed on over the plate hand-tight (1.5 turns), pressing plate and knife
home. Then the hands stand clear.
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
import grinder as G  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "meat-grinder-assemble"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 2400
OBJECT_NAMES = ("auger", "knife", "plate", "ring")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (G.towel_object(), G.mincer_object(), G.auger_object(), G.knife_object(), G.plate_object(), G.ring_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_meat_grinder_assemble", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                                          # measured state the scripted demo steers by

SEAT_TOL = 0.0035                                    # m: auger centre to its seat (axially and off the axis)
ON_TOL = 0.003                                       # m: knife/plate/ring centre off the barrel's axis
TILT_TOL = 8.0                                       # deg: a part's axis off the barrel's
RING_TOL = 0.003                                     # m: ring axially from its place on the thread
TURNS_TIGHT = 1.5                                    # hand-tight
HOLD_D = 0.05                                        # N m s / deg: a started ring left alone, held by its thread
LOCK_D = 2.0                                         # N m s / deg: tight (no position target, nothing snaps)
HOME_HOLD = 0.001                                    # m: knife/plate this close to home are held there ...
HOME_D = 1.0e5                                       # N s / m: ... by a stiff damping drive on their slide


class Dynamics:
    """What rigid bodies cannot do by themselves, from measured poses only (a disclosed stand-in).

    Each fit is a joint made at spawn, switched off, and switched on when the part is put to its
    place within tolerance; none has a position target, so nothing is pulled into place beyond the
    joint's own few millimetres of tolerance.
    * Auger: seated once its centre is within 3 mm of its seat (flights against the back wall) and of
      the axis, axis within 8 degrees: it then turns freely in its bearings (damped).
    * Knife: on the auger's square once its hub is within 3 mm of the axis, anywhere along the square
      and pin, axis within 8 degrees, the auger seated and the plate not yet on. It then slides along
      and turns with the square (damped); flat face out or flat face in, whichever way it went on.
    * Plate: on the pin once within 3 mm of the axis, along the pin, axis within 8 degrees and its rim
      notch within 8 degrees of the key at 12 o'clock, with the auger seated. It then slides along the
      pin, its notch held on the key (8 degrees play).
    * Ring: started on the thread once within 3 mm of its place and of the axis, its lip out (within 8
      degrees). It then turns about the axis; its drag is a damping drive, it stays where it is left
      (a holding drive while no hand grips it) and at 1.5 turns it is tight (held stiff). Its lip
      pushes plate and knife home; the thread's travel (4 mm over 1.5 turns) is not modelled.
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        want = {"seat": "/auger/seat_joint", "knife": "/knife/shaft_joint", "knife_rev": "/knife/shaft_joint_rev",
                "plate": "/plate/pin_joint", "plate_rev": "/plate/pin_joint_rev", "ring": "/ring/thread_joint"}
        self.j = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for k, suffix in want.items():
                if path.endswith(suffix):
                    self.j[k] = prim
        missing = set(want) - set(self.j)
        if missing:
            raise RuntimeError(f"fit joints not found on the stage: {sorted(missing)}")

    def _on(self, k, on=True):
        self.j[k].GetAttribute("physics:jointEnabled").Set(bool(on))

    def _ring_damp(self, d):
        self.j["ring"].GetAttribute("drive:rotX:physics:damping").Set(float(d))

    def reset(self):
        for k in self.j:
            self._on(k, False)
        for k in ("knife", "knife_rev", "plate", "plate_rev"):
            self.j[k].GetAttribute("drive:transX:physics:damping").Set(G.SLIDE_D)
        self.auger_on = self.knife_on = self.plate_on = self.ring_on = False
        self.knife_backwards = False
        self.knife_key = self.plate_key = None
        self.knife_held = self.plate_held = False
        self.plate_before_knife = False
        self.order = []
        self.ring_prev = None
        self.ring_turns = 0.0
        self.tight = False
        self.hand_open = True
        self._ring_damp(G.THREAD_D)

    def before_step(self, step, action):
        a = np.asarray(action, float).ravel()
        self.hand_open = a.size >= 14 and float(a[13]) > 0.9

    def after_step(self, step, objects, root_poses, robot):
        P = {k: np.asarray(root_poses[k], float) for k in OBJECT_NAMES}
        SHARED.update({k: P[k].copy() for k in OBJECT_NAMES})
        for s in ("left", "right"):
            SHARED[f"{s}_tcp"] = np.asarray(robot[s]["tcp"], float)[:3]
        # -- auger into its seat
        a_ax, a_rad = G.axial_radial(P["auger"], G.AUGER_HOME)
        a_tilt = G.tilt_deg(P["auger"], (1, 0, 0), G.AXIS)
        if not self.auger_on and abs(a_ax) < SEAT_TOL and a_rad < SEAT_TOL and a_tilt < TILT_TOL:
            self._on("seat")
            self.auger_on = True
            self.order.append("auger")
        # -- knife onto the square
        k_ax, k_rad = G.axial_radial(P["knife"], G.KNIFE_HOME)
        k_out = G.tilt_deg(P["knife"], (0, 0, 1), -G.AXIS)           # flat face out of the mouth
        k_in = G.tilt_deg(P["knife"], (0, 0, 1), G.AXIS)
        if (not self.knife_on and self.auger_on and not self.plate_on and k_rad < ON_TOL
                and -G.KNIFE_SLIDE - 0.001 <= k_ax <= 0.001 and min(k_out, k_in) < TILT_TOL):
            self.knife_backwards = k_in < k_out
            self.knife_key = "knife_rev" if self.knife_backwards else "knife"
            self._on(self.knife_key)
            self.knife_on = True
            self.order.append("knife")
        # -- plate over the pin, notch on the key
        p_ax, p_rad = G.axial_radial(P["plate"], G.PLATE_HOME)
        p_out = G.tilt_deg(P["plate"], (0, 0, 1), -G.AXIS)
        p_in = G.tilt_deg(P["plate"], (0, 0, 1), G.AXIS)
        notch = G.notch_deg(P["plate"])
        if (not self.plate_on and self.auger_on and not self.ring_on and p_rad < ON_TOL
                and -G.PLATE_SLIDE - 0.001 <= p_ax <= 0.001 and min(p_out, p_in) < TILT_TOL and abs(notch) < G.NOTCH_DEG):
            self.plate_key = "plate_rev" if p_in < p_out else "plate"
            self._on(self.plate_key)
            self.plate_on = True
            self.plate_before_knife = not self.knife_on
            self.order.append("plate")
        # -- pressed home: held there (the square's and the pin's fit; a damping drive, no position target)
        if self.knife_on and not self.knife_held and k_ax >= -HOME_HOLD:
            self.j[self.knife_key].GetAttribute("drive:transX:physics:damping").Set(HOME_D)
            self.knife_held = True
        if self.plate_on and not self.plate_held and p_ax >= -HOME_HOLD:
            self.j[self.plate_key].GetAttribute("drive:transX:physics:damping").Set(HOME_D)
            self.plate_held = True
        # -- ring onto the thread, and turned
        r_ax, r_rad = G.axial_radial(P["ring"], G.RING_HOME)
        r_tilt = G.tilt_deg(P["ring"], (0, 0, 1), -G.AXIS)
        if not self.ring_on and abs(r_ax) < RING_TOL and r_rad < ON_TOL and r_tilt < TILT_TOL:
            self._on("ring")
            self.ring_on = True
            self.order.append("ring")
            self.ring_prev = G.twist(P["ring"])
            self.ring_turns = 0.0
        elif self.ring_on:
            tw = G.twist(P["ring"])
            d = (tw - self.ring_prev + math.pi) % (2 * math.pi) - math.pi
            self.ring_prev = tw
            self.ring_turns += d / (2 * math.pi)
            if not self.tight and self.ring_turns >= TURNS_TIGHT:
                self.tight = True
                self._ring_damp(LOCK_D)
            elif not self.tight:
                self._ring_damp(HOLD_D if self.hand_open else G.THREAD_D)
        SHARED.update(auger_on=self.auger_on, knife_on=self.knife_on, plate_on=self.plate_on, ring_on=self.ring_on,
                      tight=self.tight, ring_turns=self.ring_turns)
        return {"auger_seated": self.auger_on, "knife_on": self.knife_on, "knife_backwards": self.knife_backwards,
                "plate_on": self.plate_on, "plate_before_knife": self.plate_before_knife, "ring_on": self.ring_on,
                "ring_turns": round(self.ring_turns, 2), "tight": self.tight,
                "auger_off_mm": round(1000 * max(abs(a_ax), a_rad), 1), "knife_axial_mm": round(1000 * k_ax, 1),
                "knife_radial_mm": round(1000 * k_rad, 1), "plate_axial_mm": round(1000 * p_ax, 1),
                "plate_radial_mm": round(1000 * p_rad, 1), "notch_deg": round(notch, 1),
                "ring_axial_mm": round(1000 * r_ax, 1), "ring_radial_mm": round(1000 * r_rad, 1),
                "order": ",".join(self.order), "ik_right": SHARED.get("right_ik")}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------------
# The right hand does the assembling; the left stays at its ready pose. Every placement plans the
# *part*: the hand's target is the part's planned pose carried through the hand's grasp on it as
# measured once the fingers have closed, nudged by the measured placement error (a scripted demo may
# steer by sim state).

REST = {"left": (0.16, 0.24, G.TOP + 0.20), "right": (0.16, -0.24, G.TOP + 0.20)}


def _frame(approach, up):
    z = np.asarray(approach, float)
    z = z / np.linalg.norm(z)
    u = np.asarray(up, float)
    x = u - z * float(np.dot(u, z))
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def FWD(roll_deg=0.0):
    """Hand level, pointing +x into the mouth; the fingers close along (0, sin r, cos r)."""
    r = math.radians(roll_deg)
    return _frame((1.0, 0.0, 0.0), (0.0, math.sin(r), math.cos(r)))


DOWN_X = _frame((0.0, 0.0, -1.0), (1.0, 0.0, 0.0))   # hand pointing down, fingers closing along x


def _between(a, b):
    """The smallest rotation taking unit vector a onto b."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _finger(width, squeeze=0.004):
    """Finger target to close on something ``width`` across, squeezing it ``squeeze``."""
    return float(np.clip(-width / 2 + squeeze, FINGER_OPEN, FINGER_CLOSED))


def axis_point(x):
    return np.array([x, G.AXIS_Y, G.AXIS_Z])


OPEN = FINGER_OPEN
F_AUGER = _finger(2 * G.FLIGHT_R)                     # across the flights
F_PRE_AUGER = -0.036
F_KNIFE = _finger(2 * G.KNIFE_R)                      # across two wing tips
F_PLATE = _finger(2 * G.PLATE_R)
F_RING = _finger(2 * G.RING_R + 0.002)                # across the knurled body
F_PUSH = -0.007                                       # fingertips 14 mm apart: either side of the pin
STROKE = 175.0                                        # deg of wrist roll per stroke on the ring
ROLL_HI, ROLL_LO = 10.0, 10.0 - STROKE


def demo():
    """Arm plans for the runtime (it builds IK plans from these); the controller below drives the arms."""
    return {sd: [Waypoint(1.0, REST[sd], FINGER_OPEN, "ready", DOWN),
                 Waypoint(float(MAX_STEPS) / HZ, REST[sd], FINGER_OPEN, "ready", DOWN)] for sd in ("left", "right")}


class Phase:
    def __init__(self, label, dur, goal, until=None, timeout=4.0, on_end=None, live=False):
        self.label, self.dur, self.goal, self.until, self.timeout, self.on_end, self.live = \
            label, dur, goal, until, timeout, on_end, live


class DemoController:
    def __init__(self, plans):
        self.plans = plans
        self.grasp = None                 # (p_rel, R_rel): the held part in the hand's frame
        self.held = None
        self.corr = np.zeros(3)
        self.phases = self._phases()
        self.i = 0
        self.t_start = 0.0
        self.start = None
        self.cmd = None                   # last commanded (pos, rot, finger)
        self.goal_cache = None
        self.left_q = None

    # -- grasp bookkeeping
    def _record(self, part):
        p_obj = np.asarray(SHARED[part][:3], float)
        R_obj = G.mat(SHARED[part])
        p_tcp = np.asarray(SHARED["right_tcp"], float)
        R_tcp = self.cmd[1]
        self.grasp = (R_tcp.T @ (p_obj - p_tcp), R_tcp.T @ R_obj)
        self.held = part
        self.corr = np.zeros(3)

    def _regrasp(self, part, alpha=0.25):
        """Re-measure the held part in the hand (it slips and tilts in the fingers), low-passed."""
        if self.grasp is None or part not in SHARED:
            return
        p_obj = np.asarray(SHARED[part][:3], float)
        R_obj = G.mat(SHARED[part])
        p_tcp = np.asarray(SHARED["right_tcp"], float)
        R_tcp = self.cmd[1]
        p_new, R_new = R_tcp.T @ (p_obj - p_tcp), R_tcp.T @ R_obj
        p_rel, R_rel = self.grasp
        p = (1 - alpha) * p_rel + alpha * p_new
        U, _, Vt = np.linalg.svd((1 - alpha) * R_rel + alpha * R_new)
        self.grasp = (p, U @ Vt)

    def _hand_for(self, p_obj, R_obj):
        p_rel, R_rel = self.grasp
        R_tcp = R_obj @ R_rel.T
        return p_obj - R_tcp @ p_rel, R_tcp

    def _carry(self, part, p_obj, axis_local, axis_world, roll, align_extra=None):
        """Hand pose that carries the held part to centre ``p_obj`` with its ``axis_local`` along
        ``axis_world``, the hand level at ``roll`` (corrected by the smallest turn that lines the axis up)."""
        p_rel, R_rel = self.grasp
        R0 = FWD(roll)
        pred = R0 @ R_rel @ np.asarray(axis_local, float)
        Rc = _between(pred, axis_world)
        R_tcp = Rc @ R0
        if align_extra is not None:
            R_tcp = align_extra(R_tcp) @ R_tcp
        return p_obj - R_tcp @ p_rel, R_tcp

    def _steer(self, part, target, gain=0.04, axes=(1, 2), lim=0.010):
        if part not in SHARED:
            return self.corr
        err = np.asarray(target, float) - np.asarray(SHARED[part][:3], float)
        for k in axes:
            self.corr[k] = float(np.clip(self.corr[k] + gain * err[k], -lim, lim))
        return self.corr

    # -- the plan
    def _phases(self):
        ph = []
        A = lambda: np.asarray(SHARED["auger"], float)   # noqa: E731
        Kp = lambda: np.asarray(SHARED["knife"], float)  # noqa: E731
        Pp = lambda: np.asarray(SHARED["plate"], float)  # noqa: E731
        Rp = lambda: np.asarray(SHARED["ring"], float)   # noqa: E731

        # ---------- auger: from above across its flights (a diagonal pinch); carried level into the
        # mouth until the fingers near it; then pushed home by its pin with the closed fingertips
        DIAG = _frame((0.0, 0.0, -1.0), (0.7071, 0.7071, 0.0))

        def auger_over(dz):
            a = A()
            return np.array([a[0] - AUGER_GRIP_BACK, a[1], a[2] - 0.010 + PAD_TIP + dz])

        def pin_tip():
            a = A()
            return a[:3] + G.mat(a) @ np.array([-(G.FLIGHT_L / 2 + G.FRONT_SQ[0] + G.PIN[0]), 0.0, 0.0])

        def auger_carry(c, x, steer):
            if steer:
                c._regrasp("auger")
            p = axis_point(x) + (c._steer("auger", axis_point(0.0), gain=0.10, lim=0.015) if steer else 0.0)
            p_rel, R_rel = c.grasp
            pred = DIAG @ R_rel @ np.array([1.0, 0.0, 0.0])
            R = _between(pred, G.AXIS) @ DIAG
            return p - R @ p_rel, R

        def push_pin(c, x_tip):
            t = pin_tip()
            c.pin_corr = getattr(c, "pin_corr", np.zeros(3))
            return np.array([x_tip - PAD_TIP, t[1], t[2]]), FWD(0)

        AUGER_GRIP_BACK = 0.045                         # grip this far ahead of its centre, so it can go in past its balance
        ph += [Phase("look", 1.0, lambda c: (c.cmd[0], c.cmd[1], OPEN)),
               Phase("over the auger", 2.5, lambda c: (auger_over(0.06), DIAG, F_PRE_AUGER)),
               Phase("down round its flights", 1.5, lambda c: (auger_over(0.0), DIAG, F_PRE_AUGER)),
               Phase("grip the auger", 0.8, lambda c: (c.cmd[0], c.cmd[1], F_AUGER), on_end=lambda c: c._record("auger")),
               Phase("lift it", 1.2, lambda c: (c.cmd[0] + (0.0, 0.0, 0.06), c.cmd[1], F_AUGER)),
               Phase("in line with the barrel", 3.0, lambda c: (*auger_carry(c, 0.200, False), F_AUGER)),
               Phase("square to the mouth", 1.5, lambda c: (*auger_carry(c, 0.215, True), F_AUGER), live=True,
                     until=lambda c: G.axial_radial(SHARED["auger"], G.AUGER_HOME)[1] < 0.0012, timeout=3.0),
               Phase("feed it in", 3.5, lambda c: (*auger_carry(c, 0.357, True), F_AUGER), live=True),
               Phase("let go of the auger", 0.6, lambda c: (c.cmd[0], c.cmd[1], OPEN)),
               Phase("up off it", 1.0, lambda c: (c.cmd[0] + (-0.03, 0.0, 0.06), c.cmd[1], OPEN)),
               Phase("in front of its pin", 2.0, lambda c: (*push_pin(c, pin_tip()[0] - 0.030), OPEN)),
               Phase("fingers over the pin", 1.2, lambda c: (*push_pin(c, pin_tip()[0] + 0.019), OPEN)),
               Phase("grip its square", 0.8, lambda c: (c.cmd[0], c.cmd[1], _finger(G.FRONT_SQ[1])),
                     on_end=lambda c: c._record("auger")),
               Phase("push it home", 4.0, lambda c: (*c._carry("auger", axis_point(G.AUGER_HOME[0] - 0.010)
                                                              + c._steer("auger", axis_point(0.0), gain=0.08), (1, 0, 0),
                                                              G.AXIS, 0.0), _finger(G.FRONT_SQ[1])), live=True,
                     until=lambda c: SHARED.get("auger_on"), timeout=0.5),
               Phase("let go of its square", 0.6, lambda c: (c.cmd[0], c.cmd[1], OPEN),
                     until=lambda c: True),
               Phase("back to its pin, fingers closed", 1.2, lambda c: (*push_pin(c, pin_tip()[0] - 0.010), FINGER_CLOSED)),
               Phase("push it onto its seat", 2.5,
                     lambda c: (*push_pin(c, G.AUGER_HOME[0] - (G.FLIGHT_L / 2 + G.FRONT_SQ[0] + G.PIN[0]) + 0.001),
                                FINGER_CLOSED), until=lambda c: SHARED.get("auger_on"), timeout=2.5),
               Phase("back out", 1.5, lambda c: (c.cmd[0] - (0.07, 0.0, 0.0), FWD(0), OPEN))]

        # ---------- knife: from above by two wing tips; turned to face the mouth; onto the pin and square
        def knife_over(dz):
            k = Kp()
            return np.array([k[0], k[1], k[2] + 0.001 + PAD_TIP - 0.0005 + dz])

        ph += [Phase("over the knife", 2.5, lambda c: (knife_over(0.06), DOWN_X, -0.032)),
               Phase("down round its wing tips", 1.5, lambda c: (knife_over(0.0), DOWN_X, -0.032)),
               Phase("grip the knife", 0.8, lambda c: (c.cmd[0], c.cmd[1], F_KNIFE), on_end=lambda c: c._record("knife")),
               Phase("lift it", 1.2, lambda c: (c.cmd[0] + (0.0, 0.0, 0.06), c.cmd[1], F_KNIFE)),
               Phase("face it to the mouth", 2.5,
                     lambda c: (*c._carry("knife", axis_point(0.27), (0, 0, 1), -G.AXIS, 0.0), F_KNIFE)),
               Phase("in line with the pin", 1.5,
                     lambda c: (*c._carry("knife", axis_point(0.315) + c._steer("knife", axis_point(0.0)), (0, 0, 1), -G.AXIS, 0.0),
                                F_KNIFE), live=True),
               Phase("onto the pin", 2.0,
                     lambda c: (*c._carry("knife", axis_point(G.KNIFE_HOME[0] - 0.016) + c._steer("knife", axis_point(0.0)),
                                          (0, 0, 1), -G.AXIS, 0.0), F_KNIFE), live=True,
                     until=lambda c: SHARED.get("knife_on"), timeout=3.0),
               Phase("let go of the knife", 0.6, lambda c: (c.cmd[0], c.cmd[1], OPEN)),
               Phase("back off it", 1.2, lambda c: (c.cmd[0] - (0.05, 0.0, 0.0), FWD(0), OPEN)),
               Phase("close to push", 0.6, lambda c: (axis_point(0.29), FWD(0), F_PUSH)),
               Phase("push the knife home", 2.0,
                     lambda c: (axis_point(G.KNIFE_HOME[0] - G.KNIFE_HUB[1] / 2 - PAD_TIP + 0.001), FWD(0), F_PUSH)),
               Phase("back off", 1.0, lambda c: (axis_point(0.27), FWD(0), OPEN))]

        # ---------- plate: from above by its rim; turned to face the mouth, notch up; onto the pin
        def plate_over(dz):
            p = Pp()
            return np.array([p[0], p[1], p[2] - G.PLATE_T / 2 + 0.0012 + PAD_TIP + dz])

        def notch_up(c):
            """Pick the wrist roll that brings the plate's notch to 12 o'clock."""
            best = None
            for r in [float(v) for v in range(-120, 21, 2)] + [float(v) for v in range(160, 301, 2)]:
                _, R = c._carry("plate", np.zeros(3), (0, 0, 1), -G.AXIS, r)
                n = R @ c.grasp[1] @ np.array([0.0, 1.0, 0.0])
                e = math.degrees(math.atan2(-n[1], n[2]))
                if best is None or abs(e) < abs(best[1]):
                    best = (r, e)
            c.plate_roll = best[0]
            return best[0]

        def plate_pose(c, x, steer):
            roll = getattr(c, "plate_roll", None)
            if roll is None:
                roll = notch_up(c)
            p = axis_point(x) + (c._steer("plate", axis_point(0.0)) if steer else 0.0)
            return c._carry("plate", p, (0, 0, 1), -G.AXIS, roll)

        ph += [Phase("over the plate", 2.5, lambda c: (plate_over(0.06), DOWN_X, -0.036)),
               Phase("down round its rim", 1.5, lambda c: (plate_over(0.0), DOWN_X, -0.036)),
               Phase("grip the plate", 0.8, lambda c: (c.cmd[0], c.cmd[1], F_PLATE), on_end=lambda c: c._record("plate")),
               Phase("lift it", 1.2, lambda c: (c.cmd[0] + (0.0, 0.0, 0.06), c.cmd[1], F_PLATE)),
               Phase("face it to the mouth, notch up", 2.5, lambda c: (*plate_pose(c, 0.27, False), F_PLATE)),
               Phase("in line with the pin", 1.5, lambda c: (*plate_pose(c, 0.315, True), F_PLATE), live=True),
               Phase("over the pin", 2.0, lambda c: (*plate_pose(c, G.PLATE_HOME[0] - 0.0075, True), F_PLATE), live=True,
                     until=lambda c: SHARED.get("plate_on"), timeout=3.0),
               Phase("let go of the plate", 0.6, lambda c: (c.cmd[0], c.cmd[1], OPEN)),
               Phase("back off the plate", 1.2, lambda c: (c.cmd[0] - (0.05, 0.0, 0.0), FWD(0), OPEN)),
               Phase("close to push it", 0.6, lambda c: (axis_point(0.29), FWD(0), F_PUSH)),
               Phase("push the plate flush", 2.0,
                     lambda c: (axis_point(G.PLATE_HOME[0] - G.PLATE_T / 2 - PAD_TIP + 0.001), FWD(0), F_PUSH)),
               Phase("off the plate", 1.0, lambda c: (axis_point(0.27), FWD(0), OPEN))]

        # ---------- ring: from above round its body; turned lip out; onto the thread; screwed up in strokes
        def ring_over(dz):
            r = Rp()
            return np.array([r[0], r[1], r[2] - (G.RING_DEPTH - G.RING_LIP / 2) + 0.007 + PAD_TIP + dz])

        def ring_pose(c, x, roll, steer=True):
            p = axis_point(x) + (c._steer("ring", axis_point(0.0)) if steer else 0.0)
            return c._carry("ring", p, (0, 0, 1), -G.AXIS, roll)

        ph += [Phase("over the ring", 2.5, lambda c: (ring_over(0.06), DOWN_X, -0.044)),
               Phase("down round its body", 1.5, lambda c: (ring_over(0.0), DOWN_X, -0.044)),
               Phase("grip the ring", 0.8, lambda c: (c.cmd[0], c.cmd[1], F_RING), on_end=lambda c: c._record("ring")),
               Phase("lift it", 1.2, lambda c: (c.cmd[0] + (0.0, 0.0, 0.07), c.cmd[1], F_RING)),
               Phase("face it to the mouth, lip out", 2.5, lambda c: (*ring_pose(c, 0.27, ROLL_HI, False), F_RING)),
               Phase("in line with the thread", 1.5, lambda c: (*ring_pose(c, 0.305, ROLL_HI), F_RING), live=True),
               Phase("onto the thread", 2.5, lambda c: (*ring_pose(c, G.RING_HOME[0] + 0.0005, ROLL_HI), F_RING), live=True,
                     until=lambda c: SHARED.get("ring_on"), timeout=3.0)]

        def stroke(c, roll):
            """The hand turned to ``roll`` about the ring's axis, the ring's centre staying on its thread."""
            return c._carry("ring", G.RING_HOME, (0, 0, 1), -G.AXIS, roll)

        for k in range(5):
            ph += [Phase("screw it on", 2.5, lambda c: (*stroke(c, ROLL_LO), F_RING), until=None),
                   Phase("let go", 0.5, lambda c: (*stroke(c, ROLL_LO), OPEN)),
                   Phase("turn the open hand back", 1.6, lambda c: (*stroke(c, ROLL_HI), OPEN)),
                   Phase("grip again", 0.6, lambda c: (*stroke(c, ROLL_HI), F_RING), on_end=lambda c: c._record("ring"))]
        ph += [Phase("let go of the ring", 0.6, lambda c: (c.cmd[0], c.cmd[1], OPEN)),
               Phase("back off the ring", 1.5, lambda c: (c.cmd[0] - (0.10, 0.0, 0.0), c.cmd[1], OPEN)),
               Phase("stand clear", 2.0, lambda c: (np.array(REST["right"]), DOWN, OPEN)),
               Phase("hold clear", 5.0, lambda c: (np.array(REST["right"]), DOWN, OPEN))]
        return ph

    def _advance(self, t):
        self.i += 1
        self.t_start = t
        self.start = self.cmd
        self.goal_cache = None

    def action(self, t):
        from yam_arena import ik
        plan_r, plan_l = self.plans["right"], self.plans["left"]
        if self.cmd is None:
            p, R = ik.tcp_pose(plan_r.q, plan_r.base_pos)
            self.cmd = (np.asarray(p, float), np.asarray(R, float), OPEN)
            self.start = self.cmd
            self.t_start = t
        # tight: no more strokes -- straight on to letting go, the wrist where it is
        if SHARED.get("tight") and self.i < len(self.phases) and self.phases[self.i].label in (
                "screw it on", "let go", "turn the open hand back", "grip again"):
            while self.phases[self.i].label != "let go of the ring":
                self.i += 1
            self.t_start, self.start, self.goal_cache = t, self.cmd, None
        if self.i >= len(self.phases):
            pos, rot, fin = self.cmd
        else:
            ph = self.phases[self.i]
            u = t - self.t_start
            if ph.live or self.goal_cache is None:
                self.goal_cache = ph.goal(self)
            gp, gR, gf = self.goal_cache
            s = ik.smoothstep(min(1.0, u / ph.dur))
            pos = self.start[0] + s * (np.asarray(gp, float) - self.start[0])
            rot = ik.slerp_matrix(self.start[1], np.asarray(gR, float), s)
            fin = self.start[2] + s * (gf - self.start[2])
            self.cmd = (pos, rot, fin)
            done = u >= ph.dur and (ph.until is None or ph.until(self) or u >= ph.dur + ph.timeout)
            if done:
                if ph.on_end is not None:
                    ph.on_end(self)
                self._advance(t)
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), plan_r.q, base_pos=plan_r.base_pos, iters=120)
        plan_r.q = q
        label = self.phases[min(self.i, len(self.phases) - 1)].label
        SHARED["right_ik"] = (round(1000 * pe, 1), round(math.degrees(re), 1), label)
        pl, Rl, fl, _ = plan_l.target(t)
        ql, _, _ = ik.solve(np.asarray(pl, float), ik.matrix_to_quat_xyzw(Rl), plan_l.q, base_pos=plan_l.base_pos, iters=60)
        plan_l.q = ql
        g = lambda f: float(np.clip(f / FINGER_OPEN, 0.0, 1.0))   # noqa: E731
        return np.concatenate([np.r_[ql, g(fl)], np.r_[q, g(fin)]])


def make_demo_controller(env, plans):
    return DemoController(plans)
