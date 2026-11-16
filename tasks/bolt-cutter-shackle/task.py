"""Facilities maintenance: "Cut the padlock off the cage gate."

A storage cage's gate is locked with a 40 mm brass padlock whose key is lost; a 14 in bolt cutter
lies shut on a tool case in front of it. Both hands take the cutter by its grips, lift it to the lock,
open it and set its jaws square across one shackle leg, and close the handles together, so the head stays
where it is and does not drag the lock or slip off the round leg, until the leg is cut through. The
cutter is then laid back down and the hands stand clear.
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
import cutter as C  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "bolt-cutter-shackle"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 1600
OBJECT_NAMES = ("cutter", "handle_b", "padlock")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = (C.cage_object(), C.case_object(), C.lock_object(), C.cutter_object(), C.handle_b_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_bolt_cutter_shackle", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


SHARED = {}                                          # measured state a scripted demo may steer by


class Dynamics:
    """The cut, from measured poses only (a disclosed stand-in for shearing steel).

    A shackle leg is cut once it has been between the blades, square to them (within 20 degrees),
    with the blades closed down onto it (their gap at the leg no more than its 6 mm plus 0.5 mm: the
    hands squeezing the handles against it) for 0.25 s. The leg's cut zone (the 14 mm the jaws take)
    is then no longer collided with and drawn gone, so the blades close through it.
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        self.stage = omni.usd.get_context().get_stage()
        self.zone = {}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            for sy, k in C.CUT_PIECE.items():
                if path.endswith(f"/padlock/piece{k}"):
                    self.zone[sy] = prim
        if len(self.zone) != 2:
            raise RuntimeError("padlock shackle pieces not found on the stage")

    def _cut_leg(self, sy):
        from pxr import UsdGeom, UsdPhysics
        prim = self.zone[sy]
        for p in [prim, *prim.GetAllChildren(), *[g for c in prim.GetAllChildren() for g in c.GetAllChildren()]]:
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(False)
        UsdGeom.Imageable(prim).MakeInvisible()

    def reset(self):
        from pxr import UsdGeom, UsdPhysics
        self.cut = None
        self.squeeze = 0
        for prim in self.zone.values():
            for p in [prim, *prim.GetAllChildren(), *[g for c in prim.GetAllChildren() for g in c.GetAllChildren()]]:
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Set(True)
            UsdGeom.Imageable(prim).MakeVisible()

    def before_step(self, step, action):
        pass

    def after_step(self, step, objects, root_poses, robot):
        cutter = np.asarray(root_poses["cutter"], float)
        handle_b = np.asarray(root_poses["handle_b"], float)
        lock = np.asarray(root_poses["padlock"], float)
        b = C.bite(cutter, handle_b, lock)
        if self.cut is None and b is not None and b[2] <= C.CUT_GAP:
            self.squeeze += 1
            if self.squeeze >= C.CUT_HOLD_S * HZ:
                self.cut = b[0]
                self._cut_leg(b[0])
        elif self.cut is None:
            self.squeeze = 0
        SHARED["bite"] = b
        SHARED["cutter"], SHARED["handle_b"], SHARED["padlock"] = cutter, handle_b, lock
        for s in ("left", "right"):
            SHARED[f"{s}_tcp"] = np.asarray(robot[s]["tcp"], float)[:3]
        return {"cut": self.cut is not None, "cut_leg": self.cut, "squeeze_frames": self.squeeze,
                "handles_open_deg": round(C.opening_deg(cutter, handle_b), 1)}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ----------------------------------------------------------------------------------------
# Both hands go to the grips on their own plans and take hold. From then on the demo plans the
# *cutter* (its pivot, body A's yaw and how far the handles are open), and each hand's target is that
# cutter pose carried through the hand's grasp on its handle, measured on the real bodies as it closed.
# So the two hands' targets always agree with the pivot between them: neither pulls against the other.
# The pivot target is nudged by the measured pivot error (arm sag under the 0.75 kg tool), as a scripted
# demo may steer by sim state.

REST = {"left": (0.16, 0.24, C.TOP + 0.16), "right": (0.16, -0.24, C.TOP + 0.16)}
SIDE_SIGN = {"right": -1, "left": +1}                 # right hand on handle A, left on handle B
GRIP_FINGER = -(2 * C.GRIP[0] - 0.010) / 2            # squeeze the 28 mm rubber grips


def _tcp(tip, rot, label, seconds, finger):
    tip = np.asarray(tip, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def _inward(yaw, sign):
    """The hand's approach: straight in across the bench toward the cutter's centreline (+y for the
    right hand, -y for the left), which each arm reaches well; it crosses the splayed grip at about
    60 degrees, the pads still clamping it top and bottom."""
    a = math.radians(APPROACH_FWD_DEG)
    return np.array([math.sin(a), -float(sign) * math.cos(a), 0.0])


# the hands take the shut cutter turned this far forward of straight in, so that with the handles
# opened (each turning half the opening outward) they come square in across the bench
APPROACH_FWD_DEG = C.OPEN_DEG / 2


def _grip_rot(yaw, sign):
    """Side grip: the hand comes at the handle from outside, fingers closing top and bottom, so the
    pads clamp the grip along their width and hold the cutter's weight ahead of them."""
    z = _inward(yaw, sign)
    x = np.array([0.0, 0.0, -float(sign)])          # mirror images: each wrist turns its own way
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


DEEP = 0.012                                          # pad tips this far past the grip's axis


def _grip_wp(pivot, a_yaw_deg, open_deg, sign, finger, label, seconds, lift=0.0, up=0.0):
    """Hand on handle A (sign -1, right arm) or B (+1, left arm) at its grip's middle, for the cutter's
    pivot at ``pivot`` lying flat, body A turned ``a_yaw_deg`` and the handles ``open_deg`` open."""
    yaw = C.grip_yaw(sign, open_deg, a_yaw_deg)
    g = np.asarray(pivot, float) + C.HANDLE_L * np.array([math.cos(yaw), math.sin(yaw), 0.0])
    n = _inward(yaw, sign)
    return _tcp(g + n * (DEEP - 0.7 * lift) + np.array([0.0, 0.0, 0.6 * lift + up]), _grip_rot(yaw, sign), label, seconds,
                GRIP_FINGER if finger is None else finger)


CUT_PIVOT = np.array([C.NEAR_LEG[0] - C.JAW_CUT, C.NEAR_LEG[1], C.CUT_Z])
BITE_DEG = math.degrees(math.atan(2 * C.LEG_R / C.JAW_CUT))     # the blades on the 6 mm leg
APPROACH = [                                                      # per hand, before the cutter plan takes over
    ("look", 1.5, None), ("beside its grip", 2.0, 0.05), ("in round it", 1.2, 0.0), ("take it", 0.8, 0.0),
    ("hold it", 0.5, 0.0)]
T_GRASP = sum(d for _, d, _ in APPROACH)
# the cutter plan: (seconds, pivot offset or absolute, body A yaw (deg) or None = half the opening,
# opening (deg), label). A pivot of None keeps the last one.
LAY = np.array([C.PIVOT0[0], C.PIVOT0[1], C.LIE_Z + 0.004])
CUTTER_PLAN = [
    (1.5, ("rel", (-0.030, 0.0, 0.030)), C.REST_OPEN, "lift it off the case"),
    (1.5, ("abs", (0.445, -0.030, C.LIE_Z + 0.050)), C.REST_OPEN, "draw it back, clear of the lock"),
    (2.5, ("abs", CUT_PIVOT - (0.060, 0.0, 0.0)), C.OPEN_DEG, "open the handles wide, jaws level with the leg"),
    (1.0, None, C.OPEN_DEG, "jaws level with the leg, in front of it"),
    (2.0, ("abs", CUT_PIVOT), C.OPEN_DEG, "jaws round the leg, square to it"),
    (1.5, None, BITE_DEG + 1.0, "close both handles together onto it"),
    (1.5, None, BITE_DEG - 2.0, "squeeze: cut it"),
    (0.8, None, 4.0, "through"),
    (1.5, ("abs", CUT_PIVOT - (0.070, 0.0, -0.010)), 4.0, "draw the jaws back off the lock"),
    (2.0, ("abs", LAY + (-0.060, -0.010, 0.040)), C.REST_OPEN, "back over the case, handles shut"),
    (1.5, ("abs", LAY + (0.0, 0.0, 0.004)), C.REST_OPEN, "lower it onto the case"),
    (1.0, ("abs", LAY), C.REST_OPEN, "lay it down on the case"),
]
T_LAID = T_GRASP + sum(p[0] for p in CUTTER_PLAN)
AFTER = [("let go", 0.8), ("draw the open hands back", 1.0), ("stand clear", 1.8), ("done", 2.0)]
CUT_LABELS = ("jaws round the leg, square to it", "close both handles together onto it", "squeeze: cut it", "through")
# phases steered by where the near leg sits in the jaws (lateral: centred in the gap; along: at the
# cutting point), and phases where the cutter is held where it is (squeezing and cutting through)
LEG_STEER = {"jaws round the leg, square to it": True, "close both handles together onto it": False}
HOLD_STEER = ("squeeze: cut it", "through")


def demo():
    """Each arm's plan. Up to the grasp the hands follow these waypoints; after it the DemoController
    places them from the cutter plan (the later waypoints here only carry the timing)."""
    p0, o = C.PIVOT0, C.REST_OPEN
    plans = {}
    for side, sign in SIDE_SIGN.items():
        # the hand starts turned for its side grip, above and outside the grip, where each arm reaches
        wps = [_grip_wp(p0, o / 2, o, sign, FINGER_OPEN, "look", APPROACH[0][1], lift=0.05, up=0.10)]
        for label, dur, lift in APPROACH[1:]:
            finger = FINGER_OPEN if label in ("beside its grip", "in round it") else None
            wps.append(_grip_wp(p0, o / 2, o, sign, finger, label, dur, lift=lift))
        for dur, _, _, label in CUTTER_PLAN:
            wps.append(_grip_wp(p0, o / 2, o, sign, None, label, dur))
        for label, dur in AFTER:
            wps.append(Waypoint(dur, REST[side], FINGER_OPEN, label, DOWN))
        plans[side] = wps
    return plans


def _ease(u):
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def _yaw_of(pose):
    r = C.quat_matrix(np.asarray(pose, float)[3:7])
    return math.atan2(r[1, 0], r[0, 0])


class DemoController:
    """Hands to the grips on their waypoints; then both hands placed from one planned cutter pose."""

    def __init__(self, plans):
        self.plans = plans
        self.q_grasp = {}
        self._seed()
        self.local = None          # each hand's TCP and rotation in its handle's frame (yaw only)
        self.keys = None           # cutter plan as absolute (t_end, pivot, opening)
        self.corr = np.zeros(3)
        self.pitch = 0.0           # head-up turn of both hands about the line between the grips (rad)
        self.yawc = 0.0            # turn of both hands about the pivot, keeping the jaws' centre line on its plan
        self.last = {}
        self.hand_off = {}

    def _seed(self):
        """Start each arm's warm-started IK on the elbow branch that reaches its grasp (from READY it
        can settle on one that cannot), using a multi-seed solve of the grasp pose."""
        from yam_arena import ik
        seeds = ([0, 1, 1, -0.6, 0, 0], [0.5, 1.5, 1.2, -1, 0, 0], [-0.5, 1.5, 1.2, -1, 0, 0], [0, 2, 2, -1.5, 0, 0],
                 [0.8, 1.2, 0.8, -0.5, 1, 0], [-0.8, 1.2, 0.8, -0.5, -1, 0], [0.8, 1.2, 0.8, -0.5, -1, 0],
                 [-0.8, 1.2, 0.8, -0.5, 1, 0])
        for side, plan in self.plans.items():
            wp = next(w for w in plan.waypoints if w.label == "take it") if hasattr(plan, "waypoints") else None
            if wp is None:
                continue
            pos, rot = np.asarray(wp.pos, float), np.asarray(wp.yaw, float)   # the patched Waypoint carries a rotation
            best = None
            for s in seeds:
                q, _, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), np.asarray(s, float), base_pos=plan.base_pos, iters=200)
                p, _ = ik.tcp_pose(q, plan.base_pos)
                e = float(np.linalg.norm(p - pos))
                if best is None or e < best[0]:
                    best = (e, q)
            plan.q = best[1]
            self.q_grasp[side] = best[1].copy()

    def _start(self):
        cut, hb = SHARED["cutter"], SHARED["handle_b"]
        self.local = {}
        for side, body in (("right", cut), ("left", hb)):
            yaw = _yaw_of(body)
            rz = simkit.rot_z(yaw)
            pos, rot = self.last[side]
            self.local[side] = (rz.T @ (pos - body[:3]), rz.T @ rot)
        p0 = np.array(cut[:3], float)
        o0 = C.opening_deg(cut, hb)
        self.yaw0 = math.degrees(_yaw_of(cut)) - o0 / 2      # carried along: body A yaw = this + opening / 2
        keys, t, piv, op = [], T_GRASP, p0, o0
        keys.append((t, piv, op, "hold it"))
        for dur, where, opening, label in CUTTER_PLAN:
            if where is not None:
                piv = p0 + np.asarray(where[1]) if where[0] == "rel" else np.asarray(where[1], float)
            t += dur
            keys.append((t, np.array(piv, float), float(opening), label))
        self.keys = keys

    def _cutter_at(self, t):
        k = self.keys
        for i in range(1, len(k)):
            if t <= k[i][0] or i == len(k) - 1:
                t0, p0, o0, _ = k[i - 1]
                t1, p1, o1, label = k[i]
                u = _ease((t - t0) / max(1e-6, t1 - t0))
                return p0 + (p1 - p0) * u, o0 + (o1 - o0) * u, label

    def action(self, t):
        from yam_arena import ik
        FC, FO = FINGER_CLOSED, FINGER_OPEN
        if t >= T_GRASP and self.keys is None and "cutter" in SHARED:
            self._start()
        out = []
        cutter_pose = self._cutter_at(t) if self.keys is not None and t <= T_LAID + AFTER[0][1] + AFTER[1][1] else None
        if cutter_pose is not None:
            piv, opening, label = cutter_pose
            if t <= T_LAID - 1.0:
                # steer the pivot by its measured error (arm sag), the whole cutter at once, and keep the
                # cutter level: the heavy head turns down in the fingers, so both wrists roll it back up
                err = piv - np.asarray(SHARED["cutter"][:3], float)
                step = 0.12 * err
                if label in LEG_STEER:
                    # across the bench, steer by the near leg's place in the jaws instead
                    a = np.asarray(SHARED["cutter"], float)
                    ra = C.quat_matrix(a[3:7])
                    pt, _ = C.leg_axes(SHARED["padlock"])[-1]
                    q = ra.T @ (pt - a[:3])
                    phi = math.radians(max(0.0, C.opening_deg(a, SHARED["handle_b"])))
                    e_lat = float(q[1]) + float(q[0]) * math.tan(phi) / 2      # off the gap's centre line
                    move = ra[:, 1] * e_lat
                    if LEG_STEER[label]:
                        move = move + ra[:, 0] * (float(q[0]) - C.JAW_CUT)     # leg at the cutting point
                    step[:2] = 0.06 * move[:2]
                elif label in HOLD_STEER:
                    step[:2] = 0.0
                self.corr = np.clip(self.corr + step, -0.08, 0.08)
                r = C.quat_matrix(np.asarray(SHARED["cutter"][3:7], float))
                droop = math.asin(float(np.clip(r[2, 0], -1, 1)))         # < 0: head down
                self.pitch = float(np.clip(self.pitch - 0.10 * droop, -0.05, 0.45))
                # the cutter also turns in the fingers about the vertical: steer its centre line back
                meas_c = _yaw_of(SHARED["cutter"]) - math.radians(C.opening_deg(SHARED["cutter"], SHARED["handle_b"])) / 2
                plan_c = math.radians(self.yaw0)
                if label not in HOLD_STEER:
                    d = (plan_c - meas_c + math.pi) % (2 * math.pi) - math.pi
                    self.yawc = float(np.clip(self.yawc + 0.10 * d, -0.35, 0.35))
            else:
                self.corr *= 0.95
                self.pitch *= 0.95
                self.yawc *= 0.95
            a_yaw = math.radians(self.yaw0 + opening / 2)
        for side in ("left", "right"):
            plan = self.plans[side]
            pos, rot, finger, wlabel = plan.target(t)
            pos = np.array(pos, float)
            if cutter_pose is not None:
                yaw = a_yaw if side == "right" else a_yaw - math.radians(opening)
                # the cutter's planned frame: yawed, and pitched head-up by the droop correction about a
                # horizontal axis square to the cutter (the line between the grips)
                tilt = simkit.rot_z(self.yawc) @ simkit.rot_z(a_yaw - math.radians(opening / 2)) @ simkit.rot_y(-self.pitch) \
                    @ simkit.rot_z(-(a_yaw - math.radians(opening / 2)))
                rz = tilt @ simkit.rot_z(yaw)
                lp, lr = self.local[side]
                pos = piv + self.corr + rz @ lp
                rot = rz @ lr
                finger = GRIP_FINGER
                if t > T_LAID:
                    finger = FO
                    if t > T_LAID + AFTER[0][1]:
                        # the open fingers clear the grip: up a little (the lower finger stays under it) and out
                        back = -(np.asarray(rot) @ np.array([0.0, 0.0, 1.0]))
                        pos = pos + (back * 0.05 + np.array([0.0, 0.0, 0.015])) * _ease((t - T_LAID - AFTER[0][1]) / AFTER[1][1])
            elif self.keys is not None and t > T_LAID:
                # the open hands rise away from the laid-down cutter, from wherever they are
                if side not in self.hand_off:
                    self.hand_off[side] = (self.last[side][0].copy(), t)
                p_from, t_from = self.hand_off[side]
                u = _ease((t - t_from) / AFTER[2][1])
                back = -(np.asarray(self.last[side][1]) @ np.array([0.0, 0.0, 1.0]))
                pos = p_from + (back * 0.02 + np.array([0.0, 0.0, 0.12])) * u        # up and away, clear of the cutter
                rot = self.last[side][1]
                finger = FO
            self.last[side] = (pos.copy(), np.array(rot, float))
            quat = ik.matrix_to_quat_xyzw(rot)
            q, pe, re = ik.solve(pos, quat, plan.q, base_pos=plan.base_pos, iters=60)
            if side in self.q_grasp and float(np.linalg.norm(ik.tcp_pose(q, plan.base_pos)[0] - pos)) > 0.003:
                # the warm start drifted onto a branch that cannot reach: restart from the grasp's branch
                q2, _, _ = ik.solve(pos, quat, self.q_grasp[side], base_pos=plan.base_pos, iters=200)
                if np.linalg.norm(ik.tcp_pose(q2, plan.base_pos)[0] - pos) < np.linalg.norm(ik.tcp_pose(q, plan.base_pos)[0] - pos):
                    q = q2
            plan.q = q
            grip = float(np.clip(finger / FO, 0.0, 1.0)) if FO != FC else 1.0
            out.append(np.r_[q, grip])
        SHARED["demo_label"] = cutter_pose[2] if cutter_pose is not None else None
        return np.concatenate(out)


def make_demo_controller(env, plans):
    return DemoController(plans)
