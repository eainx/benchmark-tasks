"""Pack bench: make up a shipping carton's bottom.

An 8 x 6 x 2 in regular slotted carton, opened into a tube and stood upside down, racks back toward
flat on its two knocked-down creases unless it is held. One hand squares it and keeps it square while
the other folds the near short flap in; the folded short flaps, exactly as wide as the inside of the
tube, then hold it square themselves. The far short flap goes in, the two long flaps over both short
flaps, and the seam is taped with a hand tape gun, the long flaps held flat at the far end, the tape
run on past the far edge, torn on the blade and both hanging ends smoothed down the end walls.
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
import parameters as B  # noqa: E402
from scoring import TrialScorer  # noqa: E402

from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP as PAD_TIP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

NAME = "carton-bottom-square-tape"
INSTRUCTION = json.loads((_here / "task.json").read_text())["instruction"]
HZ = 20
MAX_STEPS = 2400
OBJECT_NAMES = B.WALLS + B.FLAPS + ("gun", "roller")

DOWN = simkit.patch_demo()


def make_environment(embodiment, device, control_hz=HZ):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    simkit.install()
    from yam_arena import scene as yscene

    objects = yscene.task_room("station", embodiment.rig.top_cam)
    specs = tuple(B.wall_objects()) + tuple(B.flap_objects()) + (B.gun_object(), B.roller_object())
    objects += [yscene.task_object(s) for s in specs]
    cfg = IsaacLabArenaEnvironment(name="yam_carton_bottom_square_tape", scene=Scene(objects), embodiment=embodiment,
                                   env_cfg_callback=yscene.env_cfg_callback(control_hz))
    return ArenaEnvBuilder(cfg, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device=device)).make_registered()


class Dynamics:
    """Packing tape, from measured poses only (the accepted carton-sealing task's tape model).

    The strip is 8 mm cells along the seam and down both end walls. The gun dispenses tape as its roller
    moves along the strip; a cell is stuck where the roller pressed it on (axle within 3 mm of its working
    height on the long flaps). Cells dispensed with the roller lifted bridge, and free ends hang: both are
    drawn 4 mm off the board. A hanging cell on an end wall sticks when a fingertip is on it and the cell
    between it and the edge is already down. The gun carries 75 mm of free tape past the roller: set down
    within 30 mm of the near edge, it hangs down the near end. Run on 8 mm past the far edge and lifted
    12 mm, the tape tears on the blade (a position rule), leaving a tail hanging down the far end. Once
    three cells are stuck the tape holds the flaps down: their crease springs are set to folded. The tube's
    squareness is never set by this model: it comes only from the walls, flaps and hands in contact.
    """

    def __init__(self, env):
        import omni.usd
        self.base = env.unwrapped
        stage = omni.usd.get_context().get_stage()
        self.cells = [None] * B.CELLS
        self.flap_target = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for k in range(B.CELLS):
                if path.endswith(f"/carton/piece{B.TAPE_PIECE0 + k}"):
                    self.cells[k] = prim
            for n in B.FLAPS:
                if path.endswith(f"/{n}/hinge_joint"):
                    self.flap_target[n] = prim.GetAttribute("drive:angular:physics:targetPosition")
        if any(c is None for c in self.cells) or len(self.flap_target) != len(B.FLAPS):
            raise RuntimeError("tape cells or flap creases not found on the stage")

    def _show(self, k, offset, hanging=False):
        from pxr import Gf, UsdGeom
        prim = self.cells[k]
        local, where = B.cell_local(k, offset)
        if hanging:
            local = local + {"near": np.array([-0.004, 0.0, 0.0]), "far": np.array([0.004, 0.0, 0.0]),
                             "top": np.array([0.0, 0.0, 0.004])}[where]
        ops = [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        attr = ops[0].GetAttr()
        attr.Set(Gf.Vec3f(*map(float, local)) if attr.GetTypeName() == "float3" else Gf.Vec3d(*map(float, local)))
        UsdGeom.Imageable(prim).MakeVisible()

    def reset(self):
        from pxr import UsdGeom
        for prim in self.cells:
            UsdGeom.Imageable(prim).MakeInvisible()
        self.state = ["none"] * B.CELLS
        self.offset = [0.0] * B.CELLS
        self.gap = [0.0] * B.CELLS
        self.rack_at = [0.0] * B.CELLS
        self.started = False
        self.cut = False
        self.head = None
        self.held_down = False
        self.set = {n: 0.0 for n in B.FLAPS}          # each crease's current set (fold angle it rests at)
        for n in B.FLAPS:
            self.flap_target[n].Set(0.0)

    def before_step(self, step, action):
        pass

    def _hang(self, k, offset):
        if 0 <= k < B.CELLS and self.state[k] == "none":
            self.state[k] = "hanging"
            self.offset[k] = offset
            self._show(k, offset, hanging=True)

    def _stick(self, k, offset, gap, rack):
        if 0 <= k < B.CELLS and self.state[k] != "stuck":
            self.state[k] = "stuck"
            self.offset[k] = offset
            self.gap[k] = gap
            self.rack_at[k] = rack
            self._show(k, offset)

    def after_step(self, step, objects, root_poses, robot):
        near = B.seam_frame(root_poses)
        gun = np.asarray(root_poses["gun"], float)
        roller, blade, fwd, up = B.gun_points(gun)
        gap = max(90.0 - B.flap_fold(n, root_poses[n]) for n in B.MAJORS)
        rack = B.rack_deg(root_poses)
        contact = B.roller_contact(near, roller)
        if contact is not None and not self.cut:
            s, off = contact
            if not self.started and 0.0 <= s <= 0.03:
                self.started = True
                self.head = s
                for k in range(B.cell_of(s - B.LEADER), B.cell_of(s) + 1):
                    if 0 <= k < B.CELLS:
                        if B.cell_s(k) >= 0:
                            self._stick(k, off, gap, rack)
                        else:
                            self._hang(k, off)
            if self.started:
                lo, hi = sorted((self.head, s))
                for k in range(B.cell_of(lo), B.cell_of(hi) + 1):
                    self._hang(k, off)
                self._stick(B.cell_of(s), off, gap, rack)
                self.head = max(self.head, s)
        if self.started and not self.cut:
            bite = B.blade_cut(near, roller, blade, fwd)
            if bite is not None and bite > self.head - 0.004:
                self.cut = True
                for k in range(B.cell_of(self.head), B.cell_of(bite) + 1):
                    self._hang(k, self.offset[B.cell_of(self.head)] if 0 <= B.cell_of(self.head) < B.CELLS else 0.0)
        for side in ("left", "right"):
            tcp = np.asarray(robot[side]["tcp"], float)[:3]
            s = B.hand_press(near, tcp - np.array([0.0, 0.0, PAD_TIP]))
            if s is None:
                continue
            k = B.cell_of(s)
            if not (0 <= k < B.CELLS) or self.state[k] != "hanging":
                continue
            if s < 0:
                inner = k + 1
                ok = inner == B.TAB_CELLS or self.state[inner] == "stuck"
            else:
                inner = k - 1
                ok = inner == B.TAB_CELLS + B.TOP_CELLS - 1 or self.state[inner] == "stuck"
            if ok:
                self._stick(k, self.offset[k], 0.0, rack)
        if not self.held_down:
            # the creases take a set: a flap pushed to fold a rests, let go, at a - SPRING_BACK
            for n in B.FLAPS:
                back = B.SPRING_BACK["major" if n in B.MAJORS else "minor"]
                fold = B.flap_fold(n, root_poses[n])
                if fold - back > self.set[n] + 1.0:
                    self.set[n] = fold - back
                    self.flap_target[n].Set(float(B.flap_sign(n) * self.set[n]))
        top = [k for k in range(B.TAB_CELLS, B.TAB_CELLS + B.TOP_CELLS) if self.state[k] == "stuck"]
        if not self.held_down and len(top) >= 3:
            self.held_down = True
            for n in B.FLAPS:
                self.flap_target[n].Set(float(B.flap_sign(n) * 90.0))
        left = sum(1 for k in range(B.TAB_CELLS) if all(self.state[B.TAB_CELLS - 1 - j] == "stuck" for j in range(k + 1)))
        right = sum(1 for k in range(B.TAB_CELLS) if all(self.state[B.TAB_CELLS + B.TOP_CELLS + j] == "stuck"
                                                          for j in range(k + 1)))
        offs = [abs(self.offset[k]) for k in top]
        return {"top_stuck": len(top), "top_cells": B.TOP_CELLS,
                "max_offset_mm": round(1000 * max(offs), 1) if offs else 0.0,
                "max_gap_deg": round(max((self.gap[k] for k in top), default=0.0), 1),
                "max_rack_taped_deg": round(max((self.rack_at[k] for k in range(B.CELLS) if self.state[k] == "stuck"),
                                                default=0.0), 2),
                "near_tab_mm": left * B.CELL * 1000, "far_tab_mm": right * B.CELL * 1000,
                "started": bool(self.started), "cut": bool(self.cut), "flaps_held": bool(self.held_down),
                "rack_deg": round(rack, 2), "crease_set_deg": {n: round(v, 1) for n, v in self.set.items()},
                "fold_deg": {n: round(B.flap_fold(n, root_poses[n]), 1) for n in B.FLAPS}}


def make_dynamics(env):
    return Dynamics(env)


def make_scorer():
    return TrialScorer()


# -- demo ---------------------------------------------------------------------------------------

REST = {"left": (0.16, 0.24, B.TOP + 0.20), "right": (0.16, -0.24, B.TOP + 0.20)}
FINGER_HALF = 0.010                                   # closed fingers' half thickness along the jaw axis
ALONG_X = simkit.rot_z(math.pi / 2) @ DOWN            # jaws along x: the closed pair's flat face pushes along x


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


BRACE = simkit.rot_y(0.45) @ ALONG_X                  # fingers leaning in at the far end wall, the palm out over the bench
BRACE_NEAR = simkit.rot_y(-0.45) @ ALONG_X            # the same at the near end wall, leaning the other way
HOLD = _rot_x(-0.40) @ DOWN                           # fingertips down on a foot, the palm leaning out away from the carton
HANDLE_FINGER = -(2 * B.HANDLE[0] - 0.010) / 2
GUN0 = B.gun_rot()
GRIP_LOCAL = np.array([B.HANDLE_AT[0], 0.0, B.HANDLE_AT[1]])
ROLLER_LOCAL = np.array([B.ROLLER_AT[0], 0.0, B.ROLLER_AT[1]])


def _gun_hand(seconds, roller_xyz, finger, label, lift=0.0):
    rg = B.gun_rot()
    centre = np.asarray(roller_xyz, float) - rg @ ROLLER_LOCAL
    grip = centre + rg @ GRIP_LOCAL + np.array([0.0, 0.0, lift])
    tcp = grip - DOWN @ np.array([0.0, 0.0, PAD_TIP])
    return Waypoint(seconds, tuple(map(float, tcp)), finger, label, DOWN)


def _tip(seconds, xyz, label, rot=None, finger=FINGER_CLOSED):
    """Closed fingertips (the pads' lower ends) at ``xyz``."""
    rot = DOWN if rot is None else rot
    tip = np.asarray(xyz, float)
    return Waypoint(seconds, tuple(map(float, tip - rot @ np.array([0.0, 0.0, PAD_TIP]))), finger, label, rot)


def _z(h):
    return B.TOP + h


class _Arm:
    def __init__(self, side):
        self.side = side
        self.w = [Waypoint(1.5, REST[side], FINGER_OPEN, "look", DOWN)]

    @property
    def t(self):
        return sum(w.duration for w in self.w)

    def add(self, *wps):
        self.w.extend(wps)
        return self

    def until(self, t, label="wait"):
        """Hold the last pose until time t."""
        last = self.w[-1]
        if t > self.t + 1e-6:
            self.w.append(Waypoint(t - self.t, last.pos, last.finger, label, last.yaw))
        return self


def demo():
    cx = B.C[0]
    x_near, x_far = cx - B.L / 2, cx + B.L / 2           # outside faces of the end (short) walls
    y_out = B.W / 2                                       # outside faces of the long walls
    minor_top = B.H + B.T + 0.0005                        # a folded short flap's top face
    major_top = B.TOP_SURF - B.TOP
    left, right = _Arm("left"), _Arm("right")
    seam_y = -0.009                                       # where the seam ends up after the squaring (measured, fixed start)

    # 1. Left braces the carton from beyond its far end, low on the far end wall, so it cannot slide away
    #    while right pushes the near short flap in; that flap, as wide as the inside, wedges the racked
    #    tube square as it goes in.
    by = 0.030
    hi = _z(0.150)
    left.add(_tip(1.5, (x_far + 0.06, by, hi), "high over beyond the far end", BRACE),
             _tip(0.8, (x_far + 0.06, by, _z(0.07)), "beyond the far end wall", BRACE),
             _tip(1.0, (x_far + 0.03, by, _z(0.024)), "down beside it", BRACE),
             _tip(1.5, (x_far + FINGER_HALF + 0.002, by, _z(0.024)), "against it", BRACE),
             _tip(0.5, (x_far + FINGER_HALF - 0.001, by, _z(0.024)), "brace it", BRACE))
    t_square = left.t

    # 2. Right folds the near short flap in and holds it down: it only goes in square. Come in over the
    #    top of the flaps (they stand 76 mm), on the carton's centre line, and sweep the flap over in one
    #    motion: push its top in, then follow it round and down onto it.
    right.add(_tip(1.5, (x_near - 0.030, 0.0, hi), "over beyond the near short flap", ALONG_X))
    right.until(t_square - 1.2)
    right.add(_tip(0.8, (x_near - 0.035, 0.0, _z(0.108)), "down well outside it", ALONG_X),
              _tip(0.6, (x_near - FINGER_HALF - 0.003, 0.0, _z(0.108)), "against it", ALONG_X),
              _tip(1.2, (x_near + 0.030, 0.0, _z(0.100)), "push it over", ALONG_X),
              _tip(1.0, (x_near + 0.060, 0.0, _z(0.075)), "on over the opening", ALONG_X),
              _tip(0.8, (x_near + 0.045, 0.0, _z(minor_top + 0.004)), "onto it"),
              _tip(0.6, (x_near + 0.045, 0.0, _z(minor_top - 0.0015)), "hold it down"))
    t_near = right.t

    # 3. Right now braces the near end wall from outside, low down, so the carton cannot slide toward the
    #    arms; left lets the far wall go and folds the far short flap in, which wedges the tube the rest
    #    of the way square.
    right.add(_tip(0.6, (x_near + 0.045, 0.0, hi), "up off the short flap"),
              _tip(1.2, (x_near - 0.060, 0.0, hi), "high beyond the near end", BRACE_NEAR),
              _tip(0.8, (x_near - 0.030, 0.0, _z(0.024)), "down beside the near end wall", BRACE_NEAR),
              _tip(1.0, (x_near - FINGER_HALF - 0.002, 0.0, _z(0.024)), "against it", BRACE_NEAR),
              _tip(0.4, (x_near - FINGER_HALF + 0.001, seam_y, _z(0.024)), "brace it", BRACE_NEAR))
    t_rb = right.t
    # 3. Left lets the wall go (the folded flap keeps the tube square) and folds the far short flap in.
    fy = 0.015
    left.until(t_rb - 2.8)
    left.add(_tip(0.8, (x_far + 0.04, by, _z(0.07)), "off the wall", BRACE),
             _tip(0.8, (x_far + 0.04, by, hi), "up"),
             _tip(1.5, (x_far + 0.025, fy, hi), "over beyond the far short flap", ALONG_X),
             _tip(0.8, (x_far + 0.035, fy, _z(0.108)), "down well outside it", ALONG_X),
             _tip(0.6, (x_far + FINGER_HALF + 0.003, fy, _z(0.108)), "against it", ALONG_X),
             _tip(1.2, (x_far - 0.030, fy, _z(0.100)), "push it over", ALONG_X),
             _tip(1.0, (x_far - 0.060, fy, _z(0.075)), "on over the opening", ALONG_X),
             _tip(0.8, (x_far - 0.045, fy, _z(minor_top + 0.004)), "onto it"),
             _tip(0.6, (x_far - 0.045, fy, _z(minor_top - 0.0015)), "press it flat"),
             _tip(0.6, (x_far - 0.045, fy, hi), "up off it"))
    t_far = left.t

    # 4. The long flaps over both short flaps: left folds the left one, then right lets the near short
    #    flap go and folds the right one, each swept over in one motion from outside.
    mx = cx + 0.005
    left.add(_tip(1.4, (mx, y_out + 0.030, hi), "over beyond the left long flap"),
             _tip(0.8, (mx, y_out + 0.035, _z(0.118)), "down well outside it"),
             _tip(0.6, (mx, y_out + FINGER_HALF + 0.003, _z(0.118)), "against it"),
             _tip(1.2, (mx, y_out - 0.030, _z(0.110)), "push it over"),
             _tip(1.0, (mx, y_out - 0.060, _z(0.078)), "on over the short flaps"),
             _tip(0.8, (mx, 0.035, _z(major_top + 0.004)), "onto it"),
             _tip(0.6, (mx, 0.035, _z(major_top - 0.0015)), "press it flat"),
             _tip(0.6, (mx, 0.035, hi), "up off it"))
    t_ml = left.t
    rx = cx - 0.050
    rhi = _z(0.136)
    right.until(t_ml - 0.6)
    right.add(_tip(0.6, (x_near - 0.040, 0.0, _z(0.07)), "off the near end wall", BRACE_NEAR),
              _tip(1.0, (x_near - 0.040, 0.0, hi), "up"),
              _tip(2.2, (rx, -y_out - 0.025, rhi), "over beyond the right long flap"),
              _tip(1.2, (rx, -y_out - 0.025, _z(0.118)), "down well outside it"),
              _tip(0.6, (rx, -y_out - FINGER_HALF - 0.003, _z(0.118)), "against it"),
              _tip(1.2, (rx, -y_out + 0.030, _z(0.110)), "push it over"),
              _tip(1.0, (rx, -y_out + 0.060, _z(0.078)), "on over the short flaps"),
              _tip(0.8, (rx, -0.035, _z(major_top + 0.004)), "onto it"),
              _tip(0.6, (rx, -0.035, _z(major_top - 0.0015)), "press it flat"))
    t_mr = right.t

    # 5. Left holds the empty carton down at the near end, beside the seam, while right takes the gun and
    #    runs the tape away from it; after the tear, left smooths the near tab and right, its gun put down,
    #    the far tab.
    hold_x, hold_y = x_near + 0.045, y_out + 0.034          # on the left wall's foot (its top flap, folded out flat)
    left.until(t_mr - 1.5)
    left.add(_tip(1.4, (hold_x, hold_y + 0.02, _z(0.06)), "over the left wall's foot", HOLD),
             _tip(0.8, (hold_x, hold_y, _z(B.T + 0.002)), "onto it", HOLD),
             _tip(0.6, (hold_x, hold_y, _z(B.T - 0.0015)), "hold the carton down by it", HOLD))
    r = B.ROLLER[0]
    top = B.TOP_SURF
    rest_roller = np.array([B.GUN_REST[0], B.GUN_REST[1], B.TOP + r]) + GUN0 @ np.array([B.ROLLER_AT[0], 0.0, 0.0])
    press = 0.002
    x_start, x_end = x_near + 0.018, x_far - 0.004
    f = HANDLE_FINGER
    right.add(_tip(0.6, (rx, -0.035, hi), "up off the flap"),
              _gun_hand(1.8, rest_roller, FINGER_OPEN, "over the tape gun's handle", lift=0.05),
              _gun_hand(1.2, rest_roller, FINGER_OPEN, "down round it"),
              _gun_hand(0.8, rest_roller, f, "take it"),
              _gun_hand(1.5, rest_roller + np.array([0.0, 0.0, 0.06]), f, "lift it"),
              _gun_hand(2.0, (x_start, seam_y, top + r + 0.03), f, "over the near end of the seam"),
              _gun_hand(1.6, (x_start, seam_y, top + r - press), f, "roller down on the seam, leader over the end"),
              _gun_hand(0.6, (x_start, seam_y, top + r - press), f, "press the leader end on"))
    n = 10
    for k in range(1, n + 1):
        x = x_start + (x_end - x_start) * k / n
        right.add(_gun_hand(0.5, (x, seam_y - 0.05 * (x - x_start), top + r - press), f, "run it along the seam"))
    right.add(_gun_hand(1.2, (x_far + 0.040, seam_y - 0.010, top + r + 0.004), f, "on past the far edge, the tape stretched over it"),
              _gun_hand(0.8, (x_far + 0.040, seam_y - 0.010, top + r + 0.024), f, "snap it up: the blade tears the tape"),
              _gun_hand(1.2, (x_far + 0.03, seam_y - 0.010, top + r + 0.07), f, "lift it away, up clear of the carton"))
    t_cut = right.t
    right.add(_gun_hand(2.0, rest_roller + np.array([0.0, 0.0, 0.04]), f, "back to the bench"),
              _gun_hand(1.0, rest_roller, f, "stand it down"),
              _gun_hand(0.6, rest_roller, FINGER_OPEN, "let go"),
              _gun_hand(0.8, rest_roller, FINGER_OPEN, "lift off", lift=0.06),
              _tip(2.0, (x_near - 0.04, seam_y, _z(major_top + 0.04)), "out beyond the near end", ALONG_X),
              _tip(0.8, (x_near - FINGER_HALF, seam_y, _z(major_top - 0.004)), "fingers on the hanging leader", ALONG_X),
              _tip(3.0, (x_near - FINGER_HALF + 0.001, seam_y, _z(major_top - 0.050)), "smooth it down the end", ALONG_X),
              _tip(0.8, (x_near - 0.05, seam_y, _z(major_top + 0.05)), "off it", ALONG_X),
              Waypoint(2.0, REST["right"], FINGER_OPEN, "stand clear", DOWN))
    left.until(t_cut + 0.3)
    left.add(_tip(0.6, (hold_x, hold_y + 0.02, _z(0.06)), "let the carton go", HOLD),
             _tip(1.6, (x_far + 0.03, 0.06, _z(major_top + 0.05)), "round to beyond the far end", ALONG_X),
             _tip(0.8, (x_far + FINGER_HALF - 0.002, seam_y, _z(major_top + 0.01)), "to the hanging tail", BRACE),
             _tip(0.8, (x_far + FINGER_HALF - 0.004, seam_y, _z(major_top - 0.003)), "fingers on the tail", BRACE),
             _tip(3.0, (x_far + FINGER_HALF - 0.005, seam_y, _z(major_top - 0.050)), "smooth it down the end", BRACE),
             _tip(0.8, (x_far + 0.05, 0.06, _z(major_top + 0.05)), "off it", ALONG_X),
             Waypoint(1.5, REST["left"], FINGER_OPEN, "stand clear", DOWN))
    end = max(left.t, right.t) + 2.0
    left.until(end, "done")
    right.until(end, "done")
    return {"left": left.w, "right": right.w}
