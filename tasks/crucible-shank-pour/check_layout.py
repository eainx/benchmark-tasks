"""Offline checks (no simulator): starting overlaps, the crucible lip over a pouring cup through the
tip, and the demo's IK on both arms. Run with /isaac-sim/python.sh check_layout.py"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shank as S  # noqa: E402
import task  # noqa: E402

from yam_arena import ik, station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402


def main():
    problems = 0
    beads = S.bead_objects()
    for a, b in itertools.combinations(beads, 2):
        if math.dist(a.pos, b.pos) < 2 * S.BEAD_R + 0.0005:
            print("OVERLAP", a.name, b.name); problems += 1
    for b in beads:
        loc = np.asarray(b.pos) - np.array([S.CRU_XY[0], S.CRU_XY[1], S.TOP])
        r = math.hypot(loc[0], loc[1])
        bore = S.CRU_BORE + (S.CRU_R - S.CRU_BORE) * loc[2] / S.CRU_H - 0.003
        if r + S.BEAD_R > bore:
            print("BEAD IN THE LINING", b.name, round(r + S.BEAD_R, 4), round(bore, 4)); problems += 1
    # crucible clear of the mould at rest
    if S.CRU_XY[0] + S.RING_R + 0.01 > S.MOULD_X - S.MOULD[0] / 2:
        print("CRUCIBLE ON THE MOULD"); problems += 1
    # lip through the tip
    axis = task.pour_axis(task.DRY)
    for deg in (40, 60, 80, 88):
        r = task.ry(-math.radians(deg))
        lip = axis + r @ np.array([-(S.CRU_R + S.CRU_WALL), 0.0, S.CRU_H - S.SHANK_Z])
        foot = axis + r @ np.array([S.RING_R, 0.0, -S.SHANK_Z])
        d = math.dist(lip[:2], S.basin_xy(task.DRY))
        print(f"tilt {deg}: lip {lip.round(3)} {d*1000:.0f} mm from cup centre, {1000*(lip[2]-S.CUP_TOP):.0f} mm above rim;"
              f" lowest foot z {1000*(foot[2]-S.CUP_TOP):.0f} mm above rim")
        if d > S.BASIN_TOP_R - 0.010 and deg >= 60:
            print("LIP OUTSIDE THE CUP"); problems += 1
    # the foot's near edge dips as the crucible turns about the handle line: it must clear the cup
    dip = min((axis + task.ry(-math.radians(d)) @ np.array([sx * S.RING_R, 0.0, -S.SHANK_Z]))[2]
              for d in range(0, 91, 2) for sx in (-1, 1))
    print(f"lowest foot edge through the tip: {1000*(dip - S.CUP_TOP):.0f} mm above the cup rim")
    if dip < S.CUP_TOP + 0.004:
        print("FOOT HITS THE CUP"); problems += 1
    for side, wps in task.demo().items():
        plan = ArmPlan(wps, station.BASES[side])
        t, bad = 0.0, {}
        while t < plan.duration + 0.05:
            _, label = plan.action7(t)
            pos, _, _, _ = plan.target(t)
            p, _ = ik.tcp_pose(plan.q, plan.base_pos)
            e = np.linalg.norm(p - pos) * 1000
            if e > 3:
                bad.setdefault(label, []).append(round(e, 1))
            t += 1 / task.HZ
        for k, v in bad.items():
            print(f"   {side} '{k}': {len(v)} steps, max {max(v)} mm")
        print(f"{side}: {plan.duration:.1f} s, worst IK {plan.worst_ik_mm:.1f} mm")
        if plan.worst_ik_mm > 8:
            problems += 1
    print("PROBLEMS:", problems)
    return problems


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
