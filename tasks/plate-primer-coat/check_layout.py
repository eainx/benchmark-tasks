"""Offline checks (no simulator): the brush handle clears the tray rim while dipping, a stroke's
reach over the patches, one dip short of the whole plate, and the demo's IK.
Run with /isaac-sim/python.sh check_layout.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paint as P  # noqa: E402
import task  # noqa: E402

from yam_arena import ik, station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402


def main():
    problems = 0
    handle_bottom = task.DIP_Z + P.HANDLE_Z - P.HANDLE_R
    rim = P.TOP + P.TRAY_WALL + P.TRAY_IN[2]
    print(f"dipping: handle bottom {1000*(handle_bottom - rim):.1f} mm above the tray rim")
    problems += handle_bottom < rim + 0.002
    if not task.DIP_Z < P.TOP + P.TRAY_WALL + P.PRIMER_DEPTH - 0.003:
        print("DIP TOO SHALLOW"); problems += 1
    # one stroke from the far edge to the near edge passes over every patch
    seen = set()
    for x in np.linspace(task.FAR_X, task.NEAR_X, 200):
        seen.update(P.tiles_under([x, 0.0, task.STROKE_Z + P.BRISTLE_BLOCK[2] / 2, 0, 0, 0, 1]))
    print(f"one stroke passes over {len(seen)}/{P.TILES} patches; one dip carries {P.LOAD}")
    problems += len(seen) != P.TILES or P.LOAD >= P.TILES
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
        problems += plan.worst_ik_mm > 8
    print("PROBLEMS:", problems)
    return problems


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
