"""Offline checks (no simulator): the start is unsolved, the demo's turns solve it with no open end,
and the demo's IK. Run with /isaac-sim/python.sh check_layout.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipes as P  # noqa: E402
import task  # noqa: E402

from yam_arena import ik, station  # noqa: E402
from yam_arena.demo import ArmPlan  # noqa: E402


def main():
    problems = 0
    start = {rc: P.LAYOUT[rc][1] for rc in P.CELLS}
    wet, out, leaks = P.flow(start)
    print("start:", len(wet), "wet, through", out)
    problems += out
    end = dict(start)
    for side in task.TURNS.values():
        for rc, q in side:
            end[rc] = (end[rc] + q) % 4
    wet, out, leaks = P.flow(end)
    print("after the demo's turns:", len(wet), "wet, through", out, "open ends", leaks)
    problems += not out or leaks
    for side, wps in task.demo().items():
        plan = ArmPlan(wps, station.BASES[side])
        t, bad = 0.0, {}
        while t < plan.duration + 0.05:
            _, label = plan.action7(t)
            pos = plan.target(t)[0]
            p, _ = ik.tcp_pose(plan.q, plan.base_pos)
            e = np.linalg.norm(p - pos) * 1000
            if e > 3:
                bad.setdefault(label, []).append(round(e, 1))
            t += 1 / task.HZ
        for k, v in bad.items():
            print(f"   {side} '{k}': {len(v)} steps, max {max(v)} mm")
        worst = max([max(v) for v in bad.values()], default=0.0)
        print(f"{side}: {plan.duration:.1f} s, worst IK {worst:.1f} mm")
        problems += worst > 8
    print("PROBLEMS:", problems)
    return problems


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
