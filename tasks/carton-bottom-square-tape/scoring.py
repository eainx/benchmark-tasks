"""Stateful scorer for making up a carton's bottom: squareness from the measured wall poses, flap folds
from the measured flap poses, tape from the cells reported by task.Dynamics, hands from measured TCPs."""
from __future__ import annotations

import math

import numpy as np

import parameters as B

HZ = 20
HOLD_S = 1.0
SQUARE_DEG = 5.0                     # square enough while the flaps are folded and taped
FINAL_SQUARE_DEG = 3.0               # square enough at the end
FOLDED_DEG = 80.0                    # a flap within 10 degrees of flat
CENTRE_MM = 10.0                     # the strip's centre within this of the seam all along
CRUSH_DEG = 15.0                     # a wall tipped this far out of vertical: crushed


class TrialScorer:
    def __init__(self):
        self.held = 0
        self.step = 0
        self.failed = None
        self.max_rack_folded = 0.0

    def _layered(self, poses):
        """Both long flaps lie over both short flaps: at each short flap's middle, under each long flap,
        the long flap is higher."""
        ok = True
        for minor in B.MINORS:
            mp = np.asarray(poses[minor], float)[:3]
            for major in B.MAJORS:
                jp = np.asarray(poses[major], float)[:3]
                ok &= jp[2] > mp[2] + 0.001
        return bool(ok)

    def update(self, step, objects, root_poses, robot, mechanics=None, **_):
        if step != self.step + 1:
            raise ValueError("scorer needs one update per control step")
        if mechanics is None:
            raise ValueError("tape measurements missing")
        self.step = step
        m = mechanics
        rack = B.rack_deg(root_poses)
        fold = {n: B.flap_fold(n, root_poses[n]) for n in B.FLAPS}
        minors_in = all(fold[n] >= FOLDED_DEG for n in B.MINORS)
        majors_in = all(fold[n] >= FOLDED_DEG for n in B.MAJORS)
        layered = self._layered(root_poses)
        if minors_in:
            self.max_rack_folded = max(self.max_rack_folded, rack)
        crushed = B.wall_out_of_plane(root_poses) > CRUSH_DEG
        near = np.asarray(root_poses["carton"], float)
        centre, _ = B.carton_frame(near)
        off_table = any(np.asarray(root_poses[n], float)[2] < B.TOP - 0.02 for n in B.WALLS)
        if self.failed is None and (crushed or off_table):
            self.failed = "crushed" if crushed else "off the table"
        frac = m["top_stuck"] / m["top_cells"]
        centred = m["max_offset_mm"] <= CENTRE_MM
        tabs = min(m["near_tab_mm"], m["far_tab_mm"]) >= B.TAB_MIN * 1000
        gun = np.asarray(root_poses["gun"], float)
        free = all(math.dist(np.asarray(robot[s]["tcp"])[:3], gun[:3]) > 0.10 or float(np.mean(robot[s]["fingers"])) < -0.040
                   for s in ("left", "right"))
        gun_down = gun[2] < B.TOP + 0.10 and math.dist(gun[:2], centre[:2]) > 0.13
        clear = all(math.dist(np.asarray(robot[s]["tcp"])[:2], centre[:2]) > 0.16 or
                    float(np.asarray(robot[s]["tcp"])[2]) > B.TOP_SURF + 0.10 for s in ("left", "right"))
        square_kept = self.max_rack_folded <= SQUARE_DEG and m["max_rack_taped_deg"] <= SQUARE_DEG
        score = 0.0
        if minors_in and rack <= SQUARE_DEG:
            score = 0.25
        if score == 0.25 and majors_in and layered:
            score = 0.5
        if score == 0.5 and m["started"] and frac >= 0.9 and centred and square_kept:
            score = 0.75
        full = (score == 0.75 and frac >= 1.0 and m["cut"] and tabs and rack <= FINAL_SQUARE_DEG
                and free and gun_down and clear)
        self.held = self.held + 1 if full else 0
        if full and self.held >= HOLD_S * HZ:
            score = 1.0
        if self.failed is not None:
            score = 0.0
        return {"score": score, "success": score == 1.0, "rack_deg": round(rack, 2),
                "max_rack_folded_deg": round(self.max_rack_folded, 2), "fold_deg": {k: round(v, 1) for k, v in fold.items()},
                "minors_in": bool(minors_in), "majors_in": bool(majors_in), "layered": bool(layered),
                "top_covered": round(frac, 3), "centred": bool(centred), "cut": bool(m["cut"]),
                "near_tab_mm": m["near_tab_mm"], "far_tab_mm": m["far_tab_mm"], "gun_released": bool(free),
                "gun_down": bool(gun_down), "hands_clear": bool(clear), "failed": self.failed, "held_frames": self.held}
