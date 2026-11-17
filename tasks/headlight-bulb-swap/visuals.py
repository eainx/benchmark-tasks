"""Detailed visual meshes for the headlamp scene (no collision; the colliders are simple primitives
hidden inside these). Each builder returns vmesh tuples (name, points, triangles, colour, material,
opacity) in the owning Part's frame."""
from __future__ import annotations

import math

import numpy as np

import headlamp as H
import meshlib as M

HOUSING = (0.055, 0.055, 0.06)       # black textured PP (matte)
SEAM = (0.03, 0.03, 0.03)
CHROME = (0.86, 0.87, 0.89)
LENS = (0.88, 0.91, 0.95)
GREY = (0.50, 0.51, 0.53)            # mid-grey PBT bulb base
RUBBER = (0.06, 0.06, 0.065)
BLUE_CAP = (0.13, 0.36, 0.72)
AMBER = (0.98, 0.52, 0.05)
STEEL = (0.70, 0.71, 0.73)
GLASS = (0.96, 0.97, 0.99)
COPPER = (0.82, 0.50, 0.25)
BLACK = (0.04, 0.04, 0.045)
WHITE = (0.93, 0.93, 0.92)


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _shift(vms, d):
    d = np.asarray(d, float)
    return tuple((n, p + d, t, c, m, o) for n, p, t, c, m, o in vms)


# -- headlamp --------------------------------------------------------------------------------------
def housing_vmeshes(part_pos):
    """The Camry-type lamp, back toward the arms: a black PP shell lofted along its width (swept and
    narrowing toward the fender), the clear lens over its front with chrome reflectors behind it, the two
    ribbed port cans on its back (low beam open, high beam with its rubber cover), top bolt flanges, the
    outboard ear, beam adjusters and the main plug. World-frame meshes, returned in the Part's frame."""
    V = []
    secs_shell, secs_lens = [], []
    for y, d, h, zb in H.SECTIONS:
        zc = H.TOP + zb + h / 2
        secs_shell.append((y, d - 0.045, h, 5.0, H.BACK_X + (d - 0.045) / 2, zc))
        secs_lens.append((y, 0.062, h * 0.97, 5.0, H.BACK_X + d - 0.031, zc))
    V.append(_vm("shell", M.loft(secs_shell, axis="Y", n=64), HOUSING, "matte"))
    V.append(_vm("lens", M.loft(secs_lens, axis="Y", n=64), LENS, "acrylic", 0.22))
    # seam band where the lens is bonded on
    V.append(_vm("seam", M.loft([(y, 0.006 + 0.0, h + 0.004, 5.0, H.BACK_X + d - 0.045, H.TOP + zb + h / 2)
                                  for y, d, h, zb in H.SECTIONS], axis="Y", n=64), SEAM))
    # reflectors seen through the lens: low beam (large, outboard), high beam (inboard), turn signal
    for i, (y, z, r) in enumerate(((H.PORT_Y, H.PORT_Z, 0.085), (H.HIGH_Y, H.HIGH_Z, 0.070), (-0.330, H.TOP + 0.13, 0.050))):
        d = [s_[1] for s_ in H.SECTIONS if abs(s_[0] - y) < 0.08]
        front = H.BACK_X + (d[0] if d else 0.24) - 0.035
        prof = [(r - (r - 0.012) * k / 8, -0.05 * (1 - ((r - (r - 0.012) * k / 8) / r) ** 2)) for k in range(9)]
        bowl = M.revolve(prof, n=48, axis="X")
        V.append(_vm(f"reflector{i}", bowl, CHROME, "aluminium", offset=(front + 0.005, y, z)))
        V.append(_vm(f"shield{i}", M.revolve([(0.0, 0.0), (0.012, 0.0), (0.014, 0.004), (0.014, 0.010), (0.0, 0.012)],
                                             n=32, axis="X"), STEEL, "steel", offset=(front - 0.03, y, z)))
    V.append(_vm("turn_bulb", M.revolve([(0.0, 0.0), (0.006, 0.002), (0.0065, 0.010), (0.004, 0.016), (0.0, 0.017)], n=24,
                                        axis="X"), AMBER, "acrylic", 0.9, offset=(H.BACK_X + 0.17, -0.330, H.TOP + 0.13)))
    V.append(_vm("bezel", M.rbox((0.004, 0.012, 0.200), 6, axis="X"), BLACK, offset=(H.BACK_X + 0.250, 0.010, H.TOP + 0.150)))
    V.append(_vm("marker", M.rbox((0.060, 0.004, 0.026), 8, axis="Y"), AMBER, "acrylic", 0.85,
                 offset=(H.BACK_X + 0.090, -0.451, H.TOP + 0.150)))
    # stiffening ribs down the back
    for k, y in enumerate((-0.30, -0.18, 0.02, 0.20)):
        V.append(_vm(f"back_rib{k}", M.rbox((0.006, 0.004, 0.200), 4, axis="X"), HOUSING, "matte",
                     offset=(H.BACK_X - 0.002, y, H.TOP + 0.150)))
    # ribbed port cans; the low-beam one carries the open PGJ19 boss and its bayonet notches
    for i, (y, z) in enumerate(((H.PORT_Y, H.PORT_Z), (H.HIGH_Y, H.HIGH_Z))):
        prof = [(0.050, 0.0), (0.046, -0.002), (0.0425, -0.006), (H.CAN_R, -0.010), (H.CAN_R, -H.CAN_H + 0.002),
                (H.CAN_R - 0.003, -H.CAN_H)]
        if i == 0:
            prof += [(H.BOSS_R + 0.001, -H.CAN_H), (H.BOSS_R, -H.CAN_H - 0.002), (H.BOSS_R, -H.CAN_H - H.BOSS_L + 0.0015),
                     (H.BOSS_R - 0.0015, -H.CAN_H - H.BOSS_L), (H.HOLE_R, -H.CAN_H - H.BOSS_L), (H.HOLE_R, -0.004)]
            V.append(_vm("bore", M.revolve([(H.HOLE_R - 0.0003, -0.004), (0.0, -0.004)], n=32, axis="X"), BLACK,
                         offset=(H.BACK_X, y, z)))
            V.append(_vm("boss_rim", M.revolve([(H.BOSS_R + 0.0012, -H.CAN_H - 0.002), (H.BOSS_R + 0.0012, -H.CAN_H - H.BOSS_L),
                                                (H.HOLE_R - 0.0004, -H.CAN_H - H.BOSS_L - 0.0006),
                                                (H.HOLE_R - 0.0004, -H.CAN_H - H.BOSS_L + 0.001)], n=64, axis="X"),
                         (0.30, 0.30, 0.32), "satin", offset=(H.BACK_X, y, z)))
            for k in range(3):
                t = 2 * math.pi * k / 3 + math.radians(30)
                V.append(_vm(f"notch{k}", M.rbox((0.0012, 0.004, 0.004), 4, axis="X"), BLACK,
                             offset=(H.PORT_FACE_X - 0.0004, y + (H.HOLE_R + 0.0015) * math.cos(t),
                                     z + (H.HOLE_R + 0.0015) * math.sin(t))))
        else:
            prof += [(0.0, -H.CAN_H)]
        V.append(_vm(f"can{i}", M.revolve(prof, n=64, axis="X"), (0.15, 0.15, 0.16), "satin", offset=(H.BACK_X, y, z)))
        V.append(_vm(f"seal{i}", M.revolve([(0.050, 0.0005), (0.0525, -0.001), (0.0525, -0.0035), (0.050, -0.0045)],
                                           n=64, axis="X"), (0.02, 0.02, 0.02), "gloss", offset=(H.BACK_X, y, z)))
        for k in range(8):
            t = 2 * math.pi * k / 8 + math.pi / 8
            V.append(_vm(f"rib{i}_{k}", M.rbox((H.CAN_H - 0.004, 0.006, 0.0035), 4, axis="X"), HOUSING,
                         offset=(H.BACK_X - H.CAN_H / 2 - 0.001, y + (H.CAN_R + 0.0012) * math.cos(t),
                                 z + (H.CAN_R + 0.0012) * math.sin(t))))
    yb = H.BACK_X - H.CAN_H
    cover = [(H.CAN_R - 0.002, 0.003), (0.0335, 0.0), (0.0335, -0.012), (0.032, -0.016), (0.026, -0.020), (0.0, -0.021)]
    V.append(_vm("cover", M.revolve(cover, n=64, axis="X"), RUBBER, "satin", offset=(yb, H.HIGH_Y, H.HIGH_Z)))
    V.append(_vm("cover_tab", M.rbox((0.003, 0.012, 0.016), 4, axis="X"), RUBBER, offset=(yb - 0.004, H.HIGH_Y, H.HIGH_Z + 0.040)))
    # top bolt flanges (three) and the outboard ear, steel-bushed
    for k, (x, y, z) in enumerate(H.BOLTS):
        ear = (0.040, 0.034, 0.010) if k < 3 else (0.040, 0.010, 0.034)
        V.append(_vm(f"ear{k}", M.rbox(ear, 3.0), HOUSING, "matte", offset=(x, y if k < 3 else y + 0.010, z - 0.008 if k < 3 else z - 0.010)))
    # beam adjusters (blue) and the main plug on its back
    for k, y in enumerate((-0.25, 0.18)):
        V.append(_vm(f"adjuster{k}", M.revolve([(0.0, 0.0), (0.009, 0.0), (0.009, 0.004), (0.006, 0.006), (0.006, 0.012), (0.0, 0.012)],
                                               n=24, axis="Z"), BLUE_CAP, offset=(H.BACK_X + 0.030, y, H.TOP + 0.292)))
    V.append(_vm("plug", M.rbox((0.024, 0.044, 0.030), 5, axis="X"), (0.16, 0.16, 0.17), offset=(H.BACK_X - 0.012, 0.02, H.TOP + 0.050)))
    return _shift(tuple(V), -np.asarray(part_pos, float))


# -- H11 bulb --------------------------------------------------------------------------------------
def bulb_vmeshes(R, prefix="b"):
    """The bulb in its own frame (a: axis into the lamp, s: socket direction, t = a x s), mapped by R.
    The grip block's centre is the origin (the Part's root)."""
    V = []
    fa = H.GRIP[0] / 2 + H.FLANGE_T / 2
    a0 = fa - H.FLANGE_T / 2

    def add(name, mesh, color, material=None, opacity=1.0):
        V.append(_vm(f"{prefix}_{name}", mesh, color, material, opacity, R=R))

    def rev(profile, n=48):
        return M.revolve(profile, n=n, axis="X")       # about a

    add("flange", rev(M.fillet_profile(0.0080, H.FLANGE_R, a0, a0 + H.FLANGE_T, f=0.0008)), GREY)
    add("oring", rev([(0.0168, a0 + H.FLANGE_T), (0.0182, a0 + H.FLANGE_T + 0.0004), (0.0186, a0 + H.FLANGE_T + 0.0012),
                      (0.0182, a0 + H.FLANGE_T + 0.0020), (0.0168, a0 + H.FLANGE_T + 0.0024)]), BLACK)
    for k in range(3):
        t = 2 * math.pi * k / 3 + math.radians(30)
        lug = M.transform(M.rbox((0.0030, 0.0040, 0.0030), 4), None,
                          (a0 + H.FLANGE_T + 0.0016, H.TAB_R * math.cos(t), H.TAB_R * math.sin(t)))
        add(f"lug{k}", lug, GREY)
    ring0 = a0 + H.FLANGE_T
    add("ring", rev([(0.0060, ring0), (0.0085, ring0), (0.0088, ring0 + 0.002), (0.0088, ring0 + 0.007),
                     (0.0078, ring0 + 0.008), (0.0055, ring0 + 0.008)]), STEEL, "steel")
    c0 = ring0 + 0.008
    L = H.CAPSULE_L
    cap = [(0.0040, c0), (0.0050, c0 + 0.002), (0.0050, c0 + L - 0.004)]
    for k in range(1, 7):
        ang = (math.pi / 2) * k / 6
        cap.append((0.0050 * math.cos(ang), c0 + L - 0.004 + 0.0050 * math.sin(ang)))
    add("capsule", rev(cap), (0.90, 0.95, 1.0), "acrylic", 0.22)
    add("tip", rev([(0.0034, c0 + L - 0.0005), (0.0022, c0 + L + 0.0006), (0.0, c0 + L + 0.0010)], n=24), (0.12, 0.12, 0.13))
    coil = [(c0 + 0.008 + 0.010 * k / 60, 0.0011 * math.cos(k * 0.9), 0.0011 * math.sin(k * 0.9)) for k in range(61)]
    add("filament", M.tube(coil, 0.00045, n=6), (0.20, 0.14, 0.08))
    for d in (-0.0016, 0.0016):                                          # the filament's two supports
        add(f"lead{d > 0}", M.tube([(c0 - 0.001, 0.0, d), (c0 + 0.008, 0.0, d * 0.7)], 0.00035, n=6), STEEL, "steel")
    # the grip: a flat moulded handle with two knurled wings, and the L-shaped socket
    add("grip", M.rbox(H.GRIP, 4.0, axis="X"), GREY, "satin")
    for sgn in (-1, 1):
        wing = M.transform(M.rbox((0.016, 0.003, 0.010), 4, axis="Y"), None, (-0.001, sgn * 0.0, sgn * 0.0))
        wing = M.transform(M.rbox((0.016, 0.006, 0.0025), 4), None, (-0.001, 0.0, sgn * (H.GRIP[2] / 2 + 0.001)))
        add(f"wing{sgn}", wing, GREY)
        for k in range(5):
            ridge = M.transform(M.rbox((0.0008, 0.020, 0.0012), 3), None,
                                (-0.008 + 0.004 * k, 0.0, sgn * (H.GRIP[2] / 2 + 0.0026)))
            add(f"ridge{sgn}_{k}", ridge, (0.70, 0.71, 0.72))
    sk = np.array([-0.002, H.GRIP[1] / 2 + H.SOCKET[1] / 2, 0.0])
    add("socket", M.transform(M.rbox(H.SOCKET, 5, axis="Y"), None, sk), GREY)
    add("socket_mouth", M.transform(M.rbox((H.SOCKET[0] - 0.003, 0.0012, H.SOCKET[2] - 0.003), 6, axis="Y"), None,
                                    sk + np.array([0.0, H.SOCKET[1] / 2 + 0.0002, 0.0])), BLACK)
    for d in (-0.003, 0.003):
        add(f"blade{d > 0}", M.transform(M.rbox((0.0008, 0.006, 0.0050), 3), None,
                                         sk + np.array([0.0, H.SOCKET[1] / 2 - 0.002, d])), COPPER)
    return tuple(V)


# -- connector -------------------------------------------------------------------------------------
def connector_vmeshes(part_pos):
    """The connector body; it mates on its +y face, its leads leave its -y end (tab and leads are separate parts)."""
    V = [_vm("body", M.rbox(H.CONN, 5, axis="Y"), (0.025, 0.025, 0.028), "gloss"),
         _vm("seal", M.rbox((H.CONN[0] - 0.002, 0.0025, H.CONN[2] - 0.002), 6, axis="Y"), (0.45, 0.45, 0.46), "matte",
             offset=(0.0, H.CONN[1] / 2 + 0.0008, 0.0)),
         _vm("tab_hinge", M.rbox((0.012, 0.004, 0.002), 4), (0.10, 0.10, 0.11), offset=(0.0, -0.009, H.CONN[2] / 2 + 0.001)),
         _vm("boot", M.revolve([(0.0, 0.0), (0.0065, 0.0), (0.0058, -0.004), (0.0050, -0.008), (0.0045, -0.012),
                                (0.0, -0.012)], n=24, axis="Y"), (0.02, 0.02, 0.02), "gloss",
             offset=(0.0, -H.CONN[1] / 2 + 0.006, 0.0))]
    return tuple(V)


def tab_vmeshes():
    TAB = (0.30, 0.30, 0.32)
    V = [_vm("key", M.rbox(H.TAB, 4), TAB, "satin"),
         _vm("ramp", M.rbox((H.TAB[0], 0.003, 0.0042), 4), TAB, "satin", offset=(0.0, 0.0065, 0.0008))]
    for k in range(3):
        V.append(_vm(f"ridge{k}", M.rbox((0.009, 0.0010, 0.0010), 3), (0.55, 0.55, 0.57),
                     offset=(0.0, -0.005 + 0.004 * k, H.TAB[2] / 2 + 0.0005)))
    return tuple(V)


def lead_vmeshes(axis, L, i):
    """A link of the taped lead pair: one black and one white 18 AWG lead side by side, a wrap of tape."""
    ax = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[axis]
    side = np.array([0, 0, 1.0]) if axis != "Z" else np.array([0, 1.0, 0])
    V = []
    for k, (d, col) in enumerate(((-0.0012, (0.02, 0.02, 0.02)), (0.0012, WHITE))):
        p0 = -ax * (L / 2 + 0.0006) + side * d
        p1 = ax * (L / 2 + 0.0006) + side * d
        V.append(_vm(f"wire{k}", M.tube([p0, p1], 0.0013, n=10), col, "gloss"))
    if i % 2 == 0:
        V.append(_vm("tape", M.revolve([(0.0027, -0.003), (0.0027, 0.003)], n=16, axis=axis), (0.08, 0.08, 0.08), "matte"))
    return tuple(V)


# -- dust cover on the towel (upside down: its open side up) ---------------------------------------------
def dust_cover_vmeshes():
    """The low-beam port's rubber boot, lying dome-up: a skirt, a stepped dome with moulded rings, a pull tab."""
    prof = [(0.0335, -0.010), (0.0340, -0.006), (0.0330, -0.002), (0.0300, 0.000), (0.0290, 0.003), (0.0250, 0.006),
            (0.0240, 0.008), (0.0180, 0.010), (0.0100, 0.011), (0.0, 0.0112)]
    V = [_vm("cap", M.revolve(prof, n=64, cap0=True), RUBBER, "satin")]
    for k, rr in enumerate((0.0310, 0.0205, 0.0120)):
        z = {0: -0.001, 1: 0.0085, 2: 0.0108}[k]
        V.append(_vm(f"ring{k}", M.revolve([(rr - 0.0010, z), (rr, z + 0.0010), (rr + 0.0010, z)], n=48), (0.10, 0.10, 0.11),
                     "satin"))
    V.append(_vm("tab", M.rbox((0.018, 0.012, 0.0030), 4), RUBBER, "satin", offset=(0.041, 0.0, -0.0085)))
    V.append(_vm("tab_hole", M.revolve([(0.0, 0.0), (0.003, 0.0), (0.003, 0.0006), (0.0, 0.0006)], n=16), (0.6, 0.55, 0.5),
                 offset=(0.045, 0.0, -0.0069)))
    return tuple(V)


# -- blister pack --------------------------------------------------------------------------------------
def blister_vmeshes():
    th = H.CARD[2]
    V = [_vm("card", M.loft([(-th / 2, H.CARD[0], H.CARD[1], 10), (th / 2, H.CARD[0], H.CARD[1], 10)], axis="Z", n=64),
             (0.08, 0.20, 0.52)),
         _vm("label", M.rbox((0.086, 0.052, 0.0004), 10), WHITE, offset=(0.0, -0.050, th / 2 + 0.0002)),
         _vm("band", M.rbox((0.086, 0.012, 0.0005), 10), (0.82, 0.10, 0.10), offset=(0.0, -0.034, th / 2 + 0.0004)),
         _vm("bar", M.rbox((0.040, 0.010, 0.0005), 10), (0.15, 0.15, 0.15), offset=(0.0, -0.066, th / 2 + 0.0004)),
         _vm("hang", M.rbox((0.024, 0.007, 0.0008), 10), (0.82, 0.76, 0.62), offset=(0.0, 0.071, th / 2 + 0.0004))]
    for k, (w, y) in enumerate(((0.060, -0.036), (0.070, -0.042), (0.050, -0.048), (0.064, -0.054), (0.036, -0.060))):
        V.append(_vm(f"text{k}", M.rbox((w, 0.0022, 0.0005), 10), (0.25, 0.25, 0.28), offset=(-0.008 - (0.070 - w) / 2,
                                                                                               y + 0.004, th / 2 + 0.0005)))
    # a small bulb picture on the label: grey base, glass, and a yellow glow ring
    V.append(_vm("pic_glow", M.revolve([(0.0, 0.0), (0.009, 0.0), (0.009, 0.0004), (0.0, 0.0004)], n=24), (0.98, 0.82, 0.25),
                 offset=(0.032, -0.046, th / 2 + 0.0005)))
    V.append(_vm("pic_base", M.rbox((0.006, 0.010, 0.0006), 6), (0.45, 0.45, 0.47), offset=(0.032, -0.054, th / 2 + 0.0008)))
    V.append(_vm("pic_glass", M.rbox((0.004, 0.010, 0.0006), 6), (0.95, 0.96, 0.98), offset=(0.032, -0.044, th / 2 + 0.0008)))
    V.append(_vm("brand_band", M.rbox((0.100, 0.020, 0.0005), 10), (0.95, 0.80, 0.10), offset=(0.0, 0.048, th / 2 + 0.0003)))
    # clear PETG pocket (open top), its flange, and the torn-back lid
    pocket = M.loft([(th / 2 + 0.0005, 0.066, 0.086, 6, 0.0, 0.020), (th / 2 + 0.017, 0.074, 0.094, 6, 0.0, 0.020)],
                    axis="Z", n=48, cap0=True, cap1=False)
    V.append(_vm("pocket", pocket, (0.92, 0.95, 0.98), "acrylic", 0.30))
    for k, (sx, sy, ox, oy) in enumerate(((0.090, 0.008, 0.0, 0.071), (0.090, 0.008, 0.0, -0.031),
                                           (0.008, 0.110, 0.041, 0.020), (0.008, 0.110, -0.041, 0.020))):
        V.append(_vm(f"flange{k}", M.rbox((sx, sy, 0.0006), 8), (0.92, 0.95, 0.98), "acrylic", 0.30,
                     offset=(ox, oy, th / 2 + 0.0173)))
    lid = M.rbox((0.050, 0.094, 0.0005), 8)
    c, s = math.cos(-0.9), math.sin(-0.9)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    V.append(_vm("lid", lid, (0.92, 0.95, 0.98), "acrylic", 0.30, offset=(0.060, 0.020, th / 2 + 0.036), R=Ry))
    return tuple(V)


# -- towel -----------------------------------------------------------------------------------------------
def towel_vmeshes():
    """A folded cotton shop towel: soft rumples, a fold ridge, the folded layers showing at one edge, hems."""
    x, y, z = H.TOWEL

    def top(X, Y):
        r = (0.0012 * np.sin(X * 37.0 + 0.6) * np.sin(Y * 29.0) + 0.0008 * np.sin(X * 71.0 + Y * 43.0)
             + 0.0022 * np.exp(-((X + x / 2 - 0.030) / 0.010) ** 2))
        edge = np.minimum.reduce([X + x / 2, x / 2 - X, Y + y / 2, y / 2 - Y])
        return z / 2 + r - 0.003 * np.exp(-edge / 0.004)

    V = [_vm("cloth", M.heightfield(x, y, top, -z / 2, nx=72, ny=56), (0.66, 0.10, 0.08), "matte")]
    for k, dz in enumerate((-0.0025, 0.0010)):                            # folded layers along the -x edge
        V.append(_vm(f"layer{k}", M.rbox((0.006, y - 0.006 - 0.004 * k, 0.0030), 8), (0.58, 0.08, 0.07), "matte",
                     offset=(-x / 2 + 0.002, 0.0, dz)))
    for k, oy in enumerate((0.115, -0.125)):
        V.append(_vm(f"stripe{k}", M.rbox((x - 0.020, 0.008, 0.0008), 10), (0.92, 0.88, 0.80), "matte",
                     offset=(0.0, oy, z / 2 + 0.0012)))
    for k, oy in enumerate((y / 2 - 0.006, -y / 2 + 0.006)):
        V.append(_vm(f"hem{k}", M.rbox((x - 0.006, 0.004, 0.0012), 10), (0.55, 0.07, 0.06), "matte",
                     offset=(0.0, oy, z / 2 - 0.0008)))
    return tuple(V)



# -- engine-bay section ------------------------------------------------------------------------------
def bay_vmeshes(part_pos):
    E_COAT = (0.06, 0.06, 0.065)          # black e-coated steel
    BODY = (0.55, 0.57, 0.60)             # body colour on the fender apron
    ALU = (0.62, 0.64, 0.66)
    PP = (0.08, 0.08, 0.085)
    V = []

    def box(name, key, color, material=None, e=6.0, axis="Z"):
        x0, x1, y0, y1, z0, z1 = H.BAY[key]
        V.append(_vm(name, M.rbox((x1 - x0, y1 - y0, z1 - z0), e, axis=axis), color, material,
                     offset=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)))

    box("tie_bar", "tie_bar", E_COAT, "satin", e=8, axis="Y")
    x0, x1, y0, y1, z0, z1 = H.BAY["tie_bar"]
    for k in range(9):                                                   # its lightening holes
        V.append(_vm(f"hole{k}", M.revolve([(0.0, 0.0), (0.009, 0.0), (0.009, 0.0008), (0.0, 0.0008)], n=24), (0.02, 0.02, 0.02),
                     offset=((x0 + x1) / 2, y0 + 0.06 + 0.105 * k, z1 + 0.0002)))
    V.append(_vm("flange", M.rbox((0.012, y1 - y0, 0.003), 8, axis="Y"), E_COAT, "satin", offset=(x0 - 0.006, (y0 + y1) / 2, z0 + 0.0015)))
    box("core", "radiator", ALU, "aluminium_brushed", e=10, axis="Y")
    x0, x1, y0, y1, z0, z1 = H.BAY["radiator"]
    for k in range(28):
        V.append(_vm(f"tube{k}", M.rbox((0.0012, y1 - y0, 0.0016), 10, axis="Y"), (0.30, 0.31, 0.33),
                     offset=(x0 - 0.0004, (y0 + y1) / 2, z0 + 0.008 + 0.0098 * k)))
    box("end_tank", "end_tank", PP, "satin", e=5)
    box("fender", "fender", BODY, "satin", e=10, axis="Y")
    box("fender_rail", "fender_rail", BODY, "satin", e=8, axis="X")
    box("ear_bracket", "ear_bracket", E_COAT, "satin", e=6)
    x0, x1, y0, y1, z0, z1 = H.AIRBOX
    cxa, cya = (x0 + x1) / 2, (y0 + y1) / 2
    V.append(_vm("airbox", M.loft([(z0, x1 - x0, y1 - y0, 5, cxa, cya), (z1 - 0.040, x1 - x0, y1 - y0, 5, cxa, cya),
                                    (z1 - 0.036, x1 - x0 + 0.010, y1 - y0 + 0.010, 5, cxa, cya),
                                    (z1 - 0.032, x1 - x0 + 0.010, y1 - y0 + 0.010, 5, cxa, cya),
                                    (z1 - 0.004, x1 - x0, y1 - y0, 5, cxa, cya), (z1, x1 - x0 - 0.004, y1 - y0 - 0.004, 6, cxa, cya)],
                                   axis="Z", n=48), PP, "matte"))
    for k, (x, y) in enumerate(((x0 - 0.003, y0 + 0.04), (x0 - 0.003, y1 - 0.04), (x1 + 0.003, y0 + 0.04), (x1 + 0.003, y1 - 0.04))):
        V.append(_vm(f"clip{k}", M.rbox((0.003, 0.012, 0.030), 4), STEEL, "steel", offset=(x, y, z1 - 0.034)))
    snork = M.bend([(cxa, y1, z0 + 0.09), (cxa, y1 + 0.05, z0 + 0.09), (cxa - 0.10, y1 + 0.09, z0 + 0.09)], radius=0.03)
    V.append(_vm("snorkel", M.tube(snork, 0.024, n=24), PP, "matte"))
    for k, (x, y, z) in enumerate(H.BOLTS):
        ax = "Z" if k < 3 else "Y"
        bolt = M.revolve([(0.0, 0.0), (0.0095, 0.0), (0.0095, 0.0015), (0.0065, 0.0015), (0.0065, 0.0065), (0.0, 0.0065)], n=6, axis=ax)
        V.append(_vm(f"bolt{k}", bolt, (0.70, 0.66, 0.40), "steel", offset=(x, y if k < 3 else y + 0.0, z)))
    loom = M.bend([(H.CLIP[0] - 0.004, H.CLIP[1], H.CLIP[2]), (H.CLIP[0] - 0.01, -0.05, H.TOP + 0.02),
                   (H.CLIP[0] - 0.01, 0.02, H.TOP + 0.03)], radius=0.04)
    V.append(_vm("loom", M.tube(loom, 0.0065, n=14), (0.03, 0.03, 0.03), "satin"))
    V.append(_vm("hclip", M.rbox((0.012, 0.014, 0.016), 4), (0.10, 0.10, 0.11), "satin", offset=H.CLIP))
    return _shift(tuple(V), -np.asarray(part_pos, float))
