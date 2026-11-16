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
    """World-frame meshes, returned in the housing Part's frame."""
    bot = H.BOT_Z
    xc = H.LAMP_X
    V = []
    # shell: tapered toward the back and swept in on the outboard (+x) side, bottoms level. Over its front
    # 115 mm the outboard corner is lens: the clear lens wraps round that corner, as on most cars.
    y1 = H.LENS_Y - 0.115
    secs = []
    for y, w, h, e, du in ((H.BACK_Y, 0.300, 0.140, 4.0, -0.020), (H.BACK_Y + 0.04, 0.322, 0.155, 4.5, -0.016),
                           (H.BACK_Y + 0.10, 0.352, 0.172, 5.0, -0.010), (y1, 0.384, 0.188, 6.0, -0.004)):
        secs.append((y, w, h, e, xc + du, bot + h / 2))
    V.append(_vm("shell", M.loft(secs, axis="Y", n=64), HOUSING, "matte"))
    V.append(_vm("shell_front", M.loft([(y1 - 0.001, 0.300, 0.190, 6.0, xc - 0.046, bot + 0.095),
                                        (H.LENS_Y - 0.030, 0.304, 0.198, 7.0, xc - 0.048, bot + 0.099)], axis="Y", n=64),
                 HOUSING, "matte"))
    V.append(_vm("seam", M.loft([(y1 - 0.006, 0.392, 0.194, 6.0, xc - 0.002, bot + 0.097),
                                  (y1 - 0.001, 0.392, 0.194, 6.0, xc - 0.002, bot + 0.097)], axis="Y", n=64), SEAM))
    V.append(_vm("lens", M.loft([(y1, 0.396, 0.196, 6.0, xc, bot + 0.098),
                                  (H.LENS_Y - 0.040, 0.400, 0.200, 7.0, xc, bot + 0.100),
                                  (H.LENS_Y - 0.010, 0.394, 0.194, 6.5, xc, bot + 0.100),
                                  (H.LENS_Y, 0.374, 0.178, 5.0, xc, bot + 0.100)], axis="Y", n=64),
                 LENS, "acrylic", 0.22))
    # inside the corner wrap: a chrome turn-signal reflector with its amber bulb, a chrome trim strip
    xw = xc + 0.104
    prof = [(0.046 - 0.036 * k / 7, 0.030 * (1 - ((0.046 - 0.036 * k / 7) / 0.046) ** 2)) for k in range(8)]
    V.append(_vm("turn_reflector", M.revolve(prof, n=48, axis="X"), CHROME, "aluminium",
                 offset=(xw, H.LENS_Y - 0.062, bot + 0.110)))
    V.append(_vm("turn_bulb", M.revolve([(0.0, 0.0), (0.006, 0.002), (0.0065, 0.010), (0.004, 0.016), (0.0, 0.017)],
                                        n=24, axis="X"), AMBER, "acrylic", 0.9, offset=(xw + 0.006, H.LENS_Y - 0.062,
                                                                                         bot + 0.110)))
    V.append(_vm("drl_strip", M.rbox((0.004, 0.090, 0.010), 6, axis="X"), CHROME, "steel",
                 offset=(xw + 0.004, H.LENS_Y - 0.060, bot + 0.040)))
    V.append(_vm("wrap_bezel", M.rbox((0.003, 0.100, 0.160), 6, axis="X"), (0.13, 0.13, 0.14), "satin",
                 offset=(xw - 0.002, H.LENS_Y - 0.064, bot + 0.098)))
    # moulded stiffening ribs over the top and down the back
    for k in range(6):
        V.append(_vm(f"top_rib{k}", M.rbox((0.270 - 0.012 * k, 0.004, 0.005), 4), HOUSING, "matte",
                     offset=(xc - 0.020, H.BACK_Y + 0.040 + 0.030 * k, bot + 0.156 + 0.006 * k + 0.002)))
    for k, x in enumerate((xc - 0.140, xc, xc + 0.105)):                # between and beside the port cans
        V.append(_vm(f"back_rib{k}", M.rbox((0.004, 0.006, 0.110), 4, axis="Y"), HOUSING, "matte",
                     offset=(x, H.BACK_Y - 0.002, bot + 0.068)))
    # aluminised reflector bowls (paraboloids) with a black inner bezel between them, seen through it
    for i, x in enumerate((H.PORT_X, H.HIGH_X)):
        prof = []
        for k in range(9):
            r = 0.068 - 0.054 * k / 8
            prof.append((r, -0.050 * (1 - (r / 0.068) ** 2)))
        bowl = M.revolve(prof, n=48, axis="Y")
        V.append(_vm(f"reflector{i}", bowl, CHROME, "aluminium", offset=(x, H.LENS_Y - 0.032, H.PORT_Z)))
        V.append(_vm(f"shield{i}", M.revolve([(0.0, 0.0), (0.012, 0.0), (0.014, -0.004), (0.014, -0.010), (0.0, -0.012)],
                                             n=32, axis="Y"), STEEL, "steel", offset=(x, H.LENS_Y - 0.040, H.PORT_Z)))
    V.append(_vm("bezel", M.rbox((0.012, 0.004, 0.170), 6, axis="Y"), BLACK, offset=(xc, H.LENS_Y - 0.040, bot + 0.100)))
    V.append(_vm("marker", M.rbox((0.004, 0.080, 0.024), 8, axis="X"), AMBER, "acrylic", 0.85,
                 offset=(xc + 0.199, H.LENS_Y - 0.060, bot + 0.040)))
    # moulded port cans on the back, ribbed; the low-beam one carries the open PGJ19 boss
    for i, x in enumerate((H.PORT_X, H.HIGH_X)):
        prof = [(0.050, 0.0), (0.046, -0.002), (0.0425, -0.006), (H.CAN_R, -0.010), (H.CAN_R, -H.CAN_H + 0.002),
                (H.CAN_R - 0.003, -H.CAN_H)]
        if i == 0:
            prof += [(H.BOSS_R + 0.001, -H.CAN_H), (H.BOSS_R, -H.CAN_H - 0.002), (H.BOSS_R, -H.CAN_H - H.BOSS_L + 0.0015),
                     (H.BOSS_R - 0.0015, -H.CAN_H - H.BOSS_L), (H.HOLE_R, -H.CAN_H - H.BOSS_L), (H.HOLE_R, -0.004)]
            V.append(_vm("bore", M.revolve([(H.HOLE_R - 0.0003, -0.004), (0.0, -0.004)], n=32, axis="Y"), BLACK,
                         offset=(x, H.BACK_Y, H.PORT_Z)))
            V.append(_vm("boss_rim", M.revolve([(H.BOSS_R + 0.0012, -H.CAN_H - 0.002), (H.BOSS_R + 0.0012, -H.CAN_H - H.BOSS_L),
                                                (H.HOLE_R - 0.0004, -H.CAN_H - H.BOSS_L - 0.0006),
                                                (H.HOLE_R - 0.0004, -H.CAN_H - H.BOSS_L + 0.001)], n=64, axis="Y"),
                         (0.30, 0.30, 0.32), "satin", offset=(x, H.BACK_Y, H.PORT_Z)))
        else:
            prof += [(0.0, -H.CAN_H)]
        V.append(_vm(f"can{i}", M.revolve(prof, n=64, axis="Y"), (0.15, 0.15, 0.16), "satin", offset=(x, H.BACK_Y, H.PORT_Z)))
        V.append(_vm(f"seal{i}", M.revolve([(0.050, 0.0005), (0.0525, -0.001), (0.0525, -0.0035), (0.050, -0.0045)],
                                           n=64, axis="Y"), (0.02, 0.02, 0.02), "gloss", offset=(x, H.BACK_Y, H.PORT_Z)))
        for k in range(8):
            t = 2 * math.pi * k / 8 + math.pi / 8
            rib = M.rbox((0.0035, H.CAN_H - 0.004, 0.006), 4, axis="Y")
            V.append(_vm(f"rib{i}_{k}", rib, HOUSING,
                         offset=(x + (H.CAN_R + 0.0012) * math.cos(t), H.BACK_Y - H.CAN_H / 2 - 0.001,
                                 H.PORT_Z + (H.CAN_R + 0.0012) * math.sin(t))))
        if i == 0:                                                       # the boss's three bayonet notches
            for k in range(3):
                t = 2 * math.pi * k / 3 + math.radians(30)
                V.append(_vm(f"notch{k}", M.rbox((0.004, 0.0012, 0.004), 4, axis="Y"), BLACK,
                             offset=(x + (H.HOLE_R + 0.0015) * math.cos(t), H.PORT_FACE_Y - 0.0004,
                                     H.PORT_Z + (H.HOLE_R + 0.0015) * math.sin(t))))
    # high-beam port's rubber dust cover (on), with its pull tab and moulded grip rings
    yb = H.BACK_Y - H.CAN_H
    cover = [(H.CAN_R - 0.002, 0.003), (0.0335, 0.0), (0.0335, -0.012), (0.032, -0.016), (0.026, -0.020), (0.0, -0.021)]
    V.append(_vm("cover", M.revolve(cover, n=64, axis="Y"), RUBBER, offset=(H.HIGH_X, yb, H.PORT_Z)))
    for k, rr in enumerate((0.018, 0.010)):
        V.append(_vm(f"cover_ring{k}", M.revolve([(rr + 0.0015, -0.0205), (rr + 0.0015, -0.022), (rr - 0.0015, -0.022),
                                                  (rr - 0.0015, -0.0205)], n=48, axis="Y"), RUBBER,
                     offset=(H.HIGH_X, yb, H.PORT_Z)))
    V.append(_vm("cover_tab", M.rbox((0.012, 0.003, 0.016), 4, axis="Y"), RUBBER,
                 offset=(H.HIGH_X, yb - 0.004, H.PORT_Z + 0.040)))
    # beam adjusters (blue gear caps) on top, vent cap, mounting ears with steel bushings, main plug
    top_back = bot + 0.140
    for k, x in enumerate((xc - 0.120, xc + 0.100)):
        gear = [(0.0, 0.0), (0.009, 0.0), (0.009, 0.004), (0.006, 0.006), (0.006, 0.012), (0.0, 0.012)]
        V.append(_vm(f"adjuster{k}", M.revolve(gear, n=24, axis="Z"), BLUE_CAP,
                     offset=(x, H.BACK_Y + 0.030, bot + 0.150 + 0.0)))
        V.append(_vm(f"adj_hex{k}", M.revolve([(0.0, 0.0), (0.004, 0.0), (0.004, 0.005), (0.0, 0.005)], n=6, axis="Z"),
                     STEEL, "steel", offset=(x, H.BACK_Y + 0.030, bot + 0.162)))
    V.append(_vm("vent", M.revolve([(0.0, 0.0), (0.008, 0.0), (0.009, 0.006), (0.006, 0.010), (0.0, 0.011)], n=24),
                 BLACK, offset=(xc + 0.020, H.BACK_Y + 0.060, bot + 0.164)))
    for k, (x, y, z) in enumerate(((xc - 0.200, H.LENS_Y - 0.150, bot + 0.150), (xc + 0.182, H.LENS_Y - 0.150, bot + 0.150),
                                   (xc + 0.166, H.BACK_Y + 0.060, bot + 0.020))):
        V.append(_vm(f"ear{k}", M.rbox((0.040, 0.034, 0.009), 3.0), HOUSING, "matte", offset=(x, y, z)))
        V.append(_vm(f"bush{k}", M.revolve([(0.0045, -0.006), (0.0075, -0.006), (0.0080, -0.0055), (0.0080, 0.0055),
                                            (0.0075, 0.006), (0.0045, 0.006), (0.0045, -0.006)], n=32), (0.85, 0.85, 0.82),
                     "steel", offset=(x, y, z)))
    V.append(_vm("plug", M.rbox((0.044, 0.024, 0.030), 5, axis="Y"), (0.16, 0.16, 0.17),
                 offset=(xc + 0.000, H.BACK_Y - 0.012, bot + 0.040)))
    V.append(_vm("plug_latch", M.rbox((0.014, 0.010, 0.004), 4), (0.16, 0.16, 0.17),
                 offset=(xc + 0.000, H.BACK_Y - 0.018, bot + 0.057)))
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
    cx, cy, cz = part_pos
    TAB = (0.30, 0.30, 0.32)
    V = [_vm("body", M.rbox(H.CONN, 5, axis="X"), (0.025, 0.025, 0.028), "gloss"),
         _vm("seal", M.rbox((0.0025, H.CONN[1] - 0.002, H.CONN[2] - 0.002), 6, axis="X"), (0.45, 0.45, 0.46), "matte",
             offset=(H.CONN[0] / 2 + 0.0008, 0.0, 0.0)),
         _vm("tab", M.rbox((0.015, 0.011, 0.0028), 4), TAB, "satin", offset=(-0.001, 0.0, H.CONN[2] / 2 + 0.0020)),
         _vm("tab_ramp", M.rbox((0.003, 0.011, 0.0045), 4), TAB, "satin", offset=(0.0060, 0.0, H.CONN[2] / 2 + 0.0030)),
         _vm("boot", M.revolve([(0.0, 0.0), (0.0065, 0.0), (0.0058, -0.004), (0.0050, -0.008), (0.0045, -0.012),
                                (0.0, -0.012)], n=24, axis="X"), (0.02, 0.02, 0.02), "gloss",
             offset=(-H.CONN[0] / 2, 0.0, -0.002))]
    for k in range(3):                                                   # grip ridges on the tab
        V.append(_vm(f"tab_ridge{k}", M.rbox((0.0010, 0.010, 0.0010), 3), (0.55, 0.55, 0.57),
                     offset=(-0.006 + 0.004 * k, 0.0, H.CONN[2] / 2 + 0.0036)))
    for k, d in enumerate((-0.004, 0.004)):                              # the two cavity windows on its back
        V.append(_vm(f"cavity{k}", M.rbox((0.0006, 0.0045, 0.006), 6, axis="X"), (0.35, 0.22, 0.12),
                     offset=(-H.CONN[0] / 2 - 0.0002, d, 0.006)))
    # two 18 AWG leads out of the boot, down to the towel and along the lamp's back to its main plug
    for k, (dy, col) in enumerate(((-0.0024, (0.02, 0.02, 0.02)), (0.0024, WHITE))):
        x0 = cx - H.CONN[0] / 2 - 0.012
        pts = [(x0 + 0.004, cy + dy, cz - 0.002), (x0 - 0.010, cy + dy, cz - 0.004), (x0 - 0.016, cy + dy, H.BOT_Z + 0.004),
               (x0 + 0.020, cy + dy - 0.004, H.BOT_Z + 0.002), (H.LAMP_X - 0.030, H.BACK_Y - 0.026 + dy, H.BOT_Z + 0.002),
               (H.LAMP_X - 0.024, H.BACK_Y - 0.020 + dy, H.BOT_Z + 0.030)]
        path = M.bend(pts, radius=0.012, steps=6)
        V.append(_vm(f"lead{k}", M.tube(path, 0.0020, n=10), col, "gloss", offset=(-cx, -cy, -cz)))
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
