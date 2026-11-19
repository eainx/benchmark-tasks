"""Geometry and parts for chipping the slag off a stick weld (real size).

A 3/8 in (9.5 mm) x 3 in x 8 in mild-steel weld coupon is tacked flat to a steel welding fixture plate on the
bench. A 150 mm stringer bead runs along it (y), still covered by its slag: a dark glassy crust 1.2 mm thick that
has cracked across into six pieces about 25 mm long as it cooled. A welder's chipping hammer (chisel end, point
end, coil-spring handle, about 0.40 kg) lies on the bench to the right of the fixture. The job: knock the slag
off the bead, then put the hammer down.

Physics: the coupon, bead and fixture plate are one fixed body. Each slag piece is a free rigid body held where
it lies by a PhysX fixed joint to the world that breaks at SLAG_BREAK_N newtons or SLAG_BREAK_NM newton-metres
(simkit Anchor). The piece's own collider sits on the bead with a 0.3 mm gap all round, so a load pressing it
down onto the bead is carried by contact, while a sideways or lifting load is carried by the joint until it
breaks: a blow driven into the toe of the slag (where it meets the plate) pops the piece off; nothing else
frees it. Frame: metres, x away from the arm bases, y left, z up; TOP is the bench.
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace

import numpy as np

import meshlib as M
from simkit import Anchor, Part, Piece

TOP = 0.7475

# -- fixture plate, coupon, bead ------------------------------------------------------------------------
FIX_C = (0.340, -0.080)          # fixture plate centre
FIX_SX, FIX_SY, FIX_T = 0.260, 0.260, 0.012
FIX_TOP = TOP + FIX_T
CPN_SX, CPN_SY, CPN_T = 0.076, 0.200, 0.0095          # 3 x 8 in coupon, 3/8 in plate
CPN_C = (0.340, -0.085)
CT = FIX_TOP + CPN_T             # coupon top
BX = 0.340                       # bead line (along y)
BEAD_Y0, BEAD_Y1 = -0.160, -0.010
BEAD_W, BEAD_H = 0.009, 0.003    # collider: 9 mm wide, 3 mm high crown (visual 10 mm)

# -- slag ---------------------------------------------------------------------------------------------
N_SLAG = 6
SEG_L = 0.024                    # each piece 24 mm long, 1 mm crack between pieces
GAP = 0.0003                     # piece collider to bead / coupon
SHELL = 0.0020
SLAG_MASS = 0.004                # kg (slag about 3 g/cm3; 1.2 mm crust over a 10 mm bead, 24 mm long)
SLAG_BREAK_N = 5.0               # N: the crust's bond to the bead, sideways or lifting
SLAG_BREAK_NM = 0.05             # N m
SLAGS = tuple(f"slag{i}" for i in range(N_SLAG))


def slag_y(i):
    return BEAD_Y0 + (i + 0.5) * (BEAD_Y1 - BEAD_Y0) / N_SLAG


CAP_Z = CT + BEAD_H + GAP + SHELL / 2          # cap collider centre
CAP_W = 0.012                                  # the crust's collider: a 12 x 2 mm cap over the bead's crown ...
STRIP_H = 0.001                                # the bead's collider: only its lowest 1 mm (see fixture_object)
SKIRT_X = CAP_W / 2 - SHELL / 2                # ... and a 2 mm skirt down each side, to 1.3 mm over the plate
SKIRT_B = CT + STRIP_H + GAP
SKIRT_H = (CAP_Z - SHELL / 2) - SKIRT_B
SKIRT_Z = SKIRT_B + SKIRT_H / 2
SLAG_Z = CAP_Z                                 # body origin at the cap collider centre

# -- chipping hammer ------------------------------------------------------------------------------------
# Hammer frame (as it lies on the bench, on its side): origin at the head's centre, the head along y (chisel
# end toward -y, point toward +y), the handle along -x from the head, the head's 16 mm depth along z.
HEAD_L = 0.110                   # head body between the chisel and point tapers
HEAD_S = 0.016
CHISEL_TIP = 0.081               # chisel edge this far from the head centre (-y)
POINT_TIP = 0.080
SHANK = (-0.008, -0.048, 0.005)  # x0, x1, radius
HANDLE = (-0.048, -0.163, 0.010)
GRIP_X = -0.075                  # where the hand takes the handle (from the head centre)
HAMMER_MASS = 0.40
HAMMER_C = (0.270, -0.300, TOP + HANDLE[2])   # head centre as it lies (resting on its handle)

# colours
MILL = (0.20, 0.21, 0.23)
FIXTURE = (0.30, 0.31, 0.33)
HOLE = (0.03, 0.03, 0.035)
WELD = (0.62, 0.62, 0.63)
TINT_STRAW = (0.60, 0.48, 0.26)
TINT_BLUE = (0.26, 0.24, 0.42)
SLAG = (0.17, 0.145, 0.12)
SLAG_EDGE = (0.30, 0.25, 0.18)
HEAD = (0.16, 0.16, 0.17)
GROUND = (0.70, 0.70, 0.72)
CHROME = (0.80, 0.81, 0.83)


def _vm(name, mesh, color, material=None, opacity=1.0, offset=(0, 0, 0), R=None):
    p, t = M.transform(mesh, R, offset)
    return (name, p, t, color, material, opacity)


def _colliders(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def disc(r, n=20):
    return M.revolve([(0.0, 0.0), (r, 0.0)], n=n, axis="Z")


# ------------------------------------------------------------------------------- fixed: fixture + coupon
def bead_mesh(y0, y1, w=0.010, h=0.0034, n_rip=None):
    """A stringer bead: a low rounded ridge along y with the ripples a stick electrode leaves (crescents every
    ~2.5 mm), lower half buried in the plate."""
    L = y1 - y0
    n = max(8, int(L / 0.0006))
    secs = []
    for k in range(n + 1):
        s = k / n
        y = y0 + s * L
        rip = 0.5 + 0.5 * math.cos(2 * math.pi * (y - y0) / 0.0025)
        end = min(1.0, (s * L) / 0.006, ((1 - s) * L) / 0.006) ** 0.5
        ww = w * (0.94 + 0.06 * rip) * max(end, 0.25)
        hh = h * (0.93 + 0.07 * rip) * max(end, 0.15)
        secs.append((y, ww, 2 * hh, 2.3, 0.0, 0.0))
    return M.loft(secs, axis="Y", n=24)


def fixture_object() -> Part:
    """Fixture plate (bolted to the bench), the coupon tacked to it, and the bead: one fixed body."""
    V = []
    fz = TOP + FIX_T / 2
    c = (FIX_C[0], FIX_C[1], fz)
    V.append(_vm("fixplate", M.rbox((FIX_SX, FIX_SY, FIX_T), e=14, axis="Z"), FIXTURE, "steel_painted"))
    # the fixture plate's 16 mm hole grid (50 mm pitch), as on a welding table
    for i in range(5):
        for j in range(5):
            x = -0.100 + 0.050 * i
            y = -0.100 + 0.050 * j
            if abs(x + c[0] - CPN_C[0]) < CPN_SX / 2 + 0.010 and abs(y + c[1] - CPN_C[1]) < CPN_SY / 2 + 0.010:
                continue
            V.append(_vm(f"hole{i}{j}", disc(0.008), HOLE, "matte", offset=(x, y, FIX_T / 2 + 0.0002)))
    # bolts holding the plate down at its corners
    for k, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        head = M.revolve(M.fillet_profile(0.0, 0.009, FIX_T / 2, FIX_T / 2 + 0.006, f=0.0015), n=6, axis="Z")
        V.append(_vm(f"bolt{k}", head, (0.12, 0.12, 0.13), "satin", offset=(sx * (FIX_SX / 2 - 0.016), sy * (FIX_SY / 2 - 0.016), 0)))
    # coupon: mill-scale plate with sheared edges
    oc = (CPN_C[0] - c[0], CPN_C[1] - c[1], FIX_T / 2 + CPN_T / 2)
    V.append(_vm("coupon", M.rbox((CPN_SX, CPN_SY, CPN_T), e=16, axis="Z"), MILL, "steel_painted", offset=oc))
    top_rel = FIX_T / 2 + CPN_T
    # heat tint either side of the bead (straw nearest, blue further out)
    for s in (-1, 1):
        for k, (dx, w, col) in enumerate(((0.0085, 0.006, TINT_STRAW), (0.0150, 0.008, TINT_BLUE))):
            V.append(_vm(f"tint{s}{k}", M.rbox((w, BEAD_Y1 - BEAD_Y0 + 0.010, 0.0003), e=4, axis="Z"), col, "matte", 0.75,
                         offset=(BX - c[0] + s * dx, (BEAD_Y0 + BEAD_Y1) / 2 - c[1], top_rel + 0.00015)))
    # the bead (bright weld metal where the slag comes off); its crater at the far end
    bm = bead_mesh(BEAD_Y0, BEAD_Y1)
    V.append(_vm("bead", bm, WELD, "steel", offset=(BX - c[0], -c[1], top_rel)))
    # tack welds joining the coupon to the fixture plate at both ends
    for k, y in enumerate((CPN_C[1] - CPN_SY / 2, CPN_C[1] + CPN_SY / 2)):
        tk = M.loft([(-0.009, 0.006, 0.006, 2.2), (0.0, 0.008, 0.009, 2.2), (0.009, 0.006, 0.006, 2.2)], axis="X", n=16)
        V.append(_vm(f"tack{k}", tk, (0.34, 0.33, 0.33), "satin", offset=(CPN_C[0] - c[0], y - c[1], FIX_T / 2 + 0.0005)))
    # weld spatter on the coupon
    rnd = np.random.default_rng(5)
    for k in range(26):
        x = BX + rnd.choice((-1, 1)) * rnd.uniform(0.010, 0.032)
        y = rnd.uniform(BEAD_Y0 - 0.008, BEAD_Y1 + 0.008)
        r = rnd.uniform(0.0005, 0.0012)
        sp = M.revolve([(0.0, -r), (r * 0.8, -r * 0.6), (r, 0.0), (r * 0.8, r * 0.5), (0.0, r * 0.7)], n=10, axis="Z")
        V.append(_vm(f"spat{k}", sp, (0.28, 0.27, 0.27), "satin", offset=(x - c[0], y - c[1], top_rel)))
    P = [Piece("box", (CPN_SX, CPN_SY, CPN_T), oc),
         Piece("box", (BEAD_W, BEAD_Y1 - BEAD_Y0, STRIP_H), (BX - c[0], (BEAD_Y0 + BEAD_Y1) / 2 - c[1], top_rel + STRIP_H / 2))]
    return Part("fixture", pos=c, size=(FIX_SX, FIX_SY, FIX_T), exact_z=True, color=FIXTURE, friction=(0.6, 0.5),
                restitution=0.2, pieces=_colliders(P), hide_root=True, vmeshes=tuple(V), contact_offset=0.002)


# ------------------------------------------------------------------------------------------ slag pieces
def slag_vmeshes(i):
    """A slag piece in its frame (origin at its cap collider's centre): a glassy dark crust over the bead,
    flowing out thin at the toes, with ragged cracked ends."""
    rnd = np.random.default_rng(100 + i)
    L = SEG_L
    secs = []
    n = 14
    for k in range(n + 1):
        s = k / n
        y = -L / 2 + s * L
        rag = 1.0 - 0.10 * rnd.random()
        secs.append((y, 0.0124 * rag, 2 * (BEAD_H + 0.0022) * (0.96 + 0.04 * rnd.random()), 2.0, 0.0, 0.0))
    p, t = M.loft(secs, axis="Y", n=28)
    p = p + np.array([0.0, 0.0, CT - SLAG_Z])        # its base on the coupon top
    out = [("crust", p, t, SLAG, "gloss", 1.0)]
    # a lighter, rougher rim where it cracked from its neighbours
    for k, y in enumerate((-L / 2, L / 2)):
        e = M.loft([(y - 0.0004, 0.0124, 2 * (BEAD_H + 0.0022), 2.0), (y + 0.0004, 0.0124, 2 * (BEAD_H + 0.0022), 2.0)], axis="Y", n=28)
        pe = e[0] * np.array([1.004, 1.0, 1.02]) + np.array([0.0, 0.0, CT - SLAG_Z])
        out.append((f"crack{k}", pe, e[1], SLAG_EDGE, "matte", 1.0))
    return tuple(out)


def slag_objects():
    out = []
    for i in range(N_SLAG):
        P = [Piece("box", (SHELL, SEG_L, SKIRT_H), (-SKIRT_X, 0.0, SKIRT_Z - SLAG_Z)),
             Piece("box", (SHELL, SEG_L, SKIRT_H), (SKIRT_X, 0.0, SKIRT_Z - SLAG_Z))]
        out.append(Part(SLAGS[i], pos=(BX, slag_y(i), SLAG_Z), size=(CAP_W, SEG_L, SHELL),
                        exact_z=True, mass=SLAG_MASS, color=SLAG, friction=(0.6, 0.5), restitution=0.1,
                        contact_offset=0.001, max_depenetration=0.3, solver_iters=(24, 4),
                        pieces=_colliders(P), hide_root=True, vmeshes=slag_vmeshes(i),
                        anchor=Anchor(break_force=SLAG_BREAK_N, break_torque=SLAG_BREAK_NM)))
    return out


# --------------------------------------------------------------------------------------- chipping hammer
def hammer_vmeshes():
    V = []
    hs = HEAD_S
    # head: a forged square bar, its middle where the shank enters a little swollen
    V.append(_vm("head", M.loft([(-HEAD_L / 2, hs, hs, 8.0), (-0.012, hs, hs, 8.0), (0.0, hs * 1.12, hs * 1.08, 8.0),
                                 (0.012, hs, hs, 8.0), (HEAD_L / 2, hs, hs, 8.0)], axis="Y", n=32), HEAD, "satin"))
    # chisel end: tapers in x to a ground edge along z
    ch = M.loft([(-HEAD_L / 2, hs, hs, 8.0), (-0.068, 0.009, hs, 6.0), (-CHISEL_TIP + 0.002, 0.0025, hs, 4.0),
                 (-CHISEL_TIP, 0.0012, hs * 0.96, 4.0)], axis="Y", n=32)
    V.append(_vm("chisel", ch, HEAD, "satin"))
    edge = M.loft([(-0.074, 0.0062, hs * 1.004, 4.0), (-CHISEL_TIP - 0.0002, 0.0014, hs * 0.97, 4.0)], axis="Y", n=32)
    V.append(_vm("ground_edge", edge, GROUND, "steel"))
    # point end: a square-to-round taper to a point
    pt = M.loft([(HEAD_L / 2, hs, hs, 8.0), (0.064, 0.012, 0.012, 3.0), (0.074, 0.005, 0.005, 2.0), (POINT_TIP, 0.0012, 0.0012, 2.0)],
                axis="Y", n=32)
    V.append(_vm("point", pt, HEAD, "satin"))
    # shank and the coil-spring handle round a steel core, with an end cap
    x0, x1, r = SHANK
    V.append(_vm("shank", M.loft([(x1, 2 * r, 2 * r, 2.0), (x0, 2 * r * 1.15, 2 * r * 1.15, 2.0)], axis="X", n=20), HEAD, "satin"))
    h0, h1, R = HANDLE
    V.append(_vm("core", M.loft([(h1, 0.010, 0.010, 2.0), (h0, 0.010, 0.010, 2.0)], axis="X", n=16), (0.25, 0.25, 0.26), "satin"))
    turns = (h0 - h1) / 0.0042
    ts = np.linspace(0.0, turns * 2 * math.pi, int(turns * 14))
    rc = R - 0.0019
    path = np.stack([h1 + (h0 - h1) * ts / ts[-1], rc * np.cos(ts), rc * np.sin(ts)], axis=1)
    V.append(_vm("spring", M.tube(path, 0.0019, n=7), CHROME, "steel"))
    cap = M.revolve(M.fillet_profile(0.0, R + 0.0005, -0.006, 0.0, f=0.002), n=24, axis="X")
    V.append(_vm("endcap", cap, CHROME, "steel", offset=(h1, 0, 0)))
    ferrule = M.loft([(h0 - 0.006, 2 * R + 0.001, 2 * R + 0.001, 2.0), (h0 + 0.004, 0.013, 0.013, 2.0)], axis="X", n=20)
    V.append(_vm("ferrule", ferrule, CHROME, "steel"))
    return tuple(V)


def hammer_object() -> Part:
    x0, x1, r = SHANK
    h0, h1, R = HANDLE
    P = [Piece("box", (0.006, CHISEL_TIP - HEAD_L / 2, HEAD_S), (0.0, -(HEAD_L / 2 + CHISEL_TIP) / 2, 0.0)),
         Piece("box", (0.008, POINT_TIP - HEAD_L / 2, 0.008), (0.0, (HEAD_L / 2 + POINT_TIP) / 2, 0.0)),
         Piece("cylinder", (r, x0 - x1, r), ((x0 + x1) / 2, 0.0, 0.0), axis="X"),
         Piece("cylinder", (R, h0 - h1, R), ((h0 + h1) / 2, 0.0, 0.0), axis="X")]
    return Part("hammer", pos=HAMMER_C, size=(HEAD_S, HEAD_L, HEAD_S), exact_z=True, mass=HAMMER_MASS,
                com=(-0.030, 0.0, 0.0), color=HEAD, friction=(0.8, 0.7), restitution=0.2, contact_offset=0.002,
                pieces=_colliders(P), hide_root=True, vmeshes=hammer_vmeshes(), solver_iters=(24, 4))


def scene_objects():
    return [fixture_object(), hammer_object()] + slag_objects()
