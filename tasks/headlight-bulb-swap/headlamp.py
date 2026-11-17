"""Geometry and measurements for replacing the H11 low-beam bulb in a car headlamp (real size).

Everything is real size; the bench is only the working height. A halogen headlamp unit, taken off the
car, stands on a folded shop towel on its bottom edge as it sits in the car, its back toward the
right arm. In its low-beam port sits the old H11 bulb: a quarter-turn bayonet base whose L-shaped socket
points toward the arms, with the lamp's own pigtail connector latched onto it. The port's rubber dust
cover has already been taken off and lies on the towel. The new H11 bulb lies in its opened blister
pack on the bench. The connector is unlatched (tab pressed) and pulled off, the old bulb is turned a
quarter turn counter-clockwise (seen from above) and drawn out, the new bulb goes in by its base (tabs
in the port's notches) and is turned a quarter turn clockwise to lock, and the connector is pushed back
on until its latch clicks. The new bulb's glass capsule is not to be touched (skin oil on quartz
halogen glass makes it fail early).
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The lamp stands as in the
car, lens toward +y, back toward -y: a bulb goes in along +y.

Real parts:
  headlamp   halogen reflector headlamp unit for a compact car, about 400 W x 200 H x 250 D mm
             (lying lens-down its depth is vertical), black polypropylene housing, clear polycarbonate
             lens, aluminised reflectors, 1.6 kg.
  H11 bulb   12 V 55 W, PGJ19-2 base: 33.5 mm across its three locating tabs, grey PBT plastic base
             and grip, L-shaped socket with two 2.8 mm blade terminals; quartz capsule 10 mm across.
             25 g. Locks with a quarter turn (90 degrees) clockwise.
  connector  H11 pigtail connector, black PA66, 22 x 18 x 22 mm, a press-to-release locking tab on
             top; two 18 AWG leads to the lamp's main plug.
"""
from __future__ import annotations

import math

import numpy as np

from dataclasses import replace as _replace

from simkit import Anchor, Hinge, Part, Piece, Spring


def _colliders(pieces):
    """Keep only the colliding pieces, hidden: the look comes from the detailed meshes in visuals.py."""
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)

TOP = 0.7475

HOUSING_COL = (0.11, 0.11, 0.12)      # black PP (textured, reads dark grey)
GREY_BASE = (0.74, 0.75, 0.77)       # light-grey PBT bulb base
CHROME = (0.85, 0.86, 0.88)
LENS_COL = (0.86, 0.89, 0.93)
TOWEL_COL = (0.70, 0.12, 0.10)       # red shop towel
CARD_COL = (0.10, 0.22, 0.55)        # printed blister card
RUBBER = (0.08, 0.08, 0.09)
AMBER = (0.95, 0.55, 0.08)
COPPER = (0.80, 0.45, 0.20)
WIRE_A = (0.08, 0.08, 0.08)
WIRE_B = (0.85, 0.85, 0.85)

# -- towel and headlamp (fixed) ---------------------------------------------------------------------
# The lamp stands on the towel the way it sits in the car: lens toward +y, back toward -y (toward the
# right arm), its 200 mm height vertical.
TOWEL = (0.460, 0.330, 0.010)        # x, y, z: folded shop towel
TOWEL_C = (0.400, 0.055)
LAMP_W = 0.400                       # along x
LAMP_H = 0.200                       # up
LAMP_D = 0.250                       # along y, back to lens
LAMP_X = 0.400                       # centre along x
BACK_Y = -0.060                      # the back face
LENS_Y = BACK_Y + LAMP_D
BOT_Z = TOP + TOWEL[2]
PORT_X = 0.330                       # low-beam (H11) port, open
HIGH_X = 0.470                       # high-beam (9005) port, cover on
PORT_Z = TOP + 0.115
CAN_R = 0.040                        # the moulded "can" round each port
CAN_H = 0.018
BOSS_L = 0.010                       # the port's boss stands this far out of the can
BOSS_R = 0.024
HOLE_R = 0.0180                      # the PGJ19 port: 36 mm
PORT_FACE_Y = BACK_Y - CAN_H - BOSS_L   # the bulb's flange seats on this face


def towel_object() -> Part:
    import visuals
    return Part("towel", pos=(TOWEL_C[0], TOWEL_C[1], TOP + TOWEL[2] / 2), size=TOWEL, exact_z=True,
                color=TOWEL_COL, friction=(0.9, 0.8), hide_root=True, vmeshes=visuals.towel_vmeshes())


def housing_object() -> Part:
    """Headlamp standing on the towel: a clear lens with a chrome bezel at the front, aluminised
    reflectors behind it, the black housing tapering back in three steps, two moulded port cans on the
    back, adjuster screws on top, mounting ears, an amber side marker and the main plug. Colliders leave
    the low-beam port's column open so a bulb fits into it."""
    cx = LAMP_X
    cy = (BACK_Y + LENS_Y) / 2
    cz = BOT_Z + LAMP_H / 2
    P = []
    ly = LENS_Y - 0.012 - cy
    P.append(Piece("box", (LAMP_W, 0.024, LAMP_H), (0.0, ly, 0.0), color=LENS_COL, material="acrylic", opacity=0.30,
                   collide=False))                                       # lens
    for xx in (PORT_X, HIGH_X):
        P.append(Piece("cylinder", (0.072, 0.004, 0.072), (xx - cx, ly - 0.016, PORT_Z - cz), axis="Y", color=CHROME,
                       material="aluminium", collide=False))             # reflector bowls behind the lens
        P.append(Piece("cylinder", (0.018, 0.012, 0.018), (xx - cx, ly - 0.010, PORT_Z - cz), axis="Y",
                       color=(0.30, 0.30, 0.32), collide=False))         # bulb shield caps
    P.append(Piece("box", (LAMP_W + 0.004, 0.006, LAMP_H + 0.004), (0.0, ly - 0.015, 0.0), color=CHROME,
                   material="aluminium_brushed", collide=False))         # bezel
    steps = ((0.075, LAMP_W - 0.004, LAMP_H - 0.004), (0.080, LAMP_W - 0.040, LAMP_H - 0.030),
             (LAMP_D - 0.024 - 0.075 - 0.080 - 0.006, LAMP_W - 0.090, LAMP_H - 0.060))
    y = LENS_Y - 0.024 - 0.006
    for d, sx, sz in steps:
        P.append(Piece("box", (sx, d, sz), (0.0, y - d / 2 - cy, (sz - LAMP_H) / 2), color=HOUSING_COL, collide=False))
        y -= d
    P.append(Piece("box", (0.006, 0.090, 0.020), (-LAMP_W / 2 + 0.003, ly - 0.05, -LAMP_H / 2 + 0.040), color=AMBER,
                   collide=False))                                       # amber side marker
    by = BACK_Y - cy
    for xx in (PORT_X, HIGH_X):                                          # port cans
        P.append(Piece("cylinder", (CAN_R, CAN_H, CAN_R), (xx - cx, by - CAN_H / 2, PORT_Z - cz), axis="Y",
                       color=HOUSING_COL, collide=False))
    n = 16
    r = (HOLE_R + BOSS_R) / 2
    w = 2 * BOSS_R * math.tan(math.pi / n) * 0.95
    for k in range(n):                                                   # the low-beam boss: a ring round the open port
        t = 2 * math.pi * k / n
        P.append(Piece("box", (BOSS_R - HOLE_R, BOSS_L + CAN_H, w),
                       (PORT_X - cx + r * math.cos(t), by - (BOSS_L + CAN_H) / 2, PORT_Z - cz + r * math.sin(t)),
                       rot=(0.0, -t, 0.0), color=HOUSING_COL))
    P.append(Piece("cylinder", (0.031, 0.020, 0.031), (HIGH_X - cx, by - CAN_H - 0.010, PORT_Z - cz), axis="Y",
                   color=RUBBER))                                        # high-beam dust cover, on
    P.append(Piece("box", (0.010, 0.004, 0.006), (HIGH_X - cx, by - CAN_H - 0.022, PORT_Z - cz + 0.034),
                   color=RUBBER, collide=False))                         # its pull tab
    for xx in (-0.14, 0.14):                                             # beam adjuster screws (blue caps)
        P.append(Piece("cylinder", (0.007, 0.012, 0.007), (xx, by + 0.050, LAMP_H / 2 - 0.054), axis="Z",
                       color=(0.15, 0.40, 0.75), collide=False))
    for sx in (-1, 1):                                                   # mounting ears with steel bushings
        P.append(Piece("box", (0.030, 0.030, 0.008), (sx * (LAMP_W / 2 + 0.012), ly - 0.10, LAMP_H / 2 - 0.040),
                       color=HOUSING_COL, collide=False))
        P.append(Piece("cylinder", (0.006, 0.010, 0.006), (sx * (LAMP_W / 2 + 0.016), ly - 0.10, LAMP_H / 2 - 0.040),
                       axis="Z", color=CHROME, material="steel", collide=False))
    P.append(Piece("box", (0.040, 0.022, 0.030), (0.0, by - 0.011, -LAMP_H / 2 + 0.050), color=(0.20, 0.20, 0.22),
                   collide=False))                                       # main plug
    # colliders: the back layer round a window at the low-beam port, and the body ahead of it
    layer = 0.050
    win = 0.022
    bx0, bx1 = -(LAMP_W - 0.09) / 2, (LAMP_W - 0.09) / 2
    bz0, bz1 = -(LAMP_H - 0.06) / 2 - 0.0, (LAMP_H - 0.06) / 2
    bz0 = bz1 - (LAMP_H - 0.06)
    pzc = PORT_Z - cz
    # back steps are bottom-aligned: shift the window layer to sit where the top step is
    zoff = (LAMP_H - 0.06 - LAMP_H) / 2
    bz0, bz1 = zoff - (LAMP_H - 0.06) / 2, zoff + (LAMP_H - 0.06) / 2
    pxc = PORT_X - cx
    yc = by + layer / 2
    for (a0, a1, b0, b1) in ((bx0, pxc - win, bz0, bz1), (pxc + win, bx1, bz0, bz1),
                             (pxc - win, pxc + win, bz0, pzc - win), (pxc - win, pxc + win, pzc + win, bz1)):
        P.append(Piece("box", (a1 - a0, layer, b1 - b0), ((a0 + a1) / 2, yc, (b0 + b1) / 2), color=HOUSING_COL))
    root_y0 = BACK_Y + layer
    root_d = LENS_Y - root_y0
    rc = root_y0 + root_d / 2
    from dataclasses import replace
    P = [replace(p, offset=(p.offset[0], p.offset[1] + cy - rc, p.offset[2])) for p in P]
    import visuals
    return Part("housing", pos=(cx, rc, cz), size=(LAMP_W, root_d, LAMP_H), exact_z=True, color=HOUSING_COL,
                friction=(0.5, 0.4), pieces=_colliders(P), hide_root=True,
                vmeshes=visuals.housing_vmeshes((cx, rc, cz)))


# -- H11 bulbs -----------------------------------------------------------------------------------
# Built in the bulb's own frame: +a is its axis into the lamp (toward the capsule), +s is where its
# L-shaped socket points, t = a x s. A placement maps (a, s, t) onto world axes.
FLANGE_R = 0.0200                    # flange with its O-ring: seats on the port's boss
FLANGE_T = 0.0030
TAB_R = 0.0160                       # the three bayonet lugs under the flange
GRIP = (0.020, 0.026, 0.024)         # grip block behind the flange: along a, s, t
SOCKET = (0.014, 0.016, 0.016)       # the L-shaped socket out of the grip toward +s: along a, s, t
CAPSULE_R = 0.0050
CAPSULE_L = 0.026
BULB_MASS = 0.025
_AX = {(1, 0, 0): "X", (0, 1, 0): "Y", (0, 0, 1): "Z"}


def _place(R):
    """Return helpers mapping bulb-frame offsets/sizes/axes to the world for a rotation R (columns a, s, t)."""
    R = np.asarray(R, float)

    def off(a, s, t):
        return tuple(float(v) for v in R @ np.array([a, s, t]))

    def size(a, s, t):
        return tuple(float(v) for v in np.abs(R) @ np.array([a, s, t]))

    def axis(col):
        v = tuple(int(round(abs(x))) for x in R[:, col])
        return _AX[v]

    return off, size, axis


def _bulb(R):
    """Pieces and root size for a bulb whose frame maps by R. The root is the grip block."""
    off, size, axis = _place(R)
    A = axis(0)
    fa = GRIP[0] / 2 + FLANGE_T / 2
    P = [Piece("cylinder", (FLANGE_R, FLANGE_T, FLANGE_R), off(fa, 0, 0), axis=A)]
    for k in range(3):                                                   # the three locating tabs
        t = 2 * math.pi * k / 3 + math.radians(30)
        P.append(Piece("cylinder", (0.0015, 0.003, 0.0015), off(fa + FLANGE_T / 2 + 0.0015, TAB_R * math.cos(t),
                                                                  TAB_R * math.sin(t)), axis=A))
    ring = fa + FLANGE_T / 2 + 0.004
    P.append(Piece("cylinder", (0.0085, 0.008, 0.0085), off(ring, 0, 0), axis=A, color=CHROME, material="steel"))
    cap = fa + FLANGE_T / 2 + 0.008 + CAPSULE_L / 2
    P.append(Piece("cylinder", (CAPSULE_R, CAPSULE_L, CAPSULE_R), off(cap, 0, 0), axis=A, color=(0.95, 0.96, 0.97),
                   material="glass", opacity=0.45))
    P.append(Piece("sphere", (CAPSULE_R, CAPSULE_R, CAPSULE_R), off(cap + CAPSULE_L / 2, 0, 0), color=(0.25, 0.25, 0.27),
                   collide=False))                                       # blackened tip
    P.append(Piece("cylinder", (0.0012, 0.010, 0.0012), off(cap, 0, 0), axis=A, color=(0.55, 0.40, 0.20),
                   collide=False))                                       # filament
    for k in (-0.008, 0.0, 0.008):                                       # grip ribs
        for sgn in (-1, 1):
            P.append(Piece("box", size(GRIP[0] * 0.8, 0.0015, 0.0015), off(0, k, sgn * (GRIP[2] / 2 + 0.0005)),
                           collide=False))
    sk = (-0.002, GRIP[1] / 2 + SOCKET[1] / 2, 0)
    P.append(Piece("box", size(*SOCKET), off(*sk)))                      # the socket
    for d in (-0.0025, 0.0025):                                          # its two blade terminals (drawn)
        P.append(Piece("box", size(0.006, 0.006, 0.0008), off(sk[0], sk[1] + SOCKET[1] / 2 - 0.004, d), color=COPPER,
                       collide=False))
    return tuple(P), size(*GRIP)


# old bulb: axis into the lamp (+y), socket toward the arms (-x)
R_OLD = np.column_stack([(0, 1, 0), (-1, 0, 0), np.cross((0, 1, 0), (-1, 0, 0))])
# new bulb lying in the blister: axis along +y, socket pointing down (-z) into the tray: the orientation
# it goes into the port in (its lugs in the notches), before the quarter turn that brings the socket to -x
R_NEW = np.column_stack([(0, 1, 0), (0, 0, -1), np.cross((0, 1, 0), (0, 0, -1))])
LOCK_DEG = 90.0                      # quarter turn clockwise seen from behind: +90 deg about +y
DETENT_D = 200.0                     # N m s / deg: a locked bayonet (its detent spring), turned only deliberately
TURN_D = 0.002                       # N m s / deg: turning in the port (the O-ring's drag)


def bulb_seated_pos():
    """The old bulb's grip block when its flange sits on the port's face."""
    return (PORT_X, PORT_FACE_Y - FLANGE_T - GRIP[0] / 2, PORT_Z)


def old_bulb_object() -> Part:
    """The old bulb, locked in the port (its bayonet a hinge about the port axis, held at 0 till turned)."""
    pos = bulb_seated_pos()
    pieces, root = _bulb(R_OLD)
    import visuals
    return Part("old_bulb", pos=pos, size=root, exact_z=True, mass=BULB_MASS, color=GREY_BASE,
                friction=(0.8, 0.7), restitution=0.05, contact_offset=0.0015, pieces=_colliders(pieces),
                hide_root=True, vmeshes=visuals.bulb_vmeshes(R_OLD, "old"),
                hinge=Hinge(pivot=pos, axis="Y", damping=DETENT_D, lower_deg=-LOCK_DEG - 0.5, upper_deg=0.5))


# -- the pigtail connector, latched onto the old bulb's socket --------------------------------------
CONN = (0.022, 0.018, 0.022)         # along x (its length off the socket), y, z


def connector_pos():
    bx, by, bz = bulb_seated_pos()
    sx = bx - (GRIP[1] / 2 + SOCKET[1])                                  # the socket's end (toward -x)
    return (sx - CONN[0] / 2, by - 0.002, bz)                            # pushed on up to the socket's shoulder


def _connector_on_new():
    """The connector latched on the new bulb's socket once that bulb is locked in the port: a joint to the
    new bulb, made at spawn and switched on by task.Dynamics when the connector is pushed home. Its frame in
    the new bulb is the connector's seated pose relative to the bulb locked (turned +90 deg about +y)."""
    seat = np.asarray(bulb_seated_pos(), float)
    c = np.asarray(connector_pos(), float)
    a = math.radians(LOCK_DEG)
    Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    rel = Ry.T @ (c - seat)
    q = (math.cos(-a / 2), 0.0, math.sin(-a / 2), 0.0)                 # w x y z: the connector's turn in the bulb
    return Hinge(pivot=tuple(c), axis="X", damping=1.0, lower_deg=-0.5, upper_deg=0.5, parent="new_bulb",
                 parent_pos=new_bulb_pos(), local0=tuple(rel), local1=(0.0, 0.0, 0.0), rot0=q,
                 rot1=(1.0, 0.0, 0.0, 0.0), enabled=False)


def connector_object() -> Part:
    """Black connector on the socket. Its latch tab is a separate sprung part (tab_object); its leads are a
    chain of flexible links (lead_objects) from its back end to the harness clip on the lamp."""
    cx, cy, cz = connector_pos()
    return Part("connector", pos=(cx, cy, cz), size=CONN, exact_z=True, mass=0.012, color=(0.10, 0.10, 0.11),
                friction=(0.8, 0.7), restitution=0.05, contact_offset=0.0015, hide_root=True,
                vmeshes=__import__("visuals").connector_vmeshes((cx, cy, cz)),
                anchor=Anchor(break_force=1e6, break_torque=1e6), hinge=_connector_on_new())


# -- the latch tab: a sprung key on top of the connector ---------------------------------------------
TAB = (0.015, 0.011, 0.0028)
TAB_TRAVEL = 0.0020                  # it presses down 2 mm to its stop
TAB_RELEASE = 0.0012                 # pressed this far, the latch's hook clears the socket's ramp
TAB_K = 3500.0                       # N/m: about 4 N to press it to release, 7 N to its stop


def tab_rest():
    c = np.asarray(connector_pos(), float)
    return tuple(float(v) for v in c + np.array([-0.001, 0.0, CONN[2] / 2 + TAB[2] / 2 + 0.0002]))


def tab_object() -> Part:
    pos = tab_rest()
    return Part("tab", pos=pos, size=TAB, exact_z=True, mass=0.002, color=(0.30, 0.30, 0.32), friction=(0.9, 0.8),
                restitution=0.0, contact_offset=0.001, hide_root=True,
                vmeshes=__import__("visuals").tab_vmeshes(),
                spring=Spring(axis="Z", lower=-TAB_TRAVEL, upper=0.0, stiffness=TAB_K, damping=4.0, target=0.0,
                              parent="connector", parent_pos=connector_pos()))


# -- the connector's leads: a chain of flexible links to the harness clip ------------------------------
LEAD_R = 0.0025                      # the two 18 AWG leads taped together: about 5 mm across
LEAD_K = 0.0003                      # N m / deg: bend stiffness at each link joint
LEAD_D = 0.00003                     # N m s / deg
CLIP = (0.235, -0.071, TOP + 0.030)  # the harness clip on the lamp's back, low on its inboard corner


def _lead_path():
    """Axis-aligned runs from the connector's back end: back along -x, down, then to the clip along +y."""
    cx, cy, cz = connector_pos()
    e = np.array([cx - CONN[0] / 2, cy, cz])
    corner1 = np.array([0.235, cy, cz])
    corner2 = np.array([0.235, cy, CLIP[2]])
    clip = np.asarray(CLIP, float)
    runs = [(e, corner1, 2, "X"), (corner1, corner2, 4, "Z"), (corner2, clip, 1, "Y")]
    links = []
    for a, b, n, ax in runs:
        for k in range(n):
            p0 = a + (b - a) * k / n
            p1 = a + (b - a) * (k + 1) / n
            links.append((p0, p1, ax))
    return links


def lead_objects():
    """Seven capsule links, each jointed (D6, sprung) to the next; the first to the connector's back end,
    the last to the clip (the world). They collide with the lamp, the hands and each other's neighbours'
    neighbours; adjacent links do not."""
    links = _lead_path()
    out = []
    for i, (p0, p1, ax) in enumerate(links):
        c = (p0 + p1) / 2
        L = float(np.linalg.norm(p1 - p0))
        f3 = lambda v: tuple(float(x) for x in v)
        parent, ppos = ("connector", f3(connector_pos())) if i == 0 else (f"lead{i - 1}", f3((links[i - 1][0] + links[i - 1][1]) / 2))
        j = [("j_prev", parent, ppos, f3(p0), LEAD_K, LEAD_D)]
        if i == len(links) - 1:
            j.append(("j_clip", None, None, f3(p1), LEAD_K * 3, LEAD_D))
        out.append(Part(f"lead{i}", pos=f3(c), size=(LEAD_R, float(max(L - 2 * LEAD_R, 0.002)), LEAD_R), shape="capsule",
                        axis=ax, exact_z=True, mass=0.003, color=(0.05, 0.05, 0.05), friction=(0.6, 0.5),
                        restitution=0.0, contact_offset=0.0015,
                        vmeshes=__import__("visuals").lead_vmeshes(ax, L, i), d6=tuple(j)))
    return out


# -- the low-beam port's dust cover (off, on the towel) ----------------------------------------------
def dust_cover_object() -> Part:
    pos = (0.215, -0.085, TOP + TOWEL[2] + 0.010)
    return Part("dust_cover", pos=pos, size=(0.031, 0.020, 0.031), shape="cylinder", axis="Z", exact_z=True,
                mass=0.020, color=RUBBER, friction=(1.0, 0.9), hide_root=True,
                vmeshes=__import__("visuals").dust_cover_vmeshes())


# -- the new bulb in its opened blister pack (bench, in front of the right arm) ----------------------
CARD = (0.110, 0.160, 0.0012)
CARD_C = (0.225, -0.300)


def blister_object() -> Part:
    """Printed card with its clear formed tray (lid torn back): a trough the bulb lies in."""
    cx, cy = CARD_C
    cz = TOP + CARD[2] / 2
    P = [Piece("box", (0.080, 0.050, 0.0004), (0.0, -0.050, CARD[2] / 2 + 0.0002), color=(0.92, 0.92, 0.95),
               collide=False),
         Piece("box", (0.045, 0.012, 0.0004), (0.0, -0.068, CARD[2] / 2 + 0.0004), color=(0.85, 0.10, 0.10),
               collide=False)]
    tray = dict(color=(0.90, 0.93, 0.96), material="acrylic", opacity=0.35)
    for sx in (-1, 1):
        P.append(Piece("box", (0.003, 0.090, 0.018), (sx * 0.036, 0.020, CARD[2] / 2 + 0.009), **tray))
    P.append(Piece("box", (0.075, 0.003, 0.018), (0.0, 0.066, CARD[2] / 2 + 0.009), **tray))
    P.append(Piece("box", (0.075, 0.003, 0.010), (0.0, -0.026, CARD[2] / 2 + 0.005), **tray))
    # the moulded cradle the bulb's flange rests in (the bulb stands on its socket, flange on the cradle)
    nb = new_bulb_pos()
    fy = nb[1] + GRIP[0] / 2 + FLANGE_T / 2 - cy
    fh = nb[2] - FLANGE_R - (TOP + CARD[2])
    P.append(Piece("box", (0.030, 0.006, fh), (nb[0] - cx, fy, CARD[2] / 2 + fh / 2), **tray))
    P.append(Piece("box", (0.012, 0.030, nb[2] - CAPSULE_R - 0.004 - (TOP + CARD[2])), (nb[0] - cx, fy + 0.030,
                   CARD[2] / 2 + (nb[2] - CAPSULE_R - 0.004 - (TOP + CARD[2])) / 2), **tray))
    P.append(Piece("box", (0.050, 0.090, 0.0006), (0.040, 0.020, CARD[2] / 2 + 0.018), rot=(0.0, -0.9, 0.0),
                   collide=False, **tray))                              # its torn-back lid
    import visuals
    return Part("blister", pos=(cx, cy, cz), size=CARD, exact_z=True, color=CARD_COL, friction=(0.6, 0.5),
                pieces=_colliders(P), hide_root=True, vmeshes=visuals.blister_vmeshes())


def new_bulb_pos():
    """The new bulb on its side in the trough: its grip block's centre (socket toward -x, lying flat)."""
    cx, cy = CARD_C
    return (cx + 0.004, cy + 0.008, TOP + CARD[2] + GRIP[1] / 2 + SOCKET[1] + 0.0005)


def new_bulb_object() -> Part:
    pieces, root = _bulb(R_NEW)
    import visuals
    seat = bulb_seated_pos()
    return Part("new_bulb", pos=new_bulb_pos(), size=root, exact_z=True, mass=BULB_MASS, color=GREY_BASE,
                friction=(0.8, 0.7), restitution=0.05, contact_offset=0.0015, pieces=_colliders(pieces),
                hide_root=True, vmeshes=visuals.bulb_vmeshes(R_NEW, "new"),
                # its bayonet in the port: made at spawn, switched on when the bulb is put into the port
                # square (task.Dynamics); the joint frame sits on the port axis at the seat
                hinge=Hinge(pivot=seat, axis="Y", damping=TURN_D, lower_deg=-0.5, upper_deg=LOCK_DEG + 0.5,
                            local0=seat, local1=(0.0, 0.0, 0.0), rot0=(1.0, 0.0, 0.0, 0.0), rot1=(1.0, 0.0, 0.0, 0.0),
                            enabled=False))


# -- measurements ----------------------------------------------------------------------------------
def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def port_axis():
    return np.array([0.0, 1.0, 0.0])


def capsule_segment(pose):
    """End points (world) of the quartz capsule's axis for a bulb at ``pose`` (its Part frame)."""
    p = np.asarray(pose, float)
    R = mat(p)
    fa = GRIP[0] / 2 + FLANGE_T / 2
    c0 = fa + FLANGE_T / 2 + 0.008
    a = R @ np.array([0.0, 1.0, 0.0])                                    # the bulb's axis: +y in its Part frame
    return p[:3] + a * c0, p[:3] + a * (c0 + CAPSULE_L)


# -- the engine-bay section the lamp is bolted into (fixed; real size, cut at the bench) ---------------
# Boxes (x0, x1, y0, y1, z0, z1) in world metres: the colliders. visuals.bay_vmeshes draws the parts.
BAY = {
    "tie_bar": (-0.060, 0.215, 0.100, 0.150, TOP + 0.195, TOP + 0.225),      # upper radiator support (steel hat channel)
    "radiator": (-0.060, 0.150, 0.135, 0.157, TOP + 0.000, TOP + 0.195),     # radiator core (aluminium fins)
    "end_tank": (0.150, 0.172, 0.128, 0.162, TOP + 0.010, TOP + 0.195),      # its plastic end tank
    "bracket_in": (0.170, 0.222, 0.030, 0.110, TOP + 0.160, TOP + 0.164),    # lamp's inboard top bracket
    "fender": (0.640, 0.644, -0.220, 0.250, TOP + 0.000, TOP + 0.260),       # fender apron (inner panel)
    "fender_rail": (0.598, 0.644, -0.220, 0.250, TOP + 0.255, TOP + 0.260),  # its top bolt rail
    "bracket_out": (0.574, 0.640, 0.020, 0.060, TOP + 0.150, TOP + 0.154),   # lamp's outboard bracket to the fender
    "lower_support": (0.548, 0.640, -0.020, 0.020, TOP + 0.000, TOP + 0.022),  # lamp's lower mount
    "airbox": (0.470, 0.620, -0.310, -0.130, TOP + 0.000, TOP + 0.150),      # air filter box beside the lamp's back
}
BOLTS = ((0.200, 0.040, TOP + 0.164), (0.582, 0.040, TOP + 0.154), (0.566, 0.000, TOP + 0.022))


def bay_object() -> Part:
    import visuals
    pos = (0.0775, 0.125, TOP + 0.210)                                   # the tie bar's centre (root collider)
    P = []
    for key, (x0, x1, y0, y1, z0, z1) in BAY.items():
        if key == "tie_bar":
            continue
        P.append(Piece("box", (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2 - pos[0], (y0 + y1) / 2 - pos[1],
                                                           (z0 + z1) / 2 - pos[2]), hidden=True))
    x0, x1, y0, y1, z0, z1 = BAY["tie_bar"]
    return Part("bay", pos=pos, size=(x1 - x0, y1 - y0, z1 - z0), exact_z=True, color=(0.1, 0.1, 0.1),
                friction=(0.6, 0.5), pieces=tuple(P), hide_root=True, vmeshes=visuals.bay_vmeshes(pos))
