"""Geometry and measurements for replacing the H11 low-beam bulb in a car headlamp, installed (real size).

Everything is real size; the bench stands for the floor of a real-size front engine-bay section built
on it (an aluminium-extrusion fixture, see README). The headlamp is a 2015-2017 Toyota Camry LE/XLE/
Hybrid halogen assembly, driver side (Toyota 81150-06860, Partslink TO2502222, CAPA Depo 312-11F4L-AC1),
about 750 mm wide, 300 mm tall and 300 mm deep, bolted in as in the car: three bolts along its top into
the upper radiator support and one through its outboard ear into the fender apron. Its back faces the
arms (the arms stand where the engine is). In its low-beam (H11) port sits the old bulb, locked by its
quarter-turn bayonet, its L-shaped socket pointing outboard (-y) with the pigtail connector latched on
it; the connector's two taped leads run as seven colliding articulated links to a harness clip low on
the lamp's back. The air-filter box stands in front of the lamp's inboard half; the new bulb lies in its
opened blister pack on the air box's lid. The radiator is inboard of the lamp, the fender apron outboard.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. Bulbs go in along +x.

Dimensions marked (est.) are estimated from published box sizes and photos: the fixture's T-slot rails
let every mounting point be set to the measured part (README).
"""
from __future__ import annotations

import math
from dataclasses import replace as _replace

import numpy as np

from simkit import Anchor, Hinge, Part, Piece, Spring


def _colliders(pieces):
    """Keep only the colliding pieces, hidden: the look comes from the detailed meshes in visuals.py."""
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


TOP = 0.7475

HOUSING_COL = (0.11, 0.11, 0.12)
GREY_BASE = (0.74, 0.75, 0.77)
CHROME = (0.85, 0.86, 0.88)
CARD_COL = (0.10, 0.22, 0.55)
COPPER = (0.80, 0.45, 0.20)

# -- the headlamp (fixed: bolted in) ------------------------------------------------------------------
LAMP_Y = (-0.450, 0.300)             # 750 mm wide (est.): outboard end at -y (the fender), inboard at +y
BACK_X = 0.400                       # its back face (toward the arms)
LAMP_D = 0.300                       # back to lens (est.)
LAMP_H = 0.300                       # tallest section (est.); it stands on the fixture's floor
FRONT_X = BACK_X + LAMP_D
PORT_Y = -0.080                      # low-beam (H11) port (est., measure on the part)
PORT_Z = TOP + 0.150
HIGH_Y = 0.100                       # high-beam (9005) port, its rubber cover on (est.)
HIGH_Z = TOP + 0.165
CAN_R = 0.040                        # the moulded can round each port
CAN_H = 0.018
BOSS_L = 0.010
BOSS_R = 0.024
HOLE_R = 0.0180                      # PGJ19 port, 36 mm
PORT_FACE_X = BACK_X - CAN_H - BOSS_L   # the bulb's flange seats on this face

# the shell's sections along its width: (y, depth, height, z of its bottom), swept toward the fender
SECTIONS = ((0.300, 0.200, 0.240, 0.020), (0.150, 0.270, 0.290, 0.005), (0.000, 0.300, 0.300, 0.000),
            (-0.150, 0.290, 0.285, 0.006), (-0.300, 0.240, 0.220, 0.040), (-0.450, 0.140, 0.130, 0.090))


def housing_object() -> Part:
    """The lamp: colliders are its back wall round the low-beam port (open) and its body behind it."""
    import visuals
    y0, y1 = -0.400, 0.270
    z0, z1 = TOP + 0.015, TOP + 0.285
    layer = 0.040
    win = 0.022
    body_x0 = BACK_X + layer
    root = ((body_x0 + FRONT_X - 0.02) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    size = (FRONT_X - 0.02 - body_x0, y1 - y0, z1 - z0)
    P = []
    xc = BACK_X + layer / 2 - root[0]
    for (a0, a1, b0, b1) in ((y0, PORT_Y - win, z0, z1), (PORT_Y + win, y1, z0, z1),
                             (PORT_Y - win, PORT_Y + win, z0, PORT_Z - win), (PORT_Y - win, PORT_Y + win, PORT_Z + win, z1)):
        P.append(Piece("box", (layer, a1 - a0, b1 - b0), (xc, (a0 + a1) / 2 - root[1], (b0 + b1) / 2 - root[2]), hidden=True))
    # the port's boss ring (staves) so a bulb is located by it
    n = 16
    r = (HOLE_R + BOSS_R) / 2
    w = 2 * BOSS_R * math.tan(math.pi / n) * 0.95
    for k in range(n):
        t = 2 * math.pi * k / n
        P.append(Piece("box", (BOSS_L + CAN_H, BOSS_R - HOLE_R, w),
                       (BACK_X - (BOSS_L + CAN_H) / 2 - root[0], PORT_Y + r * math.cos(t) - root[1], PORT_Z + r * math.sin(t) - root[2]),
                       rot=(t, 0.0, 0.0), hidden=True))
    return Part("housing", pos=root, size=size, exact_z=True, color=HOUSING_COL, friction=(0.5, 0.4),
                pieces=tuple(P), hide_root=True, vmeshes=visuals.housing_vmeshes(root))


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


# old bulb: axis into the lamp (+x), socket outboard (-y)
R_OLD = np.column_stack([(1, 0, 0), (0, -1, 0), np.cross((1, 0, 0), (0, -1, 0))])
# new bulb: axis +x, socket up (+z): how it lies in the blister and goes into the port (lugs in the
# notches), before the quarter turn (+90 deg about +x) that brings its socket outboard (-y)
R_NEW = np.column_stack([(1, 0, 0), (0, 0, 1), np.cross((1, 0, 0), (0, 0, 1))])
LOCK_DEG = 90.0
DETENT_D = 200.0                     # N m s / deg: a locked bayonet (its detent), turned only deliberately
TURN_D = 0.002                       # N m s / deg: turning in the port (the O-ring's drag)


def bulb_seated_pos():
    """A bulb's grip block when its flange sits on the port's face."""
    return (float(PORT_FACE_X - FLANGE_T - GRIP[0] / 2), PORT_Y, PORT_Z)


def old_bulb_object() -> Part:
    """The old bulb, locked in the port: its bayonet a hinge about the port axis (+x), at its stop at 0."""
    import visuals
    pos = bulb_seated_pos()
    pieces, root = _bulb(R_OLD)
    return Part("old_bulb", pos=pos, size=root, exact_z=True, mass=BULB_MASS, color=GREY_BASE,
                friction=(0.8, 0.7), restitution=0.05, contact_offset=0.0015, pieces=_colliders(pieces),
                hide_root=True, vmeshes=visuals.bulb_vmeshes(R_OLD, "old"),
                hinge=Hinge(pivot=pos, axis="X", damping=DETENT_D, lower_deg=-LOCK_DEG - 0.5, upper_deg=0.5))


# -- the pigtail connector, latched onto the old bulb's socket ------------------------------------------
CONN = (0.018, 0.022, 0.022)         # along x, y (its length off the socket), z


def connector_pos():
    bx, by, bz = bulb_seated_pos()
    end = by - (GRIP[1] / 2 + SOCKET[1])                                 # the socket's end (toward -y)
    return (float(bx - 0.002), float(end - CONN[1] / 2), float(bz))


def _connector_on_new():
    """The connector latched on the new bulb's socket once that bulb is locked: a joint to the new bulb,
    made at spawn, switched on by task.Dynamics when the connector is pushed home."""
    seat = np.asarray(bulb_seated_pos(), float)
    c = np.asarray(connector_pos(), float)
    a = math.radians(LOCK_DEG)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    rel = Rx.T @ (c - seat)
    q = (math.cos(-a / 2), math.sin(-a / 2), 0.0, 0.0)
    return Hinge(pivot=tuple(float(v) for v in c), axis="X", damping=1.0, lower_deg=-0.5, upper_deg=0.5,
                 parent="new_bulb", parent_pos=new_bulb_pos(), local0=tuple(float(v) for v in rel),
                 local1=(0.0, 0.0, 0.0), rot0=q, rot1=(1.0, 0.0, 0.0, 0.0), enabled=False)


def connector_object() -> Part:
    """Black connector on the socket; its latch tab (tab_object) and leads (lead_objects) are their own parts."""
    cx, cy, cz = connector_pos()
    return Part("connector", pos=(cx, cy, cz), size=CONN, exact_z=True, mass=0.012, color=(0.10, 0.10, 0.11),
                friction=(0.8, 0.7), restitution=0.05, contact_offset=0.0015, hide_root=True,
                vmeshes=__import__("visuals").connector_vmeshes((cx, cy, cz)),
                anchor=Anchor(break_force=1e6, break_torque=1e6), hinge=_connector_on_new())


# -- the latch tab: a sprung key on top of the connector -------------------------------------------------
TAB = (0.011, 0.015, 0.0028)
TAB_TRAVEL = 0.0020                  # it presses down 2 mm to its stop
TAB_RELEASE = 0.0012                 # pressed this far, the latch's hook clears the socket's ramp
TAB_K = 3500.0                       # N/m: about 4 N to press it to release, 7 N to its stop


def tab_rest():
    c = np.asarray(connector_pos(), float)
    return tuple(float(v) for v in c + np.array([0.0, -0.001, CONN[2] / 2 + TAB[2] / 2 + 0.0002]))


def tab_object() -> Part:
    pos = tab_rest()
    return Part("tab", pos=pos, size=TAB, exact_z=True, mass=0.002, color=(0.30, 0.30, 0.32), friction=(0.9, 0.8),
                restitution=0.0, contact_offset=0.001, hide_root=True,
                vmeshes=__import__("visuals").tab_vmeshes(),
                spring=Spring(axis="Z", lower=-TAB_TRAVEL, upper=0.0, stiffness=TAB_K, damping=4.0, target=0.0,
                              parent="connector", parent_pos=connector_pos()))


# -- the connector's leads: seven colliding articulated links to the harness clip ------------------------
LEAD_R = 0.0025                      # the two 18 AWG leads taped together: about 5 mm across
LEAD_K = 0.0003                      # N m / deg: bend stiffness at each link joint
LEAD_D = 0.00003                     # N m s / deg
CLIP = (BACK_X - 0.006, -0.172, TOP + 0.030)   # the harness clip on the lamp's back, low


def _lead_path():
    """Axis-aligned runs from the connector's back end: outboard along -y, down, then to the clip along +x."""
    cx, cy, cz = connector_pos()
    e = np.array([cx, cy - CONN[1] / 2, cz])
    corner1 = np.array([cx, CLIP[1], cz])
    corner2 = np.array([cx, CLIP[1], CLIP[2]])
    clip = np.asarray(CLIP, float)
    runs = [(e, corner1, 2, "Y"), (corner1, corner2, 4, "Z"), (corner2, clip, 1, "X")]
    links = []
    for a, b, n, ax in runs:
        for k in range(n):
            links.append((a + (b - a) * k / n, a + (b - a) * (k + 1) / n, ax))
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


# -- the new bulb in its opened blister pack, on the air box's lid --------------------------------------
CARD = (0.110, 0.160, 0.0012)        # card's own frame: the pack is turned 90 deg, its long side along x
AIRBOX = (0.170, 0.350, 0.060, 0.240, TOP, TOP + 0.140)   # x0, x1, y0, y1, z0, z1: 180 x 180 x 140 mm
CARD_C = (0.246, 0.150)
CARD_YAW = math.pi / 2


def _to_card(dx, dy):
    """World offset from the card's centre to the card's own frame (turned CARD_YAW about z)."""
    return (dy, -dx)


def blister_object() -> Part:
    """Printed card with its clear formed tray (lid torn back, its near end open); a small moulded block
    under the bulb's grip keeps it level, its flange resting on the card."""
    import visuals
    cx, cy = CARD_C
    cz = AIRBOX[5] + CARD[2] / 2
    tray = dict(color=(0.90, 0.93, 0.96), material="acrylic", opacity=0.35)
    P = []
    for sx in (-1, 1):
        P.append(Piece("box", (0.003, 0.090, 0.018), (sx * 0.036, 0.020, CARD[2] / 2 + 0.009), **tray))
    P.append(Piece("box", (0.075, 0.003, 0.010), (0.0, -0.026, CARD[2] / 2 + 0.005), **tray))   # far end; near end open
    nb = new_bulb_pos()
    top = cz + CARD[2] / 2
    h = nb[2] - GRIP[1] / 2 - top                                      # under the grip's bottom face
    for dy in (-0.0095, 0.0095):                                        # the moulded cradle's two rails
        lx, ly = _to_card(nb[0] - 0.003 - cx, nb[1] + dy - cy)
        P.append(Piece("box", (0.005, 0.016, h), (lx, ly, CARD[2] / 2 + h / 2), **tray))
    fx = nb[0] + GRIP[0] / 2 + FLANGE_T / 2
    for dy in (-0.015, 0.015):                                          # chocks either side of the flange
        lx, ly = _to_card(fx - cx, nb[1] + dy - cy)
        P.append(Piece("box", (0.006, 0.008, 0.006), (lx, ly, CARD[2] / 2 + 0.003), **tray))
    return Part("blister", pos=(cx, cy, cz), size=CARD, exact_z=True, color=CARD_COL, friction=(0.6, 0.5),
                yaw=CARD_YAW, pieces=_colliders(P), hide_root=True, vmeshes=visuals.blister_vmeshes())


def new_bulb_pos():
    """The new bulb lying in the tray, axis +x, socket up: its grip block's centre; its flange on the card."""
    cx, cy = CARD_C
    return (float(cx - 0.050), float(cy), float(AIRBOX[5] + CARD[2] + FLANGE_R + 0.0008))


def new_bulb_object() -> Part:
    import visuals
    pieces, root = _bulb(R_NEW)
    seat = bulb_seated_pos()
    return Part("new_bulb", pos=new_bulb_pos(), size=root, exact_z=True, mass=BULB_MASS, color=GREY_BASE,
                friction=(0.8, 0.7), restitution=0.05, contact_offset=0.0015, pieces=_colliders(pieces),
                hide_root=True, vmeshes=visuals.bulb_vmeshes(R_NEW, "new"),
                # its bayonet in the port: made at spawn, switched on when the bulb is put into the port
                # square (task.Dynamics); the joint frame sits on the port axis at the seat
                hinge=Hinge(pivot=seat, axis="X", damping=TURN_D, lower_deg=-0.5, upper_deg=LOCK_DEG + 0.5,
                            local0=seat, local1=(0.0, 0.0, 0.0), rot0=(1.0, 0.0, 0.0, 0.0), rot1=(1.0, 0.0, 0.0, 0.0),
                            enabled=False))


# -- measurements ----------------------------------------------------------------------------------
def mat(pose):
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def port_axis():
    return np.array([1.0, 0.0, 0.0])


def capsule_segment(pose):
    """End points (world) of the quartz capsule's axis for a bulb at ``pose`` (its axis is +x in its frame)."""
    p = np.asarray(pose, float)
    a = mat(p) @ np.array([1.0, 0.0, 0.0])
    c0 = GRIP[0] / 2 + FLANGE_T + 0.008
    return p[:3] + a * c0, p[:3] + a * (c0 + CAPSULE_L)


# -- the engine-bay section the lamp is bolted into (fixed; real size, built on the bench) ----------------
# Boxes (x0, x1, y0, y1, z0, z1): the colliders. visuals.bay_vmeshes draws the parts.
BAY = {
    "tie_bar": (0.395, 0.460, -0.330, 0.640, TOP + 0.300, TOP + 0.330),     # upper radiator support (hat channel)
    "radiator": (0.440, 0.470, 0.315, 0.640, TOP + 0.020, TOP + 0.300),     # radiator core, inboard of the lamp
    "end_tank": (0.430, 0.475, 0.300, 0.330, TOP + 0.020, TOP + 0.300),     # its plastic end tank
    "fender": (0.250, 0.800, -0.476, -0.472, TOP + 0.000, TOP + 0.330),     # fender apron (inner panel)
    "fender_rail": (0.250, 0.800, -0.520, -0.472, TOP + 0.326, TOP + 0.330),  # its top bolt rail
    "ear_bracket": (0.590, 0.630, -0.472, -0.444, TOP + 0.180, TOP + 0.184),  # the lamp's side ear bolts here
    "airbox": AIRBOX[:2] + AIRBOX[2:4] + AIRBOX[4:],
}
BOLTS = ((0.430, -0.300, TOP + 0.330), (0.430, -0.020, TOP + 0.330), (0.430, 0.240, TOP + 0.330),
         (0.610, -0.452, TOP + 0.184))


def bay_object() -> Part:
    import visuals
    x0, x1, y0, y1, z0, z1 = BAY["tie_bar"]
    pos = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    P = []
    for key, (a0, a1, b0, b1, c0, c1) in BAY.items():
        if key == "tie_bar":
            continue
        P.append(Piece("box", (a1 - a0, b1 - b0, c1 - c0), ((a0 + a1) / 2 - pos[0], (b0 + b1) / 2 - pos[1],
                                                           (c0 + c1) / 2 - pos[2]), hidden=True))
    return Part("bay", pos=pos, size=(x1 - x0, y1 - y0, z1 - z0), exact_z=True, color=(0.1, 0.1, 0.1),
                friction=(0.6, 0.5), pieces=tuple(P), hide_root=True, vmeshes=visuals.bay_vmeshes(pos))
