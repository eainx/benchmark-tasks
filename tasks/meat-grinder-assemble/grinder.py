"""Geometry and measurements for putting a washed #10 hand meat grinder back together (real size).

Everything is real size; the bench is the kitchen worktop. A tinned cast-iron #10 hand mincer stands
on its suction-lock foot with its barrel level and its mouth toward the arms; its crank is on the
back. Its parts lie washed and dried on a cotton tea towel beside it: the auger (worm), the four-wing
cross knife, the 4.5 mm grinding plate and the threaded retaining ring. They go back in order: auger
into the barrel, knife onto the auger's square with its flat cutting face out, plate over the auger's
pin with its notch on the barrel's key, ring screwed on over the plate hand-tight. The knife and plate
only sit on the auger's square and pin, so one hand keeps them home while the other starts the ring.
Frame: metres, x away from the arm bases, y left, z up; TOP is the bench. The barrel's axis runs
along x; its mouth faces -x.

Each part collides through simple hidden shapes (staves, discs, boxes); what is drawn is a separate
visual mesh of the real part's shape (meshes.py).
"""
from __future__ import annotations

import math

import numpy as np

import meshes as M
from simkit import Part, Piece

TOP = 0.7475

TIN = (0.74, 0.75, 0.76)              # tinned cast iron, satin
TIN_DARK = (0.46, 0.47, 0.49)
STEEL = (0.86, 0.87, 0.89)            # stainless knife and plate
WOOD = (0.66, 0.45, 0.26)             # beech crank handle
RUBBER = (0.07, 0.07, 0.08)
DARK = (0.05, 0.05, 0.06)

LOOK_TIN = {"color": TIN, "metallic": 0.75, "roughness": 0.42}
LOOK_TIN_DARK = {"color": TIN_DARK, "metallic": 0.6, "roughness": 0.5}
LOOK_STEEL = {"color": STEEL, "metallic": 0.7, "roughness": 0.30}
LOOK_EDGE = {"color": (0.95, 0.95, 0.97), "metallic": 1.0, "roughness": 0.08}
LOOK_WOOD = {"color": WOOD, "metallic": 0.0, "roughness": 0.45}
LOOK_RUBBER = {"color": RUBBER, "metallic": 0.0, "roughness": 0.75}
LOOK_DARK = {"color": DARK, "metallic": 0.0, "roughness": 0.8}


def _rel(mesh, origin):
    p, t = mesh
    return np.asarray(p, float) - np.asarray(origin, float), t


# -- the mincer (fixed: its suction foot is locked down) ---------------------------------------
# A #10 hand mincer: 2-1/8 in (54 mm) plate size. Barrel bore 50 mm, wall 7 mm; its mouth (front
# face) has a 66 mm external thread for the ring and, inside, a 6 mm counterbore the plate sits in,
# with a key at 12 o'clock that the plate's rim notch fits over.
AXIS_Y = 0.040
AXIS_Z = TOP + 0.115
MOUTH_X = 0.340                       # the barrel's front face
BORE_R = 0.025
WALL = 0.007
BARREL_L = 0.150                      # mouth to the back bearing wall
THREAD_R = 0.033                      # 66 mm over the ring thread
THREAD_L = 0.018
PLATE_SEAT = 0.006                    # counterbore depth: the plate sits flush in the mouth
KEY = (0.008, 0.006, 0.005)           # x deep, y wide, z tall: the key at 12 o'clock
BACK_X = MOUTH_X + BARREL_L
SOCKET_X = BACK_X + 0.008             # the crank's square socket behind the back wall
HOPPER_X = MOUTH_X + 0.085
CRANK_R = 0.150                       # crank arm: axis to handle


def _mincer_visuals(foot_c):
    def W(mesh):                                         # world-frame mesh -> the part's frame
        return _rel(mesh, foot_c)

    ax = np.array([MOUTH_X, AXIS_Y, AXIS_Z])
    ro = BORE_R + WALL
    # barrel: one closed lathe section (bore, front face, threaded collar, body, back wall)
    prof = [(BORE_R - 0.0002, PLATE_SEAT + 0.0005), (BORE_R + 0.0012, PLATE_SEAT), (BORE_R + 0.0012, 0.0006),
            (BORE_R + 0.0020, 0.0), (THREAD_R - 0.0016, 0.0), (THREAD_R - 0.0010, 0.0008), (THREAD_R - 0.0010, THREAD_L),
            (ro + 0.0006, THREAD_L + 0.0015), (ro + 0.0006, BARREL_L - 0.004), (ro - 0.0030, BARREL_L + 0.006),
            (0.009, BARREL_L + 0.009), (0.009, BARREL_L), (BORE_R - 0.0002, BARREL_L - 0.002),
            (BORE_R - 0.0002, PLATE_SEAT + 0.0005)]
    barrel = M.place(M.lathe(prof, n=96), M.Z_TO_X, ax)
    thread = M.place(M.helix_flight(THREAD_R - 0.0012, THREAD_R + 0.0004, 0.0028, THREAD_L - 0.003, 0.0011, per_turn=72),
                     M.Z_TO_X, ax + (0.0015, 0.0, 0.0))
    # casting seam (parting line) along each side of the barrel and a raised boss under the hopper
    seams = [M.box((BARREL_L - THREAD_L - 0.008, 0.0012, 0.0016),
                   (MOUTH_X + THREAD_L + (BARREL_L - THREAD_L) / 2, AXIS_Y + s * (ro + 0.0004), AXIS_Z), bevel=0.0004)
             for s in (-1, 1)]
    # hopper: a round flared hopper with a rolled rim, on a neck into the barrel
    hx, hy = HOPPER_X, AXIS_Y
    neck_z0 = AXIS_Z + ro - 0.006
    top_z = AXIS_Z + 0.132
    hp = [(0.0215, neck_z0 + 0.034), (0.0215, neck_z0), (0.0275, neck_z0), (0.0275, neck_z0 + 0.030),
          (0.058, top_z - 0.004), (0.0605, top_z - 0.001), (0.0608, top_z + 0.003), (0.0585, top_z + 0.005),
          (0.0560, top_z + 0.003), (0.0555, top_z), (0.0245, neck_z0 + 0.036), (0.0215, neck_z0 + 0.034)]
    hopper = M.place(M.lathe(hp, n=80), None, (hx, hy, 0.0))
    throat = M.place(M.lathe([(0.0, neck_z0 + 0.010), (0.021, neck_z0 + 0.010)], n=40), None, (hx, hy, 0.0))
    # pedestal from the barrel down to the foot, waisted, with a flange the screws hold
    pz0 = TOP + 0.024
    pz1 = AXIS_Z - ro + 0.004
    ped_prof = [(0.0, pz0), (0.046, pz0), (0.046, pz0 + 0.006), (0.034, pz0 + 0.010), (0.022, pz0 + 0.026),
                (0.017, (pz0 + pz1) / 2), (0.019, pz1 - 0.012), (0.026, pz1), (0.0, pz1)]
    pedestal = M.place(M.lathe(ped_prof, n=64), None, (hx + 0.005, hy, 0.0))
    screws = [M.place(M.lathe([(0.0, pz0 + 0.0115), (0.0045, pz0 + 0.0115), (0.0045, pz0 + 0.006), (0.0, pz0 + 0.006)], n=6),
                      None, (hx + 0.005 + 0.036 * math.cos(a), hy + 0.036 * math.sin(a), -0.002))
              for a in (math.radians(45), math.radians(135), math.radians(225), math.radians(315))]
    # suction foot: a domed rubber cup, a tinned cover plate and the cam lever
    cup = M.place(M.lathe([(0.0, TOP + 0.020), (0.050, TOP + 0.020), (0.064, TOP + 0.010), (0.066, TOP + 0.002),
                           (0.068, TOP + 0.0), (0.0, TOP + 0.0)], n=80), None, (hx + 0.005, hy, 0.0))
    cover = M.place(M.lathe([(0.0, TOP + 0.025), (0.047, TOP + 0.025), (0.052, TOP + 0.020), (0.0, TOP + 0.020)], n=80),
                    None, (hx + 0.005, hy, 0.0))
    lever = M.merge(M.tapered_bar((hx + 0.005, hy - 0.030, TOP + 0.024), (hx - 0.020, hy - 0.085, TOP + 0.018),
                                  0.018, 0.014, 0.005),
                    M.place(M.lathe([(0.0, -0.004), (0.010, -0.004), (0.010, 0.004), (0.0, 0.004)], n=24), None,
                            (hx - 0.022, hy - 0.088, TOP + 0.018)))
    # crank: hub with a wing nut on the drive square, a curved tapered arm in the y-z plane, a turned
    # beech handle parallel to the axis
    hub_c = np.array([SOCKET_X, AXIS_Y, AXIS_Z])
    hub = M.place(M.lathe([(0.0, 0.0), (0.016, 0.0), (0.016, 0.016), (0.012, 0.022), (0.0, 0.022)], n=48), M.Z_TO_X, hub_c)
    a1, a2 = math.radians(20), math.radians(-6)
    mid = hub_c + (0.020, -0.075 * math.sin(a1), 0.075 * math.cos(a1))
    end = hub_c + (0.020, -CRANK_R * math.sin(a2) - 0.020, CRANK_R * math.cos(a2))
    arm = M.merge(M.tapered_bar(hub_c + (0.020, 0.0, 0.0), mid, 0.028, 0.020, 0.008, up=(1.0, 0.0, 0.0)),
                  M.tapered_bar(mid, end, 0.020, 0.016, 0.008, up=(1.0, 0.0, 0.0)),
                  M.place(M.lathe([(0.0, 0.0), (0.011, 0.0), (0.011, 0.008), (0.0, 0.008)], n=32), M.Z_TO_X,
                          end + (-0.004, 0.0, 0.0)))
    hprof = [(0.0, 0.0), (0.0085, 0.0), (0.0110, 0.010), (0.0135, 0.035), (0.0140, 0.055), (0.0125, 0.078),
             (0.0100, 0.088), (0.0060, 0.092), (0.0, 0.092)]
    handle = M.place(M.lathe(hprof, n=40), M.Z_TO_X, end + (0.006, 0.0, 0.0))
    wings = M.merge(M.lathe([(0.0, 0.0), (0.009, 0.0), (0.007, 0.010), (0.0, 0.012)], n=24),
                    M.tapered_bar((0.0, 0.0, 0.005), (0.0, 0.024, 0.008), 0.010, 0.016, 0.003, up=(1.0, 0.0, 0.0)),
                    M.tapered_bar((0.0, 0.0, 0.005), (0.0, -0.024, 0.008), 0.010, 0.016, 0.003, up=(1.0, 0.0, 0.0)))
    wingnut = M.place(wings, M.Z_TO_X, hub_c + (0.022, 0.0, 0.0))
    return [
        ("v_barrel", *W(barrel), LOOK_TIN),
        ("v_thread", *W(thread), LOOK_TIN),
        ("v_seams", *W(M.merge(*seams)), LOOK_TIN),
        ("v_hopper", *W(hopper), LOOK_TIN),
        ("v_throat", *W(throat), LOOK_DARK),
        ("v_pedestal", *W(pedestal), LOOK_TIN),
        ("v_screws", *W(M.merge(*screws)), LOOK_TIN_DARK),
        ("v_cup", *W(cup), LOOK_RUBBER),
        ("v_cover", *W(cover), LOOK_TIN),
        ("v_lever", *W(lever), LOOK_RUBBER),
        ("v_hub", *W(hub), LOOK_TIN),
        ("v_arm", *W(arm), LOOK_TIN),
        ("v_handle", *W(handle), LOOK_WOOD),
        ("v_wingnut", *W(wingnut), LOOK_TIN_DARK),
    ]


def mincer_object() -> Part:
    """Barrel (bored: staves round the bore), threaded mouth, key, hopper, column, suction foot and the
    crank on its back. The colliders are hidden; the visual meshes carry the look. Root: the foot."""
    foot_c = np.array([HOPPER_X + 0.005, AXIS_Y, TOP + 0.012])

    def at(x, y, z):
        return tuple(float(v) for v in np.array([x, y, z]) - foot_c)

    pieces = []
    n = 16
    for k in range(n):
        th = 2 * math.pi * (k + 0.5) / n
        r = BORE_R + WALL / 2
        w = 2 * (BORE_R + WALL) * math.tan(math.pi / n) + 0.001
        pieces.append(Piece("box", (BARREL_L, w, WALL), at(MOUTH_X + BARREL_L / 2, AXIS_Y + r * math.cos(th),
                                                              AXIS_Z + r * math.sin(th)),
                            rot=(th - math.pi / 2, 0.0, 0.0), visible=False))
        rt = BORE_R + (THREAD_R - BORE_R) / 2
        wt = 2 * THREAD_R * math.tan(math.pi / n) + 0.001
        pieces.append(Piece("box", (THREAD_L, wt, THREAD_R - BORE_R), at(MOUTH_X + THREAD_L / 2, AXIS_Y + rt * math.cos(th),
                                                                         AXIS_Z + rt * math.sin(th)),
                            rot=(th - math.pi / 2, 0.0, 0.0), visible=False))
    pieces.append(Piece("box", KEY, at(MOUTH_X + KEY[0] / 2, AXIS_Y, AXIS_Z + BORE_R - KEY[2] / 2), color=TIN_DARK,
                        collide=False))                  # the key: the plate's notch rule reads it
    for k in range(n):                                   # back wall: an annulus round the drive square's bearing
        th = 2 * math.pi * (k + 0.5) / n
        rb = (0.013 + BORE_R + WALL) / 2
        pieces.append(Piece("box", (0.008, 2 * (BORE_R + WALL) * math.tan(math.pi / n) + 0.001, BORE_R + WALL - 0.013),
                            at(BACK_X + 0.004, AXIS_Y + rb * math.cos(th), AXIS_Z + rb * math.sin(th)),
                            rot=(th - math.pi / 2, 0.0, 0.0), visible=False))
    neck_z = AXIS_Z + BORE_R + WALL
    pieces.append(Piece("box", (0.050, 0.044, 0.030), at(HOPPER_X, AXIS_Y, neck_z + 0.010), visible=False))
    col_top = AXIS_Z - BORE_R - WALL
    pieces.append(Piece("box", (0.040, 0.034, col_top - (TOP + 0.024)), at(HOPPER_X + 0.005, AXIS_Y, (col_top + TOP + 0.024) / 2),
                        visible=False))
    return Part("mincer", pos=tuple(float(v) for v in foot_c), size=(0.066, 0.024, 0.066), shape="cylinder",
                exact_z=True, color=TIN, friction=(0.6, 0.5), contact_offset=0.002, pieces=tuple(pieces),
                root_visible=False, vmeshes=tuple(_mincer_visuals(foot_c)))


# -- loose parts on the tea towel --------------------------------------------------------------
TOWEL_C = (0.285, -0.185)
TOWEL_SIZE = (0.230, 0.200, 0.003)
ON_TOWEL = TOP + TOWEL_SIZE[2]

# auger: 162 mm long; flights 46 mm across on a 26 mm core; a 10 mm square for the knife and an
# 8 mm pin for the plate at the front, an 11 mm square at the back into the crank's socket.
FLIGHT_R = 0.023
CORE_R = 0.013
FLIGHT_L = 0.125
FLIGHT_PITCH = 0.034
FRONT_SQ = (0.019, 0.010)             # length, across flats
PIN = (0.014, 0.004)                  # length, radius
REAR_SQ = (0.016, 0.011)
AUGER_MASS = 0.52
AUGER_C = (0.280, -0.112)             # lying along x on the towel


def auger_object() -> Part:
    """Root: the core, centred on the flights. +x is its back (drive) end. Colliders: the core and a
    row of discs (hidden); drawn: a continuous helical flight on a tapered core."""
    c = np.array([AUGER_C[0], AUGER_C[1], ON_TOWEL + FLIGHT_R])
    pieces = []
    nf = 9
    for k in range(nf):
        x = -FLIGHT_L / 2 + 0.008 + k * (FLIGHT_L - 0.016) / (nf - 1)
        pieces.append(Piece("cylinder", (FLIGHT_R - 0.0010, 0.0035, FLIGHT_R - 0.0010), (x, 0.0, 0.0), axis="X",
                            visible=False))
    x0 = -FLIGHT_L / 2
    pieces.append(Piece("box", (FRONT_SQ[0], FRONT_SQ[1], FRONT_SQ[1]), (x0 - FRONT_SQ[0] / 2, 0.0, 0.0), color=TIN_DARK))
    pieces.append(Piece("cylinder", (PIN[1], PIN[0], PIN[1]), (x0 - FRONT_SQ[0] - PIN[0] / 2, 0.0, 0.0), axis="X",
                        color=TIN_DARK))
    x1 = FLIGHT_L / 2
    pieces.append(Piece("box", (REAR_SQ[0], REAR_SQ[1], REAR_SQ[1]), (x1 + REAR_SQ[0] / 2, 0.0, 0.0), color=TIN_DARK))
    core = M.place(M.lathe([(0.0, -0.001), (CORE_R - 0.004, -0.001), (CORE_R, 0.006), (CORE_R + 0.001, FLIGHT_L * 0.5),
                            (CORE_R + 0.003, FLIGHT_L - 0.004), (CORE_R + 0.004, FLIGHT_L), (0.0, FLIGHT_L)], n=48),
                   M.Z_TO_X, (x0, 0.0, 0.0))
    flight = M.place(M.helix_flight(CORE_R - 0.001, FLIGHT_R, FLIGHT_PITCH, FLIGHT_L - 0.010, 0.006, per_turn=64),
                     M.Z_TO_X, (x0 + 0.004, 0.0, 0.0))
    return Part("auger", pos=tuple(float(v) for v in c), size=(CORE_R, FLIGHT_L, CORE_R), shape="cylinder", axis="X",
                exact_z=True, color=TIN, mass=AUGER_MASS, friction=(0.6, 0.5), contact_offset=0.002,
                pieces=tuple(pieces), root_visible=False, joints=(_seat_joint(),),
                vmeshes=(("v_core", *core, LOOK_TIN), ("v_flight", *flight, LOOK_TIN)))


# knife: four-wing cross knife, 52 mm across, 10 mm thick at the hub, a 10 mm square hole; one face
# is flat and ground (the cutting face, toward the plate), the wings are bevelled on the other.
KNIFE_R = 0.024
KNIFE_HUB = (0.0095, 0.010)           # radius, thickness
KNIFE_MASS = 0.045
KNIFE_C = (0.365, -0.235)             # lying flat, cutting face up
KNIFE_YAW = 0.0                       # its wings along x and y


def _knife_visuals():
    """In the knife's frame: hub centre at the origin, the flat cutting face at z = +hub/2 (it lies
    cutting face up); the wings are flush with that face, the hub stands 6 mm proud of them below."""
    zc = -KNIFE_HUB[1] / 2
    meshes = []
    for k in range(4):
        a = k * math.pi / 2
        # a wing: a curved blade from the hub to the rim, its leading edge (cutting) straight, its
        # back swept; 3.5 mm thick, ground flat underneath, bevelled toward the leading edge on top
        out = []
        for i in range(9):
            r = KNIFE_HUB[0] - 0.001 + (KNIFE_R - KNIFE_HUB[0] + 0.001) * i / 8
            out.append((r, 0.0045))                        # leading edge
        back = []
        for i in range(9):
            r = KNIFE_R - (KNIFE_R - KNIFE_HUB[0] + 0.001) * i / 8
            w = 0.0060 + 0.0030 * math.sin(math.pi * (1 - i / 8))
            back.append((r, -w - 0.0020 * (1 - i / 8) ** 2))
        poly = [(r * math.cos(a) - y * math.sin(a), r * math.sin(a) + y * math.cos(a)) for r, y in out + back]
        poly = poly[::-1] if k % 2 else poly
        body = M.extrude(poly[::-1], zc, zc + 0.0040, top_inset=0.22)
        meshes.append(body)
        edge = [(r * math.cos(a) - 0.0028 * math.sin(a), r * math.sin(a) + 0.0028 * math.cos(a))
                for r in (KNIFE_HUB[0] + 0.001, KNIFE_R - 0.0005)]
        e0, e1 = np.array([*edge[0], zc + 0.0002]), np.array([*edge[1], zc + 0.0002])
        meshes.append(M.tapered_bar(e0, e1, 0.0014, 0.0012, 0.0004 + 0.0001))
    hub = M.lathe([(0.0071, zc), (KNIFE_HUB[0], zc), (KNIFE_HUB[0], -zc - 0.001), (KNIFE_HUB[0] - 0.001, -zc),
                   (0.0071, -zc), (0.0071, zc)], n=48)
    sq = [M.box((0.010, 0.0024, KNIFE_HUB[1]), (0.0, s * 0.0062, 0.0)) for s in (-1, 1)] + \
         [M.box((0.0024, 0.010, KNIFE_HUB[1]), (s * 0.0062, 0.0, 0.0)) for s in (-1, 1)]
    blades = M.merge(*[meshes[i] for i in range(0, 8, 2)])
    edges = M.merge(*[meshes[i] for i in range(1, 8, 2)])
    flip = np.diag([1.0, -1.0, -1.0])                    # lying cutting face up: turned over about x
    return [("v_blades", *M.place(blades, flip), LOOK_STEEL), ("v_edges", *M.place(edges, flip), LOOK_EDGE),
            ("v_hub", *M.place(M.merge(hub, *sq), flip), LOOK_STEEL)]


def knife_object() -> Part:
    """Root: the hub, axis Z as it lies. Its own +z is the flat cutting face (lying face up); on the
    auger that face must point out of the mouth (-x)."""
    c = np.array([KNIFE_C[0], KNIFE_C[1], ON_TOWEL + 0.002])
    pieces = []
    for k in range(4):
        a = k * math.pi / 2
        mid = (KNIFE_HUB[0] + KNIFE_R) / 2
        pieces.append(Piece("box", (KNIFE_R - KNIFE_HUB[0] + 0.002, 0.008, 0.004),
                            (mid * math.cos(a), mid * math.sin(a), KNIFE_HUB[1] / 2 - 0.002),
                            rot=(0.0, 0.0, a), visible=False))
    pos = (float(c[0]), float(c[1]), float(c[2] - 0.002 + KNIFE_HUB[1] / 2))
    return Part("knife", pos=pos, size=(KNIFE_HUB[0], KNIFE_HUB[1], KNIFE_HUB[0]), shape="cylinder", axis="Z",
                exact_z=True, color=STEEL, mass=KNIFE_MASS, friction=(0.6, 0.5), contact_offset=0.0015, yaw=KNIFE_YAW,
                pieces=tuple(pieces), root_visible=False, vmeshes=tuple(_knife_visuals()), joints=(_shaft_joint(), _shaft_joint_rev()),
                filter_with=("auger", "mincer"))


# plate: 2-1/8 in (54 mm), 6 mm thick, 4.5 mm holes, 8.5 mm centre hole, a 6 mm notch in its rim.
PLATE_R = 0.027
PLATE_T = 0.006
PLATE_MASS = 0.085
PLATE_C = (0.290, -0.240)
NOTCH = (0.006, 0.0035)               # width, depth
PLATE_YAW = math.radians(-35)


def _plate_outline(r):
    pts = []
    n = 120
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = r * math.cos(a), r * math.sin(a)
        if abs(x) < NOTCH[0] / 2 and y > 0:              # the notch at local +y
            y = r - NOTCH[1]
        pts.append((x, y))
    return pts


def plate_object() -> Part:
    """Root: the plate disc (hidden collider), axis Z as it lies flat. The rim notch is at local +y."""
    h = PLATE_T / 2
    body = M.extrude(_plate_outline(PLATE_R - 0.0006), -h + 0.0006, h - 0.0006)
    top = M.extrude(_plate_outline(PLATE_R - 0.0006), h - 0.0006, h, top_inset=0.025)
    bot = M.place(M.extrude(_plate_outline(PLATE_R - 0.0006), h - 0.0006, h, top_inset=0.025), M.rot(roll=math.pi))
    holes = [(0.0, 0.0)]
    for ring_r, n in ((0.0085, 6), (0.0140, 12), (0.0195, 18)):
        holes += [(ring_r * math.cos(2 * math.pi * k / n + 0.1 * ring_r), ring_r * math.sin(2 * math.pi * k / n + 0.1 * ring_r))
                  for k in range(n)]
    dots = []
    for i, (hx, hy) in enumerate(holes):
        r = 0.00425 if i == 0 else 0.00225
        for s in (1, -1):
            dots.append(M.place(M.lathe([(0.0, 0.0), (r, 0.0)], n=20), None, (hx, hy, s * (h + 0.00015))))
    pos = (PLATE_C[0], PLATE_C[1], ON_TOWEL + PLATE_T / 2)
    return Part("plate", pos=pos, size=(PLATE_R, PLATE_T, PLATE_R), shape="cylinder", axis="Z", exact_z=True,
                color=STEEL, mass=PLATE_MASS, friction=(0.6, 0.5), contact_offset=0.0015, root_visible=False,
                joints=(_pin_joint(), _pin_joint_rev()), filter_with=("auger", "mincer"),
                yaw=PLATE_YAW,
                vmeshes=(("v_plate", *M.merge(body, top, bot), LOOK_STEEL), ("v_holes", *M.merge(*dots), LOOK_DARK)))


# ring: tinned cast iron, 72 mm body with two lugs, 20 mm deep; inside, 66 mm thread; its front lip
# leaves a 46 mm opening over the plate.
RING_R = 0.036
RING_DEPTH = 0.020
RING_LIP = 0.004
RING_OPEN_R = 0.023
RING_MASS = 0.190
RING_C = (0.205, -0.235)


def _ring_visuals():
    """In the ring's frame: lip centred at z = 0, the skirt down to -(RING_DEPTH - lip/2) (lying lip up)."""
    z0, z1 = -RING_LIP / 2, RING_DEPTH - RING_LIP / 2
    ti = THREAD_R + 0.0006
    prof = [(RING_OPEN_R + 0.0012, z0), (RING_R - 0.0015, z0), (RING_R, z0 + 0.0015), (RING_R, z1 - 0.0012),
            (RING_R - 0.0012, z1), (ti, z1), (ti, z0 + RING_LIP), (RING_OPEN_R, z0 + RING_LIP),
            (RING_OPEN_R, z0 + 0.0012), (RING_OPEN_R + 0.0012, z0)]

    def knurl(a, k):                                     # straight knurl on the outer face
        return 1.0 + (0.012 * (0.5 + 0.5 * math.cos(64 * a)) if k in (2, 3) else 0.0)

    body = M.lathe(prof, n=256, radial=knurl)
    thread = M.place(M.helix_flight(ti - 0.0013, ti + 0.0001, 0.0028, RING_DEPTH - RING_LIP - 0.003, 0.0011, per_turn=72),
                     None, (0.0, 0.0, z0 + RING_LIP + 0.001))
    lugs = []
    for a in (0.0, math.pi):
        L = M.box((0.010, 0.016, RING_DEPTH - 0.005), (0.0, 0.0, 0.0), bevel=0.0015)
        lugs.append(M.place(L, M.rot(yaw=a), ((RING_R + 0.004) * math.cos(a), (RING_R + 0.004) * math.sin(a),
                                              (z0 + z1) / 2)))
    flip = np.diag([1.0, -1.0, -1.0])                    # lying lip up: turned over about x
    return [("v_ring", *M.place(body, flip), LOOK_TIN), ("v_thread", *M.place(thread, flip), LOOK_TIN_DARK),
            ("v_lugs", *M.place(M.merge(*lugs), flip), LOOK_TIN)]


def ring_object() -> Part:
    """Root: the lip (a flat annulus of hidden staves), lying lip up on its skirt. Local +z: from the
    skirt out through the lip (out of the mouth, -x, once on)."""
    pieces = []
    n = 16
    for k in range(n):
        th = 2 * math.pi * (k + 0.5) / n
        rl = (RING_OPEN_R + RING_R) / 2
        pieces.append(Piece("box", (RING_R - RING_OPEN_R, 2 * RING_R * math.tan(math.pi / n) + 0.001, RING_LIP),
                            (rl * math.cos(th), rl * math.sin(th), 0.0), rot=(0.0, 0.0, th), visible=False))
        rs = THREAD_R + (RING_R - THREAD_R) / 2 + 0.0004
        pieces.append(Piece("box", (RING_R - THREAD_R - 0.0004, 2 * RING_R * math.tan(math.pi / n) + 0.001,
                                    RING_DEPTH - RING_LIP),
                            (rs * math.cos(th), rs * math.sin(th), -RING_DEPTH / 2), rot=(0.0, 0.0, th), visible=False))
    for a in (0.0, math.pi):
        pieces.append(Piece("box", (0.010, 0.014, RING_DEPTH - 0.004), ((RING_R + 0.004) * math.cos(a),
                                                                       (RING_R + 0.004) * math.sin(a), -RING_DEPTH / 2),
                            rot=(0.0, 0.0, a), visible=False))
    pos = (RING_C[0], RING_C[1], ON_TOWEL + RING_DEPTH - RING_LIP / 2)
    return Part("ring", pos=pos, size=(0.003, RING_LIP, 0.003), shape="cylinder", axis="Z", exact_z=True, color=TIN,
                mass=RING_MASS, friction=(0.6, 0.5), contact_offset=0.0015, pieces=tuple(pieces), root_visible=False,
                joints=(_thread_joint(),), filter_with=("mincer", "auger"),
                yaw=math.radians(90), vmeshes=tuple(_ring_visuals()))


def towel_object() -> Part:
    """A cotton tea towel folded once (fixed): soft folds, a woven check border and two blue bands."""
    sx, sy = TOWEL_SIZE[0] + 0.004, TOWEL_SIZE[1] + 0.004
    pts, tris = M.cloth((sx, sy), 0.0012, nx=46, ny=40, seed=7, droop=TOWEL_SIZE[2])
    pts = pts + (0.0, 0.0, TOWEL_SIZE[2] / 2 + 0.0004)
    cen = pts[tris].mean(axis=1)
    band = np.zeros(len(tris), bool)
    for x0 in (-sx / 2 + 0.020, sx / 2 - 0.032):
        band |= (cen[:, 0] > x0) & (cen[:, 0] < x0 + 0.012)
    thin = np.zeros(len(tris), bool)
    for x0 in (-sx / 2 + 0.036, sx / 2 - 0.040):
        thin |= (cen[:, 0] > x0) & (cen[:, 0] < x0 + 0.004)
    plain = ~(band | thin)
    lift = (0.0, 0.0, 0.0002)
    return Part("towel", pos=(TOWEL_C[0], TOWEL_C[1], TOP + TOWEL_SIZE[2] / 2), size=TOWEL_SIZE, exact_z=True,
                color=(0.93, 0.92, 0.88), friction=(0.9, 0.8), root_visible=False,
                vmeshes=(("v_cloth", pts, tris[plain], {"color": (0.93, 0.92, 0.88), "roughness": 0.95}),
                         ("v_band", pts + lift, tris[band], {"color": (0.17, 0.30, 0.56), "roughness": 0.95}),
                         ("v_line", pts + lift, tris[thin], {"color": (0.62, 0.14, 0.14), "roughness": 0.95})))


# -- assembled poses, the seating rules' joints and measurements -----------------------------------
AXIS = np.array([1.0, 0.0, 0.0])
AUGER_HOME = np.array([BACK_X - FLIGHT_L / 2 - 0.0005, AXIS_Y, AXIS_Z])      # flights against the back wall
KNIFE_HOME = np.array([MOUTH_X + PLATE_SEAT + KNIFE_HUB[1] / 2, AXIS_Y, AXIS_Z])  # against the plate, on the square
PLATE_HOME = np.array([MOUTH_X + PLATE_T / 2, AXIS_Y, AXIS_Z])                 # flush in the counterbore
RING_HOME = np.array([MOUTH_X - RING_LIP / 2, AXIS_Y, AXIS_Z])                 # its lip on the plate
# knife, plate and ring as fitted: their own +z out of the mouth (-x), the plate's notch (+y) up on the key
R_HOME = np.column_stack([(0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)])
KNIFE_SLIDE = 0.019                   # how far out along the square/pin the knife can start and still be on it
PLATE_SLIDE = 0.011                   # the plate on the pin, proud of its seat by up to this
NOTCH_DEG = 8.0                       # the notch's play round the key
SEAT_D = 0.002                        # N m s / deg: the auger turning in its bearings
SLIDE_D = 40.0                        # N s / m: a part sliding along the square/pin (stays where left)
THREAD_D = 0.0008                     # N m s / deg: the ring running up its thread
SPIN_D = 0.0005


def quat_wxyz(R):
    """(w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return (w, x, y, z)


def mat(pose):
    """Rotation matrix of a root pose (x, y, z, qx, qy, qz, qw)."""
    x, y, z, w = (float(v) for v in pose[3:7])
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


_LOCK = "lock"
_ROT1 = quat_wxyz(R_HOME.T)


def _seat_joint():
    return dict(name="seat_joint", parent=None, local0=tuple(AUGER_HOME), enabled=False,
                limits={"transX": _LOCK, "transY": _LOCK, "transZ": _LOCK, "rotY": _LOCK, "rotZ": _LOCK},
                drives={"rotX": (0.0, SEAT_D)})


def _shaft_joint():
    return dict(name="shaft_joint", parent="auger", local0=tuple(KNIFE_HOME - AUGER_HOME), rot1=_ROT1, enabled=False,
                limits={"transX": (-KNIFE_SLIDE, 0.0), "transY": _LOCK, "transZ": _LOCK, "rotY": _LOCK, "rotZ": _LOCK},
                drives={"transX": (0.0, SLIDE_D), "rotX": (0.0, SPIN_D)})


R_REV = np.column_stack([(0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)])   # the same, its +z into the barrel
_ROT1_REV = quat_wxyz(R_REV.T)


def _shaft_joint_rev():
    """The knife put on backwards (flat cutting face into the barrel): it still slides on the square."""
    j = _shaft_joint()
    return dict(j, name="shaft_joint_rev", rot1=_ROT1_REV)


def _pin_joint_rev():
    """The plate turned over: plates are flat both sides, so either way round is right."""
    j = _pin_joint()
    return dict(j, name="pin_joint_rev", rot1=_ROT1_REV)


def _pin_joint():
    return dict(name="pin_joint", parent=None, local0=tuple(PLATE_HOME), rot1=_ROT1, enabled=False,
                limits={"transX": (-PLATE_SLIDE, 0.0), "transY": _LOCK, "transZ": _LOCK, "rotX": (-NOTCH_DEG, NOTCH_DEG),
                        "rotY": _LOCK, "rotZ": _LOCK},
                drives={"transX": (0.0, SLIDE_D), "rotX": (0.0, SPIN_D)})


def _thread_joint():
    return dict(name="thread_joint", parent=None, local0=tuple(RING_HOME), rot1=_ROT1, enabled=False,
                limits={"transX": _LOCK, "transY": _LOCK, "transZ": _LOCK, "rotY": _LOCK, "rotZ": _LOCK},
                drives={"rotX": (0.0, THREAD_D)})


def axial_radial(pose, home):
    """(signed offset along +x from ``home``, distance off the barrel's axis) of a part's centre (m)."""
    rel = np.asarray(pose[:3], float) - home
    ax = float(rel[0])
    return ax, float(math.hypot(rel[1], rel[2]))


def tilt_deg(pose, local_axis, want):
    """Angle (deg) between a part's local axis, in the world, and ``want``."""
    v = mat(pose) @ np.asarray(local_axis, float)
    return math.degrees(math.acos(float(np.clip(np.dot(v, want), -1.0, 1.0))))


def notch_deg(pose):
    """How far the plate's notch is turned off 12 o'clock about the barrel's axis (deg)."""
    n = mat(pose) @ np.array([0.0, 1.0, 0.0])
    return math.degrees(math.atan2(-n[1], n[2]))


def twist(pose):
    """A part's angle about the barrel's axis (rad): its local x axis's direction in the y-z plane."""
    v = mat(pose) @ np.array([1.0, 0.0, 0.0])
    return math.atan2(v[2], v[1])


def auger_home():
    return AUGER_HOME.copy()


def knife_home():
    return KNIFE_HOME.copy()


def plate_home():
    return PLATE_HOME.copy()


def ring_home():
    return RING_HOME.copy()
