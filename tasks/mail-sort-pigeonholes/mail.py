"""Geometry for sorting incoming letters into a desktop mail sorter's labelled pigeonholes (real size).

Everything is real size; the bench is the desk top. Against the back of the desk stands a wooden
desktop mail sorter (literature organizer): 2 columns x 3 rows of open pigeonholes, 255 mm wide,
95 mm high and 240 mm deep inside, built of 15 mm oak-veneered boards with a 6 mm hardboard back. Its
openings face -y (toward the right arm and the overview camera). Each opening has a printed room
label on the edge of the board below it: 101 102 (bottom), 201 202, 301 302 (top). On the desk to the
right of it stands a black wire-mesh step file organizer with four rising sections; in each stands one
#10 business envelope (241 x 105 mm, about 4 mm thick: a few folded sheets inside), its addressed face
toward -y. Each address names a room in large type. One by one the letters are taken out of the
organizer by their top edge, turned flat and slid fully into the pigeonhole labelled with their room.

Frame: metres, x away from the arm bases, y left, z up; TOP is the desk top.

Real parts:
  mail sorter   desktop literature organizer, 6 compartments (2 x 3), oak-veneered 15 mm boards,
                555 W x 345 H x 246 D mm, compartments 255 W x 95 H x 240 D mm, about 6 kg.
  organizer     wire-mesh step file organizer, black powder-coated steel, 4 sections stepped
                35 mm, 300 L x 135 D x 200 H mm, 0.45 kg.
  envelope      #10 business envelope, 4 1/8 x 9 1/2 in (105 x 241 mm), 24 lb white wove, with
                three folded A4/letter sheets inside: about 4 mm thick, 28 g.
"""
from __future__ import annotations

import math

import numpy as np

from simkit import Part, Piece
import meshlib as M

TOP = 0.7475

OAK = (0.70, 0.52, 0.33)
HDF = (0.42, 0.29, 0.18)
WIRE = (0.05, 0.05, 0.055)
PAPER = (0.93, 0.91, 0.84)                    # off-white wove
INK = (0.06, 0.06, 0.08)
LABEL = (0.97, 0.97, 0.96)
FRAME = (0.78, 0.79, 0.81)

# -- mail sorter (fixed) --------------------------------------------------------------------------------
BOARD = 0.015
BACK = 0.006
SLOT_W, SLOT_H, SLOT_D = 0.255, 0.095, 0.240
COLS, ROWS = 2, 3
SORT_W = COLS * SLOT_W + (COLS + 1) * BOARD           # 0.555 along x
SORT_H = ROWS * SLOT_H + (ROWS + 1) * BOARD           # 0.345
SORT_D = SLOT_D + BACK                                # 0.246 along y
SORT_X = 0.420                                        # centre x
FRONT_Y = 0.000                                       # the openings' plane (faces -y)
X0 = SORT_X - SORT_W / 2
ROOMS = (("101", "102"), ("201", "202"), ("301", "302"))   # [row][col], row 0 at the bottom


def slot_centre(row, col):
    """Centre of a pigeonhole's floor at its opening (world): x mid-width, y at the opening, z on the floor."""
    x = X0 + BOARD + SLOT_W / 2 + col * (SLOT_W + BOARD)
    z = TOP + BOARD + row * (SLOT_H + BOARD)
    return np.array([x, FRONT_Y, z])


def slot_of(room):
    for r, row in enumerate(ROOMS):
        for c, name in enumerate(row):
            if name == room:
                return r, c
    raise KeyError(room)


# -- envelopes ---------------------------------------------------------------------------------------------
ENV = (0.241, 0.004, 0.105)                            # x (long), y (thick), z (tall) standing in the organizer
ENV_MASS = 0.028
LETTERS = (                                            # front section first: (room, addressee, company line)
    ("202", "Dana Whitfield", "Harbor Point Business Center"),
    ("102", "Marcus Ellery", "Harbor Point Business Center"),
    ("302", "Priya Nandakumar", "Harbor Point Business Center"),
    ("202", "Owen Castellano", "Harbor Point Business Center"),     # a second letter for room 202
)
ENV_NAMES = tuple("env%d_%s" % (i + 1, room) for i, (room, _, _) in enumerate(LETTERS))
SENDERS = ("Lakeside Dental Group", "County Water Utility", "Brightline Insurance", "Office Supply Depot")

# -- wire step organizer (free) ----------------------------------------------------------------------------
ORG_X, ORG_Y = 0.245, -0.450                           # centre of its footprint
ORG_YAW = math.pi / 2                                  # turned so its sections run along x, low front toward +x
ORG_L = 0.300                                          # its length (along world y once turned)
SEC_W = 0.0335                                         # each section's width along y
NSEC = 4
ORG_D = NSEC * SEC_W + 0.001
STEP = 0.010                                           # each section's floor 10 mm above the one in front
BASE_H = 0.012
WALL_T = 0.003
ORG_MASS = 0.45


def section_floor(i):
    """Floor height (world z) of section i (0 = front, toward +x)."""
    return TOP + BASE_H + i * STEP


def section_local(i):
    """Section i's centre across the organizer, in its own frame (front = local -y)."""
    return -ORG_D / 2 + SEC_W * (i + 0.5)


def envelope_pos(i):
    """World centre of letter i, standing in its section, leaning toward the section's front divider. The
    organizer is turned 90 degrees: its local -y (front) is world +x, its local x (length) world y."""
    ly = section_local(i) - 0.006
    return (ORG_X - ly, ORG_Y, section_floor(i) + ENV[2] / 2 + 0.0005)


# -- printed text as geometry -------------------------------------------------------------------------------
_TEXT_CACHE = {}


def text_mesh(text, height, bold=False, px=None):
    """Flat ink for ``text`` in the (u, v) plane: u along the line, v up, the baseline-left corner at the
    origin; ``height`` is the capital height (m). Rasterised with matplotlib's Agg (DejaVu Sans), each row
    of ink pixels a run of quads. Returns (points (N, 3) with w = 0, triangles), and the line's width."""
    key = (text, round(height, 6), bold)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    fp = FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")
    cap = TextPath((0, 0), "H", size=100, prop=fp).get_extents().height       # cap height at size 100
    path = TextPath((0, 0), text, size=100, prop=fp)
    ext = path.get_extents()
    n_rows = 34 if px is None else px
    scale_px = n_rows / cap                                                    # pixels per text unit
    w_px = int(math.ceil((ext.x1 - min(ext.x0, 0)) * scale_px)) + 4
    h_px = int(math.ceil((ext.y1 - min(ext.y0, 0)) * scale_px)) + 4
    fig = Figure(figsize=(w_px / 100, h_px / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(min(ext.x0, 0) - 2 / scale_px, min(ext.x0, 0) - 2 / scale_px + w_px / scale_px)
    ax.set_ylim(min(ext.y0, 0) - 2 / scale_px, min(ext.y0, 0) - 2 / scale_px + h_px / scale_px)
    ax.axis("off")
    from matplotlib.patches import PathPatch
    ax.add_patch(PathPatch(path, facecolor="black", edgecolor="none"))
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[:, :, 0] < 128                      # True = ink, row 0 at the top
    m_per_px = height / n_rows
    u0 = (min(ext.x0, 0) - 2 / scale_px) * scale_px * m_per_px
    v0 = (min(ext.y0, 0) - 2 / scale_px) * scale_px * m_per_px
    pts, tris = [], []
    H = img.shape[0]
    for r in range(H):
        row = img[r]
        if not row.any():
            continue
        v_top = v0 + (H - r) * m_per_px
        v_bot = v_top - m_per_px
        c = 0
        while c < len(row):
            if row[c]:
                c2 = c
                while c2 < len(row) and row[c2]:
                    c2 += 1
                ua, ub = u0 + c * m_per_px, u0 + c2 * m_per_px
                k = len(pts)
                pts += [(ua, v_bot, 0.0), (ub, v_bot, 0.0), (ub, v_top, 0.0), (ua, v_top, 0.0)]
                tris += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
                c = c2
            else:
                c += 1
    out = (np.asarray(pts, float).reshape(-1, 3), np.asarray(tris, int).reshape(-1, 3), (ext.x1 - min(ext.x0, 0)) * height / cap)
    _TEXT_CACHE[key] = out
    return out


def place(mesh2d, origin, u, v, n=0.0003, normal=None):
    """Put a (u, v) mesh on a plane in a Part's frame: ``origin`` + u * U + v * V, lifted ``n`` along the
    plane's normal (U x V unless ``normal`` is given)."""
    p, t = mesh2d[0], mesh2d[1]
    U, V = np.asarray(u, float), np.asarray(v, float)
    N = np.cross(U, V) if normal is None else np.asarray(normal, float)
    N = N / np.linalg.norm(N)
    q = np.asarray(origin, float) + np.outer(p[:, 0], U) + np.outer(p[:, 1], V) + n * N
    return q, t


def _quad(origin, u, v, w, h, n=0.0003, normal=None):
    p = np.array([(0, 0, 0), (w, 0, 0), (w, h, 0), (0, h, 0)], float)
    return place((p, np.array([(0, 1, 2), (0, 2, 3)])), origin, u, v, n, normal)


# -- sorter -------------------------------------------------------------------------------------------------
def sorter_object():
    """The wooden sorter as one fixed body: its boards are its colliders; labels are drawn on the front."""
    pos = (SORT_X, FRONT_Y + (SORT_D - BACK) / 2, TOP + BOARD / 2)   # the bottom board's centre
    pieces = []
    z_base = -BOARD / 2                                                        # Part frame: the bottom board's centre
    # shelves (the bottom board is the root)
    for k in range(1, ROWS + 1):
        z = z_base + k * (SLOT_H + BOARD) + BOARD / 2
        pieces.append(Piece("box", (SORT_W, SORT_D - BACK, BOARD), (0.0, 0.0, z), material="oak", color=OAK))
    # sides and the middle upright
    for c in range(COLS + 1):
        x = -SORT_W / 2 + BOARD / 2 + c * (SLOT_W + BOARD)
        pieces.append(Piece("box", (BOARD, SORT_D - BACK, SORT_H - 2 * BOARD),
                            (x, 0.0, z_base + SORT_H / 2), material="oak", color=OAK))
    # hardboard back
    pieces.append(Piece("box", (SORT_W, BACK, SORT_H), (0.0, (SORT_D - BACK) / 2 + BACK / 2, z_base + SORT_H / 2), color=HDF,
                        material="satin"))
    # front edge banding: a lighter strip on every board's front edge
    vms = []
    front = -(SORT_D - BACK) / 2
    for k in range(ROWS + 1):
        z = z_base + k * (SLOT_H + BOARD) + BOARD / 2
        vms.append(("band_h%d" % k, *M.transform(M.rbox((SORT_W, 0.0008, BOARD - 0.001), e=8.0, n=16),
                                               offset=(0.0, front - 0.0004, z)), (0.80, 0.63, 0.42), "satin", 1.0))
    # label holders and printed room labels under each opening (on the board below it)
    for r in range(ROWS):
        for c in range(COLS):
            x = -SORT_W / 2 + BOARD + SLOT_W / 2 + c * (SLOT_W + BOARD)
            z = z_base + r * (SLOT_H + BOARD) + BOARD / 2
            lw, lh = 0.070, 0.0135
            fr = M.transform(M.rbox((lw + 0.004, 0.0012, lh + 0.0012), e=6.0, n=16), offset=(x, front - 0.0012, z))
            vms.append(("holder_%d%d" % (r, c), *fr, FRAME, "aluminium", 1.0))
            card = _quad((x - lw / 2, front - 0.0019, z - lh / 2), (1, 0, 0), (0, 0, 1), lw, lh, n=0.0, normal=(0, -1, 0))
            vms.append(("card_%d%d" % (r, c), *card, LABEL, "matte", 1.0))
            tm = text_mesh(ROOMS[r][c], 0.0095, bold=True, px=30)
            tw = tm[2]
            ink = place(tm, (x - tw / 2, front - 0.0024, z - 0.0095 / 2), (1, 0, 0), (0, 0, 1), n=0.0, normal=(0, -1, 0))
            vms.append(("room_%d%d" % (r, c), *ink, INK, "matte", 1.0))
    return Part("sorter", pos=pos, size=(SORT_W, SORT_D - BACK, BOARD), shape="box", exact_z=True, color=OAK,
                material="oak", friction=(0.45, 0.35), restitution=0.2, contact_offset=0.002,
                pieces=tuple(pieces), vmeshes=tuple(vms))


# -- wire step organizer --------------------------------------------------------------------------------------
def organizer_object():
    """Black wire-mesh step organizer: a stepped floor and wire dividers, drawn as wire; colliders hidden."""
    pos = (ORG_X, ORG_Y, TOP + BASE_H / 2)
    pieces = []
    y_front = -ORG_D / 2
    # stepped floors (colliders, hidden under the wire mesh)
    for i in range(1, NSEC):
        h = i * STEP
        pieces.append(Piece("box", (ORG_L, SEC_W, h), (0.0, y_front + SEC_W * (i + 0.5), BASE_H / 2 + h / 2),
                            hidden=True))
    # dividers: a front lip and one behind every section, rising with the steps
    tops = []
    for i in range(NSEC + 1):
        y = y_front + SEC_W * i
        top = 0.040 if i == 0 else 0.045 + i * STEP
        tops.append(top)
        pieces.append(Piece("box", (ORG_L, WALL_T, top), (0.0, y, BASE_H / 2 + top / 2), hidden=True))
    # open ends: only a low base rail closes each end, so a letter slides out sideways along its section
    for sx in (-1, 1):
        pieces.append(Piece("box", (WALL_T, ORG_D, 0.006), (sx * (ORG_L / 2 - WALL_T / 2), 0.0, BASE_H / 2 + 0.003),
                            hidden=True))
    # wire drawing
    vms = []
    r = 0.0016
    segs = []
    zb = -BASE_H / 2
    # base frame
    segs.append([(-ORG_L / 2, y_front, zb + r), (ORG_L / 2, y_front, zb + r), (ORG_L / 2, -y_front, zb + r),
                 (-ORG_L / 2, -y_front, zb + r), (-ORG_L / 2, y_front, zb + r)])
    # dividers: top rails, vertical wires every 15 mm, a mid rail
    for i in range(NSEC + 1):
        y = y_front + SEC_W * i
        top = BASE_H / 2 + tops[i]
        segs.append([(-ORG_L / 2, y, top), (ORG_L / 2, y, top)])
        segs.append([(-ORG_L / 2, y, (zb + top) / 2), (ORG_L / 2, y, (zb + top) / 2)])
        for x in np.arange(-ORG_L / 2 + 0.010, ORG_L / 2 - 0.005, 0.015):
            segs.append([(x, y, zb + 0.002), (x, y, top)])
    # stepped floors: wires across each step
    for i in range(NSEC):
        z = BASE_H / 2 + i * STEP
        for x in np.arange(-ORG_L / 2 + 0.010, ORG_L / 2 - 0.005, 0.015):
            segs.append([(x, y_front + SEC_W * i, z), (x, y_front + SEC_W * (i + 1), z)])
        segs.append([(-ORG_L / 2, y_front + SEC_W * (i + 0.5), z), (ORG_L / 2, y_front + SEC_W * (i + 0.5), z)])
    # open ends: the dividers' end uprights and a low end rail
    for sx in (-1, 1):
        x = sx * (ORG_L / 2 - 0.001)
        segs.append([(x, y_front, BASE_H / 2 + 0.006), (x, -y_front, BASE_H / 2 + 0.006)])
        for i in range(NSEC + 1):
            y = y_front + SEC_W * i
            segs.append([(x, y, zb), (x, y, BASE_H / 2 + tops[i])])
    meshes = [M.tube(np.asarray(s, float), r, n=6) for s in segs]
    p, t = M.merge(*meshes)
    vms.append(("wire", p, t, WIRE, "satin", 1.0))
    # rubber feet
    for sx in (-1, 1):
        for sy in (-1, 1):
            foot = M.transform(M.revolve([(0.0, 0.0), (0.006, 0.0), (0.006, 0.003), (0.0, 0.003)], n=16),
                               offset=(sx * (ORG_L / 2 - 0.012), sy * (ORG_D / 2 - 0.008), zb - 0.0005))
            vms.append(("foot_%d_%d" % (sx + 1, sy + 1), *foot, (0.12, 0.12, 0.12), "matte", 1.0))
    return Part("organizer", pos=pos, size=(ORG_L, ORG_D, BASE_H), shape="box", exact_z=True, mass=ORG_MASS, yaw=ORG_YAW,
                color=WIRE, friction=(0.5, 0.4), restitution=0.1, contact_offset=0.002, pieces=tuple(pieces),
                vmeshes=tuple(vms), hide_root=True)


# -- envelopes --------------------------------------------------------------------------------------------------
def envelope_object(i):
    """One #10 envelope standing on its long edge, addressed face toward -y."""
    room, name, company = LETTERS[i]
    L, T, Hh = ENV
    fy = -T / 2 - 0.0002                                    # front (address) face, Part frame
    by = T / 2 + 0.0002
    X, Z = (1, 0, 0), (0, 0, 1)
    vms = []
    # paper body with softly rounded edges
    vms.append(("paper", *M.rbox((L, T, Hh), e=24.0, n=32, axis="Y"), PAPER, "matte", 1.0))
    # back flap: a V of slightly darker paper on the back face
    flap = (np.array([(-L / 2, Hh / 2 - 0.002, 0), (L / 2, Hh / 2 - 0.002, 0), (0.0, -0.002, 0)], float),
            np.array([(0, 2, 1)]))
    fp = place(flap, (0, by, 0), (1, 0, 0), (0, 0, 1), n=0.0, normal=(0, 1, 0))
    vms.append(("flap", *fp, (0.90, 0.90, 0.86), "matte", 1.0))
    # the folded-over flap shows as a fold shadow along the front's top edge
    vms.append(("fold", *_quad((-L / 2 + 0.002, fy - 0.0001, Hh / 2 - 0.0035), X, Z, L - 0.004, 0.0009, n=0.0,
                                normal=(0, -1, 0)), (0.80, 0.78, 0.72), "matte", 1.0))
    # return address (top left)
    sender = SENDERS[i]
    y0 = Hh / 2 - 0.010
    lines = [(sender, 0.0026, True), ("PO Box %d" % (2400 + 37 * i), 0.0024, False), ("Springfield, OR 97477", 0.0024, False)]
    for k, (s, h, b) in enumerate(lines):
        tm = text_mesh(s, h, bold=b, px=14)
        vms.append(("ret%d" % k, *place(tm, (-L / 2 + 0.010, fy, y0 - k * 0.0045), X, Z, n=0.0, normal=(0, -1, 0)),
                    (0.20, 0.22, 0.30), "matte", 1.0))
    # stamp (top right) with a white border and a little picture; postmark lines beside it
    sx0, sz0 = L / 2 - 0.008 - 0.022, Hh / 2 - 0.008 - 0.026
    vms.append(("stamp_border", *_quad((sx0, fy - 0.0001, sz0), X, Z, 0.022, 0.026, n=0.0, normal=(0, -1, 0)),
                (0.98, 0.98, 0.97), "matte", 1.0))
    colors = [(0.15, 0.35, 0.65), (0.70, 0.18, 0.20), (0.15, 0.50, 0.30), (0.55, 0.30, 0.60)]
    vms.append(("stamp", *_quad((sx0 + 0.002, fy - 0.0002, sz0 + 0.002), X, Z, 0.018, 0.022, n=0.0, normal=(0, -1, 0)),
                colors[i % 4], "matte", 1.0))
    tm = text_mesh("USA", 0.0022, bold=True, px=10)
    vms.append(("stamp_txt", *place(tm, (sx0 + 0.004, fy - 0.0003, sz0 + 0.004), X, Z, n=0.0, normal=(0, -1, 0)),
                (0.97, 0.97, 0.95), "matte", 1.0))
    # postmark ring overlapping the stamp's left edge, with the date line inside it
    ring = []
    cx, cz, R0, R1 = sx0 - 0.004, sz0 + 0.012, 0.0095, 0.0083
    for j in range(48):
        a0, a1 = 2 * math.pi * j / 48, 2 * math.pi * (j + 1) / 48
        k = len(ring)
        ring += [(cx + R1 * math.cos(a0), cz + R1 * math.sin(a0)), (cx + R0 * math.cos(a0), cz + R0 * math.sin(a0)),
                 (cx + R0 * math.cos(a1), cz + R0 * math.sin(a1)), (cx + R1 * math.cos(a1), cz + R1 * math.sin(a1))]
    P = np.array([(x, fy - 0.0005, z) for x, z in ring])
    T4 = np.array([(4 * j, 4 * j + 1, 4 * j + 2) for j in range(48)] + [(4 * j, 4 * j + 2, 4 * j + 3) for j in range(48)])
    vms.append(("pm_ring", P, T4, (0.12, 0.12, 0.18), "matte", 1.0))
    tm = text_mesh("NOV 14", 0.0018, bold=True, px=10)
    vms.append(("pm_date", *place(tm, (cx - 0.0055, fy - 0.0005, cz - 0.0009), X, Z, n=0.0, normal=(0, -1, 0)),
                (0.12, 0.12, 0.18), "matte", 1.0))
    # postmark: three wavy cancel lines to the stamp's left
    for k in range(3):
        xs = np.linspace(sx0 - 0.046, sx0 - 0.015, 30)
        zs = sz0 + 0.008 + k * 0.005 + 0.0012 * np.sin(xs * 900.0)
        pts = []
        tris = []
        for j in range(len(xs) - 1):
            a = len(pts)
            pts += [(xs[j], zs[j] - 0.0004), (xs[j + 1], zs[j + 1] - 0.0004), (xs[j + 1], zs[j + 1] + 0.0004),
                    (xs[j], zs[j] + 0.0004)]
            tris += [(a, a + 1, a + 2), (a, a + 2, a + 3)]
        P = np.array([(x, 0.0, z) for x, z in pts])
        mesh = (np.column_stack([P[:, 0], np.zeros(len(P)), P[:, 2]]) + np.array([0, fy - 0.0004, 0]), np.array(tris))
        vms.append(("cancel%d" % k, *mesh, (0.15, 0.15, 0.20), "matte", 1.0))
    # address block, centre-left; the room line in large bold type
    ax0 = -0.050
    tm = text_mesh("ROOM " + room, 0.0120, bold=True, px=30)          # interoffice style: the room first
    vms.append(("addr_room", *place(tm, (ax0, fy, 0.029), X, Z, n=0.0, normal=(0, -1, 0)), INK, "matte", 1.0))
    tm = text_mesh(name, 0.0042, px=16)
    vms.append(("addr0", *place(tm, (ax0, fy, 0.017), X, Z, n=0.0, normal=(0, -1, 0)), INK, "matte", 1.0))
    tm = text_mesh(company, 0.0034, px=14)
    vms.append(("addr2", *place(tm, (ax0, fy, 0.007), X, Z, n=0.0, normal=(0, -1, 0)), INK, "matte", 1.0))
    tm = text_mesh("1200 Commerce Ave, Portland OR 97204", 0.0030, px=12)
    vms.append(("addr3", *place(tm, (ax0, fy, -0.001), X, Z, n=0.0, normal=(0, -1, 0)), INK, "matte", 1.0))
    # barcode stripe along the bottom
    bars = []
    rng = np.random.RandomState(7 + i)
    x = -0.040
    for k in range(52):
        h = 0.0032 if rng.rand() < 0.5 else 0.0016
        bars.append(_quad((x, fy, -Hh / 2 + 0.006), X, Z, 0.0006, h, n=0.0, normal=(0, -1, 0)))
        x += 0.0013
    p, t = M.merge(*bars)
    vms.append(("barcode", p, t, (0.10, 0.10, 0.12), "matte", 1.0))
    return Part(ENV_NAMES[i], pos=envelope_pos(i), size=ENV, shape="box", exact_z=True, mass=ENV_MASS, yaw=ORG_YAW,
                color=PAPER, friction=(0.45, 0.35), restitution=0.05, contact_offset=0.0015,
                vmeshes=tuple(vms), hide_root=True)


def quat_matrix(q):
    """Rotation matrix of a (x, y, z, w) quaternion."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
