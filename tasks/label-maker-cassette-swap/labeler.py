"""Geometry for putting a new tape cassette in a desktop label maker (real size).

Everything is real size; the bench is an office desk at working height. A desktop label maker
(keyboard at the front, colour LCD at the back left, cassette bay at the back right under a flip-up hood
cover) stands on the desk with its keyboard toward the arms. Its bay holds the used-up 24 mm tape
cassette, the cover latched shut. The new cassette lies face up on the desk beside its opened carton.
The cover's latch tab is pressed and the cover swung up past upright (it stays open), the old cassette
lifted out and set down on the desk, the new one put in the one way it fits and pressed down until it
clicks, and the cover swung shut until it latches. The label maker is free on its rubber feet: pushed or
lifted at, it slides unless the other hand steadies it.
Frame: metres, x away from the arm bases, y left, z up; TOP is the desk top.

Real parts (a PT-D600-class desktop labeller for 3.5-24 mm laminated tape cassettes):
  label maker  201 W x 192 D x 88 H mm (to the top of the cover hood; the real unit is 86 mm), ABS shell, 0.95 kg plus six AA
               cells (0.14 kg): 1.09 kg on four rubber feet.
  cassette     24 mm laminated tape cassette, 88 x 66 x 26 mm, ABS shell, 8 m of tape; 32 g full, 22 g used
               up. Its tape-exit corner is cut away round the print head, so it drops into the bay only one
               way round; the top 18 mm stands above the deck, inside the cover's hood, which is how
               fingers take it by its ends.
  carton       printed paperboard carton for one cassette, 105 x 68 x 29 mm, top flap open; 14 g.
"""
from __future__ import annotations

from dataclasses import replace as _replace

from simkit import Hinge, Part, Piece

TOP = 0.7475

# -- label maker (free body) --------------------------------------------------------------------------
BODY = (0.192, 0.201)                 # x (depth), y (width)
BODY_C = (0.426, 0.050)
X0, X1 = BODY_C[0] - BODY[0] / 2, BODY_C[0] + BODY[0] / 2        # front (keyboard edge), back
Y0, Y1 = BODY_C[1] - BODY[1] / 2, BODY_C[1] + BODY[1] / 2        # right (-y), left (+y) sides
SHELL_TOP = TOP + 0.066               # the top deck
CASSETTE = (0.066, 0.088, 0.026)      # x, y, z of a cassette
PROUD = 0.018                         # how far a seated cassette stands above the deck
FLOOR_Z = SHELL_TOP + PROUD - CASSETTE[2]                        # the bay floor
BASE_H = FLOOR_Z - TOP                # the lower shell up to the bay floor (the body's root collider)
LABELER_MASS = 1.09

# cassette bay (2 mm clearance round the cassette), the print head post in its +x -y corner
BAY_X = (0.440, 0.510)
BAY_Y = (-0.026, 0.066)
BAY_C = ((BAY_X[0] + BAY_X[1]) / 2, (BAY_Y[0] + BAY_Y[1]) / 2)
POST = (0.012, 0.012, 0.012)          # x, y, z: the print head block the cassette's cut-away corner goes round
POST_C = (BAY_X[1] - 0.002 - POST[0] / 2, BAY_Y[0] + 0.002 + POST[1] / 2)
NOTCH = 0.014                         # the cassette's cut-away corner (+x, -y), square

# the cassette's root collider is its body less the cut-away strip along +x; its reference point (the Part
# position) is that block's centre, CASS_OFF toward -x of the cassette's middle
CASS_OFF = NOTCH / 2
SEAT = (BAY_C[0] - CASS_OFF, BAY_C[1], FLOOR_Z + CASSETTE[2] / 2)   # a seated cassette's reference point

# hood cover over the bay: a 3 mm plate on 19 mm skirts, hinged along y at its back bottom edge, the latch
# tab at the foot of its front skirt over a finger recess in the deck
COVER_X = (0.430, 0.518)
COVER_Y = (BAY_C[1] - 0.054, BAY_C[1] + 0.054)
COVER_PLATE = (COVER_X[1] - COVER_X[0], COVER_Y[1] - COVER_Y[0], 0.003)
COVER_H = 0.022                       # hood height above the deck
COVER_POS = ((COVER_X[0] + COVER_X[1]) / 2, BAY_C[1], SHELL_TOP + COVER_H - COVER_PLATE[2] / 2)
HINGE_PIVOT = (COVER_X[1], BAY_C[1], SHELL_TOP + 0.0005)
COVER_OPEN_DEG = 110.0                # its stop: past upright, so it stays open on its own
TAB = (0.006, 0.024, 0.005)
TAB_C = (COVER_X[0] - TAB[0] / 2, BAY_C[1], SHELL_TOP + 0.0035)
RECESS_X = (0.410, 0.440)             # finger recess in the deck in front of the cover
RECESS_W = 0.044
RECESS_D = 0.012

# front: keyboard; back left: LCD
KEYS_X = (0.338, 0.404)
KEYS_Y = (Y0 + 0.008, Y1 - 0.008)
LCD_C = (0.474, 0.108)
LCD = (0.055, 0.060)

# -- the new cassette and its carton -------------------------------------------------------------------
CARTON = (0.068, 0.105, 0.029)
CARTON_C = (0.330, 0.345)
NEW_C = (0.325, 0.232)
CARD_T = 0.0012

DARK = (0.17, 0.18, 0.20)


def _colliders(pieces):
    return tuple(_replace(p, hidden=True) for p in pieces if p.collide)


def labeler_pos():
    return (BODY_C[0], BODY_C[1], TOP + BASE_H / 2)


def _box(x0, x1, y0, y1, z0, z1, c):
    return Piece("box", (x1 - x0, y1 - y0, z1 - z0), ((x0 + x1) / 2 - c[0], (y0 + y1) / 2 - c[1], (z0 + z1) / 2 - c[2]))


def labeler_object() -> Part:
    """The body: the lower shell up to the bay floor (root), the deck round the bay, the finger recess in
    front of the cover and the print head post in the bay."""
    import visuals
    c = labeler_pos()
    f, t = FLOOR_Z, SHELL_TOP
    ry0, ry1 = BAY_C[1] - RECESS_W / 2, BAY_C[1] + RECESS_W / 2
    P = [_box(X0, RECESS_X[0], Y0, Y1, f, t, c),                                # keyboard deck
         _box(RECESS_X[0], BAY_X[0], Y0, ry0, f, t, c),                         # beside the recess
         _box(RECESS_X[0], BAY_X[0], ry1, Y1, f, t, c),
         _box(RECESS_X[0], BAY_X[0], ry0, ry1, f, t - RECESS_D, c),             # the recess floor
         _box(BAY_X[1], X1, Y0, Y1, f, t, c),                                    # back rim
         _box(BAY_X[0], BAY_X[1], Y0, BAY_Y[0], f, t, c),                        # right of the bay
         _box(BAY_X[0], BAY_X[1], BAY_Y[1], Y1, f, t, c),                        # LCD side
         _box(POST_C[0] - POST[0] / 2, POST_C[0] + POST[0] / 2, POST_C[1] - POST[1] / 2, POST_C[1] + POST[1] / 2,
              f, f + POST[2], c)]                                                # print head post
    return Part("labeler", pos=c, size=(BODY[0], BODY[1], BASE_H), exact_z=True, color=DARK, mass=LABELER_MASS,
                friction=(0.8, 0.7), restitution=0.05, pieces=_colliders(P), hide_root=True, com=(0.0, 0.0, -0.006),
                vmeshes=visuals.labeler_vmeshes())


def cover_object() -> Part:
    """The hood cover, shut, hinged to the body along its back bottom edge (0 to 110 degrees)."""
    import visuals
    sx, sy, sz = COVER_PLATE
    skirt_h = COVER_H - sz
    zs = -sz / 2 - skirt_h / 2
    w = 0.003
    P = [Piece("box", (w, sy, skirt_h), (-sx / 2 + w / 2, 0.0, zs)),
         Piece("box", (w, sy, skirt_h), (sx / 2 - w / 2, 0.0, zs)),
         Piece("box", (sx, w, skirt_h), (0.0, -sy / 2 + w / 2, zs)),
         Piece("box", (sx, w, skirt_h), (0.0, sy / 2 - w / 2, zs)),
         Piece("box", TAB, (TAB_C[0] - COVER_POS[0], 0.0, TAB_C[2] - COVER_POS[2]))]
    return Part("cover", pos=COVER_POS, size=COVER_PLATE, exact_z=True, color=DARK, mass=0.035, friction=(0.9, 0.8),
                pieces=_colliders(P), hide_root=True, vmeshes=visuals.cover_vmeshes(),
                hinge=Hinge(pivot=HINGE_PIVOT, axis="Y", damping=0.0004, lower_deg=0.0, upper_deg=COVER_OPEN_DEG,
                            parent="labeler", parent_pos=labeler_pos()))


def cassette_object(name, pos, full, hinge=None) -> Part:
    """A cassette: its root is the body less the cut-away strip along +x; the rest of that strip (beside the
    cut-away corner) is a second block."""
    import visuals
    sx, sy, sz = CASSETTE
    root = (sx - NOTCH, sy, sz)
    strip = Piece("box", (NOTCH, sy - NOTCH, sz), (sx / 2 - CASS_OFF, NOTCH / 2, 0.0), hidden=True)
    return Part(name, pos=pos, size=root, exact_z=True, color=DARK, mass=0.032 if full else 0.022,
                friction=(0.9, 0.8), restitution=0.05, pieces=(strip,), hide_root=True,
                vmeshes=visuals.cassette_vmeshes(full), hinge=hinge)


def old_cassette_object() -> Part:
    return cassette_object("old_cassette", (SEAT[0], SEAT[1], SEAT[2] + 0.0002), full=False)


def new_cassette_pos():
    return (NEW_C[0] - CASS_OFF, NEW_C[1], TOP + CASSETTE[2] / 2 + 0.0002)


def new_cassette_object() -> Part:
    """The new cassette; its click-in seat is a joint to the body made at spawn and switched on by the rule."""
    seat = Hinge(pivot=SEAT, axis="Z", damping=0.0, lower_deg=0.0, upper_deg=0.0, parent="labeler",
                 parent_pos=labeler_pos(), local1=(0.0, 0.0, 0.0), enabled=False)
    return cassette_object("new_cassette", new_cassette_pos(), full=True, hinge=seat)


def carton_pos():
    return (CARTON_C[0], CARTON_C[1], TOP + CARD_T / 2)


def carton_object() -> Part:
    """Open paperboard carton (empty): floor (root) and four low walls; the top flap stands folded back."""
    import visuals
    cx, cy, cz = carton_pos()
    sx, sy, sz = CARTON
    zc = sz / 2 - CARD_T / 2
    P = [Piece("box", (CARD_T, sy, sz), (-sx / 2 + CARD_T / 2, 0.0, zc)),
         Piece("box", (CARD_T, sy, sz), (sx / 2 - CARD_T / 2, 0.0, zc)),
         Piece("box", (sx, CARD_T, sz), (0.0, -sy / 2 + CARD_T / 2, zc)),
         Piece("box", (sx, CARD_T, sz), (0.0, sy / 2 - CARD_T / 2, zc))]
    return Part("carton", pos=(cx, cy, cz), size=(sx, sy, CARD_T), exact_z=True, color=(0.92, 0.92, 0.90),
                mass=0.014, friction=(0.6, 0.5), pieces=_colliders(P), hide_root=True,
                vmeshes=visuals.carton_vmeshes())


# -- a label it printed, lying on the desk (set dressing that says "label maker") --------------------------------
LABEL_STRIP = (0.024, 0.090, 0.0004)
LABEL_C = (0.285, -0.060)


def label_strip_object() -> Part:
    import visuals
    return Part("label_strip", pos=(LABEL_C[0], LABEL_C[1], TOP + LABEL_STRIP[2] / 2), size=LABEL_STRIP, exact_z=True,
                color=(0.97, 0.97, 0.96), hide_root=True, vmeshes=visuals.label_strip_vmeshes())
