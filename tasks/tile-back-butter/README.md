# tile-back-butter

**Instruction:** "Replace the missing tile on the step."

A raised step is tiled with 75 x 150 mm glazed ceramic tiles, four rows deep and three across, the
spacers still in the joints. One tile of the front row was cracked and has been chipped out and its old
adhesive scraped off, leaving a bare gap boxed in by its neighbours. On a sheet of cardboard on the
floor in front of the step the spare tiles are stacked, the new tile standing on edge leaning against
them, and an open tub of mixed tile adhesive stands by with a notched trowel resting across its rim.

## Why this job

Spot-repairing a cracked floor or step tile is routine work for general maintenance and repair workers
(about 1.5 million in the US) and construction laborers, and one of the commonest household fixes. The
gap is boxed in by tiles, so adhesive cannot be troweled onto the floor: it is combed onto the back of the
new tile (back-buttering) while the other hand holds the tile still, to full coverage, or the tile sounds
hollow and later cracks. The tile is then turned over, set between the spacers level with its neighbours
and pressed down to bed it.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, floor (bench top) z = 0.

| object | type | size / mass | pose |
|---|---|---|---|
| step | fixed, concrete plinth, plastered riser | 136 mm high; tiles 75 x 150 x 8 mm, 2 mm joints, 6 mm bed, face 150 mm up; front row overhangs the riser 18 mm | riser at x 0.304, y -0.131 to 0.343 |
| gap | bare concrete, old adhesive scraped off | one tile's place, front row, middle | centre (0.3195, 0.106) |
| cardboard | fixed, on the floor | 150 x 210 x 2 mm | centre (0.195, 0.10) |
| spare tiles | part of the cardboard | five tiles stacked face-down | (0.215, 0.10) |
| new tile | free | 75 x 150 x 8 mm, 200 g | standing on a long edge in front of the stack, leaning back 15 degrees against it, glaze toward the arms; centre (0.172, 0.10) |
| adhesive tub | fixed, open | 120 mm dia, 150 mm high, mixed grey adhesive 30 mm below the rim | (0.24, -0.20) |
| notched trowel | free | 70 x 140 mm steel blade, 6 mm square notches on its right edge, wooden handle along x, 260 g | resting across the tub's rim |

## Scoring

Measured on the tile:
- **0.25** adhesive over at least half of its back.
- **0.5** over at least 80 % of its back, the ANSI A108.5 minimum for dry interiors (real trial: photo of
  the back on a 12.5 mm grid before setting).
- **0.75** also set face-up in the gap: centre within 3 mm, square within 3 degrees, lippage against the
  neighbours at most 1 mm (straightedge and feeler gauge).
- **1.0** also pressed down to bed it, the trowel let go of, both hands 8 cm clear, held 1 s.
- A tile set with under 80 % coverage caps the score at 0.5 (a hollow tile).

Run limit: 2,000 control steps (100 s). The scripted demo's plan is about 38 s; it reaches 1.0 and the
trial ends on success at step 739 (37 s).

## Scripted demo

The left arm takes the leaning tile by the middle of its top edge, fingers straight down either side
of it 18 mm deep, lifts it clear of the stack (it swings plumb between the fingers) and turns it back up,
holding it out by its near edge in front of the step. The right arm takes the trowel off the tub,
tilts the blade 45 degrees and drags it across the tile's back in two passes, handle leading and the
notched edge trailing, the second pass with the blade shifted clear of the left fingertips (the strip
under them stays bare: 87.5 % coverage); it lays the trowel back across the rim and stands clear. The
left arm lifts the tile, rolls it over half a turn, lowers it into the gap (the lower finger passes
under the 22 mm nosing), lets go, and presses the set tile down with its closed fingertips before
standing clear. The controller steers by measured state, as a scripted demo may: it holds the tile's
height against the trowel, keeps the notched edge on the layer, and squares and centres the tile over
the gap.

## Simulation approximations

- **Adhesive** is 72 cells of 12.5 mm on the tile's back, not a paste. A cell is buttered where the
  trowel's notched edge passes over it with the blade 25-75 degrees to the back and the edge within 6.5 mm
  of the layer's top, while the trowel carries adhesive (a load covers 84 cells; digging the edge into the
  tub's adhesive reloads it). A hidden slab inset 25 mm from the tile's edges stands for the layer's body,
  so the trowel rides on it and a set tile sits level. No squeeze-out, slump or adhesion force.
- **Bedding** is a rule: shut fingertips, pointing down, on the set tile's face for 0.5 s.
- **The new tile** spawns clear and is stood against the stack at reset (a spawned body can only be
  yawed). The nosing overhangs the riser by 22 mm so a hand can hold the tile by its front edge while
  setting it.

## Files

`task.py` (scene, adhesive model, scripted demo and its controller), `scoring.py` (stateful scorer), `tile.py` (geometry
and measurements), `simkit.py` (compound rigid parts, joints and the spline demo controller). No
third-party assets. No hardware validation has been performed.
