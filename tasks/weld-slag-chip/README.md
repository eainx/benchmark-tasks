# weld-slag-chip

**Instruction:** "Clean the slag off the weld."

A 150 mm stick-weld stringer bead runs along a 3/8 in mild-steel coupon. The coupon is tacked flat to a steel welding fixture plate on the bench. The bead is still covered by its slag: a dark, glassy crust about 1.2 mm thick that cracked across into six pieces about 25 mm long as it cooled. A welder's chipping hammer lies on the bench to the right of the fixture. It has a chisel end, a point end and a coil-spring handle.

The welder knocks the slag off piece by piece. The chisel end is driven into the toe of the slag, where the crust meets the plate, so each piece pops off. Then the hammer is laid back down. The crust is hot and sharp, so it is never knocked off by hand. A blow on top of the crust only presses it onto the bead.

## Why it matters

Chipping slag is core, daily work for about 420,000 US welders, cutters and welder fitters (BLS OES 51-4121). It is also routine for about 75,000 structural metal fabricators and fitters and about 65,000 ironworkers. Every stick (SMAW) and flux-cored pass is chipped before the next pass, and again before inspection. O*NET lists "Chip or grind off excess weld, slag, or spatter, using hand scraper or power chipper" among welders' core tasks.

What makes it hard for a robot:
- The hammer has to be taken by its handle and turned chisel end down.
- The blow has to land on a 2–3 mm toe along a 150 mm line, at an angle that drives into the crust's edge rather than onto its top or the plate.
- A parallel gripper holds a hammer only by friction. A blow that twists the handle turns the hammer in the jaws, so the blow has to be driven along the line from the grip to the edge.

## Demo

The demo uses the right hand only; the left arm stays at rest.

1. The hand takes the spring handle straight down, jaws across it, 75 mm from the head.
2. It lifts the hammer and rolls it 60 deg about the handle, so the chisel end points down, with its edge running along the bead.
3. Each slag piece gets two blows, 10 mm apart along it. For each blow:
   - The chisel is set on the plate 10–30 mm short of the crust's near edge, its lowest corner about 0.5 mm over the plate.
   - It is driven 60 mm across the bead in +x in 0.25 s (about 0.24 m/s), which pops the piece off.
   - The chisel is lifted 20 mm off the plate before it is drawn back, so it does not drag a loose piece back onto the bead.
4. The hammer is rolled back and laid where it was, and the hand goes back to rest.

The arms are soft (joint PD control), and the 0.4 kg hammer hangs 0.1 m from the grip. With the chisel down, the hand settles about 10 mm short, 6 mm right and 16 mm low of its target. These offsets were measured in the first runs, and the demo's targets are shifted by them.

**Forces:**
- The blow's reaction acts about 73 mm below the grip, along the blow. At the 5 N the crust's bond takes, that is about 0.4 N m on the grip.
- The jaws close fully on the 20 mm handle, about 50 N per pad (the effort cap), with rubber pads (friction coefficient about 1.2). That holds roughly 1.2 N m against the handle turning in the grip.
- The hammer did not move in the jaws in any of the runs.

## Scoring

A piece is off when it has been knocked loose (moved 2 mm or more, so its joint has let go) and its centre now lies 9 mm or more to the side of the bead line, or past the bead's ends. This is judged on the current state at every step, so a loose piece still lying on the bead does not count.

| Score | Condition |
|---|---|
| 0.15 each | A slag piece off the bead (0.90 for all six) |
| 1.0 | All six off, and the hammer put down: lying still on the bench or fixture (under 2 cm/s), with no finger pad within 30 mm of its handle |

Rules:
- A piece that first moves while a finger pad is within 15 mm of it was knocked off by hand. It earns nothing.
- A hammer that ends up below the bench top is `dropped`, and there is no credit for putting it down.

**Real trial:** the six zones are marked on the coupon with soapstone. A zone counts when its bead metal is bare (photo). The hammer and the hands are judged from the video.

## Real parts

| Part | Specification |
|---|---|
| Coupon | ASTM A36 hot-rolled steel, 3/8 x 3 x 8 in (9.5 x 76 x 200 mm), mill scale on. Tacked at both ends to the fixture plate. |
| Bead | One flat stringer, 150 mm long, about 10 mm wide and 3 mm high, 1/8 in E7018 at about 120 A. Slag left on and cooled. Any welding shop or school makes these in minutes; a fresh coupon is used for each trial. |
| Fixture plate | 260 x 260 x 12 mm steel plate with a 16 mm hole grid at 50 mm pitch (a welding-table fixture plate offcut), bolted to the bench at its corners. |
| Chipping hammer | Standard welder's chipping hammer, about 0.40 kg (14 oz), 280 mm long. One-piece forged head 16 mm square with a chisel end and a point end, and a coil-spring handle 20 mm across. Sold under many brands. |

## Simulation and approximations

- The coupon, bead and fixture plate are one fixed body.
- Each slag piece is a free rigid body (4 g) held where it lies by a PhysX fixed joint to the world. The joint breaks at 5 N or 0.05 N m.
- The piece's collider is a 12 x 2 mm cap over the bead's crown and a 2 mm skirt down each side, ending 1.3 mm over the plate.
- The bead's collider is only its lowest 1 mm, a 9 mm strip, so the crust rests on its joint 0.3 mm clear of the bead. Once loose, it can slide off sideways over the strip.
- Any load past the joint's limits frees the piece. That includes a hard blow on its top (real slag also cracks off under a blow on top), but nothing short of it does.
- Real slag's bond varies with the electrode and the cooling. This one value stands in for a crust that a light chisel blow at the toe removes.
- The crust's cracking into six pieces is fixed in advance. Real slag cracks into irregular pieces.
- The hammer's spring handle is a rigid cylinder.

## Results

The scripted demo scored 1.0 in 764 steps (38.2 s), run 20261119-151809-839a03. All six pieces ended on the coupon 50–57 mm beyond the bead line, and the hammer was back on the bench with the hand clear.
