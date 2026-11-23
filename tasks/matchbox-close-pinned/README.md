# Close a kitchen matchbox: pin the sleeve, push the tray home flush

Instruction: *Close the box of matches so the tray is flush with the end of the box, then move your hands away.*

## The real task
A box of kitchen matches is in most homes, for candles, stoves and grills, and it is shut after every use. The empty sleeve weighs 17 g. Pushing the tray on its own just slides the whole box across the counter.

So one hand holds the sleeve down while the other pushes the tray in:
- Push too far, and the tray comes out of the other end.
- Push off-square, and the tray catches on the sleeve's edge.

## Real items (in the scene at real size)
- **Box:** a 250-count household box of kitchen matches.
- **Sleeve:** board, 110 x 58 x 27 mm, 17 g, with striking strips on both long sides.
- **Tray:** 108 mm long, about 120 matches, 55 g.
  - 0.6 mm play to the sleeve each side.
  - At the start it is pulled 50 mm out of the sleeve's right end.

## Simulation (parameters.py, task.py)
- **Bodies:** the sleeve and the tray are free rigid bodies that move by contact only, with no joints, rules or attachments.
  - Board on board: mu 0.45/0.38. Board on the bench: mu 0.40/0.33.
- **Approximations:**
  - The walls are 1.5 mm thick; real board is about 1 mm.
  - The matches are part of the tray's mass.
- **Scoring (scoring.py):** from the measured sleeve and tray poses and the measured TCPs.
  - How far the tray's end stands out of, or in from, the sleeve's end.
  - How far the sleeve has moved, and any tilt.
  - Hand clearance.
  - Every tier needs the tray seated in the sleeve: centred across it within 1.5 mm, on its floor within 2 mm, square within 3 degrees, and its far end not out of the sleeve's other end. A tray pushed through, or pushed in past flush by more than 1 mm, earns nothing.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.

## Status and checks
- **Scripted demo:** 1.0 in 233 control steps.
  - The left fingertips pin the sleeve's top near its left end.
  - The right fingertips push the tray's end in, leading with the tips, part way.
  - The demo, which may read ground truth, pauses to check where the tray's end is and corrects the rest of the push.
  - The tray's end ends 0.36 mm out of the sleeve's end, 0.08 mm off centre and seated; the sleeve moved 0.0 mm.
- **Earlier attempts:**
  - With the jaws across the push, the gripper body met the sleeve's end 30 mm short.
  - A full correction overshot by 2 mm, so the tray went in past flush.
- **Model run (Dream-1, revised source):** 0 after 369 steps. The Dream session expired; the tray was not moved.

## Revised after review (TK-TLDTMW)
- The end offset no longer subtracts the 1 mm by which the tray is shorter than the sleeve: it is now the plain distance between the two ends.
- A seating check was added (lateral, height, squareness, far end), and the offset has a lower limit, so a tray pushed out of the far end or displaced from the sleeve gets no credit.
- The demo was re-recorded with the corrected scorer.
