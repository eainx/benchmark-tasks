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
  - The tray ends 0.34 mm inside flush; the sleeve moved 0.0 mm.
- **Earlier attempts:**
  - With the jaws across the push, the gripper body met the sleeve's end 30 mm short.
  - A full correction overshot by 2 mm, so the tray went in past flush.
- **Model run (Dream-1):** 0 after 369 steps. The Dream session expired; the tray was not moved.
