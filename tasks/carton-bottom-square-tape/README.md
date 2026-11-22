# Make up a shipping carton's bottom

Instruction: *Make up the carton's bottom: keep it square, fold the two short flaps in, then the two long flaps over them, and tape along the seam with the tape gun. Then put the tape gun down and let go.*

## The real task
Hand packers erect regular slotted cartons (RSC) at a pack station many times an hour. O*NET 53-7064.00 Packers and Packagers, Hand, has the task "Assemble, line, or pad cartons, crates, or containers, using hand tools."

The steps, in order:
1. The knocked-down carton is popped open into a tube and stood upside down.
2. The two creases that were folded flat rack it back toward flat.
3. A short flap is folded in. It is barely narrower than the inside of the tube, so it only goes in as it is pushed square: it wedges the tube square, and once in, its edges bear on the long walls and keep it square.
4. The second short flap goes in, then both long flaps go over the short flaps and meet at the seam.
5. The seam is taped with a hand tape gun, and the tape is lapped down both end walls.

The empty carton weighs 160 g, so it slides away from any push. The far short flap has to be pushed toward the packer, so the other hand braces the carton's near end while it is. During taping the other hand holds the carton down, or the gun drags it along.

## Real items (in the scene at real size)
- **Carton:** stock 8 x 6 x 2 in RSC (203 x 152 x 51 mm outside), 32 ECT C-flute board 4 mm thick, about 160 g.
  - Long flaps: 189 x 71 mm (they meet at the middle of the inside).
  - Short flaps: 141 x 76 mm (1.5 mm narrower each side than the inside).
- **Tape gun:** a standard pistol-grip hand dispenser (350 g) with 48 mm polypropylene carton-sealing tape. It is the same gun model as an accepted carton-sealing task.

## Simulation (parameters.py, task.py)
- **Walls:** four rigid wall panels joined by vertical revolute crease joints, which close the loop.
  - The two knocked-down creases (near-left and far-right) have a spring toward a 15 degree racked rest, 0.003 N·m per degree.
  - Unheld on the bench, the tube settles about 10.5 degrees out of square, because the friction of its feet on the bench stops it short of the rest angle.
  - The top flaps are folded out flat under the walls as feet.
  - The long walls' creases sit one board thickness higher than the end walls', as in a folded RSC, so the long flaps lie over the short ones.
- **Flaps:** four rigid flaps on crease hinges.
  - The crease takes a set: a flap pushed to a fold angle and let go springs back 20 degrees (short) or 25 degrees (long) from it and stays there.
  - The set is updated from the measured flap angle every control step (task.Dynamics).
- **Squareness:** never set by a rule.
  - A short flap has 1.5 mm clearance to the long walls each side, so with the tube racked more than about 1.2 degrees it bears on a long wall. Pushed in, it wedges the tube square.
  - Once a short flap is in, its contact with the long walls holds the tube square.
- **Tape:** the 8 mm-cell model of an accepted carton-sealing task.
  - A cell sticks where the measured roller axle runs within 3 mm of its height over the long flaps.
  - Hanging ends stick where a fingertip smooths them down an end wall, working outward from the edge.
  - The tape tears when the gun is run 8 mm past the far edge and lifted 12 mm (a position rule).
  - Once three cells hold, the flaps' creases are set to folded, standing in for the tape holding them down.
- **Scoring (scoring.py):**
  - Rack is the largest corner's departure from 90 degrees, from the measured wall poses.
  - Folds come from the flap poses.
  - Long-over-short layering comes from the flap heights.
  - Tape comes from the cells.
  - Hand clearance comes from the measured TCPs.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed, and seed 0 is used.
- A 60-step passive settle takes about 13 s of wall time on one GPU. A full 2,400-step trial is expected to take about 5 minutes.

## Status and checks
- **Scripted demo:** 1.0 in 1,347 control steps.
  - Rack 1.16 degrees, all four flaps folded long over short.
  - Tape along the whole seam (centreline within 10 mm), torn off, and 48 mm down both end walls.
  - Gun put down, hands clear.
- **One-hand control:** the left hand never braces, and the right hand folds the near short flap.
  - It still goes in and squares the tube (10.6 to 1.3 degrees): the flap's own wedging does it.
  - Bracing is not what squares the carton.
- **Earlier failed attempts:**
  - Pushing the far short flap with nothing bracing the near end slid the whole carton about 10 cm toward the arms.
  - Taping with the carton unheld let the gun drag it along the bench.
  - These are the two places a second hand is needed.
- **Tape measurement:** the tape's offset is measured from the seam (centred between the long walls and turned with them).
- **Model run (Dream-1):** 0 after 561 steps; the run stopped when the Dream session expired.
