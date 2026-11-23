# Fish a bottle cap out from under a toaster oven with a wooden spoon

Instruction: *A bottle cap has rolled under the toaster oven. Get it out with the wooden spoon without moving the oven, then put the spoon down and move your hands away.*

## The real task
Something rolling under a countertop appliance is an everyday kitchen mishap. The usual fix is the same everywhere: slide a spoon handle or a ruler under, get it behind the object, and draw the object out.

The toaster oven makes this hard in three ways:
- It stands only 22 mm off the counter, so the cap cannot be seen from above and no finger fits under.
- A straight push only sends the cap deeper.
- Its rubber feet are set well in from the sides. The handle has to go in between the feet and past the cap, then be drawn back without knocking a foot and shoving the 6 kg oven.

## Real items (in the scene at real size)
- **Toaster oven:** a 6 kg countertop oven, 300 x 440 x 210 mm body, 22 mm off the bench on four 24 mm rubber feet.
  - The feet are 50 mm in from the front and back and 100 mm in from the sides.
- **Cap:** a 30 x 12 mm polyethylene soda-bottle cap, 4 g, open side down.
  - It lies 30 mm in from the oven's right side and 170 mm in from the front.
- **Spoon:** a 330 mm beech wooden spoon (260 x 14 x 9 mm handle, 70 x 50 mm bowl), 45 g.
  - It lies on the bench to the right of the oven, along y.

## Simulation (parameters.py, task.py)
- **Bodies:** all free rigid bodies that move by contact only. There are no rules, joints or attachments.
  - The oven is one body with four rubber feet (mu 1.0), so a knock on a foot shoves it.
  - The cap has mu 0.35 and the spoon mu 0.40.
- **Approximation:** the spoon is rigid, with no flex in the handle.
- **Scoring (scoring.py):**
  - How far the cap's centre is inside or outside the oven's footprint, from the measured cap and oven poses.
  - How far the oven has moved and turned since the start.
  - Whether the spoon is lying on the bench and released: the distance from each hand to the spoon's centre line, or open fingers.
  - Whether the hands are out from under the oven.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.
- The recorded overview is low, in front and to the right of the oven, so the gap under the oven is in view (`OVERVIEW_EYE` in task.py).

## Status and checks
- **Scripted demo:** 1.0 in 317 control steps.
  - The right hand takes the handle near the bowl and slides the spoon along, tip outside the oven, to behind the cap's line.
  - It pushes the handle in under the right side between the feet, 40 mm past the cap.
  - It draws the handle toward the front, raking the cap out.
  - It then lays the spoon down and lets go.
  - Final: cap 88 mm clear of the footprint; oven moved 0.0 mm.
- **Earlier failed attempt:** with only 6 mm of clearance behind the cap, the handle (shifted 7 mm in the grip) brushed the cap and pushed it deeper. The cap then slid off the handle tip.
- **Scorer fix:** an early scorer counted the spoon as released while the hand still held it 120 mm from its centre. It now measures to the spoon's centre line.
- **Model run (Dream-1):** 0 after 993 steps; the run stopped at the task's Dream budget limit. The cap was never moved (still 30 mm under), and the oven was not moved.
