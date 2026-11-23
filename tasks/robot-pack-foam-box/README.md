# Pack a desktop robot back in its retail box

Instruction: *Pack the robot back into its box: robot in its foam pocket, the top foam pad on it, and the lid on, then move your hands away.*

## The real task
Returned, demo and refurbished electronics are repacked by hand in their original boxes at returns centres and shops. A small robot ships in a rigid two-piece box:
1. The robot goes down into a die-cut foam pocket in the base.
2. A foam pad goes on top of it so it cannot move.
3. The deep (telescoping) lid slides down over the base until it rests on the pad.

The pocket has 4 mm of play and the lid 2 mm each side round the base. Each part must go down square and centred, and a lid lowered skew catches on the base or on the pad.

Close boards tasks: #2264 closes a telescoping lid over a product. This task packs three parts in order, each into a fit, before the lid.

## Real items (in the scene at real size)
- **Robot:** a palm-sized desktop robot, 70 x 36 x 45 mm, 150 g, rubber wheels.
- **Box base:** 115 x 56 mm, 38 mm high, rigid board, 120 g with its black foam insert.
  - The insert's pocket is 78 x 44 x 20 mm, so the seated robot stands 15 mm above the rim.
- **Top pad:** 100 x 46 x 25 mm foam, 25 g.
- **Lid:** 123 x 64 mm, 50.5 mm deep, rigid board, 100 g.

## Simulation (parameters.py, task.py)
- **Bodies:** the base with its foam insert is one free rigid body. The robot, the pad and the lid are free rigid bodies.
- **Approximations:** foam and board are rigid. The foam does not squeeze and the board does not flex, so the plays are fixed gaps.
- **Scoring (scoring.py):** from the measured poses of the four bodies, each part relative to the base, and the hands' measured TCPs.
  - Robot: on the pocket floor, within 6 mm of centre, square within 6 degrees, upright.
  - Pad: flat on the robot, within 10 mm of centre, square within 6 degrees.
  - Lid: resting on the pad, within 4 mm of centre, square within 4 degrees, level within 3 degrees.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.

## Status and checks
- **Scripted demo:** 1.0 in 817 control steps. The right hand moves all three parts; the left stays at rest.
  - The demo may read ground truth. After lifting each part over the box, it checks the part's place and turn in the hand three times (twice high, once just above the box) and corrects the set-down.
  - The pad and the lid are pressed 4 mm down before letting go, so the opening fingers do not drag them.
  - Result: robot 2.4 mm, pad 2.7 mm and lid 2.3 mm from centre, all level; the base moved 3.3 mm.
- **Earlier failures and their fixes:**
  - The fingers close one at a time, so a light part is pushed toward the slower finger.
    - For the pad and the lid, that finger comes down just outside the part's side, so the part is pushed against it, not past it.
    - Otherwise one finger ended up inside the lid and the lid was pinched by one wall.
  - With 53 mm walls, the fingers holding the robot caught on the rim. The walls are now 38 mm, below the fingertips.
  - With 2.5 mm of play, the robot jammed tilted on the pocket's edge. The pocket now has 4 mm.
  - A lid 78 mm wide was beyond what the fingers could close round. The box was narrowed.
  - The left arm could not reach the box's middle at the lid's height, so the right arm does all three parts.
- **Model run (Dream-1):** 0 after 465 steps; the Dream service refused a request and the run ended. No part was moved.
