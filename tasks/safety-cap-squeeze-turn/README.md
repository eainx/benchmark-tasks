# Open a squeeze-and-turn child-resistant cap

Instruction: *Open the bottle of cleaner: take its cap off and put the cap down on the counter, then move your hands away.*

## The real task
Household chemicals come with child-resistant caps (US Poison Prevention Packaging Act, 16 CFR 1700). Examples are drain cleaners, bleach and some medicines. Adults have to open these caps correctly every time.

A squeeze-and-turn cap works like this:
- It turns only while both side panels are squeezed in. Squeezing lifts its locking lugs out of the ratchet teeth on the neck.
- It cannot be forced, and it locks again whenever the panels are let go.
- A hand cannot turn it half a turn in one go. The hand has to let go, take a fresh grip and squeeze again for each stroke.
- Meanwhile the other hand keeps the bottle from turning with the cap.

The board already has a push-and-turn cap (#2700). The squeeze-to-unlock mechanism and the regrip under it are what this task adds.

## Real items (in the scene at real size)
- **Bottle:** a flat 750 ml HDPE bottle of household cleaner, 180 x 56 x 100 mm, 0.85 kg.
  - Its 28 mm neck is near one end.
- **Cap:** a 38 mm squeeze-and-turn cap, 26 mm high.
  - Two red ribbed panels on opposite sides press in 3 mm against springs: 2.3 N to start, 6.8 N fully in.

## Simulation (parameters.py, task.py)
- **Bodies and joints:**
  - The bottle is a free rigid body (mu 0.45 on the bench).
  - The cap turns on the neck on a revolute joint.
  - The two panels slide inward on sprung prismatic joints mounted on the cap, so they are pressed by real contact with the jaws.
- **Ratchet and thread:** these are rules on measured state (task.Dynamics).
  - With both panels in at least 2 mm, the cap's joint turns with only the thread's drag.
  - Otherwise a stiff drive holds the cap at its current angle: the lugs are in the teeth.
  - After 170 degrees anticlockwise the joint is switched off, and the cap (with its panels) is loose.
- **Scoring (scoring.py):**
  - Unlock, turn and thread-free states come from Dynamics.
  - The cap's place comes from its measured pose: off the neck and standing on the counter.
  - Bottle movement and tilt come from its measured pose.
  - Hand clearance comes from the measured TCPs.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```

## Status and checks
- **Scripted demo:** 1.0 in 422 control steps.
  - The left fingertips press down on the bottle's far end to hold it.
  - The right jaws close across the panels, which squeezes both in, then turn a quarter turn.
  - The right hand lets go, re-grips the panels (squeezing again) and turns again, to 171.7 degrees.
  - It then lifts the cap off and stands it on the counter. The bottle moved 2 mm.
- **Earlier failures and their fixes:**
  - With the bottle along x and the hands 9 cm apart, the two grippers met. The bottle was turned along y and lengthened, putting the hands 15 cm apart.
  - Stopping 9 degrees short of the thread-free angle dragged the bottle along with the cap.
- **Model run (Dream-1):** 0 after 993 steps; the run stopped at the task's Dream budget limit. The panels were never squeezed and the cap was not turned.
