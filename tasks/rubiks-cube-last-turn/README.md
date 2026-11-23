# Finish a Rubik's cube one move from solved

Instruction: *Finish solving the Rubik's cube, then move your hand away.*

## The real task
The Rubik's cube is the best-selling puzzle toy, solved by hand by millions of people. This one is a single move from solved:
- The lower two layers are done.
- The top layer is a quarter turn out, so each side shows a top row in the colour of the face beside it.

The instruction does not say which layer to turn or which way. Only the colours say it: the top layer, a quarter turn anticlockwise seen from above (U'). The layer has to be turned without the cube sliding or tipping on the mat, and left square. A layer left part-way round blocks every other face from turning.

What this adds to a plain layer turn (#736, which turned a named layer a quarter turn): the move has to be read off the sticker colours, and a turn the wrong way leaves the cube further from solved.

## Real items (in the scene at real size)
- **Cube:** a standard 57 mm 3x3 speedcube, 90 g, yellow on top, white below.
  - Lower two layers solved: red +x, orange -x, blue +y, green -y.
- **Mat:** a 200 x 260 x 3 mm speedcubing mat, rubber-backed.

## Simulation (parameters.py, task.py)
- **Bodies:**
  - The lower two layers are one free rigid body (60 g) standing on the fixed mat (mu 1.0).
  - The top layer (30 g) turns on a revolute joint on the core's vertical axis, with light damping and 0.004 N m joint friction, like a broken-in cube.
- **Approximations:**
  - The middle layer cannot turn, and the corner-cutting of a real cube is not modelled.
  - Stickers are drawn but do not collide.
- **Scoring (scoring.py):** the top layer's turn relative to the lower layers, unwrapped step by step from their measured poses.
  - Then the cube's tilt and travel on the mat, and the hand's clearance from the measured TCPs.
  - Three quarter turns the other way also solve it and count.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.

## Status and checks
- **Scripted demo:** 1.0 in 212 control steps.
  - The right hand's jaws close across the top layer, pressing the cube lightly onto the mat.
  - It turns the layer a quarter turn anticlockwise in eight steps, lets go and lifts away.
  - The layer ends 1.6 degrees from square; the cube moved 5.5 mm and did not tilt.
- **Model run (Dream-1):** 0 after 800 steps (the step limit). The top layer was not turned and the cube was not moved.
