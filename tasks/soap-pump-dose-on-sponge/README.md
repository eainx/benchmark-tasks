# One pump of dish soap onto the sponge

Instruction: *Put one pump of dish soap on the sponge, then move your hands away.*

## The real task
Pumping dish soap or hand soap onto a sponge or a hand is one of the most common things people do at a sink. A pump head swivels freely, so its spout is often left facing the wrong way.

To get one dose on the sponge:
- Swing the spout round until it overhangs the sponge before pressing, or the soap lands on the counter.
- Press the head down through its whole stroke against the spring for a full dose.
- Let it come back up.
- Do all this without sliding or knocking over the light bottle.

## Real items (in the scene at real size)
- **Bottle:** a 250 ml Boston-round dish-soap bottle, 64 mm diameter, 115 mm body, 0.30 kg nearly full.
- **Pump:** a standard 28 mm lotion pump.
  - The 34 mm head has a spout reaching 46 mm from its axis.
  - Stroke 16 mm; the spring gives about 2.6 N at rest and 6 N at the bottom.
  - The head swivels freely. At the start its spout points straight back.
- **Sponge:** a 95 x 65 x 28 mm kitchen sponge, scouring side up, lying against the bottle's right side.

## Simulation (parameters.py, task.py)
- **Free bodies:** the bottle and the sponge (mu 0.5 for the bottle on the bench).
- **Pump stem:** (drawn as a white tube under the head, riding down into the screw closure on the neck) slides in the bottle on a sprung prismatic joint (220 N/m, preloaded to 2.6 N, 16 mm stroke).
- **Pump head:** swivels freely on the stem on a revolute joint.
- **Soap:** not simulated. The dose lands where the measured spout tip is when the head first reaches 13 mm down in a stroke.
- **Scoring (scoring.py):**
  - The stroke, from the stem's measured height in the bottle.
  - Whether the dose landed on the sponge: the spout tip over the sponge's measured footprint (5 mm in from its edges).
  - The stroke count, using a 4 mm return threshold.
  - How far the bottle has shifted and tilted.
  - Hand clearance, from the measured TCPs.
  - Any stroke off the sponge (soap on the counter) scores 0.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.

## Status and checks
- **Scripted demo:** 1.0 in 373 control steps.
  - A fingertip of the right hand behind the spout swings it clockwise until it overhangs the sponge (−99 degrees).
  - The fingertips press the head straight down through the full stroke, hold, and let it come back up.
  - Result: one stroke, on the sponge; the bottle moved 0.1 mm.
- **Placement:** the bottle stands where the right arm can reach its head. A YAM's reach falls off sharply about 180 mm above the bench beyond x 0.30. With the bottle further out or taller, both arms' IK fell 55–180 mm short of the head.
- **Spring choice:** a 13 N lotion-pump spring was too stiff for the arm to press through a full stroke. A light dish-soap pump is used instead.
- **Model run (Dream-1):** 0 after 993 steps; the run stopped at the task's Dream budget limit. No stroke was made, and the spout was barely turned (1.2 degrees).
