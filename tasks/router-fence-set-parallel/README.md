# Set a router-table fence 25 mm from the bit, parallel, and lock it

Instruction: *Set the router table fence so its face is 25 mm from the centre of the bit and parallel, then lock both clamping levers and move your hands away.*

## The real task
Carpenters, cabinetmakers and joiners reset a router-table or saw fence at the start of nearly every cut. O*NET 47-2031.00 Carpenters includes the task "Shape or cut materials to specified measurements, using hand tools, machines, or power saws".

A fence clamped at both ends is the awkward kind:
- With the clamps loose, moving one end swings the fence about the other.
- So the two ends have to be brought in together to the setting and kept parallel.
- Then each clamp has to be locked without knocking its end off the setting.

## Real items (in the scene at real size)
- **Bit:** a 1/2 in (12.7 mm) straight bit, 25 mm above the table, through a router insert plate. The router hangs under the bench.
- **Fence:** a 600 mm long, 19 mm MDF face, 75 mm high, on a 110 x 12 mm MDF base. It weighs 1.9 kg.
- **Tracks and clamps:** each end of the base sits on a T-bolt in an aluminium T-track let into the bench at y = ±0.27 m.
  - Each T-bolt carries an adjustable clamping lever 75 mm behind the face: a 75 mm arm on a hub.
  - Loose, the lever points straight back, away from the bit.
- **Start:** both levers are loose. The face is 40.0 mm from the bit's centre at the left lever and 33.0 mm at the right lever.

## Simulation (parameters.py, task.py)
- **Fence:** one rigid body. Each end rides on a T-bolt carrier through a vertical pivot.
  - Each carrier slides along its track on a prismatic joint with 400 N·s/m drag, standing in for the loose T-bolt and the base dragging.
  - The right pivot sits in a 6 mm slot along the fence, so the two bolts never fight the fence as it turns.
  - Moving one end swings the fence about the other end.
- **Levers:** they turn freely on their bolts (revolute joints) until they clamp.
  - A lever swung 75° clockwise from loose clamps its end where it is: the slide gets a stiff drive holding its current position (task.Dynamics).
  - This is a rule standing in for the screw clamp.
  - Backed off below 60°, the end is loose again.
- **Setting:** comes only from the hands' contact with the fence.
- **Scoring (scoring.py):**
  - Face-to-bit-centre distance at the bit, and parallelism over 500 mm, both from the fence's measured pose.
  - Lever lock state from the measured lever angles.
  - Hand clearance from the measured TCPs.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.
- The recorded overview camera is placed in front of the bench, to the right (`OVERVIEW_EYE` in task.py).

## Status and checks
- **Scripted demo:** 1.0 in 482 control steps (24.1 s).
  - The left hand pushes the far (left) end in until the fence is parallel at 33 mm.
  - Both hands push both ends in together to 25 mm and hold while the drag settles.
  - A fingertip sweeps each lever a quarter turn clockwise.
  - Final: 24.90 mm, parallel within 0.29 mm, both levers locked, hands clear.
- **Earlier failures and their fixes:**
  - **Gripping and twisting star knobs:** the fingers close one at a time, so the grip was off-centre and the turning dragged the fence 10–15 mm. Replaced by clamping levers swept with a fingertip.
  - **Turning with the jaws:** the jaws, turned along x, struck the face board. The levers were moved 75 mm behind the face.
  - **Two fixed pivots:** without the slot, the closed loop pulled the fence back toward its racked start.
- **Model run (Dream-1):** 0 after 945 steps. The run stopped at the task's Dream budget limit; the fence did not move.
