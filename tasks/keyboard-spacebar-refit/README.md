# Put a mechanical keyboard's spacebar back on

Instruction: *Put the spacebar back on the keyboard so it sits level on its stems and springs back when pressed, then move your hands away.*

## The real task
Keycaps are pulled off to clean under them, and a spill means doing it on the whole board. The spacebar is the key people struggle to put back.

- It has three cross sockets: the switch's in the middle and two stabilizer stems 50 mm either side.
- The two stabilizer stems are linked under the plate by one wire, so they move together.
- All three sockets have to come down over their crosses at once, with the cap square and level.
- Then the cap is pressed home with both ends pushed evenly. Pressed at one end, the far socket misses or the cap cocks.
- Once on, the key must still travel and spring back up.

## Real items (in the scene at real size)
- **Keyboard:** a full-size 104-key ANSI mechanical keyboard (case 143 x 453 x 18.5 mm) with every keycap in place except the spacebar.
  - Linear MX-style switch: 4 mm travel, 0.45 N at rest to 0.9 N bottomed.
  - Plate-mounted wire stabilizers, 100 mm between their stems.
  - The bottom-row keycaps beside the spacebar are 0.75 mm further out, and the rows behind it 2 mm further back, than a stock board: a slightly roomier layout so the fingertips fit.
- **Spacebar:** a 6.25u keycap, 118.5 x 18.2 mm, 8 g, lying keycap-up on the desk in front of the keyboard.

## Simulation (parameters.py, task.py)
- **Keyboard:** fixed. Its case, plate, switch housings and all 103 other keycaps are colliders near the spacebar and visual further away.
- **Stems:** the switch slider and the two stabilizer sliders (one body, as the wire links them) slide on sprung prismatic joints. Each carries an MX + cross (4.0 x 1.17 mm arms) built from boxes.
- **Spacebar:** a free rigid body.
  - An 8 mm solid top. A real one is a 1.5 mm shell; the thick top is there so the gripper can hold it.
  - Four skirt walls and three square boss sockets whose cross slots leave 0.08 mm play each side.
  - There is no interference fit: the cap stays on by gravity and friction.
- **Scoring (scoring.py):** from the measured spacebar and stem poses and the hands' measured TCPs.
  - A socket is on its cross when the cross is 2.5 to 4.1 mm into the 3.5 mm socket (not through its floor) and centred within 1.2 mm.
  - Then the cap's tilt, whether the stems are back at rest height, and hand clearance.
  - The tiers are in task.json.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.

## Status and checks
- **Scripted demo:** 1.0 in 502 control steps.
  - The right hand picks the spacebar up by its middle, with level fingertips on the thick top.
  - It carries the cap over the stems. The demo, which may read ground truth, checks the cap in the hand three times on the way down and corrects position, yaw and tilt.
  - It lowers the cap to 1 mm above the cross tips, holds still, then sets it on them and lets go.
  - Both hands then press the two ends down together, hold, ease off slowly so the key comes back up under the fingertips, and lift away.
  - The crosses end 2.9 mm into the middle socket and 2.8 and 3.0 mm into the stabilizer sockets; offsets 0.1 mm; tilt 1.35 degrees.
- **Earlier failures and their fixes:**
  - A thin (1.5 mm) keycap top let the fingertips tunnel through. The top is now 8 mm, and the skirt walls 2 mm.
  - Letting go with the sockets over the crosses, without pressing, left the cap 21–31 degrees cocked. The two-hand press squares it.
  - An off-centre grip, with the other hand holding one end, missed the cap altogether.
  - Pressing deeper (4 mm) did not seat the cap further: it stops about 0.6 mm short of the socket floors on the cross ridges. The seated threshold is set at 2.5 mm, the depth at which a real MX cap already holds.
- **Model run (Dream-1):** 0 after 993 steps; the run stopped at the task's Dream budget limit. The spacebar was nudged about 9 mm and turned slightly on the desk but never lifted.
