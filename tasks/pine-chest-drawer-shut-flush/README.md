# Shut a wide wood-runner chest drawer that has gone in crooked

Instruction: *Shut the drawer of the chest all the way, so its front is flush with the chest, then let go and move your hands away.*

## The real task
A wide, shallow wooden drawer that rides on wooden runners and is steered only by the carcass sides is the classic sticking drawer. Shoved shut by one knob, it turns in its side play. A back corner then bears on one side panel and the front edge on the other. Past that point, pushing harder only makes the two contacts bind harder.

The everyday fix is the same everywhere: square the drawer up by pushing its trailing end, then push it home evenly until its front is flush. This is the start state many chests are left in: the drawer is part-way out and cocked, after someone shut it by one knob.

The ratio of drawer width to engaged depth is what makes it stick. Here the drawer is 581 mm wide and 280 mm deep, and starts only 150 mm engaged.

## Real items (in the scene at real size)
- **Chest:** a low one-drawer chest of 18 mm solid pine, 620 wide x 320 deep x 200 mm high. It is fixed to the bench, front at x 0.42, facing the arms.
  - Two 20 x 10 mm beech runners run along the side panels. They start 24 mm behind the face, so the inset front passes over them.
  - Stop blocks at the back stop the drawer with its front flush.
  - The top panel is the kicker above the drawer.
- **Drawer:**
  - Pine sides 14 mm thick and 135 mm high; a 20 mm inset front; a 6 mm plywood bottom.
  - 581 mm wide, which leaves 1.5 mm of play to each side panel.
  - Two 30 mm turned wooden knobs, 35 mm proud, 400 mm apart.
  - Mass 1.6 kg: 1.1 kg of drawer and 0.5 kg of folded towels lying in its back half.
- **Start:** the drawer is pulled 130 mm out at its middle and cocked 0.97 degrees. Its left end is 135 mm out and its right end 125 mm out. It rests against the side panels: the left back corner on the left panel, the right side at the mouth on the right panel.

## Simulation (parameters.py, task.py)
- **Chest:** a fixed body. It is one compound collider made of the bottom board, side panels, top, back, runners and stop blocks.
- **Drawer:** one free rigid body with no joint, made of the sides, back, bottom, front and knobs. The towels are visual only, but their mass is counted at the back of the drawer.
  - Its x, y, yaw and tilt are set only by contact.
  - It slides on the runners: waxed wood on beech, mu 0.30 static / 0.25 dynamic.
  - It is steered by the side panels: pine on pine, mu 0.45 / 0.35.
  - It is stopped by the stop blocks.
- **Binding:** comes only from those contacts and friction. No rule, joint or scripted jam produces it.
- **Approximations:** wood is rigid, with no swelling or give. The towels do not shift inside the drawer.
- **Scoring (scoring.py):** all measurements come from the drawer's measured pose.
  - The distance of each end of the drawer front from the chest's face, and the drawer's yaw and tilt.
  - Hand clearance comes from the measured TCPs.
  - The tiers are in task.json. The score is the current state, held for 1 s at full credit; a failure (tipped or pulled off) is permanent.

## Run
```
python3 run.py --policy scripted --episodes 1 --seed 0     # inside the yam-inspect-isaac:6.1.0 image
```
- No randomisation: the start is fixed.
- The recorded overview camera sits in front of the chest, off to the right and above the arms (`OVERVIEW_EYE` in task.py). The default overview looks at the back of the chest, where the drawer is hidden.

## Status and checks
- **Scripted demo:** 1.0 in 300 control steps (15 s).
  - The left hand pushes the trailing (left) end level with the right end, while the right hand rests on the drawer front.
  - Both hands then push the drawer home evenly, low on its front, inboard of the knobs.
  - Final: both ends 0.0 mm from flush, yaw 0.00 degrees, hands clear.
- **One-end check (the binding):** the right hand alone pushes the right end all the way.
  - The drawer binds at about 70 mm out: it stops, and the pushing hand is deflected up and sideways.
  - It then jumps in and ends cocked 0.39 degrees, left end 3.8 mm out. It scores 0.75, not 1.0.
- **Earlier attempts and fixes:**
  - Pushing the knob ends: the gripper body reached the drawer front and the chest's top rail before the knobs were home, leaving 4 to 12 mm.
  - The drawer front's bottom edge caught the front of the runners. The runners were set back 24 mm, as in a real inset drawer.
- **Model run (Dream-1):** 0.25 after 993 steps (the task's Dream budget limit). It pushed the drawer in to 17 mm (left end) and 23 mm (right end) out, not flush.
