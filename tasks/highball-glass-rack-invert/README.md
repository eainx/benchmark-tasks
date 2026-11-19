# highball-glass-rack-invert

**Instruction:** "Rack both glasses mouth-down for the dish machine."

Two empty 10 oz heavy-base hi-ball tumblers stand mouth-up on a stainless dish table, as they come back from
the bar. Beside them is an empty 25-compartment glass rack for a door-type dish machine. Each glass has to
end mouth-down in its own compartment, resting on the rack's floor grid, with both hands off it.

## Why this job

Glassware is racked mouth-down, one glass per compartment, so the wash and rinse reach inside and the glass
drains. Racking happens many times a shift, as a core part of these jobs:

- Dishwashers: about 390,000 in the US (BLS OEWS 2023).
- Dining room and cafeteria attendants and bartender helpers: about 440,000.
- Bartenders: about 650,000.

O*NET lists the core tasks as "Wash dishes, glassware, flatware, pots, or pans, using dishwashers or by
hand" (35-9021) and "Clean glasses, utensils, and bar equipment" (35-3011).

## What makes it hard for a robot

A person turns a glass over inside one hand. A two-finger gripper can't, and its geometry forces a regrasp:

- **Picking a standing glass.** The only grip is from the side, near the rim. Lower down, the wrist meets
  the table, which needs about 90 mm of clearance. From above, the jaws would have to go inside the mouth.
- **Placing it in the rack.** A glass can only go down between the 101 mm compartment walls when held from
  above by its foot. A side grip would put the hand into the neighbouring compartments.
- **Turning it over between the two grips.** One hand picks the glass and rolls it over. The wrist roll
  joint has only ±120°, so a 180° roll has to use the whole arm. The other hand then takes the glass by the
  foot. The glass is still tipped 30° at that point, so the second grip must match the tilt. The glass is
  then straightened over the rack's wall and lowered into a 95 mm compartment.
- **Nothing to hold it while it is turned.** The glass is never free on the table mouth-down, so a dropped
  glass is a broken glass.

The difficulty can't be read off a picture. Turning the glass by regrasping between two hands, with only a
few millimetres of clearance at each grip, is the whole task.

## Real parts (all real size)

| part | real item | size / mass |
|---|---|---|
| glasses (2) | generic 10 oz (296 mL) heavy-base hi-ball tumbler, clear soda-lime glass (restaurant-supply stock item) | 116 mm tall, 66 mm across the rim, 60 mm across the foot, 2.5 mm wall, 14 mm solid base; 0.30 kg |
| rack | 25-compartment polypropylene dish-machine glass rack (Cambro Camrack 25S318 type, 19-3/4 in square) | 500 x 500 x 101 mm; 5 x 5 compartments at 97.6 mm pitch, about 95 mm inside; open floor grid 9 mm; 2.3 kg |
| dish table | brushed stainless steel table top | 640 x 1100 x 3 mm |

For a clean trial the glasses are empty and dry. Polycarbonate tumblers of the same size can stand in for
glass to rule out breakage. No brand marks appear in the scene.

## Forces

- **Glasses:** each is 0.30 kg, which is 2.9 N.
- **Grip:** the scripted grips close 6 mm past the glass's surface. At the gripper's 5000 N/m that is about
  30 N per pad.
  - Pad-on-glass friction is about 0.6, so the friction available is well over 10 N, against the 2.9 N
    weight.
  - The weight acts along the glass's axis, and the centre of mass is below the grip, so there is no
    twisting moment on either grip.
- **Rack:** 2.3 kg. It is free on the table and only moves if it is pushed.

## Scoring

These measurements come from the glass and rack poses:

- **Turned over:** a glass's mouth has pointed within 30° of straight down at any time, even while held.
- **Racked:** a glass is racked when all of these hold:
  - it is mouth-down within 10°;
  - its mouth rim rests on the rack's floor grid, within 5 mm;
  - its mouth centre is inside one compartment, with one glass per compartment;
  - both hands' finger pads are at least 20 mm off it.
- **Knocked over:** a glass that is not held and lies tipped (between 35° and 145°) at the height of the
  table or rack, or falls off the bench.

| score | condition |
|---|---|
| 0.15 per glass | the glass has been turned over (latched) |
| + 0.30 per glass | the glass is racked (the most at once, latched) |
| 1.0 | both glasses racked and at rest, held for 1 s |
| 0 | any glass knocked over or dropped at any time |

In a real trial each condition is read from video: mouth-down, in a compartment, resting on the grid and
hands off. A ruler can check that the rim rests on the grid.

## Simulation

- PhysX at 240 Hz, real-size parts.
- **Glass:** one rigid body. Its tapered outside wall is 24 box staves on the measured cone, plus a solid
  base. It renders as a translucent glass surface (opacity 0.45, faint blue-green tint).
- **Rack:** one free rigid body (2.3 kg) made of the floor-grid bars, 4 + 4 dividers and four walls.
  - The walls' lattice of posts is visual. A solid collider stands behind it.
- **Contact only:** glasses, rack and table meet only by contact. Nothing is attached, snapped or
  triggered, and no rule fires on proximity.

## Demo (scripted, physical contacts only)

The left hand does this:

1. Comes down over glass A with its approach tipped 15° down and grips it near the rim.
2. Lifts it and carries it to the hand-over point.
3. Rolls it 180° about the approach axis in six 30° steps, ending mouth-down and tipped 30°.

Then the right hand takes over:

1. Comes down along the glass's tipped axis and grips the foot. The left hand then lets go and backs off.
2. Straightens the glass as it crosses the rack's wall, about 16 mm above it.
3. Lowers it into compartment (0, 3) and lets go.

Glass B follows into compartment (0, 4), with the left wrist rolled back first. The scripted demo scores 1.0 after 893 control steps (44.65 s; run limit 1,200).
