# tote-bail-stack

**Instruction:** "Stack the full tote on top of the other full tote."

Stack-nest bale-arm crates nest inside each other when empty. Each crate has a wire bail at each end. Before a full crate can be stacked on another, both bails of the lower crate must be swung in across its rim. The full crate is then lowered level onto the bails. If a bail is left out, the crate drops into the one below it and lands on its load.

**Scene:** two full 400 x 300 x 131 mm crates stand on the bench. Tote A, at the back, has both bails hanging out. Tote B is in front.

## Real parts

| Part | Description |
|---|---|
| Totes (2) | 12 L stack-nest bale-arm crate, 400 x 300 x 131 mm external and 364 x 264 x 121 mm internal, HDPE. Example: Engels "bale arm crate 12 l 400x300x131 mm" (https://engels.uk/bale-arm-crate-12l-400x300x131-mm-grey). The walls taper to a 330 x 230 mm base under the rim lip. Each end wall has a 120 x 33 mm hand hole under an 18 mm grip bar. About 0.6 kg empty. The model is blue. |
| Bails | Zinc-plated steel wire bails, 6 mm wire, 312 mm across, with 70 mm legs. They pivot on bosses at the ends of the long walls. About 0.11 kg each. |
| Cartons | Taped kraft corrugated cartons, 100 x 100 mm, six per crate. A's are 90 mm tall at 0.5 kg each; weight them with clean sandbags or books. B's are 70 mm tall at 0.06 kg each, light packing. |

**Placement:**

- Tote A: centred at x 0.48, y 0. Its bails hang straight down outside its end walls.
- Tote B: centred at x 0.11, y 0. Its bails hang out.
- Long sides face the arms.

## Scoring

Milestones latch once reached:

| Credit | Condition |
|---|---|
| 0.15 each | One of A's two bails is swung in. The bar lies across A's rim, with the leg at 160-180 deg from the outward horizontal. |
| 0.10 | B is lifted 20 mm clear of the bench, within 20 deg of level. |
| 0.20 | B stands on the bails. Its underside is 131-145 mm above A's underside, it is within 3 deg of level, and both bails are in. |
| 1.0 | At the end, B stands as above, both hands are 50 mm or more clear of both totes, and both totes and both bails are still for 1 s (under 1 cm/s and 0.2 rad/s). |

**Failure cases:**

- **Nested.** If B ends up in A, with its underside less than 120 mm above A's (down on A's cartons), it gets no lift or stack credit.
- **Dropped.** A tote knocked off the bench or tipped more than 30 deg is `dropped` and scores 0.

The scorer also returns `clear` and `dropped`.

**Hand clearance** is measured from the finger pads. Their positions come from the measured joints by forward kinematics at the runner's arm bases (`station.BASES`).

**In a real trial:**

- The bails' positions and the stack are checked on video.
- B's underside height above A's underside is checked with a tape measure: 137 mm when stacked, 96 mm when nested.

## Demo

**Bails.** Each arm handles the bail at its own end.

1. The hand comes in from outside the end wall, 60 deg off straight down. It closes the jaws lightly on the bail's bar near its end, about 4 N per pad. The bar is 26 mm clear of the end wall.
2. It carries the bar round the pivots. The bar turns between the pads on the way.
3. It lets go 50 deg past the top. The bail drops inward onto A's rim, into the notches.

**Tote B.** Both hands then take B by the grip bars over its end hand holes. Each hand points forward and 40 deg down. One finger goes inside the end wall and one outside, and the jaws close across the 18 mm bar at about 20 N per pad.

1. The hands lift B straight up over A's rim.
2. They carry it over A and lower it level onto the two bails.
3. They let go and move straight up, clear of the totes.

## Forces and reach

| Load | Value | YAM limit |
|---|---|---|
| Tote B | 0.96 kg, 4.7 N per hand, carried by friction on the 18 mm bar | Pinch about 20 N per pad: 2 pads x 20 N x 0.4 = 16 N per hand |
| Bail | Less than 0.05 N m at the pivots | Fine |

Every waypoint was checked with IK at the runner's arm bases.

## Simulation

- **Crates.** Each crate is one rigid body. Its detailed meshes sit over simple colliders: the tapered walls, the rim lip, the end grip bars over the hand holes, and the rim notches. The cartons are part of the crate's body.
- **Tote A's bails.** Each bail is a free rigid wire body on a revolute joint through its pivots. The joint has `physics:collisionEnabled`, so the bail still collides with its own crate. It hangs, swings and lies on the rim by contact alone, and B stands on the bails by contact. Joint damping is light. The joints have no limits; the bail's travel of about 260 deg is stopped by contact alone, hanging free outside the end wall or lying on the rim.
- **Tote B's bails.** They are drawn hanging out and are part of B's body.
- **No rules.** Nothing is moved, held or triggered by a rule.

## Reproduction

Run from the folder that contains this task folder, and write outputs outside the task folder:

```
python3 tote-bail-stack/run.py --policy scripted --output-dir ../tote-bail-stack-outputs
```

## Assets and licences

All geometry and materials are generated procedurally by `tote.py`, `meshlib.py` and `simkit.py` in this folder. No third-party meshes, textures, fonts or images are used.
