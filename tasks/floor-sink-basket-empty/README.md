# floor-sink-basket-empty

**Instruction:** "Empty the floor sink's basket into the bus tub."

Category: Societal relatability. Almost everyone has pulled the strainer or basket out of a clogged drain and shaken the
gunk into a bin, whether at a kitchen sink, a shower, a yard drain or a commercial kitchen's floor sink at close-down.
This task is the floor-sink form of that chore.

**How it differs from #2130 ("Empty the sink strainer into the bin").** That task lifts a sink strainer out by its knob.
Here the basket sits 115 mm down on a seat ledge with about 7 mm to the sink's walls, so there is no rim to take. It comes
out only by its wire bail, which lies folded flat to one side. The robot has to:
1. swing the bail up;
2. lift the shallow basket out level, so the scraps stay in;
3. carry it over the tub;
4. empty it with two hands: one holds the bail while the other lifts the basket's edge and turns it over on its pivots.

**Scope note (honest).** A real floor sink also has a grate over it. The previous versions tried two ways to remove it:
- a loose grate lifted with a grate hook;
- a grate hinged at its near edge, opened by a hand-hold rod at its free edge.

Neither reached a full-score demo. With the hook, the arm topped out at about z 1.11 m and the 6 mm rod slipped in the jaws.
With the hinged grate, the plate stalled at about 10 deg in six runs. The likely cause, not confirmed, is the gripping arm's
own wrist and forearm, which lie low over the plate. So the grate step is cut: `WITH_GRATE = False` in `drain.py`. The
hinged-grate code is kept for later work and is not in the scene.

## Real parts (all real size)

| part | real item | size / mass |
|---|---|---|
| floor sink | 12 x 12 x 6 in floor sink (e.g. Zurn Z1900 family), set flush in a tiled floor mock-up | 305 mm stainless top, 254 mm square white enamel interior, 152 mm deep |
| seat ring | four 14 mm stainless flats screwed inside, 115 mm below the floor | 3 mm thick |
| sediment basket | shallow perforated aluminium basket with a 5 mm wire bail (ears on its near and far walls, 90 mm rise, 13 mm plastic grip sleeve) | 240 x 240 x 75 mm, 0.38 kg + 0.04 kg bail |
| scraps | a lemon wedge and a lime wedge (plastic food replicas, 75 mm, 18 g), a red bottle cap (29 mm, 3 g) | clean stand-ins |
| bus tub | grey polypropylene bus box (e.g. Cambro 1520CBP / Carlisle 4401) | 508 x 381 x 127 mm, 1.1 kg |
| floor | 165 mm tiled plywood platform faced with 150 mm red quarry tile, a white-tiled wall stub on the left | 485 x 555 mm |

Frame: x away from the arm bases, y left, z up; origin midway between the arm bases at bench level (bench top
z = 0.7475). Sink centre (0.300, 0.050); bus tub centre (0.300, -0.312) on the bench, right of the platform.

## Physics (PhysX, 240 Hz)

- Things move only by gravity, contacts and one joint: the bail's pivot on the basket.
- There are no scripted forces, attachments or proximity rules.
- The bail's stop at 88 deg each side is a joint limit that stands in for the rim contact.
- Only the bail's top bar (the gripped sleeve) has a collider. Its legs run clear outside the basket and inside the sink.
- Scrap friction is 0.45/0.35 (wet plastic on aluminium).
- The basket's perforations are drawn, not cut.

## Scoring (see `scoring.py`)

| score | condition (from measured poses) |
|---|---|
| 0.25 | the basket lifted clear of the sink (its base above the floor) |
| +0.20 each | scrap lying in the bus tub (inside its inner outline less 30 mm, below its rim), counted only after the basket has been out; 3 scraps |
| 1.00 | all three in the tub and at rest for 1 s |
| cap 0.5 | any scrap outside both the tub and the basket (spilled), or the tub pushed > 50 mm / tipped > 15 deg |

Real trial: the basket clearing the floor is seen on video. Count the scraps in the tub.

## Demo

The scripted demo is in `task.py` (`full_demo`):
- **Right hand:** grips the folded bail's sleeve straight down, swings the bail up (the wire rolls in the jaws), lifts the
  basket out slowly and carries it over the tub.
- **Left hand:** its closed fingers are held flat, like a fork. It slides them in from behind under the basket's left
  bottom edge and lifts. The basket turns on its bail pivots to the 88 deg stop, then with the bail about the grip, and the
  scraps slide out into the tub.
- **Arm sag:** the arm's position servo sags under the basket, so the lift is commanded 90 mm higher than needed and the
  tipping path allows for the measured sag.

Results: the scripted demo scores 1.0 at control step 520 (the episode ends on success). The run limit is 1,200 steps.
The final run id is in the hand-off report, and the key frames are in `runs/floor-sink-basket-empty/demo_grid.png`.

## Files

- `drain.py`: geometry and parts.
- `task.py`: environment, scene/demo modes.
- `scoring.py`: scorer.
- `simkit.py`, `meshlib.py`, `runtime/`: shared helpers.
