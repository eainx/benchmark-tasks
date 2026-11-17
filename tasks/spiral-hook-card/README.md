# spiral-hook-card

**Instruction:** "Stock the new card on the spiral hook: hang it on the hook and wind it in until the coil holds it."

A retail battery bay. A white slatwall stands at the back of the bench with a spiral anti-sweep dispensing
hook on it:
- a fixed 4.8 mm chrome rod that the cards hang on;
- above the rod, a polished wire coil turned by a front knob, with a clear label strip and yellow scan label above
  the knob. The product's clear top hood is left off so the coil stays visible; the cards hang below it, so the
  hood does not change the task.

Each card's hang tab sits in a gap between coil turns, so turning the knob moves every card along the rod
by one coil pitch per turn. Two cards of AA batteries already hang in the coil. A third, new card stands in
the opened shipper on the bench.

Stocking the new card takes four steps:
1. Hang its euro slot on the rod tip.
2. Slide it back under the knob until its tab meets the coil's front end.
3. Wind the knob backwards so the coil draws the card into its first gap.
4. While winding, keep the card steady with the other hand so it neither swings nor tilts its tab out of
   the coil.

**How this differs from accepted #2009** ("Thread a blister card onto a scan hook's lower wire…"). That
task stocks a straight wire: the card is pushed along it by hand. Here the card only moves when the coil
turns. One hand guides the hung card while the other turns the knob, and the two must move in step.

## Demo (scripted, real contacts only)

1. **Right hand: take the card.** It pinches the new card's blister bubble top and bottom, from the front, and
   lifts the card off the carton's riser insert.
2. **Hang it.** It lines the euro slot up with the rod in front of the tip, threads it on, and slides it back
   under the knob into the coil's front pocket. The coil is set so this pocket is open, between the wire's free
   end and the last bottom turn. It then lowers the slot onto the rod.
3. **Left hand: guard.** It comes in level from the arms' side and holds its closed fingertips 2 mm behind the
   card, so the card cannot swing back while the right hand lets go.
4. **Right hand: wind the coil.** It pinches the knob from the front and winds it backward in three 150° strokes,
   regripping between strokes. The wire's free end comes down in front of the card's tab, and the turn behind it
   then pushes the card back along the rod.
5. **Left hand: let go.** It withdraws after the first stroke, once the free end is in front of the card and
   before the coil starts pushing the card.
6. **End.** Both hands go back to rest.

Nothing triggers, snaps or holds a card. A card moves along the rod only by contact with the coil or the hands,
and the wire crossings stop it from being slid past the coil's front end by hand.

## Scoring

Milestones latch:

| score | milestone |
|---|---|
| 0.2 | new card lifted out of the carton |
| 0.4 | hanging on the rod by its slot |
| 0.6 | in the coil, behind its free front end |
| 0.8 | wound at least 3/4 pitch (21 mm) behind the coil's front end |
| 1.0 | all three cards hanging on the rod, hands 60 mm clear, every card at rest (< 0.05 m/s, < 0.5 rad/s) for 1 s |

- A card on the bench scores 0.
- **Real trial:** read the slot on the rod and the card's position behind the coil's front end from video, or
  measure it with a ruler.

## Forces

| load | value | YAM limit |
|---|---|---|
| card | 105 g, held by a blister pinch | 50 N grip |
| knob torque | well under 0.05 N·m (axle friction 0.004 N·m, plus card-on-rod friction about 0.3 N per card at the 23 mm coil radius) | wrist joint 7 N·m |
| knob pinch | 26 mm knob; about 10 N grip gives roughly 0.15 N·m of friction torque | |

## Layout

The slatwall stands along the bench's left side, and the hook points across the bench (-y), so the hung
cards face the room camera. `spiral.py` builds the bay in a local frame and turns it 90° about z (`W()` /
`L()`).

## Real parts (all real size)

| part | real item | size / mass |
|---|---|---|
| hook | 9 in spiral anti-sweep dispensing hook with slatwall adapter (loss-prevention fixture, e.g. siffron "Spiral Anti-Sweep Hook", 6/9/12 in depths) | 4.8 mm rod, 229 mm; coil of 3 mm wire, 46 mm across, 28 mm pitch; knob 26 mm across |
| slatwall | white melamine slatwall, 3 in slat pitch, aluminium inserts | 480 x 305 x 19 mm gondola-end section |
| context | a plain 6 in chrome peg hook with two USB-C cable cards (drawn only) | |
| cards | AA 4-pack blister cards, euro slot | 100 x 180 mm, 0.8 mm board, 70 x 66 x 18 mm blister, 105 g |
| carton | brown corrugated case-pack carton (CASE PACK 12), flaps open, slotted insert on an 80 mm riser | 140 x 70 x 100 mm |

The coil's pitch and diameter are the built values. Measure the purchased hook and update `spiral.py`
(`PITCH`, `COIL_R`, `COIL_Z`) before a real trial.

The print is a made-up brand ("Brightcell"), drawn by `make_textures.py` from free fonts (Liberation: SIL
OFL 1.1; DejaVu: Bitstream Vera licence).

## Reproduction

Write outputs outside the task folder, because the runner copies the task folder into each run:

```bash
OUT=../spiral-hook-card-outputs; mkdir -p "$OUT" /tmp/rc-task-home/tmp
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$(realpath $OUT)":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
```

Seed 0, fixed poses with no randomisation, 20 Hz control and 240 Hz physics.

## Physics

- **Coil:** one rigid body made of the wire as capsule segments plus the knob, on a world revolute axle
  along x with light damping and joint friction.
- **Cards:** rigid 0.8 mm boards. Their colliders keep the euro slot's 9 mm centre hole open, so a card
  hangs on the rod.
- **No rules:** a card moves only by contact with the rod, the coil and the hands. Nothing triggers,
  snaps or holds it.
- **Card swing damping:** a hung card carries 5/s of body damping, standing in for the slot rubbing on the rod
  and air drag. A single frictional contact point does not settle a pendulum swing.
- **Calibrated hold:** the demo threads the slot using the held card's measured offset in the blister pinch
  (`HOLD_DZ`, slot top 72 mm above the pad tips).
- **Not simulated:** board flex.

## Files

- `spiral.py`: geometry, parts and drawn meshes
- `task.py`: scene; the demo stage is pending
- `scoring.py`
- `make_textures.py` and `textures/`
- `simkit.py`, `meshlib.py`, `run.py`
