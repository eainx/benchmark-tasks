# meat-grinder-assemble

**Instruction:** "Assemble the meat grinder."

A washed #10 hand mincer (tinned cast iron, 54 mm plate size) stands on its suction foot, barrel
level, mouth toward the arms. Its four loose parts lie on a tea towel beside it: the auger, the
four-wing cross knife, the 4.5 mm grinding plate and the threaded retaining ring. They go back in one
order and one way round: auger in until its flights meet the back wall, knife onto the auger's square
with its flat cutting face out, plate over the pin with its notch on the barrel's key, ring screwed on
hand-tight, pressing plate and knife home.

## Why this job

Taking a meat grinder apart to wash it and putting it back together is routine for food preparation
workers (about 900,000 in the US), cooks and butchers. The parts fit only one way: a knife put on with
its flat face in cannot cut, a plate before the knife leaves no room for it, a ring cross-started or
loose lets the plate ride off the knife.

## Real parts

| part | real size |
|---|---|
| mincer body | #10, tinned cast iron; bore 50 mm x 150 mm; 66 mm mouth thread; 6 mm counterbore; key 8 x 6 x 5 mm at 12 o'clock; suction foot 132 mm with cam lever; round hopper 112 mm; crank 150 mm with beech handle |
| auger | 174 mm long, flights 46 mm on a 26 mm core, 10 x 19 mm front square, 8 x 14 mm pin, 11 mm rear square, 520 g |
| knife | four-wing cross knife, stainless, 48 mm, 19 mm hub with a 10 mm square hole, flat ground face; lies cutting face up, 45 g |
| plate | 2-1/8 in (54 mm), 6 mm thick, 4.5 mm holes, 8.5 mm centre hole, rim notch, 85 g |
| ring | knurled 72 mm body, two lugs, 20 mm deep, 66 mm internal thread, 46 mm front opening; stands lip up, 190 g |
| tea towel | cotton, 230 x 200 mm folded, border stripes |

## Scoring

| score | condition |
|---|---|
| 0.25 | auger seated: centre within 3 mm of its seat (flights on the back wall) and of the axis |
| 0.5 | and knife on the auger's square, flat cutting face out (axis within 8 degrees) |
| 0.75 | and plate on the pin, notch on the key (within 8 degrees), after the knife |
| 1.0 | and ring on square and turned at least 1.5 turns, plate within 1 mm of flush, knife within 1 mm of home, every part at rest (under 1 cm/s and 0.3 rad/s), both hands' pads (TCP less the pads' reach: tips, measured jaw opening, half a pad) at least 8 cm from the ring for 1 s |

The knife flat face in, or the plate put on before the knife, caps the score at 0.25. Stages latch
once reached (they are fits, not where a part happens to lie); a part never fitted leaves the score at
the stage before it. The scorer also reports `clear` (hands clear) and `dropped` (a part off the bench).

## Simulation approximations

- Every part collides through hidden simple shapes (staves round the bore, discs for the flights,
  boxes for the knife's wings, staves for the ring); what is drawn is a separate mesh of the real part.
- The fits are rules, each a joint made at spawn and switched on when the part is put to its place
  within tolerance; none has a position target. Auger: seated within 3 mm, then turns freely in its
  bearings. Knife: on the square once its hub is within 3 mm of the axis anywhere along the square and
  pin, axis within 8 degrees, the auger seated and the plate not on; it then slides along and turns
  with the square (either face out, whichever way it went on). Plate: on the pin within 3 mm, notch
  within 8 degrees of the key; it then slides along the pin. Pushed to within 1 mm of home, knife and
  plate are held there by a stiff damping drive on their slide (the square's and the pin's fit). Ring: started once within 3 mm of its
  place on the thread, lip out; a damped hinge, held where it is left between strokes, tight at 1.5
  turns; the thread's travel is not modelled. Its lip pushes plate and knife home.
- Fitted parts (and the ring) do not collide with the barrel or the auger (the counterbore, square hole and thread
  clearances are below the colliders' resolution). The mincer body is fixed (suction foot locked).

## Scripted demo

The right hand does the work; the left stays at its ready pose. It takes the auger from above across
its flights, carries it level and feeds it into the mouth past its balance point, then takes it end-on
by its front square, pushes it in until the fingers reach the mouth, and pushes it onto its seat by its
pin with the fingertips closed. It takes the knife by two wing tips from above, turns it
cutting face out, puts it onto the pin and pushes it along the square with the fingertips either side
of the pin. It takes the plate by its rim, turns it so the notch is at 12 o'clock and puts it over the
pin and pushes it flush with the fingertips, then takes the ring round its body, turns it lip out, puts it to the thread (pressing plate and
knife home) and screws it up in wrist strokes with a regrip, letting go once it is tight. Each
placement steers by the part's measured pose, as a scripted demo may.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the
`yam-inspect-isaac:6.1.0` image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
OUT="$PWD/../meat-grinder-assemble-outputs"     # outside the task folder: the runner snapshots the task's source
mkdir -p /tmp/rc-task-home/tmp "$OUT"
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$OUT":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --livestream 0 --viz none
python runtime/rescore.py --task task.py "$OUT"/TASK/RUN/episode-01/trajectory.jsonl \
  --expect "$OUT"/TASK/RUN/episode-01/result.json
```

Seed 0, no randomisation; the scene settles for 30 ticks after reset before control starts. Control
is 20 Hz and physics 240 Hz; the run limit is 2,400 steps. A trial takes roughly 5 to 10 minutes of
wall time on one GPU. Each episode writes four 20 fps camera videos, setup/final PNGs, the measured
state trajectory and result.json. The scorer reads only measured part poses, the robot's TCP/gripper
state and the fit state the task's rules report.

## Files

`task.py` (scene, fit rules, scripted demo and controller), `scoring.py` (stateful scorer),
`grinder.py` (parts, colliders, fit joints, measurements), `meshes.py` (visual mesh builders),
`simkit.py` (compound rigid parts, hidden colliders, visual meshes, joints). No third-party assets.
No hardware validation has been performed.
