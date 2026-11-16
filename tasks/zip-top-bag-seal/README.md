# zip-top-bag-seal

**Instruction:** "Seal the bag."

A filled 1-quart zip-top freezer bag of chopped vegetables lies on a butcher block, its press-to-seal
track open and standing out past the block's near edge. A press-to-seal track has no slider: it
closes only where it is squeezed between finger and thumb. One hand holds the mouth's corner; the other
pinches the track beside it and slides along to the far corner, pressing the rib into its groove the
whole way; then both let go and stand clear.

## Why this job

Bagging prepped vegetables, portions and leftovers in zip-top freezer bags is routine for food
preparation workers (about 900,000 in the US) and cooks, and one of the commonest household kitchen jobs.
The holding hand is what makes it work: slid along without the corner held, the pinch drags the bag and
slews it round on the board (in this simulation, by the pads' friction on the track), and the far end
is never pressed home.

## Real parts

| object | real part | size / mass | pose (m) |
|---|---|---|---|
| bag | 1-quart zip-top freezer bag, 0.1 mm (about 2.7 mil) LDPE films | 177 x 188 mm outside (6 15/16 x 7 3/8 in); about 300 g of chopped carrot, pepper and celery 30 mm deep; 320 g in all | mouth toward the arms, centreline y 0 |
| track | press-to-seal colour-change track 10 mm in from the mouth: a rib profile on the top film over a groove profile on the bottom film | each profile 4 mm wide, 1.6 mm thick; pressed home 2.6 mm; full bag width (177 mm) | x 0.205, 30 mm out past the block; as left, the rib lies 3 mm above the groove |
| butcher block | 12 x 18 in end-grain maple block | 305 x 457 x 45 mm | centre (0.3875, 0.02), near edge x 0.235 |
| chef's knife | 8 in chef's knife | 200 x 42 mm blade, 120 mm handle | on the block's right side |
| prep bowl | 1.5-qt stainless bowl of chopped vegetables | 200 mm across the rim, 85 mm deep | (0.645, 0.20) |
| bag carton | carton of quart freezer bags | 190 x 75 x 95 mm | (0.66, -0.20) |

The track profile is typical of press-to-seal freezer bags (rib-and-groove profiles about 4 mm wide on
each film); the 2.6 mm pressed-home thickness is the two 1.6 mm profiles interlocked.

## Scoring

The track is scored in 16 cells of 11 mm.

- **0.25** at least 4 cells closed.
- **0.5** at least 12 cells closed.
- **0.75** all 16 closed, the bag never dragged more than 10 mm over the block nor turned more than
  5 degrees, and never tipped past 15 degrees while still open (the load would spill).
- **1.0** also both hands' tool centres at least 80 mm from the track, held 1 s.

Dragging or turning the bag past those limits, or spilling, caps the score at 0.5. Every quantity is
measurable in a real trial: the green (closed) length of a colour-change track, the bag's displacement
and turn on the board, and where the hands end up.

## Simulation notes

- **Bag:** one rigid body: the load's bulk, the vegetables showing on top, the film's edges, the
  mouth's films and the track. Film does not crease, stretch or puff. The renderer draws no
  see-through film, so the clear film is drawn by its edges (rolls round the load, seams, sheen) and
  the load shows as it is.
- **Closing rule:** a track cell closes where a hand's pads are squeezed shut across it to the
  pressed-home 2.6 mm to within 1 mm (open, the profiles stand 6.2 mm), with the track between the pads and the fingers' closing line
  within 30 degrees of the films' normal (the rib pressed square onto its groove). A closed cell stays
  closed and reads green. The rib does not physically snap into the groove.
- **Drag:** the pads touch the groove profile itself (LDPE, friction 0.15), so a clamped pinch slid
  along the track pulls the bag by friction; the bag slides on the block (friction 0.4). Holding the
  corner keeps it still physically, not by a rule. In a development run where the holding hand
  missed the corner, the same slide dragged the bag 33 mm and turned it 23 degrees; held, it moved
  2.3 mm and turned 0.5 degrees.

## Demo

`task.py` contains a scripted waypoint demo. The left hand comes in along the track from the left
side, pitched 25 degrees down, and pinches the mouth's left corner hard, holding it. The right hand
comes in across the mouth from in front, pitched 25 degrees down, pinches the track beside the held
corner and slides along it to the right corner over 8 s, presses the far corner home and draws back.
The left lets go of its corner; the right comes back and presses home the bit of track the left hand
was holding, and both stand clear.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the
`yam-inspect-isaac:6.1.0` image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
mkdir -p /tmp/rc-task-home/tmp ./outputs
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$PWD/outputs":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --livestream 0 --viz none
python runtime/rescore.py --task task.py outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect outputs/TASK/RUN/episode-01/result.json
```

Seed 0; fixed starting poses with no randomisation. The scene settles for 30 ticks after reset before
control starts. Control is 20 Hz and physics 240 Hz; the run limit is 1,600 control steps (80 s). The
scripted demo reaches 1.0 after about 616 steps (31 simulated seconds), about 2 minutes of wall time
on one GPU. Each episode writes four 20 fps camera videos (overview, top and both wrist cameras),
setup/final PNGs and periodic snapshots, the measured state trajectory (trajectory.jsonl) and
result.json. The scorer reads only the bag's measured pose, the robot's measured TCP/gripper state and
the task's own measured quantity (which track cells are closed); `rescore.py` replays it over the
trajectory and checks it against result.json.

## Files

`task.py` (scene, the track's closing rule, scripted demo), `scoring.py` (stateful scorer), `bag.py`
(geometry and measurements of the bag, block and set dressing), `simkit.py` (compound rigid parts,
joints and the spline demo controller). No third-party assets. No hardware validation has been
performed.
