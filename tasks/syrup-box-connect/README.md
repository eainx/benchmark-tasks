# syrup-box-connect

**Instruction:** "Connect the new syrup box."

Fast-food back room. Two 5-gallon bag-in-box soda syrup cartons stand on the bench, knock-out faces toward
the arms. The empty one's spout still wears the syrup line's quick-disconnect connector; the new one's
flap is pushed in and its spout pulled out through the opening. The connector's collar is turned a quarter
turn and the connector pulled off the empty spout; the new spout, which rides on the slack bag and gives
way when pushed, is held by its flange while the connector is pushed onto it and its collar turned a
quarter turn to lock.

## Why this job

Changing a syrup box is routine back-room work for fast food and counter workers (about 3.5 million in the
US). One hand has to hold the spout, because it sits on a floppy bag.

## Real parts

| object | real part | size / mass |
|---|---|---|
| wire rack | low chrome wire shelf, 25 mm grooved posts | 340 x 700 mm deck, 120 mm up |
| syrup cartons (2) | 5-gallon bag-in-box cola syrup carton, white clay-coated corrugated board, printed (fictional brand "Velmora"; artwork in textures/, made by make_textures.py) | 273 x 273 x 394 mm; new one about 24 kg (fixed) |
| knock-out | round perforated flap, pushed in | 82 mm, centre 65 mm up |
| bag spout | polyethylene BIB fitting with O-ring land (a QCD-type bag fitting, as used with industry-standard QCD syrup connectors) | 46 mm flange, 25 mm spout, 40 mm long |
| connector | quarter-turn QCD-type BIB coupler (as made by the main BIB suppliers), acetal body, red locking collar, 90-degree elbow barb | 32 mm body, 40 mm collar, 60 mm long, 70 g |
| syrup line | translucent PVC beverage tubing on a 90-degree 3/8 in elbow barb, stainless clamp | 9.5 mm OD, 1.14 m |
| pump board | plywood, gas-driven BIB syrup pump screwed to it | 560 x 480 x 18 mm; pump 150 x 95 x 80 mm |

## Scoring

| score | condition |
|---|---|
| 0.25 | connector off the empty carton's spout (quarter turn counter-clockwise) and at least 30 mm from where it sat |
| 0.5 | pushed onto the new spout square (within 3 mm of its axis, 4 mm of fully on, 8 degrees) |
| 0.75 | turned a quarter turn clockwise on it (85 degrees or more): locked |
| 1.0 | also every pad point of both hands at least 5 cm from the connector's surface and the connector at rest (under 0.01 m/s, 0.5 rad/s), held 1 s |

Milestones latch (0.25 stays reached wherever the connector is set down later). Caps: the new spout pushed more than 20 mm back into its carton (not held) caps the score at 0.5; the
connector dropped below the shelf's deck caps it at 0.25. Run limit: 1,200 control steps (60 s).

## Simulation notes

- Cartons, rack, pump board and pump are fixed bodies; the cartons' colliders leave the knock-out open 60 mm deep.
- The bag is not modelled as film: the new spout rides on a sprung slide along its axis (120 N/m, -15..+45 mm),
  so it gives way when pushed unless a hand holds its flange.
- The connector's quarter-turn lock is a rule: on the empty spout it turns on a hinge about the spout's axis held
  by a damping drive (the collar's detent) whose damping drops while a closed hand grips the collar (pad tips within
  25 mm of its centre); turned 85 degrees counter-clockwise the hinge is switched off and it comes off. A second hinge
  to the new spout, made at spawn, is switched on when it is pushed on square; turned 85 degrees clockwise its
  damping is made very large (locked); the line's twist can still ease it back slowly (about 12 degrees over 10 s in
  the demo) while it stays on. No position targets: nothing snaps.
- The connector is hollow at the mouth (the spout goes in 24 mm) with a stop behind it; its collar is a ring of
  colliders the fingers grip.
- The syrup line is a physical, colliding chain: 24 capsule links (9.5 mm OD, about 47 mm each, 5 g) joined by D6
  joints (no translation; bend within 60 degrees and twist within 30 degrees against a light spring-damper), fixed at
  the pump's inlet and jointed to the connector's elbow barb. It lies in a loose loop on the bench, drags and catches
  like the real line; the score is capped at 0.5 if it is pulled taut (its ends more than 95% of its length apart).
- The carton print is an image texture of a made-up brand ("Velmora"), drawn by `make_textures.py`.
- The fitting is pulled out 45 mm through the knock-out on the bag's neck (as when changing a box) so a hand can straddle its flange.

## Real trials

Use a water-filled practice bag-in-box (a non-sticky stand-in for syrup) or protect the gripper pads with
disposable covers; wipe any drips. Everything else is off the shelf: a 5-gallon BIB carton with its QCD-type
bag fitting, a matching quarter-turn QCD coupler on 3/8 in beverage tubing, and a wire shelf.

## Asset licences

- Carton artwork (`textures/`, `make_textures.py`): made for this task; the brand "Velmora" is invented.
- Fonts used to render it: Liberation Sans / Serif (SIL Open Font License 1.1) and DejaVu Sans / Sans Mono
  (Bitstream Vera licence with the DejaVu changes, public domain), both free to use and redistribute.
- All other geometry is generated by the task's own code.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the
`yam-inspect-isaac:6.1.0` image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
mkdir -p /tmp/rc-task-home/tmp ../syrup-box-connect-outputs
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$PWD/../syrup-box-connect-outputs":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
python runtime/rescore.py --task task.py ../syrup-box-connect-outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect ../syrup-box-connect-outputs/TASK/RUN/episode-01/result.json
```

Outputs go outside the task folder (the runner snapshots the task folder). Seed 0; fixed poses with no randomisation; the scene settles for 30 ticks after reset before control starts.
Control is 20 Hz and physics 240 Hz; a trial is at most 1,200 steps (60 simulated seconds), a few minutes of
wall time on one GPU. Each episode writes four 20 fps camera videos, setup/final PNGs, the measured state
trajectory and result.json. The scorer reads only measured object poses, robot TCP/gripper state and the
connector states task.Dynamics reports.

## Files

`task.py` (scene, rules, scripted demo), `scoring.py` (scorer), `syrup.py` (geometry and visual meshes),
`simkit.py` (parts, joints, textured meshes), `meshlib.py` (mesh builders), `make_textures.py` and `textures/`
(carton print; fonts: Liberation, SIL OFL 1.1, and DejaVu), `run.py`. No hardware validation has been performed.
