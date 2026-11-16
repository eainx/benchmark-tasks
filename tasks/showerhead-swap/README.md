# showerhead-swap

**Instruction:** "Replace the showerhead with the new one."

Everything is real size; the bench is only the working height. A section of shower wall tiled in
white 6 x 3 in subway tile stands on the left, its face toward the arms' right. Out of it, through a
64 mm escutcheon, comes a straight chrome 1/2 in shower arm carrying the old showerhead. Under the arm
a stainless wire shower basket hangs low on the wall; beside the arm a stainless shower shelf carries the
new head lying in its foam packaging cradle, nut toward the wall.

## Why this job

Swapping a showerhead is routine work for general maintenance and repair workers (about 1.5 million
in the US), plumbers and hotel and apartment maintenance staff, and a common household fix. The
head's swivel nut runs on the arm's 1/2"-14 NPSM thread, but the arm itself is only taped into a
drop-ear elbow behind the tiles: turning a stuck nut without holding the arm turns the arm in the
elbow and starts a leak inside the wall. So one hand holds the arm while the other turns the nut off
in strokes, letting go and taking a fresh grip as a wrist must; the old head is put in the basket,
not dropped; the new one is started square (cross-threading a fine thread is the classic mistake),
run up hand-tight on its conical washer.

## Real parts

| part | spec |
|---|---|
| shower arm | 1/2 in nominal brass, chrome, 21.3 mm OD, straight, 140 mm out of the wall; male 1/2"-14 NPSM thread at its end, 1/2"-14 NPT into a drop-ear elbow (PTFE tape) behind the tiles |
| showerhead (old and new) | compact fixed head: 32 mm knurled 1/2"-14 NPSM swivel nut with a conical rubber washer, ball joint, 76 mm face; hand-tight (no tools) at 4 turns (7.3 mm of thread at 14 TPI); 190-200 g |
| wall | 6 x 3 in (152 x 76 mm) glazed subway tile, 3 mm joints, on cement board; section 410 wide x 300 mm tall |
| basket | stainless wire shower basket, 260 x 110 x 55 mm, 4 mm wire on 20 mm centres, hung low on the wall |
| shelf | stainless shower shelf 140 x 110 mm on two brackets; the new head in a foam packaging cradle |

## Scene (station frame, m: x away from the arm bases, y left, z up; bench top z = 0.7475)

| object | type | pose |
|---|---|---|
| wall | fixed | face at y 0.14, x 0.21 to 0.62, 300 mm tall |
| shower arm | turns about its own axis in the elbow (damped) | axis along y at x 0.42, 240 mm above the bench, end at y 0.00 |
| old head | on the arm's thread (hinge to the arm) | nut against the arm's end, face level toward -y |
| basket | fixed | x 0.21 to 0.47, y 0.03 to 0.14, floor 5 mm above the bench |
| shelf | fixed | x 0.225 to 0.365, y 0.03 to 0.14, 192 mm above the bench |
| new head (nut and ball; face is a body on the ball) | free | on the shelf's cradle, axis along y level with the arm's, at x 0.30 |

## Scoring

- **0.25** the old head is off the thread (turned 4 turns back) and lying still in the basket, no hand
  within 6 cm of it.
- **0.5** and the new head's nut has been started on the arm's thread square (centre within 3 mm of
  the arm's axis, back face within 4 mm of the arm's end, axis within 6 degrees) and turned at least
  half a turn on.
- **0.75** and it is run up hand-tight (4 turns: seated on its washer), with the arm never turned
  more than 15 degrees in its elbow during the trial.
- **1.0** and both hands are at least 8 cm from the new head, held for 1 s.
- An arm turned over 15 degrees caps the score at 0.5 (it would leak in the wall). A head dropped onto
  the bench outside the basket scores 0.

Every quantity is measurable in a real trial: turns counted on the nut, the arm's angle from a mark
on the escutcheon, hand distances from the head.

## Simulation fidelity (disclosed rules)

- PhysX at 240 Hz, 20 Hz control.
- **Thread:** a nut on the arm is a revolute joint about the arm's axis. The old head's joint is
  switched off after 4 turns back. The new head's joint is made at spawn, switched off, and switched
  on when its nut is put to the arm's end square (the gate above); turned 4 turns on it is seated
  (hand-tight): its joint then takes a very heavy damping drive, so further torque turns the arm.
  While no hand grips it, a started new nut is held by a heavier damping drive (the thread's
  friction), so letting go and turning the open hand back between strokes does not turn it back.
  These drives have no position target, so nothing snaps when they switch. Turned back past the start it comes off. The
  nut's 7.3 mm of travel along the thread is not modelled: it turns where it sits.
- **Drag:** PhysX joint friction does not act on these single (non-articulation) joints, so drag is a
  damping drive: the stuck old nut five times the taped elbow's. Unscrewing the old nut without
  holding the arm therefore turns the arm.
- **Ball joint:** the new head's face is its own body on a hinge across the head limited to +-0.5
  degrees, so the head is effectively rigid, straight as out of the box; a real ball turns about
  every axis and holds by friction. Aiming the head is not part of the task.
- **Real threads were tried:** 1/2"-14 NPSM male and female thread meshes with SDF collision thread
  on physically (5 turns gave 8.97 mm, 1.78 mm per turn), but turned by the hand in regrip strokes
  they dragged the arm round in its elbow and ran three times slower, so this task uses the rule.
- No water, washer compression, tape or thread damage is modelled.

## Demo

The left hand comes round the wall's edge and holds the shower arm. The right hand grips the old
head's face rim square to the arm and unscrews it in nine wrist strokes, letting go and turning the
open hand back between strokes. It draws the head off and lets it drop into the basket; the left hand
lets go of the arm meanwhile. The right hand takes the new head by its face from the shelf (the plan waits until it
has actually come away with the hand), carries it square to the arm and puts the nut to the arm's end (steering by the measured nut-to-thread offset),
the left hand holds the arm again, and wrist strokes run it up hand-tight (once it seats, the hand lets
go where it is, opening its fingers slowly over 1.2 s, and backs off along the arm's axis). Both hands then stand clear.

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
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
python runtime/rescore.py --task task.py outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect outputs/TASK/RUN/episode-01/result.json
```

Seed 0; fixed poses with no jitter; the scene settles for 30 ticks after reset before control starts.
Control is 20 Hz and physics 240 Hz; the run limit is 2600 control steps (130 simulated seconds). The
scripted demo reaches 1.0 in about 1900 steps (95 simulated seconds); one trial takes roughly 12-20
minutes of wall time on a shared GPU, mostly rendering four cameras. Each episode writes four 20 fps
camera videos, setup/final PNGs, the measured state trajectory and result.json. The scorer reads only
measured object poses, robot TCP/gripper state and the task's own measured quantities (thread turns
and the arm's angle in its elbow), which `task.Dynamics` reports each step.

## Files

`task.py` (scene, thread and elbow rules, scripted demo and its controller), `scoring.py` (stateful
scorer), `shower.py` (geometry and measurements), `simkit.py` (compound rigid parts, joints and the
spline demo controller). No third-party assets. No hardware validation has been performed.
