# comb-bind-report

**Instruction:** "Bind the booklet with the comb binder."

A manual plastic-comb binder of the upright-slot design stands on an office desk, its front toward the
arms. A half-letter booklet stands in a desktop file sorter beside it, and a black binding comb lies on
the desk in front of it, beside its open box. The comb is laid spine down on the comb opener. The booklet
is stood binding edge down in the punch slot, held square against the side guide, and punched by pulling
the punch handle on the right forward and down. The opener lever on the left is pulled forward to spread
the comb, the punched booklet is hung on the open rings, and the lever is let back so the rings close
through the holes.

## Why this job

Comb-binding reports, manuals and handouts is routine for office clerks (about 2.5 million in the US),
administrative assistants and copy-shop staff. Both steps need two hands: the pages have to be held
square against the side guide in the slot while the other hand pulls the punch handle, and the comb has to
be held open by its lever while the other hand hangs the pages on the rings.

## Real parts

| part | real item | size / mass | in the scene |
|---|---|---|---|
| binder | manual comb binder, upright punch slot (GBC CombBind C110 / Fellowes Star+ class): light-grey ABS case, charcoal deck, printed scale, orange side guide, margin dial, 21 comb-opener fingers, white page back plate, waste drawer | 400 W x 330 D x 120 H mm, 6.8 kg | fixed, front face at x 0.250, slot (6 x 15 mm deep) at x 0.285, side guide face at y -0.120, opener at x 0.425 |
| punch handle | steel lever with black rubber grip | 190 mm, grip 85 mm; rests 75 deg up, 50 deg stroke | hinged at (0.470, -0.216, 0.085) |
| opener lever | steel lever with black knob | 120 mm; rests just past upright, 50 deg stroke | hinged at (0.480, 0.216, 0.095) |
| booklet | half-letter report: 15 sheets 80 g/m2, clear 0.2 mm PVC front cover over a printed title page, black 250 g/m2 back cover, unpunched | 140 tall x 216 (binding edge) x 3.0 mm, 75 g | standing in the sorter at (0.175, 0.105) |
| file sorter | desktop upright file sorter, black steel mesh, 3 slots | 110 x 250 x 90 mm | (0.175, 0.105) |
| comb | 1/2 in (12.7 mm) black PVC binding comb, 15 rings on 14.3 mm pitch | 216 mm, 12 g | spine down, rings up, along y at (0.205, -0.190) |
| comb box | carton of 1/2 in combs, printed paperboard, lid open | 290 x 70 x 60 mm | (0.335, -0.400) |

Positions are metres in the station frame: x away from the arm bases, y left, desk top z = 0.

## Scoring

Milestones latch once reached.

- **0.25** the booklet punched square: it stands in the slot (binding edge within 2.5 mm of the slot floor
  and of the slot line, within 6 deg of upright and square) while the punch handle passes 95% of its
  stroke, with its right edge within 1.5 mm of the side guide.
- **0.5** and the comb laid on the opener (within 12 mm and 6 deg of its place, spine on the deck) and
  spread with the opener lever (past 90% of its stroke).
- **0.75** and the punched booklet hung on the spread rings: binding edge within 9 mm (x) and 3 mm (y) of
  the rings, resting within 8 mm above them, within 25 deg of upright (leaning on the back plate).
- **1.0** and the comb closed with the booklet hung (lever back within 10% of rest); booklet and comb at
  rest (below 0.01 m/s and 0.1 rad/s); every pad point of both hands (measured joints through forward
  kinematics, jaw opening from the measured finger travel) at least 50 mm from the booklet and the comb;
  held 1 s.
- Punched off the guide (holes off the comb's pitch): 0. Booklet dropped onto the desk: at most 0.25.

Real trial: the holes must be square to the binding edge (a square against the edge), and the comb must
close through all 15 holes.

## Simulation notes

- The booklet is one rigid slab (pages do not flex) and the comb one rigid part.
- **Punch** (rule): when the booklet stands seated in the slot and the punch handle passes 95% of its
  stroke, the 15 holes appear (drawn on both faces). Square if its right edge is within 1.5 mm of the guide.
- **Comb on the opener** (rule): laid within 12 mm and 6 deg of its place on the deck, a locked slide joint
  is switched on where it lies (no position target, no snap); the opener's fingers are drawn only.
- **Comb open/closed** (rule): drawn spread while the opener lever is past 90% of its stroke, closed again
  once let back within 10% of rest.
- **Hung** (rule on measured poses): a square-punched booklet whose binding edge rests over the spread
  rings. **Bound**: when the comb closes with it hung, the closed rings hold it on the comb -- a joint to the
  comb, made at spawn and switched on where it hangs (no position target, no snap).
- The handles return by weak drive springs held on their rest stops; the grips are rigid cylinders.
- Every object is drawn with detailed visual meshes over simple hidden colliders.

## Scripted demo

The right hand takes the comb off the desk and lays it on the opener. The left hand pinches the booklet's
top edge from its left side, lifts it out of the sorter, carries it over the binder and lowers it into
the slot, then pushes it against the side guide and holds it there. The right hand pulls the punch handle
through its stroke, its wrist turning with the handle, and lets it back. The right hand then takes the
booklet's top edge; the left hand lets go, takes the opener knob and pulls the lever to spread the comb.
The right hand lifts the booklet out, carries it over the comb and lowers it onto the open rings. The left
hand lets the lever back so the comb closes, the right hand lets go, and both hands go back to rest. The
plan steers the booklet by its measured pose (its hold in the hand is measured when it is pinched).

## Reproduction

`prepare` exports the runtime into `runtime/`. From the directory that contains this task folder, with
the `yam-inspect-isaac:6.1.0` image (or `python3 comb-bind-report/runtime/build_image.py --image yam-inspect-isaac:6.1.0`).
Outputs go outside the task folder, so the runner's task-source snapshot never copies them:

```bash
mkdir -p /tmp/rc-task-home/tmp ./comb-bind-report-outputs
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD/comb-bind-report":/task:ro -v "$PWD/comb-bind-report-outputs":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
python comb-bind-report/runtime/rescore.py --task comb-bind-report/task.py \
  comb-bind-report-outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect comb-bind-report-outputs/TASK/RUN/episode-01/result.json
```

Seed 0; fixed poses, no randomisation; the scene settles for 30 ticks after reset before control starts.
Control 20 Hz, physics 240 Hz; run limit 2,400 control steps (120 s). Each episode writes four 20 fps
camera videos, setup/final PNGs and snapshots, the measured trajectory and result.json. A trial takes a
few minutes of wall time on one GPU.

## Files

`task.py` (scene, rules, scripted demo and its controller), `binder.py` (geometry, parts, measurements and
their visual meshes), `scoring.py` (stateful scorer), `simkit.py` (compound rigid parts, joints, visual
meshes, smooth arm plans), `meshlib.py` (mesh builders), `run.py`. No third-party assets.
