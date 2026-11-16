# label-maker-cassette-swap

**Instruction:** "Put the new tape cassette in the label maker."

A desktop label maker stands on an office desk, keyboard toward the arms, its cassette bay shut under a
hinged cover with the used-up 24 mm tape cassette inside. The new cassette lies face up beside its
opened carton, and a label it printed lies on the desk. The cover's latch tab is pressed and the cover swung
up, the old cassette lifted out, the new one put in the one way it fits (tape end to the print head) and
pressed down until it clicks, and the cover shut. The label maker weighs 1.09 kg and slides on the desk
unless the other hand steadies it.


## Real parts

| object | type | size / mass | pose |
|---|---|---|---|
| label maker | free, PT-D600-class desktop labeller, ABS shell on rubber feet | 201 W x 192 D mm, deck 66 mm, hood top 88 mm (real unit 86 mm), 1.09 kg | centred (0.426, 0.05), keyboard toward -x |
| hood cover | revolute joint to the body along its back bottom edge, 0-110 deg | 88 x 108 mm, 22 mm high, 35 g; push latch tab at the foot of its front edge | latched shut, hinge at x 0.518 |
| finger recess | in the deck in front of the cover | 30 x 44 x 12 mm | x 0.410-0.440 |
| used-up cassette | free | 24 mm laminated tape cassette, 88 x 66 x 26 mm ABS, cut-away corner round the print head, 22 g | in the bay, top 18 mm above the deck |
| new cassette | free | same type, full (8 m), 32 g | face up on the desk at (0.325, 0.232) |
| carton | free, printed paperboard, top flap folded back, empty | 105 x 68 x 29 mm, 14 g | (0.33, 0.345) |
| printed label | fixed, set dressing | 24 x 90 mm laminated label | (0.285, -0.06) |

Frame: metres, x away from the arm bases, y left; desk top z = 0.7475.

## Scoring

- **0.25** the cover has been swung open past 80 degrees and the old cassette set down on the desk at least
  60 mm from the bay (its middle under 30 mm up, moving under 0.01 m/s and 0.2 rad/s). Once reached this
  stays, wherever the cassette is later.
- **0.5** and the new cassette has clicked into its seat.
- **0.75** and the cover is latched shut over it.
- **1.0** and every sampled point on both hands' finger pads (across the measured jaw opening) is at least
  50 mm from the label maker's box, held 1 s.
- The label maker pushed more than 20 mm or turned more than 10 degrees caps the score at 0.5; a cassette
  off the desk scores 0. A cassette turned round cannot get down to its seat (its cut-away corner meets the
  print head post), so it never clicks. The scorer also reports `clear` and `dropped` for the runner.

## Simulation notes (disclosed rules)

- **Latch:** while latched the cover's hinge stops at 0.5 degrees. A closed hand's pad tips within 12 mm of
  the tab's top for 0.3 s release it; the hinge then opens to its 110-degree stop, past upright, where the
  cover stays. Back down within 1.5 degrees of shut for 0.2 s, it latches again.
- **Click-in:** a new cassette within 2.5 mm of its seat across the bay, 3 mm in height and 6 degrees square,
  for 0.2 s, is held there by a joint to the body (switched on; no position target).
- The cassette's top 18 mm stands above the deck inside the cover's hood, so fingers take it by its ends
  (the real cassette sits a little lower; disclosed). Tape and ink ribbon are drawn, not simulated.

## Scripted demo

The left hand presses on the front deck to steady the label maker while the right hand presses the latch
tab, puts its closed fingers into the recess under the tab and swings the cover up past upright. The right
hand then steadies the deck while the left lifts the old cassette out by its ends, sets it on the desk,
picks up the new one and lowers it into the bay (steering its last centimetre by the cassette's measured
pose) until it clicks. The left hand steadies again while the right pushes the cover back over until it
drops shut and latches; both hands then stand clear.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the
`yam-inspect-isaac:6.1.0` image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`),
with the outputs folder outside the task folder (the runner snapshots the task folder):

```bash
mkdir -p /tmp/rc-task-home/tmp ../label-maker-cassette-swap-outputs
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$PWD/../label-maker-cassette-swap-outputs":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
python runtime/rescore.py --task task.py ../label-maker-cassette-swap-outputs/TASK/RUN/episode-01/trajectory.jsonl \
  --expect ../label-maker-cassette-swap-outputs/TASK/RUN/episode-01/result.json
```

Seed 0; fixed poses with no randomisation; the scene settles for 30 ticks after reset before control starts.
Control is 20 Hz and physics 240 Hz; the run limit is 1,600 steps (80 s). The scripted demo takes about
52 s of simulated time (a few minutes of wall time per trial). Each episode writes four 20 fps camera videos,
setup/final PNGs, the measured state trajectory and result.json. The scorer reads only measured object
poses and velocities, the robot's measured joints and finger travel, and the latch / click states.

## Files

`task.py` (scene, latch and click rules, scripted demo and its controller), `scoring.py` (stateful scorer),
`labeler.py` (geometry), `visuals.py` (detailed drawn meshes), `meshlib.py` (mesh builders), `simkit.py`
(compound parts, joints, spline demo controller), `run.py`. No third-party assets.
