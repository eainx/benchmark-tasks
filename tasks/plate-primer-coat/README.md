# plate-primer-coat

**Instruction:** "Give the steel plate a coat of primer."

A bare steel plate lies on the bench. Beside it are a paint tray of red-oxide primer and a
block brush lying bristles-down.

The instruction mentions neither the brush nor the tray. The brush has to be picked up and
dipped in the primer before it puts anything on the plate, and wiped off on the tray's rim
before it is carried out; a brush lifted straight out drips primer across the bench. One dip covers only about half the
plate, so it has to be dipped again when it runs dry. The coat has to cover the whole plate,
and then the brush is laid down.

## Scene

| object | type | size / mass | pose (m) |
|---|---|---|---|
| steel plate | fixed rigid | 120 x 70 x 8 mm | centre (0.33, 0) |
| primer tray | fixed rigid | 70 x 100 mm inside, 16 mm deep, primer 10 mm deep | (0.33, 0.175) |
| block brush | rigid | 70 x 16 mm bristle head, 110 mm round wooden handle, 60 g | bristles-down at (0.22, 0.20) |

## Primer model

Coat applied is computed only from the brush's measured pose:
- **Loading:** bristle tips sunk at least 3 mm into the primer, brush within 30 degrees of upright, load the brush with 45 patches of coat. A drawn soaked band shows on the bristles while it carries any.
- **Coating:** loaded tips flat on the plate's top (within 4 mm, tilt under 20 degrees) coat each 10 mm patch the bristle footprint passes over, one unit of load per new patch. A primer tile appears on the patch.
- **Dripping:** a freshly dipped brush drips until its tips are drawn across the tray's far rim. Carried un-wiped outside the tray, it releases a 7 mm rigid primer drop at the tips every 4 control steps (0.2 s). Drops that come to rest outside the plate and tray count as drips.

## Scoring

Coverage is the fraction of the plate's 84 patches that have primer on them.
- 0.25 at 30% coverage.
- 0.5 at 60%.
- 0.75 at 95% with at most 1 drip on the bench and no primer brushed onto the bench (a loaded brush's tips flat on the bench, off the plate and tray).
- 1.0 adds the brush then put down flat off the plate (in the tray or on the bench) and let go, held 1 s. A brush still carrying primer laid on the bench counts as primer on the bench.

Run limit: 1,600 control steps (80 s).

In the scripted demo the left arm picks the brush up by its handle, dips it, and wipes it
off across the tray's far rim. It draws the
brush along the plate from the far edge toward itself; the brush runs dry about halfway.
It dips and wipes again, makes a second stroke, and rests the wet brush in the tray.

## Approximations

- Wet primer, flow and drying are not simulated. The coat is the pose-based model above.
- The bristles are one rigid low-friction block.
- Plate and tray are fixed.

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

The recording-only overview camera is pointed straight down on the bench (`top_down_overview` in `task.py`) so the layout reads at a glance; the model's cameras (top and both wrists) are the standard station's and are unchanged. Seed 0; the scene settles for 30 ticks after reset before control starts. Control is 20 Hz and
physics is 240 Hz. Each episode writes four 20 fps camera videos, setup/final PNGs, the measured
state trajectory and result.json. The scorer reads only measured object poses/velocities, robot
TCP/gripper state and the task's own measured quantities.

## Files

`task.py` (scene, primer model, scripted demo), `scoring.py` (stateful scorer),
`paint.py` (geometry and measurements), `check_layout.py` (offline dip/stroke/IK check), `simkit.py` (compound rigid parts, joints and the
spline demo controller). Nothing on the bench is decorative.

No hardware validation has been performed. The geometry is sized for the YAM's
94 mm jaw opening and the bench reach measured in simulation.
