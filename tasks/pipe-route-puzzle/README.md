# pipe-route-puzzle

**Instruction:** "Get the coolant through to the machine."

A panel of twelve rotary pipe fittings sits between the coolant supply (left) and the line
out to the machine (right). Each fitting is a disc on a swivel carrying a straight, elbow,
tee or cross pipe, with a flat T-handle on its hub. Coolant fills every fitting it can
reach from the supply, and filled fittings show blue.

The instruction gives no route and names no fitting. The model has to:
- read each fitting's shape from the camera;
- plan a route from the supply to the outlet;
- turn only the fittings that need it, by the quarter turns they need.

Five of the seven fittings on the route start turned wrong. One of them needs a half turn.

## Scene

| object | type | size / mass | pose (m) |
|---|---|---|---|
| panel with supply and outlet lines | fixed rigid | 3 x 4 cells at 95 mm pitch | centre (0.295, 0) |
| 12 rotary fittings | rigid, damped revolute about z | 88 mm disc, 26 mm pipe, 16 x 36 x 24 mm T-handle, 90 g | rows x 0.200 / 0.295 / 0.390; columns y +0.1425 / +0.0475 / -0.0475 / -0.1425 |

## Scoring

Coolant is computed from the measured fitting angles. A fitting counts only within 15 degrees
of a quarter-turn position.
- 0.25: 3 or more fittings are filled.
- 0.5: coolant reaches the outlet.
- 0.75: coolant reaches the outlet with no open end on any filled fitting.
- 1.0: adds both hands off the panel, held 1 s.

Run limit: 1,600 control steps (80 s).

In the scripted demo each arm turns the fittings on its own side, a quarter turn at a time.
For each quarter it takes the T-handle (jaws across its flats), turns the wrist 90 degrees,
lets go and lifts off; it takes the handle again before the next quarter. The plan
uses the start layout and the turns in `task.TURNS`.

## Approximations

- Coolant flow is not simulated. Connectivity comes from the measured angles and is drawn by revealing each filled fitting's blue pipe.
- The fittings turn only by gripper contact.

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

`task.py` (scene, flow display, scripted demo), `scoring.py` (stateful scorer),
`pipes.py` (geometry and flow), `check_layout.py` (offline solvability/IK check), `simkit.py` (compound rigid parts, joints and the
spline demo controller). Nothing on the bench is decorative.

No hardware validation has been performed. The geometry is sized for the YAM's
94 mm jaw opening and the bench reach measured in simulation.
