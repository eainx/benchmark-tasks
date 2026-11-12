# valve-trace-shutoff

**Instruction:** "Stop the coolant spraying on the control box."

A coolant header with three ball valves runs across the near side of the bench. Their
branch lines run away from the arms at different heights, crossing over one another, and
drip into a machine trough. Each line ends on a different side from its valve:
- the left valve's line ends on the right;
- the middle valve's line ends on the left;
- the right valve's line ends in the middle.

The left valve's line has a cracked coupling that sprays onto an electrical control panel (steel enclosure with a lightning-bolt warning sign, buttons and a lamp).

The model has to trace that line back to its own valve and shut only that one. The nearest
valve to the leak (on the right) is the wrong one. The other two lines feed running
machines, and shutting one is penalised.

## Scene

| object | type | size / mass | pose (m) |
|---|---|---|---|
| header, valve bodies, three crossing branch lines, nozzles, trough, control panel | fixed rigid | header 23 mm, branches 18 mm; panel 90 x 75 x 120 mm | header x 0.19; trough x 0.50; panel (0.425, -0.215) |
| 3 ball-valve levers | rigid, damped revolute about z, stop at 91 degrees | 75 mm red lever, 16 mm grip pin on its end, 30 g | stems at (0.255, +0.11 / 0 / -0.11) |
| 96 coolant drops | rigid spheres, pooled | 7 mm, 0.07 g | parked under the bench until released |

## Flow model

From the measured lever angles, a valve passes coolant until it is 70 degrees shut. While it
passes coolant:
- drops are released at its nozzle (every 3 steps) and, for the cracked line, as a fan at the crack toward the panel (two per step);
- each drop flies as an ordinary rigid sphere;
- a translucent stream is drawn under each running nozzle; spray that lands on or at the foot of the panel shows as a wet door and a puddle that grows in three steps (6, 25 and 60 drops).

Drops are parked for reuse once they land.

## Scoring

Valve angles are measured.
- 0.25: the spraying line's valve is turned 30 degrees.
- 0.5: it is shut (70 degrees or more, the point at which the flow model stops passing coolant).
- 0.75: it is shut and the other two valves are still open (under 20 degrees).
- 1.0: adds both hands clear of the levers, held 1 s.

Turning either other valve past 45 degrees at any time scores 0: trial-and-error closing fails.

Run limit: 600 control steps (30 s).

In the scripted demo the left arm closes on the left lever's pin and swings it a quarter turn
onto its stop along the arc. The pin turns in the jaws, so the wrist keeps its yaw. Then it lets
go and stands clear.

## Approximations

- Pipe flow is the pose-based model above.
- Drops are released with a fixed velocity.
- Streams, the wet door and the puddle are drawn.

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

`task.py` (scene, measured dynamics, scripted demo), `scoring.py` (stateful scorer),
`valves.py` (geometry and measurements), `check_layout.py` (offline IK check), `simkit.py` (compound rigid parts, joints and the
spline demo controller). Nothing on the bench is decorative.

No hardware validation has been performed. The geometry is sized for the YAM's
94 mm jaw opening and the bench reach measured in simulation.
