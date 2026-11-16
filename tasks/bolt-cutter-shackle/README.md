# bolt-cutter-shackle

**Instruction:** "Cut the padlock off the cage gate."

A run of steel-mesh storage-cage wall stands on the bench. Its gate is padlocked shut through a hasp:
a 40 mm brass padlock whose key is lost hangs from the staple on the fixed post. A 14 in (355 mm) bolt
cutter lies shut on the lid of a steel tool case in front of it.

A bolt cutter takes both hands, one on each long handle. It is lifted, its handles opened wide, and
its open jaws set square across one shackle leg, the leg as deep in the jaws as it will go. Then both
handles are closed together until the leg is cut through. Closing one handle against a held one
swings the head sideways: it drags and twists the lock on its staple and lets the blades skate off the
round leg. Once the leg is cut, the jaws are drawn back off the lock and the cutter is laid back down
on the case and let go of.

## Scene

Positions are in the station frame (m): x away from the arm bases, y to the left, bench top z = 0.

| object | type | size / mass | pose |
|---|---|---|---|
| cage wall | fixed, painted steel | 40 x 40 mm square-tube gate stile and fixed post, 400 mm tall, 8 mm apart; 6 mm mesh on a 50 mm grid either side; runs x 0.38 to 0.74 | face toward -y at y 0.04; post centred at x 0.52, stile at x 0.472 |
| hasp and staple | fixed, steel | staple base plate on the post, hasp flap from the stile over it; the staple's eye stands 40 mm proud of the post (drawn only) | eye at (0.52, 0.0), 200 mm up |
| padlock | free on the staple's eye (hinge about x) | brass body 40 x 20 x 36 mm, 220 g; 6 mm plain-steel shackle, legs 28 mm apart (22 mm inside), 30 mm of each showing between the bow and the body | hanging from the eye, face toward -y; legs at x 0.506 and 0.534 |
| tool case | fixed, painted steel | 230 x 190 x 90 mm | centred (0.515, -0.06) |
| bolt cutter | two rigid bodies on a pivot | 14 in, 0.75 kg (tubular handles; balance 85 mm behind the pivot): red steel head and handles, hardened jaws 47 mm long (cutting point 30 mm ahead of the pivot), 28 mm rubber grips 110 mm long with their middles 260 mm behind the pivot; handles open up to 26 degrees (the blades then 13.6 mm apart at the cutting point) | lying shut on the case beside the lock, pivot at (0.56, -0.04), jaws toward +x; grips' middles at about (0.31, -0.10) and (0.31, 0.02), overhanging the case's near edge |

## Scoring

Measured on the cutter's two bodies' and the padlock's poses, the cut, and the hands' TCPs and fingers:
- **0.25** a shackle leg between the jaws at their cutting point (16 to 43 mm ahead of the pivot), square
  to them (within 20 degrees).
- **0.5** the leg cut through.
- **0.75** also the lock not dragged: until the cut, it never moved more than 10 mm from where it hung
  (the head held still while the two handles closed together).
- **1.0** also the cutter laid back down (both bodies within 15 mm of their resting height), let go of
  (no closed hand within 6 cm of either grip) and both hands at least 10 cm from the lock, held for 1 s.

In a real trial: check the cut leg by eye, film the lock against a scale behind it (or a dial gauge on
its body) for the drag, and check the cutter is lying on the case with the hands off it.

Run limit: 1,600 control steps (80 s). The scripted demo's plan is about 30 s; it reaches 1.0 and the
trial ends on success at step 509 (25 s).

## Scripted demo

The two arms each come in across the bench from their own side, fingers above and below the cutter's
grip (a side grip: the pads clamp the grip top and bottom, which holds the head's weight far better than
fingers closing across it), and take it (the right hand handle A, the left handle B). From then on the demo plans the cutter,
not the hands: its pivot's position, its yaw and how far the handles are open. Each hand's target is
that planned pose carried through the hand's grasp on its handle, measured on the real bodies as the
fingers closed. The two hands' targets therefore always agree with the pivot between them, and
neither pulls against the other. The pivot's target is nudged by its measured error (the arms sag
about 3 cm under the tool), and both wrists are turned together to keep the cutter level and its jaws'
centre line on its planned heading (the head still turns a little in the fingers); while the jaws go onto
the leg, the pivot is steered by where the leg sits in the jaws (centred in the gap, at the cutting
point). A scripted demo may steer by sim state.

The cutter is lifted off the case, drawn back clear of the lock, and its handles opened wide as its
open jaws are brought level with the near shackle leg. The jaws are moved onto the leg, square. Both
handles are then closed together, symmetrically about the pivot, onto the leg and squeezed until it is
cut (2 degrees past the bite, so the grippers stay 14 cm or more apart), then a little further through.
The jaws are drawn back off the lock, the handles shut, and the cutter is laid back on the case. The
hands open, draw back off the grips and rise away.

## Simulation approximations

- **Cutting** is a rule, not fracture: a leg square between the blades at their cutting point with
  the blades closed down onto it (their gap within 0.5 mm of the 6 mm leg) for 0.25 s is cut. Its
  14 mm cut zone is then no longer collided with and is drawn gone, so the blades close through it.
  There is no cutting force: a real 14 in cutter needs about 400 N at the grips for a 6 mm plain-steel
  shackle, far more than the jaws' contact here takes.
- **Bolt cutter:** two rigid bodies on one pivot, a stand-in for the compound linkage. The jaws open
  0.115 mm per mm of grip travel (a real compound cutter is geared further down). After the cut the
  blades close only to about 3.5 mm apart (the leg's stubs above and below stay), not fully shut.
- **Padlock:** swings on the staple's eye (a hinge about the eye); it cannot slide along the staple, and
  the staple and hasp cannot bend. The eye is drawn only.
- **Grip:** the YAM's parallel fingers clamp each rubber grip top and bottom: a human hand's wrap is
  not modelled.

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

Seed 0; the scene settles for 30 ticks after reset before control starts. Control is 20 Hz and
physics 240 Hz. Each episode writes four 20 fps camera videos, setup/final PNGs, the measured state
trajectory and result.json. The scorer reads only measured object poses, robot TCP/gripper state
and the task's own measured quantity (the cut).

## Files

`task.py` (scene, cut model, scripted demo and its controller, which places both hands from one planned
cutter pose), `scoring.py` (stateful scorer), `cutter.py` (geometry and measurements), `simkit.py`
(compound rigid parts, joints and the spline demo controller). No third-party assets. No hardware
validation has been performed.
