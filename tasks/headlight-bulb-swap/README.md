# headlight-bulb-swap

**Instruction:** "Replace the headlight bulb."

A halogen headlamp unit, off the car, stands upright on a folded shop towel as it sits in the car, its
back toward the right arm. In its low-beam port the old H11 bulb is locked by its quarter-turn bayonet,
with the lamp's pigtail connector latched on its L-shaped socket. The new H11 bulb lies in its opened
blister pack on the bench, socket down in the tray. The connector's latch tab is pressed while the
connector is pulled off, the old bulb is turned a quarter turn back and drawn out, the new bulb is taken
by its base (never by its glass), put into the port and turned a quarter turn to lock, and the connector
is pushed back on until it latches.

## Why this job

Replacing a burnt-out headlight bulb is routine work for automotive service technicians and mechanics
(about 700,000 in the US), quick-lube technicians and parts-counter staff. The connector has to be
unlatched before it comes off (one hand on the tab, one pulling), the bulb is turned out of and into its
bayonet port by its plastic base, and a halogen capsule's quartz glass must not be touched: skin oil on
it makes the bulb fail early.

## Real parts

| part | spec |
|---|---|
| headlamp | halogen reflector unit for a compact car, 400 W x 200 H x 250 D mm, 1.6 kg, black PP housing, clear PC lens wrapping round its outboard corner; low-beam port 36 mm (PGJ19), high-beam port with its rubber dust cover on |
| H11 bulbs (old and new) | 12 V 55 W, PGJ19-2 base with three bayonet lugs and an O-ring, grey PBT grip, L-shaped socket with two 2.8 mm blades, 10 mm quartz capsule, 25 g; locks with a quarter turn |
| connector | H11 pigtail connector, black PA66, 22 x 18 x 22 mm, press-to-release latch tab on top, two 18 AWG leads |
| blister pack | printed paperboard card 110 x 160 mm, clear PETG tray with a moulded cradle for the bulb's flange, lid torn back |
| towel | red cotton shop towel, folded 460 x 330 x 10 mm |

## Scoring

- **0.25** the old bulb is out of the port: turned its quarter turn back (lugs clear of the notches) and
  drawn at least 20 mm back.
- **0.5** and the new bulb is locked in the port: put in square and turned at least 88 degrees.
- **0.75** and the connector is latched on the new bulb's socket.
- **1.0** and both hands are at least 80 mm from the new bulb and the connector and 50 mm from the old
  bulb, the old bulb at rest, held 1 s.
- Touching the new bulb's glass (a finger pad within 2.5 mm of its capsule) caps the score at 0.5. A bulb
  or the connector falling off the bench scores 0.

A real trial is scored by eye and with a ruler: bulb out and set down, new bulb turned home (its socket
where the old one's was), connector clicked on, hands away; glass contact is seen on video.

## Simulation rules (disclosed stand-ins)

Parts are rigid bodies at 240 Hz, drawn as detailed meshes over simple hidden colliders.

- **Latch:** the connector is held on the old bulb's socket by a fixed joint until a closed hand's pad
  tips are within 8 mm of its tab for 0.3 s; then it is free.
- **Old bulb's bayonet:** a hinge about the port axis held by a stop at 0 and a stiff damping drive (its
  detent) until the connector is 10 mm off its seat and a closed hand holds the bulb's grip. Then it
  turns against light drag to -90 deg; at -88 deg the hinge is switched off and the bulb draws out.
- **New bulb's bayonet:** a hinge made at spawn and switched on when the bulb is put to its seat square
  (grip centre within 3 mm of the port axis and 4 mm of the seat, axis within 8 deg, socket within 15
  deg of straight down: lugs in the notches). It turns to a stop at +90 deg; at +88 deg it is locked:
  its lower stop is raised behind it (the detent) and its damping drive stiffened. No position targets:
  nothing snaps.
- **Connector back on:** once the new bulb is locked, the connector pushed within 4 mm and 10 deg of its
  seat latches (a joint to the bulb is switched on).
- Finger pads are placed from the measured joints and finger travel by the YAM URDF's forward kinematics.
- The leads are drawn only; the O-ring and lug contacts are not modelled.

## Scripted demo

The left hand takes the connector level from behind its leads; the right presses the latch tab from
above with closed jaws while the left pulls the connector off its socket and holds it aside. The right
grips the old bulb's base from behind, turns it a quarter turn back, draws it out and sets it down on the
bench. It takes the new bulb from its blister by its base, carries it to the port and pushes it in
(steering the last centimetres by the measured bulb pose), lets go, regrips it from behind and turns it a
quarter turn to lock. The left pushes the connector back on until it latches, and both hands stand clear.

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

Seed 0, no randomisation; the scene settles for 30 ticks after reset before control starts. Control is
20 Hz and physics 240 Hz; run limit 1,800 steps (90 s). The scripted demo takes about 60 simulated
seconds (roughly 3–15 minutes of wall time depending on the GPU). Each episode writes four 20 fps camera
videos, setup/final PNGs, the measured state trajectory and result.json. The scorer reads only measured
object poses, the robot's joints and finger travel, and the task's own latch and bayonet states.

## Files

`task.py` (scene, latch and bayonet rules, scripted demo and its controller), `scoring.py` (stateful
scorer), `headlamp.py` (geometry, colliders, joints, real-part dimensions), `visuals.py` (detailed visual
meshes), `meshlib.py` (revolve / loft / sweep builders), `simkit.py` (parts, joints, visual meshes and
hidden colliders). No third-party assets.
