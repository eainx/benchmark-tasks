# headlight-bulb-swap

**Instruction:** "Replace the headlight bulb."

A real 2015-2017 Toyota Camry LE/XLE halogen headlamp (driver side) is bolted, as in the car, into a
real-size front engine-bay section built on the bench: three bolts along its top into the upper radiator
support, one through its outboard ear into the fender apron. The radiator stands inboard of it, the
fender apron outboard, and the air-filter box in front of its inboard half; the arms stand where the
engine would be, facing the lamp's back. In its low-beam (H11) port the old bulb is locked by its
quarter-turn bayonet with the lamp's pigtail connector latched on its socket, which points outboard; the
connector's two taped leads are seven colliding articulated links running to a harness clip low on the
lamp's back. The new H11 bulb lies in its opened blister pack on the air box's lid. One hand pinches the
connector, pressing its latch tab down, pulls it off and holds it aside on its leads; the other turns
the old bulb a quarter turn back and draws it out, sets it on the bench, takes the new bulb by its base
(never by its glass), puts it in square and turns it a quarter turn home; the connector is pushed back on
until it latches.

## Why this job

Replacing a burnt-out headlight bulb is routine work for automotive service technicians and mechanics
(about 700,000 in the US), quick-lube technicians and parts-counter staff. The connector has to be
unlatched before it comes off (one hand on the tab, one pulling), the bulb is turned out of and into its
bayonet port by its plastic base, and a halogen capsule's quartz glass must not be touched: skin oil on
it makes the bulb fail early.

## Real parts

| part | spec, part numbers and sources |
|---|---|
| headlamp | 2015-2017 Toyota Camry LE/XLE/Hybrid halogen headlamp assembly, driver (left) side, chrome reflector: Toyota OEM 81150-06860 (superseding 81150-06D90), Partslink TO2502222, CAPA Depo 312-11F4L-AC1, aftermarket TYC TY1264-B001L. Low beam H11, high beam 9005, both changed from behind with a counter-clockwise quarter turn. Shipping box 840 x 356 x 356 mm; the lamp itself about 750 W x 300 H x 300 D mm and 2.5-3 kg (modelled at that size; measure the part). Four 10 mm-head bolts: three along its top into the upper radiator support, one through its outboard ear. |
| H11 bulbs (old and new) | 12 V 55 W, PGJ19-2 base with three bayonet lugs and an O-ring, grey PBT grip, L-shaped socket with two 2.8 mm blades, 10 mm quartz capsule, 25 g; locks with a quarter turn |
| connector | H11 pigtail connector, black PA66, 22 x 18 x 22 mm, a grey sprung latch tab on top (2 mm travel, about 4 N to release) |
| leads | two 18 AWG leads taped together (5 mm across, about 200 mm) to a harness clip low on the lamp's back, then a 13 mm corrugated loom |
| blister pack | printed paperboard card 110 x 160 mm, clear PETG tray, lid torn back, its near end open |

Sources: Toyota parts catalogue https://autoparts.toyota.com/products/product/headlamp-assy-lh-8115006860 ;
CAPA unit https://www.headlightsdepot.com/headlight-halogen-left-driver-high-quality-capa-fits-2015-2017-toyota-camry-le-xle-hybrid-sku-312-11f4l-ac1.html ;
TYC unit (H11 low / 9005 high) https://www.headlightsdepot.com/sku-ty1264-b001l.html ;
a stock listing for TO2502222 https://partsgousa.com/products/2015-2017-toyota-camry-head-lamp-driver-side-halogen-chrome-le-xle-lh-to2502222 ;
bulb sizes https://lightninglab.net/toyota/camry/2016 ; bulb change from behind
https://www.youcanic.com/change-low-beam-toyota-camry-2012-2017/ ; mounting bolts
https://www.youcanic.com/replace-headlight-assembly-toyota-camry-2012-2017/ ; box size
https://www.noramstore.com/i-31654004-2015-2017-toyota-camry-drivers-headlight-assembly.html .

## The fixture: an engine-bay section on the bench

Coordinates are the task's frame: x away from the arm bases, y to the robot's left, z up from the bench
top. All parts are off the shelf; the T-slot rails let every bolt point be set to the measured lamp.

| fixture part | what to buy | size |
|---|---|---|
| frame | 80/20 1010 T-slot profile (1 in, 1/4-20) with #4119 2-hole corner brackets and #3393 screw + T-nut kits, or Misumi HFS6-3030 with HBLFSN6 brackets and M6 T-nuts | base rails 2 x 900 mm along y at x 0.38 and 0.75, 2 x 400 mm along x at y -0.50 and 0.66; four 330 mm posts at the base corners |
| upper radiator support (lamp's top bolts) | one profile rail along y on the posts, with a 3 mm steel or aluminium hat-channel strip bolted on it | 970 mm, top at 330 mm, spanning x 0.395-0.46, y -0.33 to 0.64 |
| radiator (inboard obstacle) | 3 mm aluminium sheet or 6 mm HDPE panel bolted upright to the frame | 330 W x 280 H mm at x 0.44-0.47, y 0.315-0.64 |
| fender apron (outboard obstacle) | 3 mm aluminium sheet or 6 mm HDPE panel bolted upright to the frame, with a 50 mm flange along its top | 550 L x 330 H mm in the plane y = -0.474, x 0.25-0.80 |
| air box (front obstacle) | a 180 x 180 x 140 mm plastic box (or a 6 mm HDPE box made to size) screwed to the bench | x 0.17-0.35, y 0.06-0.24 |
| lamp hardware | four M6 x 16 flange bolts (10 mm heads) and M6 T-nuts; one 3 mm steel angle bracket on the fender panel for the lamp's ear | |
| harness clip | a 13 mm P-clip on the lamp's back, low, with the leads taped to 200 mm | at x 0.394, y -0.172, z 0.030 |

Mounting: (1) build the base frame and posts square on the bench; (2) bolt the radiator-support rail
across the posts with its top 330 mm up; (3) stand the lamp on the bench with its back face in the plane
x = 0.40, its low-beam port centre at y -0.08, 150 mm up (adjust along the rails to the measured port),
and bolt its three top tabs to the rail with M6 flange bolts into T-nuts; (4) bolt the fender panel
upright at y -0.474 and the angle bracket on it under the lamp's outboard ear, and bolt the ear;
(5) bolt the radiator panel inboard and screw the air box down; (6) clip the leads low on the lamp's back
and seat the bulb and connector as in task.json. Measure the four gaps (port to air box, port to fender,
port to radiator support, port to bench) and adjust to the coordinates above before each trial.

## Scoring

- **0.25** the old bulb is out: turned its quarter turn back (lugs clear of the notches), withdrawn at
  least 20 mm off its seat (a milestone, kept once reached) and still at least 15 mm away, out of the
  port, wherever it was set down.
- **0.5** and the new bulb is locked in the port: put in square and turned at least 88 degrees.
- **0.75** and the connector is latched on the new bulb's socket.
- **1.0** and every finger pad's surface is at least 80 mm from the new bulb's and the connector's
  surfaces and 50 mm from the old bulb's, with the old bulb at rest (under 0.02 m/s and 0.2 rad/s),
  held 1 s.
- Touching the new bulb's glass caps the score at 0.5. Only actual contact counts, in both kinds of trial.
  A real trial counts it when any frame of the wrist or overview videos shows a finger on the capsule, or
  when the capsule shows a smudge afterwards (held against a lamp, before and after). In simulation it
  counts when a finger pad's surface reaches the capsule (within 0.5 mm, the simulator's contact
  tolerance). A near miss is not penalised in either. A bulb or the connector falling off the bench
  scores 0.

A real trial is scored by eye and with a ruler: bulb out and set down, new bulb turned home (its socket
where the old one's was), connector clicked on, hands away; glass contact is seen on video.

## Simulation rules (disclosed stand-ins)

Parts are rigid bodies at 240 Hz, drawn as detailed meshes over simple hidden colliders.

- **Latch:** the connector is held on the old bulb's socket by a fixed joint. Its latch tab is a
  separate sprung key on a prismatic joint (3.5 kN/m, 2 mm travel); only a finger physically pushing it
  down 1.2 mm or more (about 4 N) for 0.2 s releases the connector (in the demo, the pinching hand's
  upper pad).
- **Leads:** the two taped leads are seven colliding articulated links: 5 mm capsules jointed end to end by
  D6 joints (translations locked, rotations sprung at 0.0003 N m/deg with damping) from the connector's
  back end to the harness clip, which is fixed to the lamp. They collide with the lamp, the bay and the
  hands (jointed neighbours excepted); a freed connector hangs on them and cannot be carried further than
  they reach. Their poses are logged in the trajectory as lead0..lead6.
- **Old bulb's bayonet:** a hinge about the port axis held by a stop at 0 and a stiff damping drive (its
  detent) until the connector is 10 mm off its seat and a closed hand holds the bulb's grip. Then it
  turns against light drag to -90 deg; at -88 deg, with the connector off its socket, the hinge is
  switched off and the bulb draws out.
- **New bulb's bayonet:** a hinge made at spawn and switched on when the bulb is put to its seat square
  (grip centre within 3 mm of the port axis and 4 mm of the seat, axis within 8 deg, socket within 15
  deg of straight down: lugs in the notches). It turns to a stop at +90 deg; at +88 deg it is locked:
  its lower stop is raised behind it (the detent) and its damping drive stiffened. No position targets:
  nothing snaps.
- **Connector back on:** once the new bulb is locked, the connector pushed within 4 mm and 10 deg of its
  seat latches (a joint to the bulb is switched on).
- Finger pads are oriented boxes (8 mm thick, 24 mm wide, 45 mm back from the tip) placed from the
  measured joints and jaw opening by the YAM URDF's forward kinematics at the rig's arm bases
  (y = +-0.305 m); glass contact and hand clearance are distances from these boxes' full surfaces.
- The O-ring and lug contacts are not modelled (the bayonets are the hinges above).

## Scripted demo

The right hand comes at the connector level from behind and pinches it top to bottom, its upper finger
pad pushing the latch tab down; it pulls the connector off the socket outboard and holds it aside on its
leads. The left hand comes over the air box, grips the old bulb's base from behind, turns it a quarter
turn back, draws it out and sets it on the bench. It takes the new bulb from its pack on the air box's
lid by its base, from behind, carries it to the port, pushes it in square (steering the last centimetres
by the measured bulb pose), turns it a quarter turn home and lets go. The right pushes the connector
back onto the new bulb's socket until it latches, and both hands stand clear.

Reach was checked with the YAM's IK in this layout at the lamp's real size: every grip, turn, pick and
pinch pose solves exactly; the bay parts sit clear of the hands' paths.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the
`yam-inspect-isaac:6.1.0` image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
# outputs go OUTSIDE the task folder: the runner snapshots the task folder into each run's
# task-source, so an output directory inside it would be copied into itself
OUT="$(cd .. && pwd)/headlight-bulb-swap-outputs"
mkdir -p /tmp/rc-task-home/tmp "$OUT"
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$OUT":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --livestream 0 --viz none
python runtime/rescore.py --task task.py "$OUT"/headlight-bulb-swap/RUN/episode-01/trajectory.jsonl \
  --expect "$OUT"/headlight-bulb-swap/RUN/episode-01/result.json
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
