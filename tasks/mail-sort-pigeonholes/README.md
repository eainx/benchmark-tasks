# mail-sort-pigeonholes

**Instruction:** "Sort the letters into their room boxes."

A wooden desktop mail sorter with six labelled pigeonholes (rooms 101-302) stands on the desk. Beside the
right arm, an open-ended wire step organizer holds four incoming #10 envelopes upright, each
addressed to a room in large type (202, 102, 302, 202 from front to back: two letters for room 202; 101, 201 and 301 get none).
Each letter is lifted a few millimetres by its top edge and slid out sideways along its section, its
room is matched to a label, it is turned flat and slid all the way into that pigeonhole.

## Why this job

Sorting incoming mail into labelled pigeonholes is routine for office clerks (about 2.5 million in the
US), mail clerks, receptionists and hotel front-desk staff. It differs from bin-sorting tasks in two
ways that the score measures:

- **Reading to choose the target.** The target is not a colour or object class but a room number read
  off a printed address and matched to one of six printed labels; the left-hand column's three
  compartments (101, 201, 301) get no letter and room 202 gets two, so filing by position or one letter
  per compartment fails; filing a letter in any other room caps the score.
- **A thin flat sheet into a narrow opening.** The 4 mm letter has to be drawn out of a stepped
  organizer by its edge, turned from standing to flat, and slid fully into a 95 mm high, 255 mm wide
  opening (7 mm clearance a side) until no corner sticks out; a letter left sticking out or dropped on
  the desk does not count.

Drawing a letter straight up out of its section does not move the 0.45 kg organizer (checked in the
simulation), so the task is one-handed; knocking the organizer more than 10 mm still caps the score.

## Real parts

| object | real part | size / mass | pose |
|---|---|---|---|
| mail sorter | desktop literature organizer, 6 compartments (2 x 3), 15 mm oak-veneered boards, 6 mm hardboard back | 555 W x 345 H x 246 D mm; compartments 255 x 95 x 240 mm; fixed | centre x 0.42, front at y 0.00, openings toward -y |
| slot labels | printed card in an aluminium label holder | 70 x 13.5 mm, 9.5 mm digits | under each opening: 101 102 / 201 202 / 301 302 (left column first) |
| organizer | open-ended wire-mesh step file organizer, black powder-coated steel | 300 x 135 mm, 4 sections of 33.5 mm stepped 10 mm, dividers 40-95 mm, ends open above a 6 mm rail; 0.45 kg | centre (0.245, -0.45), sections across x, low front toward +x |
| envelopes (4) | #10 business envelope, 24 lb off-white wove, three folded sheets inside | 241 x 105 x 4 mm, 28 g | standing in the organizer, long side along y, address toward +x: ROOM 202, 102, 302, 202 front (+x) to back |

Names, companies and addresses on the envelopes are fictitious.

## Scoring

An envelope is *filed* in a pigeonhole when all eight corners of its body lie inside that compartment:
between its side boards, above its floor, below the shelf over it and at least 2 mm behind the opening.

- 0.2 for each envelope that has been filed in the pigeonhole labelled with its own room (latched;
  up to 0.8).
- 1.0: all four filed correctly at once and at rest (under 10 mm/s and 0.3 rad/s), with both hands'
  pads (each jaw's pad at its measured opening, from the measured joints) at least 30 mm from every
  envelope's body and 30 mm out of the sorter's openings, held for 1 s.
- Capped at 0.4: any envelope filed in, or resting inside, a pigeonhole of another room; any envelope
  dropped on the desk outside the organizer and the sorter (centre within 30 mm of the desk top); the
  organizer knocked more than 10 mm or tilted more than 10 degrees from where it stood, at any time.

All conditions are measurable in a real trial with a tape measure and a photo.

## Simulation notes

- PhysX 240 Hz. Each envelope is one rigid 241 x 105 x 4 mm body: the paper does not bend, and its
  contents are not modelled. Its print, stamp, postmark and barcode are drawn meshes.
- The organizer is one rigid body: hidden thin-box colliders (stepped floor, dividers, ends) under the
  drawn wire. The sorter is fixed.
- Printed text is flat ink geometry rasterised from DejaVu Sans at build time (matplotlib).
- No task rules: drawing out, turning, inserting and pushing in are all contact physics.

## Scripted demo

One hand (the right); the left stays at rest, since drawing a letter out does not move the organizer
(checked in simulation). For each letter, front first, the right hand grips it by its top edge straight
down (jaws closing across it), lifts it 12 mm clear of the end rail and slides it out sideways along its
section past the organizer's open end; turns the grip about the vertical so the letter lies along x;
carries it out to x 0.45, clear of the organizer, and tips it flat there (tool pointing +y, the letter
ahead of the fingers); carries it along the sorter's front to its pigeonhole, slides it 70 mm in (held
24 mm over the floor), opens so it drops onto the compartment floor and backs straight out; then turns
the grip back the same way for the next letter.

Every waypoint was checked by inverse kinematics before recording (all within 5 mm and 5 degrees). The
layout follows from what the YAM can reach and where its forearm goes: a straight-down grip reaches only
about 0.2 m above the desk, so letters are slid out sideways instead of lifted over each other; the flat
grip's forearm reaches back and down behind the hand, so letters are tipped flat at x 0.45 and the
organizer stands at x 0.245 with 10 mm steps, keeping the waiting letters below the forearm's path.

## Reproduction

`prepare` exports the runtime into `runtime/`. From this task directory, with the
`yam-inspect-isaac:6.1.0` image (or `python3 runtime/build_image.py --image yam-inspect-isaac:6.1.0`):

```bash
OUT="$PWD/../mail-sort-pigeonholes-outputs"     # outside the task folder: the runner snapshots /task
mkdir -p /tmp/rc-task-home/tmp "$OUT"
docker run --rm --gpus device=0 --network host --shm-size 8g \
  --user "$(id -u):$(id -g)" -v "$PWD":/task:ro -v "$OUT":/outputs:rw \
  -v /tmp/rc-task-home:/tmp/task-home:rw -e HOME=/tmp/task-home -e TMPDIR=/tmp/task-home/tmp \
  -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ACCEPT_EULA=YES -e ACCEPT_EULA=Y \
  --workdir /outputs --entrypoint /isaac-sim/python.sh yam-inspect-isaac:6.1.0 \
  /task/run.py --policy scripted --episodes 1 --seed 0 --livestream 0 --viz none
python runtime/rescore.py --task task.py "$OUT"/TASK/RUN/episode-01/trajectory.jsonl \
  --expect "$OUT"/TASK/RUN/episode-01/result.json
```

Seed 0; fixed poses, no randomisation; the scene settles for 30 ticks after reset before control
starts. Control 20 Hz, physics 240 Hz, run limit 2,400 steps (120 s). Each episode writes four 20 fps
camera videos, setup/final PNGs and snapshots, the measured state trajectory and result.json. The
scorer reads only measured object poses and velocities and the robot's measured joints and finger
travel (pad positions by forward kinematics at the runner's arm bases).

## Files

`task.py` (scene, scripted demo), `mail.py` (geometry and printed text), `scoring.py` (scorer),
`simkit.py` (compound parts, visual meshes, joints), `meshlib.py` (mesh builders), `run.py`. No
third-party assets.
