# tow-cart-hitch-pin

**Instruction:** "Hitch the cart behind to the cart in front."

Two steel tow carts stand nose to tail on their casters. The lead cart's rear has a clevis jaw: two
12 mm steel plates 45 mm apart with a 3/4 in drop pin seated through them. The trailing cart's 32 mm
drawbar sits on a vertical pivot, swung 22 degrees off line with its forged eye outside the jaw. To couple
the carts, one hand draws the pin up by its T-handle and holds it there, because it drops back under its
own weight. The other hand swings the drawbar until the eye sits in the jaw exactly under the pin. Then the
pin is let go and falls through the eye and the lower plate. A 19 mm pin only passes a 26 mm eye when
the eye is centred within about 3.5 mm. Off centre, the pin comes down on the eye and the carts stay
uncoupled.

## Why it matters

Tugger trains (tow carts behind an industrial tow tractor) move parts and goods in factories,
distribution centres and hospitals. Their carts are coupled and uncoupled by hand many times a shift.
O*NET lists "Hook tow trucks to trailer hitches and fasten attachments" as a task of Industrial Truck
and Tractor Operators (53-7051.00, about 780,000 US workers, BLS OEWS 2023). Hand laborers and freight,
stock and material movers (about 3 million) couple carts in the same way. The coupling needs two hands
working together: one keeps the pin up against gravity, while the other holds a long pivoting bar so that
its eye stays inside a 3.5 mm window. Then the pin is released without disturbing either.

## Real parts

| object | real part | size / mass | pose (m) |
|---|---|---|---|
| lead cart | steel tow/tugger cart, 50 mm box-section frame, tread-plate deck, 4 in swivel casters (cropped section) | 500 mm wide, deck 160 mm up | frame x 0.15-0.65, rear face y -0.110 |
| clevis jaw | weld-on clevis hitch plates, 1/2 in steel, for a 3/4 in pin | two plates 100 x 125 x 12 mm, 45 mm apart, 21 mm holes | pin axis (0.320, -0.040); lower plate top 80 mm up |
| hitch pin | 3/4 in zinc-plated T-handle drop hitch pin | 19 mm x 99 mm shank, 34 mm collar, 80 mm T-handle; 0.36 kg | seated in the jaw |
| drawbar | tow-cart tongue: 32 mm steel bar, forged flat eye (1 in hole), red grip sleeve | 500 mm pivot to eye; eye 25 mm thick, 72 mm across; 2.2 kg | pivot (0.320, 0.460), 102.5 mm up; swung -22 deg |
| trailing cart | the same cart, blue frame (cropped section) with the drawbar's pivot bracket | 500 mm wide | front face y 0.490 |

A real rig is two carts with the casters braked, or one clevis plate pair and one drawbar bolted to a
plywood floor panel at these positions. No liquids or consumables are used. Reset takes about 20 s:
lift the pin, swing the drawbar back to its floor mark, drop the pin.

## Scoring

The eye offset is the horizontal distance from the eye's centre to the pin's axis. It is computed from
measured poses.

- **0.25:** the eye is in the jaw under the pin (offset at most 8 mm, level within 6 mm).
- **0.5:** the pin has gone into the eye (offset at most 4.5 mm, pin end at least 3 mm below the eye's top face).
- **0.75:** the pin is fully down: its collar is within 3 mm of the upper plate.
- **1.0:** both hands have let go (each TCP at least 60 mm from the T-handle and from the drawbar), held for 1 s.

If the pin is dropped on an off-centre eye, it stands 62 mm proud and the score is 0.25 at most. A pin
seated with no eye in the jaw scores 0. A real trial checks the result from a photo, with the pin head
down on the plate and the eye round the pin. A ruler gives the pin's height.

## Simulation notes

- PhysX at 240 Hz, all rigid bodies. The pin slides on a frictionless prismatic joint along its own
  axis. This stands in for the 1 mm clearance in the plate holes, which keeps a real pin upright. The
  pin falls under gravity and stops only on contact with the upper plate (seated) or the eye.
- The drawbar turns on a damped vertical revolute joint (0.02 N m s/deg). A real tongue can also pitch
  and is usually spring-balanced level; here its pitch is locked at the jaw's mid-height.
- The carts are fixed cropped sections, standing in for braked carts. Their casters are visual only.
- Forces: the pin weighs 3.5 N, against two finger pads giving about 16 N of friction grip on the 15 mm
  crossbar. The pin slips about 13 mm in the grip but stays above the eye. Swinging the drawbar takes a
  few newtons at the sleeve.

## Results

- Scripted demo: score 1.0 after 338 control steps (16.9 s simulated) out of a 1,200-step limit.
  The pin was lifted 65 mm, the eye was placed 1.7 mm from the pin's axis, and the pin dropped
  fully through.
- Adversarial run: the same demo with the eye held 7.5 mm off centre. The pin came down on the eye and
  stopped 62 mm proud, scoring 0.25 at the end.
