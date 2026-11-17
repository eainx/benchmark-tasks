# steering-hand-over-hand

**Instruction:** "Turn the steering wheel all the way to the left and hold it there."

A 380 mm steering wheel sits on a raked column on a bench test stand, with an instrument binnacle behind it. A 25 mm band of yellow marking tape on the rim marks 12 o'clock when the wheel is straight, and a white index line on top of the column shroud is the fixed reference it is read against. A torsion spring in the column pulls it back to straight ahead, standing in for a moving vehicle's self-centring. Full lock is 1.25 turns (450 deg) left. The wheel is turned hand over hand: one hand holds the rim against the spring while the other takes a new grip further round, and both hands are never off the rim together, or the wheel spins back. At the lock it is held for 2 s.

## Demo

Each hand pinches the 28 mm rim across its section, coming in from the driver's side along the column axis. The left hand works the lower-left arc of the rim and the right hand the right side. Per quarter turn:

1. Both hands carry the rim 90 deg round together.
2. The left hand lets go and takes a new grip 90 deg back, while the right hand holds the wheel against the spring.
3. The right hand does the same while the left holds.

Five strokes reach full lock. The hands carry the rim 93, 95, 97, 100 and 100 deg (the wheel lags the hands a little more as the spring winds up; the last stroke presses it onto its 450 deg stop). The grips sit 45 deg and 30 deg off the quarter points, so a spoke (at 3, 6 and 9 o'clock of the wheel) is never where a hand takes hold. At the lock both hands hold for 3 s, then let go and move clear, and the spring returns the wheel. The hands end the last stroke crossed over (the left at the lower right of the rim, the right at the upper left), as hand-over-hand steering does; the arms pass one above the other and never touch. To leave, each hand opens its jaws fully, backs straight off the rim 80 mm, goes round in front of the wheel 100 mm toward the robot (the left under it, the right over it, so the arms pass one above the other) to where it first took hold, and then out to its own side of the wheel, to a waiting pose about 200 mm from the rim and 150 mm nearer the robot than the wheel centre.

## Scoring

The turn is followed from the wheel's pose, unwrapped across full turns, positive to the left.

| Score | Condition (milestones latch) |
|---|---|
| 0.1 each | Quarter turn reached: 85, 175, 265, 355 deg |
| 0.6 | Full lock reached: 440 deg or more |
| 0.8 | Held at full lock for 2 s |
| 1.0 | Held, then let go: both hands 50 mm or more clear of the rim, and the returned wheel at rest (under 1 cm/s and 0.2 rad/s) for 1 s |

While the wheel is being turned, a hand counts as on the rim when both of its finger pads lie within 6 mm of the rim's surface and its jaws are closed under 16 mm a side. The pads are measured from the joints at `station.BASES`. Two things count as a let-go:

- both hands off the rim for more than 0.15 s;
- the wheel slipping back more than 30 deg below the furthest it has been turned.

After a let-go, no hold credit is given (the score is capped at 0.6). These checks only observe; they trigger nothing in the simulation. A wheel torn off its column (hub moved more than 5 mm) is `dropped` and scores 0. The scorer returns `clear` and `dropped`.

**Real trial:** the rim's tape band, read against the shroud's index line (or a protractor ring on the column), gives the turn; the hold, the hands and any slip-back are seen on video.

## Real parts

| Part | Description |
|---|---|
| Wheel | 380 mm (15 in) three-spoke steering wheel, 28 mm leather-wrapped rim, about 1.4 kg; any 380 mm aftermarket wheel on a 6-bolt hub adapter. |
| Centre mark | A 25 mm band of yellow vinyl marking tape wrapped round the rim at 12 o'clock with the wheel straight, and a white paint line on top of the column shroud, as wheel-alignment shops mark a wheel. |
| Column and stand | A steering column (e.g. a salvaged van column or a universal tilt column) on a steel stand bolted to the bench, raked 25 deg (wheel plane 65 deg from horizontal), wheel centre 200 mm over the bench. |
| Centring spring | A torsion (clock) spring between column and shaft, set to about 1.6 N m at 450 deg. Check with a torque wrench on the hub, or a spring scale on the rim: about 8.4 N at the 190 mm rim radius at full lock. |
| Stops | Mechanical stops at +/-450 deg, as in the steering gear. |
| Instrument binnacle | A dark moulded two-dial cluster housing behind the wheel; recognition only. |

## Simulation

- **Wheel:** one rigid body on a free revolute axle along the column axis, with damping and bearing friction. It moves only by finger contacts on its rim. Axle damping 0.06 N·m·s/rad, bearing friction 0.02 N·m; past ±450° the end stop adds 3.0 N·m/rad with 0.3 N·m·s/rad damping. No other damping is added.
- **Spring and stops:** PhysX revolute limits and drive targets both wrap at +/-180 deg. A test with limits set to -355/+180 stopped the wheel at -180. Neither can therefore hold a 1.25-turn spring or a 450 deg stop. The centring spring (K = 1.6 N m / 450 deg) and stiff end stops past 450 deg are applied by the task's `Dynamics` each control step, as a torque about the column axis computed from the measured, unwrapped turn.
- **Colliders:** the rim collider is 32 capsules round the 28 mm torus, under a drawn mesh. The hub and spokes also collide. The column, stand and binnacle are one fixed body. The tape band and the shroud's index line are drawn only: they add no collider and no mass.
- **Simplification:** a moving vehicle's self-centring (from caster and tyre forces) is stood in for by the torsion spring.
- **Forces:** at full lock the spring needs about 9 N tangential at the rim. That is within one pinching hand's friction (two pads at about 18 N each).

## Reproduction

Run from the folder that contains this task, writing outputs outside it:

```
python3 steering-hand-over-hand/run.py --output ../steering-hand-over-hand-outputs
```
