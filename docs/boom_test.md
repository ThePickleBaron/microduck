# Boom test: a 3 ft carbon rod with a temperature sensor on the end

Added 2026-10-14. A side test, no training: can the walking policies carry a
long, thin carbon-fiber boom (up to 0.91 m / 3 ft) with a 10-20 g sensor on
the tip, clamped to the trunk? For scale, the duck is 0.74 kg and about
0.28 m tall; a 3 ft boom is over three times its height.

## The model

`evaluate.add_boom()` clamps a boom to the trunk top in the CPU evaluator. A
continuous beam is reduced to its first bending mode:

- A rigid rod plus a tip sphere carry the real masses: rod = density x area x
  length, sensor = 10-20 g.
- Two perpendicular hinges at the clamp have a rotational spring that
  reproduces the rod's first bending frequency.
  - Cantilever with a tip mass: k = 3EI/L^3, effective mass = m_tip + 0.236 m_rod.
  - Spring: k_theta = I_clamp (2 pi f1)^2.
- Each hinge has a damper set to a 2% damping ratio.
- On every reset the boom starts at its static sag, so it does not start ringing.
- No collisions: the rod passes through the head when forward and is not
  stopped by the floor.
- Mount points: front top for forward and up45, rear top for back and up,
  side top for side.

3 mm solid pultruded rod (E = 110 GPa, 1550 kg/m^3), 15 g sensor:

| Length (m) | Rod (g) | First mode (Hz) | Static sag, horizontal (mm) | Clamp moment, horizontal (N m) |
| --- | --- | --- | --- | --- |
| 0.30 | 3.3 | 8.8 | 3 | 0.049 |
| 0.45 | 4.9 | 4.8 | 11 | 0.077 |
| 0.60 | 6.6 | 3.0 | 28 | 0.108 |
| 0.75 | 8.2 | 2.2 | 57 | 0.141 |
| 0.91 | 10.0 | 1.6 | 106 | 0.178 |

A horizontal 3 ft boom shifts the robot's centre of mass about 2.4 cm
toward the boom. It also adds several times the robot's own pitch and yaw
inertia.

## Run it

```bash
uv run python scripts/boom_test.py                         # C1 + every exported c3x, c6v2, c6v3 policy, ~5 min
uv run python scripts/boom_test.py --conditions c1 c6v2    # a subset
```

Outputs `results/boom/boom.md`, `boom.csv`, `boom.png`. Runs are cached with the stress tests.

## First results (2026-10-14, vendor policy C1 only)

These ran in the cloud workspace, where only the vendor policy is available.

| Direction | Longest boom carried without failing | Falls per 10 min at 0.91 m |
| --- | --- | --- |
| up (vertical mast) | **0.91 m+** | 0 |
| side | 0.75 m | 180 |
| forward | 0.45 m | 470 |
| up45 | 0.45 m | 380 |
| back | 0.45 m | 690 |

- **Mount it vertically.** A vertical mast puts no static moment on the
  trunk, and the vendor policy carries the full 3 ft with zero falls. The
  sensor moves 15 mm RMS about its rest position.
- **It is the weight on a long lever, not the whip.** With the same masses
  made rigid, failures are similar: forward 520 vs 470, up45 280 vs 380,
  side 230 vs 180. Flexibility neither saves nor sinks it.
- **The sensor mass matters most.** Forward at 0.91 m: 10 g gives 140 falls
  per 10 min, 15 g gives 470, 20 g gives 700. Side: 0 / 180 / 340.
- **A stiffer, heavier rod is worse.** Forward at 0.91 m: 3 mm solid 470,
  4/2 mm tube 570, 5/3 mm tube 700. A 2 mm solid rod looks better (230), but it
  sags 475 mm, more than the 167 mm clamp height. A real one would drag its
  tip on the floor, which is not simulated, so that result is not valid.
- **Sensor motion** when it does not fall (RMS about rest): forward 81 mm,
  up45 42 mm, side 33 mm, up 15 mm at 0.91 m. That is tolerable for a
  temperature reading.

**Next:** run it on the desktop for C3x and C6v2. If a horizontal boom is
wanted, train a boom-aware fine-tune: randomize boom length, direction and
sensor mass in training, the same way C5 pre-trains for a site. That is the
project's question applied to an accessory fitted after delivery.

**Model limits:** one bending mode per axis (the rod's higher modes and
torsion are left out), no rod collisions, a perfectly stiff clamp, and
assumed damping.
