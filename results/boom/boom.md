# Boom test: 3 ft carbon rod with a temperature sensor

Flat floor, nominal commands, 60 s x 2 runs per setting, values pooled over a condition's seeds. Fails = 10+ falls per 10 min. Tip motion = how far the sensor moves about its static sag while the robot is upright (RMS / max).

## The boom (3 mm solid rod, 15 g sensor)

| Length (m) | Rod mass (g) | First bending mode (Hz) | Static tip sag, horizontal (mm) | Moment at the clamp, horizontal (N m) |
| --- | --- | --- | --- | --- |
| 0.30 | 3.3 | 8.8 | 3 | 0.049 |
| 0.45 | 4.9 | 4.8 | 11 | 0.077 |
| 0.60 | 6.6 | 3.0 | 28 | 0.108 |
| 0.75 | 8.2 | 2.2 | 57 | 0.141 |
| 0.91 | 10.0 | 1.6 | 106 | 0.178 |

## Longest boom each policy carries without failing (falls per 10 min in brackets at 0.91 m)

| Condition | forward | back | side | up | up45 |
| --- | --- | --- | --- | --- | --- |
| c1 | 0.45 m (470) | 0.45 m (690) | 0.75 m (180) | 0.91 m+ (0) | 0.45 m (380) |

## Falls per 10 min by length

**forward**

| Condition | no boom | 0.30 m | 0.45 m | 0.60 m | 0.75 m | 0.91 m |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | 0 | 0 | 0 | 40 | 130 | 470 |

**back**

| Condition | no boom | 0.30 m | 0.45 m | 0.60 m | 0.75 m | 0.91 m |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | 0 | 0 | 0 | 230 | 500 | 690 |

**side**

| Condition | no boom | 0.30 m | 0.45 m | 0.60 m | 0.75 m | 0.91 m |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | 0 | 0 | 0 | 0 | 0 | 180 |

**up**

| Condition | no boom | 0.30 m | 0.45 m | 0.60 m | 0.75 m | 0.91 m |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | 0 | 0 | 0 | 0 | 0 | 0 |

**up45**

| Condition | no boom | 0.30 m | 0.45 m | 0.60 m | 0.75 m | 0.91 m |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | 0 | 0 | 0 | 10 | 100 | 380 |

## Is it the weight or the whip? (0.91 m, flexible 3 mm rod vs the same masses made rigid)

| Condition | Direction | Falls: flexible | Falls: rigid | Speed err: flexible | Speed err: rigid | Tip RMS (mm): flexible |
| --- | --- | --- | --- | --- | --- | --- |
| c1 | forward | 470 | 520 | 0.177 | 0.268 | 81 |
| c1 | back | 690 | 640 | nan | nan | nan |
| c1 | side | 180 | 230 | 0.095 | 0.104 | 33 |
| c1 | up | 0 | 0 | 0.147 | 0.163 | 15 |
| c1 | up45 | 380 | 280 | 0.290 | 0.256 | 42 |

## Sensor mass and rod choice (0.91 m): falls per 10 min

| Condition | Direction | 10 g | 15 g | 20 g |
| --- | --- | --- | --- | --- |
| c1 | forward | 140 | 470 | 700 |
| c1 | back | 450 | 690 | 810 |
| c1 | side | 0 | 180 | 340 |
| c1 | up | 0 | 0 | 0 |
| c1 | up45 | 90 | 380 | 480 |

| Condition | Direction | 2mm solid | 3mm solid | 4/2mm tube | 5/3mm tube |
| --- | --- | --- | --- | --- | --- |
| c1 | forward | 230 | 470 | 570 | 700 |
| c1 | side | 0 | 180 | 320 | 340 |

Rod properties at 0.91 m with a 15 g sensor (static sag from beam theory, small-deflection; a horizontal boom is clamped about 167 mm above the floor):

| Rod | Mass (g) | First mode (Hz) | Static sag (mm) | Note |
| --- | --- | --- | --- | --- |
| 2mm solid | 4.4 | 0.7 | 475 | sag exceeds the clamp height: a real horizontal boom would rest its tip on the floor (not simulated), so treat its numbers as invalid |
| 3mm solid | 10.0 | 1.6 | 106 |  |
| 4/2mm tube | 13.3 | 2.7 | 38 |  |
| 5/3mm tube | 17.7 | 3.9 | 18 |  |

Model limits: one bending mode per axis (the rod's higher modes and torsion are left out), no collisions between the rod and the robot or floor, ideal clamp, damping ratio 2%.
