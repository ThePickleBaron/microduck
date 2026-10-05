# Hardware test protocol (after the robot arrives)

The semester deliverables rest on simulation. This test is the follow-up that answers the project's real question: **did the simulation-to-simulation gap predict the simulation-to-real gap?**

## 1. Time to first custom skill (the deployment-readiness metric)

Start a timer when the box is opened. Stop it when a policy trained in this repo is walking on the robot.

1. Unbox, charge, power on, connect the robot to Wi-Fi (Pollen's instructions).
2. Publish a pre-trained policy from this repo (needs `uv run hf auth login`):
   ```bash
   cd third_party/microduck_rl
   uv run publish --onnx ../../policies/c4_seed1.onnx --repo <hf-user>/microduck-c4-seed1 \
       --kind perpetual --slot walk
   ```
3. On the robot:
   ```bash
   sudo robotctl policy load walk <hf-user>/microduck-c4-seed1
   ```
4. Stop the timer at the first commanded steps. Log every snag and how long it took to fix.

Compare against the vendor default, which is ready the moment it powers on, and against the time it would take to tune a policy *after* delivery (record your own estimate before testing).

## 2. Same metrics, real floors

Run every condition's best seed (pick from the simulation results, decided **before** testing) plus the vendor default on:

| Surface | Mirrors simulated scenario |
| --- | --- |
| Hard floor (tile or sealed concrete) | nominal / slippery_floor |
| Low-pile carpet | rough_floor (compliance) |
| Gentle ramp or a doorway threshold | rough_floor |
| 100-150 g payload taped to the back | heavy_payload |
| Low battery (near the cutoff) | low_battery |

For each run, give the same command schedule as `sites/*.toml` with the gamepad (or a script through the runtime), for 2 minutes.

- **Falls**: count them; set the robot back up and continue.
- **Speed error**: tape a 2 m course; time the forward segments; achieved speed = 2 m / time. Turning: count rotations over a timed segment.
- **Stall**: note any segment where it stands in place when told to move.
- **Video every run** (phone on a tripod) for later review and the presentation.

Use `results/hardware_runs.csv` with columns:
`date, condition, label, surface, payload_g, battery_v, duration_s, falls, fwd_time_2m_s, stalled_segments, notes`.

## 3. Analysis

For each condition: real-world gap vs. simulated gap. A method an organization can trust is one where the cheap simulated test **ranks the conditions the same way** the expensive real one does, even if the absolute numbers differ.

## Safety

- Test on the floor, away from stairs and table edges; keep a hand near the robot on ramps.
- Payloads must be secured so they can't fall off and jam a joint.
- Stop a run if a servo gets hot or the robot repeatedly falls on its head.
