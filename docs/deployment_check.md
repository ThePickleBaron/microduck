# Deployment check: are the policies installable as they are?

Added 2026-10-10. `scripts/check_policies.py` checks every exported policy
against what the robot needs, using the shipped walk policy
(`policies/vendor/velstand.onnx`) as the reference.

Why now: Pollen's roadmap (milestone M8, "model channel") will let a robot
install a policy published outside the shipped set, and states that every
model must pass a shape check (`obs[1,61] -> actions[1,14]`) before it goes
live, with third-party loading off by default. "Deployment-ready on
delivery" should include "installable on delivery".

```bash
uv run python scripts/check_policies.py                # every policies/*.onnx -> results/policy_check.md
uv run python scripts/check_policies.py --manifest     # also policies/manifest.json (Pollen's schema 2)
```

| Group | What it checks |
| --- | --- |
| Contract | Input `obs` float[1,61], output `actions` float[1,14]; ONNX IR version and opset no newer than the reference; only operators the reference uses; the same metadata, with identical joint order, default pose, gains, observation and command layout and action scale (the runtime reads these from the file) |
| Numerics | Finite actions on zero, typical and extreme observations; action magnitudes within 3x the reference on the same inputs |
| Cost | Single-thread inference time and file size within 2x the reference (which already runs at 50 Hz on the robot's Radxa Zero 3W) |
| Behavior | In the CPU simulator, 20 s each: stands still on a zero command from rest (drift under 0.05 m/s, no falls); walks 0.3 m/s (no falls, error under 0.2 m/s, stalled under 50%) |

Exit code 1 if anything fails.

What it found on day one (cloud check):

- Policies exported with `scripts/export.py` (and so `export_latest.py`,
  `train_followup.py --export`, `train_steadycam.py --export`) have exactly
  the reference's packaging: IR 8, opset 18, the same four operators, the same
  metadata. Only `run_path` differs, which is informational.
- The snapshot ONNX files that `runs.snapshot_onnx` builds for the timelapse
  videos have the right shapes and math but are **not deployable**: no
  metadata, IR 9, and MatMul+Add instead of Gemm. Use them for videos and
  stress tests only; deploy only through `scripts/export.py`.
- Both shipped vendor policies pass every check, including behavior.

The manifest follows the layout of Pollen's shipped
`policies/vendor/manifest.json` (schema 2), adding training provenance (task,
repo, commit, run, checkpoint, export time) and the check result. The robot
does not read it yet; it is the record a model channel will need.
