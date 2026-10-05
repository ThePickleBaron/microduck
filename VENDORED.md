# Vendored code

`third_party/microduck_rl/` is a plain copy (not a submodule) of
[pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl)
at commit **1f2e4afcf33fbc75172b893c82fb66da9e3fc7ea** (branch `develop`, copied 2026-10-04).

It is vendored rather than a submodule because Hugging Face Jobs snapshots the
repo with `git ls-files`, which does not descend into submodules.

## Local patches

1. `pyproject.toml`: the BAM dependency is pinned to the commit its own
   `uv.lock` resolved (`rev = "62bd8ce..."`) instead of the moving branch
   `mjlab_frictionloss`. A newer BAM commit removed the `vin_drop_gain_range`
   actuator option the vendored code uses. Applied automatically by
   `scripts/sync_vendor_constraints.py`.

Nothing else is modified. This project's code lives in `src/microduck_pretrain/`
and plugs in through mjlab's task entry point.

## Updating

```bash
bash scripts/update_vendor.sh <commit-or-branch>   # re-copies, re-pins, re-locks
uv sync --extra dev && uv run pytest -q
```

Then update the commit above, and re-check that `tests/test_tasks.py` still
passes: it catches new upstream randomization toggles the idealized condition
must switch off.
