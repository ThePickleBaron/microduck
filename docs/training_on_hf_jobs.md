# Optional: training on Hugging Face Jobs

The desktop's RTX 3090 is the default training machine. Hugging Face Jobs is a fallback for running seeds in parallel, or when the desktop is busy. It rents a cloud GPU by the hour and is billed to your Hugging Face account; check current pricing before launching many runs.

## One-time setup

```bash
uv run hf auth login        # needs a Hugging Face account (and billing for Jobs)
uv run wandb login          # optional; forwarded to the job if present
```

## Launch

Run from the **repo root**, so the job snapshot contains this project's condition tasks:

```bash
uv run md-train c4 --seeds 1 --logger wandb --hf-jobs --detach
uv run md-train c4 --seeds 2 --logger wandb --hf-jobs --detach --flavor a10g-large
```

`md-train` pins the budget and passes `--hf-jobs` (and `--flavor`, `--timeout`, `--namespace`, ...) through to the vendored submitter. The submitter:

1. Snapshots every tracked file of this repo (vendored stack included) to a private dataset.
2. Starts a container that runs `uv sync` against this repo's `uv.lock` and trains.
3. Uploads checkpoints to a private model repo as they appear, then auto-exports `exported/policy.onnx` at the end. It uses `scripts/export.py` and `scripts/hf/uploader.py` at this repo's root, which forward to the vendored implementations.

Commit (or at least `git add`) your changes before launching: untracked files are not uploaded.

## Get the result

Download the exported policy from the model repo printed at launch and save it as `policies/c4_seed2.onnx` (the name sets its label in the report). Then:

```bash
uv run python scripts/evaluate_all.py
```

Details of the submitter: `third_party/microduck_rl/scripts/hf/README.md`.
