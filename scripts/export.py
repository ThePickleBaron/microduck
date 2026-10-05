"""Export a trained checkpoint to a deployable ONNX policy.

Thin wrapper over the vendored exporter (observation normalizer baked in):

    uv run python scripts/export.py Pretrain-C3-Standard-Flat-MicroDuck \
        --checkpoint-file logs/rsl_rl/c3_standard/<run>/model_3000.pt \
        --onnx-file policies/c3_seed1.onnx --num-envs 1

It must live at this path: the Hugging Face Jobs bootstrap calls
`scripts/export.py` from the repo root to auto-export finished runs.
"""

from mjlab_microduck.export import main

if __name__ == "__main__":
    main()
