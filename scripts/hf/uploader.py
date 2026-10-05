"""Checkpoint uploader for Hugging Face Jobs (wrapper).

The vendored HF Jobs bootstrap runs `scripts/hf/uploader.py` relative to the
repo it snapshots. When you train with `--hf-jobs` from this repo's root, that
is this file, so it forwards to the vendored implementation.
"""

import runpy
from pathlib import Path

VENDORED = Path(__file__).resolve().parents[2] / "third_party" / "microduck_rl" / "scripts" / "hf" / "uploader.py"

if __name__ == "__main__":
    runpy.run_path(str(VENDORED), run_name="__main__")
