"""Download the official Microduck policies used as the C1 vendor-default baseline.

    uv run python scripts/get_vendor_policy.py

Fetches from the public Hugging Face repo pollen-robotics/microduck-policies
into policies/vendor/:
    velstand.onnx       the robot's default walk policy (manifest slot "walk")
    alpha_walking.onnx  the earlier walking-only policy (a second reference)
    manifest.json       the shipped policy set, for provenance
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "pollen-robotics/microduck-policies"
FILES = ("velstand.onnx", "alpha_walking.onnx", "manifest.json")
DEST = Path(__file__).resolve().parents[1] / "policies" / "vendor"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = hf_hub_download(REPO_ID, name, local_dir=DEST)
        print(f"  {Path(path).relative_to(DEST.parents[1])}")
    print(f"Vendor policies saved to {DEST.relative_to(DEST.parents[1])}/")


if __name__ == "__main__":
    main()
