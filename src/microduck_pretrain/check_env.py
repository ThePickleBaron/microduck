"""`md-check`: verify this machine is ready for the project.

    uv run md-check

Checks Python, the simulation stack, the experiment tasks, the GPU (needed
only for training), a display (needed only for the 3D viewer), the vendor
baseline policy and optional logins. Training-only items are warnings on a
machine without an NVIDIA GPU, so the laptop passes as an evaluation box.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def _row(status: str, item: str, detail: str = "") -> str:
    line = f"[{status}] {item}"
    return line + (f" - {detail}" if detail else "")


def main() -> int:
    rows: list[tuple[str, str, str]] = []
    add = lambda s, i, d="": rows.append((s, i, d))  # noqa: E731

    # Platform
    v = sys.version_info
    add(OK if (v.major, v.minor) == (3, 12) else FAIL, "Python 3.12", platform.python_version())
    rel = platform.release().lower()
    is_wsl = "microsoft" in rel or "wsl" in rel
    add(OK, "Platform", f"{platform.system()} {platform.machine()}" + (" (WSL2)" if is_wsl else ""))
    if platform.system() != "Linux":
        add(FAIL, "Linux required", "run inside WSL2 Ubuntu on Windows; see docs/windows_wsl_setup.md")

    # Core imports
    for mod in ("mujoco", "onnxruntime", "bam", "mjlab", "torch", "warp"):
        try:
            m = __import__(mod)
            add(OK, f"import {mod}", getattr(m, "__version__", ""))
        except Exception as e:  # noqa: BLE001
            add(FAIL, f"import {mod}", f"{type(e).__name__}: {e}")

    # Experiment tasks registered through the mjlab plugin
    try:
        from mjlab.tasks.registry import list_tasks

        from microduck_pretrain.conditions import TRAINABLE

        have = set(list_tasks())
        missing = [c.task_id for c in TRAINABLE if c.task_id not in have]
        add(FAIL if missing else OK, "Experiment tasks registered", ", ".join(missing) or f"{len(TRAINABLE)} conditions")
    except Exception as e:  # noqa: BLE001
        add(FAIL, "Experiment tasks registered", f"{type(e).__name__}: {e}")

    # GPU (training only)
    gpu_ok = False
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 2**30
            add(OK, "CUDA GPU for training", f"{name}, {mem:.0f} GB")
            gpu_ok = True
        else:
            add(WARN, "CUDA GPU for training", "none found: evaluate here, train on the desktop or with --hf-jobs")
    except Exception as e:  # noqa: BLE001
        add(WARN, "CUDA GPU for training", str(e))
    if gpu_ok:
        try:
            import warp as wp

            wp.init()
            add(OK if wp.is_cuda_available() else FAIL, "Warp sees CUDA", str(wp.get_cuda_device_count()) + " device(s)")
        except Exception as e:  # noqa: BLE001
            add(FAIL, "Warp sees CUDA", str(e))
    if shutil.which("nvidia-smi"):
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        add(OK, "NVIDIA driver", out)

    # Display for the MuJoCo viewer (infer_policy.py); headless tools don't need it
    disp = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    add(OK if disp else WARN, "Display for 3D viewer", disp or "none: headless tools work, the viewer won't")

    # Vendor baseline policy
    vp = REPO / "policies" / "vendor" / "velstand.onnx"
    add(OK if vp.exists() else WARN, "Vendor baseline policy (C1)",
        str(vp.relative_to(REPO)) if vp.exists() else "run: uv run python scripts/get_vendor_policy.py")

    # Optional logins
    try:
        from huggingface_hub import get_token

        add(OK if get_token() else WARN, "Hugging Face login (optional: --hf-jobs, publishing)",
            "found" if get_token() else "run: uv run hf auth login")
    except Exception:  # noqa: BLE001
        pass
    netrc = Path.home() / ".netrc"
    wandb = bool(os.environ.get("WANDB_API_KEY")) or (netrc.exists() and "api.wandb.ai" in netrc.read_text())
    add(OK if wandb else WARN, "Weights & Biases login (optional)", "found" if wandb else "md-train defaults to tensorboard")

    for s, i, d in rows:
        print(_row(s, i, d))
    failed = sum(1 for s, *_ in rows if s == FAIL)
    print()
    print("All required checks passed." if not failed else f"{failed} required check(s) failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
