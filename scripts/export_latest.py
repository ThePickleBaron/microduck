"""Export the latest checkpoint of every finished training run to policies/.

    uv run python scripts/export_latest.py            # all runs under logs/rsl_rl
    uv run python scripts/export_latest.py --dry-run  # show what would be exported

Run folders look like logs/rsl_rl/c3_standard/2026-10-12_14-03-22_c3_seed1/.
Each becomes policies/c3_seed1.onnx (the run name md-train gave it), exported
through the vendored exporter so the observation normalizer is baked in.
Existing files are skipped unless --force.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from microduck_pretrain.conditions import TRAINABLE

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_TO_TASK = {
    "c2_idealized": next(c.task_id for c in TRAINABLE if c.key == "c2"),
    "c3_standard": next(c.task_id for c in TRAINABLE if c.key == "c3"),
    "c4_site": next(c.task_id for c in TRAINABLE if c.key == "c4"),
}


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    return ckpts[-1] if ckpts else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", type=Path, default=REPO / "logs" / "rsl_rl")
    p.add_argument("--out", type=Path, default=REPO / "policies")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    found = 0
    for exp, task in EXPERIMENT_TO_TASK.items():
        for run_dir in sorted((args.logs / exp).glob("*")):
            ckpt = latest_checkpoint(run_dir) if run_dir.is_dir() else None
            if ckpt is None:
                continue
            found += 1
            m = re.search(r"(c\d_seed\d+)$", run_dir.name)
            label = m.group(1) if m else run_dir.name
            dest = args.out / f"{label}.onnx"
            if dest.exists() and not args.force:
                print(f"skip   {_rel(dest)} (exists)")
                continue
            print(f"export {_rel(ckpt)} -> {_rel(dest)}")
            if args.dry_run:
                continue
            rc = subprocess.call([
                sys.executable, str(REPO / "scripts" / "export.py"), task,
                "--checkpoint-file", str(ckpt), "--onnx-file", str(dest), "--num-envs", "1",
            ])
            if rc != 0:
                print(f"  export failed ({rc})", file=sys.stderr)
    if not found:
        print(f"No checkpoints under {args.logs}. Train first: uv run md-train c3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
