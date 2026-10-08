"""Train the steady-cam walking policy by warm-starting from a finished C3 run.

    uv run python scripts/train_steadycam.py                 # newest C3 checkpoint, seed 1
    uv run python scripts/train_steadycam.py --seeds 1 2     # two seeds, one after another
    uv run python scripts/train_steadycam.py --from logs/rsl_rl/c3_standard/<run>/model_2999.pt
    uv run python scripts/train_steadycam.py --dry-run       # print what would run
    uv run python scripts/train_steadycam.py --export        # finished runs -> policies/steadycam_seedN.onnx

Why a warm start: walking already exists in C3, so this run only has to learn
camera steadiness on top of it (~1500 iterations, about 45 min on an RTX 3090,
instead of a 3000-iteration run from scratch). MICRODUCK_WARM_START=1 keeps the
C3 weights, normalizer and optimizer but restarts the step counter, so the
camera-cost curriculum starts at zero. See src/microduck_pretrain/steadycam.py.

Not part of the C1-C4 experiment; logs go to logs/rsl_rl/steadycam_walk/.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "logs" / "rsl_rl"
SOURCE_EXPERIMENT = "c3_standard"
TASK_ID = "Steadycam-Walk-Flat-MicroDuck"
EXPERIMENT = "steadycam_walk"
CKPT_RE = re.compile(r"model_(\d+)\.pt$")


def newest_c3_checkpoint() -> Path:
    """Highest-iteration checkpoint of the most recent C3 run that has one."""
    root = LOGS / SOURCE_EXPERIMENT
    runs = sorted((d for d in root.glob("*") if d.is_dir()), reverse=True) if root.exists() else []
    for run in runs:
        ckpts = [(int(m.group(1)), p) for p in run.glob("model_*.pt") if (m := CKPT_RE.search(p.name))]
        if ckpts:
            return max(ckpts)[1]
    raise SystemExit(
        f"No C3 checkpoint under {root}. Train C3 first (uv run md-train c3) "
        "or pass --from <path to model_XXXX.pt>."
    )


def stage_checkpoint(src: Path, tag: str) -> tuple[str, str]:
    """mjlab only resumes from inside the task's own log folder, so copy the
    source checkpoint into logs/rsl_rl/steadycam_walk/<tag>/ and point at it."""
    dest_dir = LOGS / EXPERIMENT / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    (dest_dir / "SOURCE.txt").write_text(f"{src.resolve()}\n")
    return tag, src.name


def export_runs(force: bool = False) -> int:
    """Export the newest checkpoint of each steady-cam run to policies/, through
    the vendored exporter (bakes in the observation normalizer)."""
    root = LOGS / EXPERIMENT
    out_dir = REPO / "policies"
    out_dir.mkdir(exist_ok=True)
    done = 0
    for run in sorted(root.glob("*_steadycam_seed*")) if root.exists() else []:
        ckpts = [(int(m.group(1)), p) for p in run.glob("model_*.pt") if (m := CKPT_RE.search(p.name))]
        if not ckpts:
            continue
        label = re.search(r"(steadycam_seed\d+)$", run.name).group(1)
        dest = out_dir / f"{label}.onnx"
        if dest.exists() and not force:
            print(f"skip   {dest.relative_to(REPO)} (exists; --force to overwrite)")
            continue
        ckpt = max(ckpts)[1]
        print(f"export {ckpt.relative_to(REPO)} -> {dest.relative_to(REPO)}")
        rc = subprocess.call([
            sys.executable, str(REPO / "scripts" / "export.py"), TASK_ID,
            "--checkpoint-file", str(ckpt), "--onnx-file", str(dest), "--num-envs", "1",
        ], cwd=REPO)
        if rc != 0:
            print(f"  export failed ({rc})", file=sys.stderr)
            return rc
        done += 1
    if not done:
        print(f"Nothing new to export under {root}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--from", dest="source", type=Path, help="C3 checkpoint (model_XXXX.pt) to start from")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export", action="store_true", help="export finished runs to policies/ and exit")
    p.add_argument("--force", action="store_true", help="with --export: overwrite existing files")
    args, extra = p.parse_known_args(argv)
    if args.export:
        return export_runs(args.force)

    src = args.source or newest_c3_checkpoint()
    if not src.exists():
        p.error(f"checkpoint not found: {src}")
    tag = "warmstart_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{src.parent.name}_{src.stem}")
    print(f"[steadycam] warm start from {src}")

    env = dict(os.environ, MICRODUCK_WARM_START="1")
    for seed in args.seeds:
        if not args.dry_run:
            load_run, load_ckpt = stage_checkpoint(src, tag)
        else:
            load_run, load_ckpt = tag, src.name
        cmd = [
            "train", TASK_ID,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max-iterations", str(args.iterations),
            "--agent.seed", str(seed),
            "--agent.run-name", f"steadycam_seed{seed}",
            "--agent.logger", args.logger,
            "--agent.resume", "True",
            "--agent.load-run", re.escape(load_run),
            "--agent.load-checkpoint", re.escape(load_ckpt),
            *extra,
        ]
        print("$ MICRODUCK_WARM_START=1 uv run " + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, env=env, cwd=REPO)
        if rc != 0:
            print(f"[steadycam] seed {seed} exited with code {rc}; stopping.", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
