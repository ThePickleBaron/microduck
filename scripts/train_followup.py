"""Train the follow-up conditions C5 and C3x by warm-starting from finished C3 runs.

    uv run python scripts/train_followup.py c5 --seeds 1 2 3     # site fine-tune
    uv run python scripts/train_followup.py c3x --seeds 1 2 3    # control: same budget, C3's world
    uv run python scripts/train_followup.py c6 --seeds 1 2 3     # careful walk: C3x + slow-command rewards
    uv run python scripts/train_followup.py c5 --dry-run         # show checkpoints and commands
    uv run python scripts/train_followup.py --export             # finished runs -> policies/c5_seedN.onnx, c3x_seedN.onnx

Seed N starts from C3 seed N (logs/rsl_rl/c3_standard/*_c3_seedN, last
checkpoint), so each C5/C3x seed has a matching C3 parent. Both conditions get
the same fine-tune budget: 4096 environments x 1500 iterations. The exported
policies drop into the normal evaluation (scripts/evaluate_all.py) and report
as groups "c5" and "c3x". Details: docs/c5_site_finetune.md.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_steadycam import CKPT_RE, LOGS, REPO, stage_checkpoint  # noqa: E402

FOLLOWUPS = {
    # key: (task id, experiment folder) - mirrors src/microduck_pretrain/followups.py
    "c5": ("Pretrain-C5-SiteFinetune-Rough-Backlash-MicroDuck", "c5_site_finetune"),
    "c3x": ("Pretrain-C3X-Standard-Extended-Flat-MicroDuck", "c3x_extended"),
    "c6": ("Pretrain-C6-Careful-Flat-MicroDuck", "c6_careful"),
}


def last_checkpoint(run: Path) -> Path | None:
    ckpts = [(int(m.group(1)), p) for p in run.glob("model_*.pt") if (m := CKPT_RE.search(p.name))]
    return max(ckpts)[1] if ckpts else None


def c3_parent(seed: int) -> Path:
    """Last checkpoint of the newest C3 run for this seed."""
    root = LOGS / "c3_standard"
    for run in sorted(root.glob(f"*_c3_seed{seed}"), reverse=True) if root.exists() else []:
        if (ck := last_checkpoint(run)) is not None:
            return ck
    raise SystemExit(f"No finished C3 seed {seed} run under {root}. Train it first: uv run md-train c3 --seeds {seed}")


def export_runs(force: bool) -> int:
    out_dir = REPO / "policies"
    out_dir.mkdir(exist_ok=True)
    done = 0
    for key, (task, experiment) in FOLLOWUPS.items():
        root = LOGS / experiment
        for run in sorted(root.glob(f"*_{key}_seed*")) if root.exists() else []:
            ck = last_checkpoint(run)
            m = re.search(rf"({key}_seed\d+)$", run.name)
            if ck is None or m is None:
                continue
            dest = out_dir / f"{m.group(1)}.onnx"
            if dest.exists() and not force:
                print(f"skip   {dest.relative_to(REPO)} (exists; --force to overwrite)")
                continue
            print(f"export {ck.relative_to(REPO)} -> {dest.relative_to(REPO)}")
            rc = subprocess.call([
                sys.executable, str(REPO / "scripts" / "export.py"), task,
                "--checkpoint-file", str(ck), "--onnx-file", str(dest), "--num-envs", "1",
            ], cwd=REPO)
            if rc != 0:
                print(f"  export failed ({rc})", file=sys.stderr)
                return rc
            done += 1
    if not done:
        print("Nothing new to export.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("condition", nargs="?", choices=sorted(FOLLOWUPS), help="c5 (site fine-tune), c3x (control) or c6 (careful walk)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    p.add_argument("--from", dest="source", type=Path, help="override the C3 parent checkpoint (all seeds)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--export", action="store_true", help="export finished C5/C3x runs to policies/ and exit")
    p.add_argument("--force", action="store_true", help="with --export: overwrite existing files")
    args, extra = p.parse_known_args(argv)
    if args.export:
        return export_runs(args.force)
    if not args.condition:
        p.error("choose c5 or c3x (or --export)")

    task, experiment = FOLLOWUPS[args.condition]
    env = dict(os.environ, MICRODUCK_WARM_START="1")
    for seed in args.seeds:
        src = args.source or c3_parent(seed)
        tag = "warmstart_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{src.parent.name}_{src.stem}")
        print(f"[{args.condition}] seed {seed}: warm start from {src}")
        if not args.dry_run:
            stage_checkpoint(src, tag, experiment)
        cmd = [
            "train", task,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max-iterations", str(args.iterations),
            "--agent.seed", str(seed),
            "--agent.run-name", f"{args.condition}_seed{seed}",
            "--agent.logger", args.logger,
            "--agent.resume", "True",
            "--agent.load-run", re.escape(tag),
            "--agent.load-checkpoint", re.escape(src.name),
            *extra,
        ]
        print("$ MICRODUCK_WARM_START=1 uv run " + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, env=env, cwd=REPO)
        if rc != 0:
            print(f"[{args.condition}] seed {seed} exited with code {rc}; stopping.", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
