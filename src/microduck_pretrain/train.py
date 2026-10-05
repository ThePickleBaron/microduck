"""`md-train`: launch one experiment condition with a fixed, comparable budget.

Every condition must get the same number of environments, PPO iterations and
seeds, or differences between conditions could just reflect training effort.
This wrapper pins those and forwards everything else to mjlab's `train`.

    uv run md-train c3                       # one run, seed 1, local GPU
    uv run md-train c4 --seeds 1 2 3         # three seeds, one after another
    uv run md-train c2 --dry-run             # print the command only
    uv run md-train c4 --hf-jobs --detach    # train on Hugging Face Jobs instead

Defaults: 4096 environments x 3000 iterations. On an RTX 3090 a usable gait
appears in roughly 1-2 hours (vendored README); budget a full run per seed.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from microduck_pretrain.conditions import TRAINABLE, by_key

DEFAULT_ENVS = 4096
DEFAULT_ITERATIONS = 3000


def build_command(
    key: str,
    *,
    seed: int,
    num_envs: int,
    iterations: int,
    logger: str,
    extra: list[str],
) -> list[str]:
    cond = by_key(key)
    if cond.task_id is None:
        raise SystemExit(
            f"{cond.key} ({cond.name}) is not trained here. "
            "Download it with: uv run python scripts/get_vendor_policy.py"
        )
    return [
        "train",
        cond.task_id,
        "--env.scene.num-envs", str(num_envs),
        "--agent.max-iterations", str(iterations),
        "--agent.seed", str(seed),
        "--agent.run-name", f"{cond.key}_seed{seed}",
        "--agent.logger", logger,
        *extra,
    ]


def main(argv: list[str] | None = None) -> int:
    keys = [c.key for c in TRAINABLE]
    p = argparse.ArgumentParser(
        prog="md-train",
        description="Train one experiment condition with a fixed budget.",
        epilog="Any unrecognised flags (e.g. --hf-jobs --detach) are passed to `train`.",
    )
    p.add_argument("condition", choices=keys, help="condition key (see `md-conditions`)")
    p.add_argument("--seeds", type=int, nargs="+", default=[1], help="random seeds, run in order")
    p.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
    p.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    p.add_argument(
        "--logger",
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="tensorboard needs no account; use wandb once you have logged in",
    )
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    args, extra = p.parse_known_args(argv)

    for seed in args.seeds:
        cmd = build_command(
            args.condition,
            seed=seed,
            num_envs=args.num_envs,
            iterations=args.iterations,
            logger=args.logger,
            extra=extra,
        )
        print("$ uv run " + shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[md-train] seed {seed} exited with code {rc}; stopping.", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
