"""Evaluate every policy in policies/ on both sites, then build the report.

    uv run python scripts/evaluate_all.py
    uv run python scripts/evaluate_all.py --quick     # 1 seed x 40 s, for a fast look

Labels come from file names (policies/c3_seed1.onnx -> c3_seed1). The vendor
baseline policies/vendor/velstand.onnx is evaluated as "c1". Results that
already exist are skipped unless --force, so this is safe to re-run as new
seeds finish training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from microduck_pretrain import evaluate, report

REPO = Path(__file__).resolve().parents[1]
SITES = ("nominal", "held_out")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policies", type=Path, default=REPO / "policies")
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--quick", action="store_true", help="1 seed x 40 s per scenario")
    p.add_argument("--force", action="store_true", help="re-run existing results")
    args = p.parse_args()

    jobs = []
    vendor = args.policies / "vendor" / "velstand.onnx"
    if vendor.exists():
        jobs.append(("c1", vendor))
    jobs += [(f.stem, f) for f in sorted(args.policies.glob("*.onnx"))]
    if not jobs:
        print("No policies found. Run scripts/get_vendor_policy.py and scripts/export_latest.py first.")
        return 1

    for label, policy in jobs:
        for site in SITES:
            out = args.results / f"{label}__{site}.json"
            if out.exists() and not args.force:
                print(f"skip {out.name}")
                continue
            argv = ["--policy", str(policy), "--label", label,
                    "--site", str(REPO / "sites" / f"{site}.toml"), "--out", str(args.results)]
            if args.quick:
                argv += ["--seeds", "1", "--duration", "40"]
            evaluate.main(argv)

    return report.main(["--results", str(args.results)])


if __name__ == "__main__":
    raise SystemExit(main())
