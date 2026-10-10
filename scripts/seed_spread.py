"""Per-seed results, spread across seeds, and paired seed-by-seed comparisons.

md-report averages the seeds of each condition. This shows what the average
hides: how much the seeds disagree, and whether a difference between two
conditions holds on every seed. C5 and C3x seed N share a parent (C3 seed N),
so comparing them seed by seed removes the parent's own luck.

    uv run python scripts/seed_spread.py                    # reads results/
    uv run python scripts/seed_spread.py --results DIR --md out.md

The evaluator is deterministic: the three runs of one policy in one scenario
are identical, so all of the spread here comes from training seeds.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PAIRS = [("c5", "c3x", "site fine-tune vs control: same parent, same extra training"),
         ("c3x", "c3", "extra training vs none"),
         ("c5", "c3", "site fine-tune vs its parent")]
METRICS = [  # key, header, format
    ("nom", "Nominal speed err (m/s)", "{:.3f}"),
    ("comb", "Held-out combined speed err (m/s)", "{:.3f}"),
    ("mean_ho", "Held-out mean speed err (m/s)", "{:.3f}"),
    ("yaw", "Held-out mean yaw err (rad/s)", "{:.3f}"),
    ("bumped", "Falls/10 min when bumped", "{:.1f}"),
    ("falls", "Held-out mean falls/10 min", "{:.1f}"),
    ("stall", "Held-out mean stall (%)", "{:.1f}"),
]


def load(results: Path) -> dict[str, dict]:
    raw: dict[str, dict] = {}
    for f in results.glob("*__*.json"):
        label, site = f.stem.split("__", 1)
        raw.setdefault(label, {})[site] = json.loads(f.read_text())["scenarios"]
    out = {}
    for label, sites in raw.items():
        if "held_out" not in sites or "nominal" not in sites:
            continue
        h, n = sites["held_out"], sites["nominal"]["nominal"]
        out[label] = {
            "nom": n["lin_vel_err_mps"],
            "comb": h["site_combined"]["lin_vel_err_mps"],
            "mean_ho": st.mean(s["lin_vel_err_mps"] for s in h.values()),
            "yaw": st.mean(s["yaw_rate_err_radps"] for s in h.values()),
            "bumped": h["bumped"]["falls_per_10min"],
            "falls": st.mean(s["falls_per_10min"] for s in h.values()),
            "stall": 100 * st.mean(s["stall_fraction"] for s in h.values()),
        }
    return out


def split(label: str) -> tuple[str, int | None]:
    m = re.fullmatch(r"(.+)_seed(\d+)", label)
    return (m.group(1), int(m.group(2))) if m else (label, None)


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def build(data: dict[str, dict]) -> str:
    by_cond: dict[str, dict[int, dict]] = {}
    for label, v in data.items():
        cond, seed = split(label)
        by_cond.setdefault(cond, {})[seed] = v

    out = ["# Seed spread", "", "## Every policy", ""]
    rows = [[label] + [fmt.format(data[label][k]) for k, _, fmt in METRICS]
            for label in sorted(data, key=lambda s: (split(s)[0], split(s)[1] or 0))]
    out += [table(["Policy"] + [h for _, h, _ in METRICS], rows), ""]

    out += ["## Mean ± standard deviation across seeds", ""]
    rows = []
    for cond in sorted(by_cond):
        seeds = by_cond[cond]
        if len(seeds) < 2:
            continue
        row = [f"{cond} (n={len(seeds)})"]
        for k, _, fmt in METRICS:
            vals = [s[k] for s in seeds.values()]
            row.append(f"{fmt.format(st.mean(vals))} ± {fmt.format(st.stdev(vals))}")
        rows.append(row)
    out += [table(["Condition"] + [h for _, h, _ in METRICS], rows), ""]

    out += ["## Paired by seed (A − B; negative = A better)", ""]
    for a, b, why in PAIRS:
        if a not in by_cond or b not in by_cond:
            continue
        seeds = sorted(set(by_cond[a]) & set(by_cond[b]))
        if not seeds:
            continue
        rows = []
        for k, h, fmt in METRICS:
            d = [by_cond[a][s][k] - by_cond[b][s][k] for s in seeds]
            better = sum(x < 0 for x in d)
            worse = sum(x > 0 for x in d)
            rows.append([h] + [("+" if x > 0 else "") + fmt.format(x) for x in d]
                        + [f"{a} better on {better}/{len(d)}, worse on {worse}/{len(d)}"])
        out += [f"### {a} vs {b}: {why}", "",
                table(["Metric"] + [f"seed {s}" for s in seeds] + ["Verdict"], rows), ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--md", type=Path, help="also write the tables to this Markdown file")
    args = p.parse_args(argv)
    data = load(args.results)
    if not data:
        print(f"No paired held_out/nominal results in {args.results}")
        return 1
    text = build(data)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
