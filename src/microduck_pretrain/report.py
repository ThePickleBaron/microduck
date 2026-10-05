"""`md-report`: combine evaluation results and compute readiness gaps.

    uv run md-report                    # reads results/*.json
    uv run md-report --results results --out results/report

Labels like `c3_seed2` are grouped by the part before `_seed` (here `c3`), so
several seeds of one condition average into one row.

Readiness gap (per condition): how much worse the policy does on the held-out
site than on the nominal site.
    tracking gap %  = (held-out lin_vel_err - nominal lin_vel_err) / nominal x 100
    falls gap       = held-out falls/10 min - nominal falls/10 min
    stall gap (pts) = held-out stall % - nominal stall %
A deployment-ready policy has small gaps: it behaves on the unfamiliar site
the way it behaves at home.

Outputs <out>.md (tables), <out>.csv (one row per condition x scenario) and
<out>_gap.png (chart of the tracking gap per held-out scenario).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
METRICS = ("falls_per_10min", "lin_vel_err_mps", "yaw_rate_err_radps", "stall_fraction", "upright_fraction")


def group_of(label: str) -> str:
    return re.split(r"_seed\d+$", label)[0]


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def load(results_dir: Path):
    """-> data[group][site][scenario][metric] = mean over labels in the group."""
    raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for f in sorted(results_dir.glob("*.json")):
        try:
            rep = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if "scenarios" not in rep:
            continue
        g = group_of(rep["label"])
        for scen, summ in rep["scenarios"].items():
            for m in METRICS:
                raw[g][rep["site"]][scen][m].append(summ.get(m))
    data = {
        g: {s: {sc: {m: _mean(v) for m, v in ms.items()} for sc, ms in scs.items()} for s, scs in sites.items()}
        for g, sites in raw.items()
    }
    return data


def gaps(data, nominal="nominal", held_out="held_out"):
    """-> rows of (group, scenario, tracking_gap_pct, falls_gap, stall_gap_pts)."""
    rows = []
    for g, sites in sorted(data.items()):
        if nominal not in sites or held_out not in sites:
            continue
        base = next(iter(sites[nominal].values()))
        for scen, m in sites[held_out].items():
            b_err = base["lin_vel_err_mps"]
            gap = (m["lin_vel_err_mps"] - b_err) / b_err * 100 if b_err else float("nan")
            rows.append((
                g,
                scen,
                gap,
                m["falls_per_10min"] - base["falls_per_10min"],
                (m["stall_fraction"] - base["stall_fraction"]) * 100,
            ))
    return rows


def _fmt(x, spec):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def write_markdown(data, gap_rows, path: Path):
    lines = ["# Deployment-readiness results", ""]
    for site in sorted({s for sites in data.values() for s in sites}):
        lines += [f"## Site: {site}", "",
                  "| Condition | Scenario | Falls / 10 min | Speed error (m/s) | Yaw-rate error (rad/s) | Stalled (%) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for g in sorted(data):
            for scen, m in data[g].get(site, {}).items():
                lines.append(
                    f"| {g} | {scen} | {_fmt(m['falls_per_10min'], '.1f')} | {_fmt(m['lin_vel_err_mps'], '.3f')} "
                    f"| {_fmt(m['yaw_rate_err_radps'], '.3f')} | {_fmt(m['stall_fraction'] * 100, '.0f')} |"
                )
        lines.append("")
    if gap_rows:
        lines += ["## Readiness gap (held-out vs nominal)", "",
                  "Positive = worse on the held-out site.", "",
                  "| Condition | Scenario | Tracking gap (%) | Extra falls / 10 min | Extra stall (pts) |",
                  "| --- | --- | --- | --- | --- |"]
        for g, scen, tg, fg, sg in gap_rows:
            lines.append(f"| {g} | {scen} | {_fmt(tg, '+.0f')} | {_fmt(fg, '+.1f')} | {_fmt(sg, '+.0f')} |")
        lines += ["", "### Mean gap per condition", "",
                  "| Condition | Mean tracking gap (%) | Mean extra falls / 10 min |", "| --- | --- | --- |"]
        by_g = defaultdict(list)
        for row in gap_rows:
            by_g[row[0]].append(row)
        for g, rows in by_g.items():
            lines.append(f"| {g} | {_fmt(_mean([r[2] for r in rows]), '+.0f')} | {_fmt(_mean([r[3] for r in rows]), '+.1f')} |")
    path.write_text("\n".join(lines) + "\n")


def write_csv(data, path: Path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "site", "scenario", *METRICS])
        for g in sorted(data):
            for site, scs in data[g].items():
                for scen, m in scs.items():
                    w.writerow([g, site, scen, *(m.get(k) for k in METRICS)])


def write_chart(gap_rows, path: Path) -> bool:
    if not gap_rows:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted({r[0] for r in gap_rows})
    scens = list(dict.fromkeys(r[1] for r in gap_rows))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(scens) + 2), 4))
    for i, g in enumerate(groups):
        vals = {r[1]: r[2] for r in gap_rows if r[0] == g}
        xs = [j + (i - (len(groups) - 1) / 2) * width for j in range(len(scens))]
        ax.bar(xs, [vals.get(s, float("nan")) for s in scens], width, label=g)
    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xticks(range(len(scens)), scens, rotation=30, ha="right")
    ax.set_ylabel("Tracking gap vs nominal (%)")
    ax.set_title("Readiness gap: speed-tracking error on the held-out site")
    ax.legend(title="Condition")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="md-report", description="Combine md-eval results.")
    p.add_argument("--results", type=Path, default=REPO / "results")
    p.add_argument("--out", type=Path, default=None, help="output path prefix (default <results>/report)")
    p.add_argument("--nominal", default="nominal", help="name of the reference site")
    p.add_argument("--held-out", default="held_out", help="name of the held-out site")
    args = p.parse_args(argv)

    data = load(args.results)
    if not data:
        p.error(f"no md-eval results in {args.results}")
    out = args.out or args.results / "report"
    out.parent.mkdir(parents=True, exist_ok=True)
    gap_rows = gaps(data, args.nominal, args.held_out)
    write_markdown(data, gap_rows, out.with_suffix(".md"))
    write_csv(data, out.with_suffix(".csv"))
    charted = write_chart(gap_rows, out.parent / (out.name + "_gap.png"))
    print(f"Wrote {out.with_suffix('.md')} and {out.with_suffix('.csv')}" + (f" and {out.name}_gap.png" if charted else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
