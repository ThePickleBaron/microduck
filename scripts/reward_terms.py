"""Where did the reward come from? Each reward term over training, from the
TensorBoard logs every run already wrote (no simulation, CPU, seconds).

    uv run python scripts/reward_terms.py                    # C3 runs (default)
    uv run python scripts/reward_terms.py --conditions c3 c6v2

For each condition: every `Episode_Reward/<term>` curve (weighted: the term's
contribution per second of episode, as the trainer logs it), averaged over
seeds and over +-25 iterations around each reported iteration. If
results/stress/budget.csv has that condition, each term's curve is lined up
with the measured nominal speed error at the same snapshots, and the
correlation from the most accurate snapshot on is reported: which terms kept rising while the
walking got less accurate.

Outputs: results/reward_terms/<condition>.md, .csv and .png.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parent
OUT = REPO / "results" / "reward_terms"
PREFIX = "Episode_Reward/"
EXTRA_TAGS = ["Train/mean_reward", "Episode_Termination/fell_over", "Train/mean_episode_length"]
WINDOW = 25


def read_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    """All scalar curves under a run folder: {tag: {step: value}}."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: dict[str, dict[int, float]] = {}
    for ev_file in sorted(run_dir.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(ev_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            if tag.startswith(PREFIX) or tag in EXTRA_TAGS:
                d = out.setdefault(tag, {})
                for e in acc.Scalars(tag):
                    d[int(e.step)] = float(e.value)
    return out


def around(curve: dict[int, float], it: int, window: int = WINDOW) -> float:
    xs = [v for s, v in curve.items() if abs(s - it) <= window and math.isfinite(v)]
    return st.mean(xs) if xs else float("nan")


def pearson(a: list[float], b: list[float]) -> float:
    pts = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 3:
        return float("nan")
    xs, ys = zip(*pts)
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in pts) / (sx * sy) if sx and sy else float("nan")


def budget_curve(key: str) -> dict[int, float]:
    """Mean nominal speed error per iteration (training-local) from the budget test."""
    f = REPO / "results" / "stress" / "budget.csv"
    if not f.exists():
        return {}
    from microduck_pretrain import runs as R

    start = next((e.start_iteration for e in R.EXPERIMENTS.values() if e.key == key), 0)
    by: dict[int, list[float]] = {}
    with f.open() as fh:
        for row in csv.DictReader(fh):
            if row["condition"] == key:
                v = float(row["nominal_speed_err"])
                if math.isfinite(v):
                    by.setdefault(int(row["iteration"]) - start, []).append(v)
    return {it: st.mean(v) for it, v in by.items()}


def analyze(key: str, runs: list) -> bool:
    curves = [read_scalars(r.path) for r in runs]
    curves = [c for c in curves if c]
    if not curves:
        print(f"[{key}] no TensorBoard event files in its run folders; skipped")
        return False
    tags = sorted({t for c in curves for t in c})
    terms = [t for t in tags if t.startswith(PREFIX)]
    last = min(max(c[t]) for c in curves for t in c if c[t])
    budget = budget_curve(key)
    marks = sorted(i for i in budget if 0 <= i <= last + WINDOW) or \
        sorted({0, 250, 500, 750, 1000, 1500, 2000, 2500, last} & set(range(last + 1)))

    def at(tag: str, it: int) -> float:
        vals = [around(c[tag], it) for c in curves if tag in c]
        vals = [v for v in vals if math.isfinite(v)]
        return st.mean(vals) if vals else float("nan")

    table = {t: [at(t, it) for it in marks] for t in tags}
    speed = [budget.get(it, float("nan")) for it in marks]
    best_i = min(range(len(marks)), key=lambda i: speed[i] if math.isfinite(speed[i]) else math.inf) \
        if any(math.isfinite(s) for s in speed) else None

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{key}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tag", *marks])
        for t in tags:
            w.writerow([t, *table[t]])
        if budget:
            w.writerow(["measured nominal speed err (m/s)", *speed])

    name = lambda t: t[len(PREFIX):] if t.startswith(PREFIX) else t  # noqa: E731
    L = [f"# Reward terms over training: {key.upper()}", "",
         f"{len(curves)} run(s); each value is the term's weighted contribution per second of episode "
         f"(as the trainer logs it), mean over runs and over +-{WINDOW} iterations. Positive = earned, "
         "negative = penalty.", ""]
    if best_i is not None:
        b, e = marks[best_i], marks[-1]
        L += [f"Measured nominal speed error is lowest at iteration {b} ({speed[best_i]:.3f} m/s) and "
              f"{speed[-1]:.3f} m/s at {e}. Columns: value at {b}, at {e}, the change, and the correlation "
              f"of the term with speed error over the snapshots from {b} on (positive r = the term rose "
              "as accuracy got worse; needs 3+ snapshots).", "",
              f"| Term | At {b} | At {e} | Change | r with speed error, {b}-{e} |", "| --- | --- | --- | --- | --- |"]
        rows = []
        for t in terms:
            v0, v1 = table[t][best_i], table[t][-1]
            rows.append((v1 - v0, t, v0, v1, pearson(table[t][best_i:], speed[best_i:])))
        for d, t, v0, v1, r in sorted(rows, key=lambda x: -abs(x[0]) if math.isfinite(x[0]) else 0):
            L.append(f"| {name(t)} | {v0:+.3f} | {v1:+.3f} | {d:+.3f} | {r:+.2f} |")
        for t in EXTRA_TAGS:
            if t in table:
                L.append(f"| *{t}* | {table[t][best_i]:.3f} | {table[t][-1]:.3f} | "
                         f"{table[t][-1] - table[t][best_i]:+.3f} | {pearson(table[t][best_i:], speed[best_i:]):+.2f} |")
    else:
        L += ["No budget-test results for this condition (run `stress_tests.py budget`), so terms are "
              "shown without the measured speed error.", "",
              "| Term | " + " | ".join(str(m) for m in marks) + " |",
              "| --- | " + " | ".join("---" for _ in marks) + " |"]
        for t in terms + [t for t in EXTRA_TAGS if t in table]:
            L.append(f"| {name(t)} | " + " | ".join(f"{v:+.3f}" for v in table[t]) + " |")
    (OUT / f"{key}.md").write_text("\n".join(L) + "\n")
    _plot(key, marks, table, terms, speed)
    print("\n".join(L))
    print(f"\n[{key}] wrote {OUT}/{key}.md, .csv, .png")
    return True


def _plot(key, marks, table, terms, speed) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(terms)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), squeeze=False)
    for ax, t in zip(axs.flat, terms):
        ax.plot(marks, table[t], marker="o", ms=3, color="#1f77b4")
        ax.set_title(t[len(PREFIX):], fontsize=9)
        ax.tick_params(labelsize=7)
        if any(math.isfinite(s) for s in speed):
            ax2 = ax.twinx()
            ax2.plot(marks, speed, color="#d62728", lw=1, ls="--")
            ax2.tick_params(labelsize=7, colors="#d62728")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"{key.upper()}: reward terms (blue) vs measured nominal speed error (red dashed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{key}.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    from microduck_pretrain import runs as R

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conditions", nargs="+", default=["c3"])
    args = p.parse_args(argv)
    found = R.discover()
    ok = False
    for key in args.conditions:
        runs = [r for r in found if r.experiment.key == key and r.seed is not None]
        if not runs:
            print(f"[{key}] no runs under {R.LOGS}; skipped")
            continue
        ok |= analyze(key, runs)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
