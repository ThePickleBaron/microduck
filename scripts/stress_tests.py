"""Three follow-up experiments that need no new training (CPU only).

    uv run python scripts/stress_tests.py budget      # readiness vs. training / fine-tuning iterations
    uv run python scripts/stress_tests.py envelope    # each policy's breaking point under each stress
    uv run python scripts/stress_tests.py careful     # every policy on a slow-and-careful site
    uv run python scripts/stress_tests.py all         # all three
    uv run python scripts/stress_tests.py all --quick # short runs, to check everything works

budget    Every saved snapshot (every 250 iterations) of the C3, C5 and C3x runs,
          evaluated on the nominal and held-out sites. C3 covers iterations
          0-3000; C5 and C3x continue from it to 4500, so the plot shows one
          story: learning, then fine-tuning on the site vs. simply training on.
envelope  Each exported policy (and C1) on the nominal floor while one stress is
          ramped up step by step: foot friction, payload, command delay, shoves,
          battery voltage, floor bumps. The "rating" is the last level before
          the policy fails (falls >= 10 per 10 min or stalled >= 50%).
careful   Each exported policy on sites/careful.toml: slow moves (0.08-0.15 m/s,
          0.3 rad/s) from a standstill, on the nominal floor and on the
          combined held-out site.

Outputs go to results/stress/: <test>.csv, <test>.md and <test>.png.
Every simulated run is cached (results/stress/cache/), so re-running only
does new work, and an interrupted run picks up where it stopped.

Options: --conditions c2 c3 ... (which conditions), --workers N (parallel
processes; default half the CPU cores, up to 12), --quick, --force (ignore cache).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import statistics as st
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from microduck_pretrain import evaluate as ev  # noqa: E402
from microduck_pretrain import runs as R  # noqa: E402

OUT = REPO / "results" / "stress"
POLICIES = REPO / "policies"
FAIL_FALLS_PER_10MIN = 10.0
FAIL_STALL = 0.5
ORDER = {"c1": 10, "c2": 20, "c3": 30, "c4": 40, "c5": 50, "c3x": 60, "c6": 65}

# Each axis: (label, unit, levels, how to apply a level to the nominal scenario)
AXES = {
    "friction": ("Foot friction", "", [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15],
                 lambda sc, x: replace(sc, foot_friction=x)),
    "payload": ("Payload", "g", [0, 50, 100, 150, 200, 250, 300, 400],
                lambda sc, x: replace(sc, payload_kg=x / 1000.0)),
    "delay": ("Command delay", "ms", [0, 20, 40, 60, 80, 100, 120],
              lambda sc, x: replace(sc, delay_steps=int(round(x / 20.0)))),
    "shove": ("Shove speed", "m/s", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
              lambda sc, x: replace(sc, push_speed_mps=x, push_interval_s=(2.0, 4.0))),
    "battery": ("Battery", "V", [7.4, 7.0, 6.6, 6.3, 6.0, 5.7, 5.4],
                lambda sc, x: replace(sc, vin=x, vin_drop_gain=0.25)),
    "bumps": ("Floor bumps", "mm", [0, 4, 8, 12, 16, 20, 25],
              lambda sc, x: replace(sc, rough_height_mm=float(x))),
}


# --------------------------------------------------------------------------
# Running simulations: cached, in parallel
# --------------------------------------------------------------------------
def _key(policy: Path, sc: ev.Scenario, seed: int) -> str:
    stat = policy.stat()
    blob = json.dumps({"p": str(policy.resolve()), "m": stat.st_mtime_ns, "s": stat.st_size,
                       "sc": asdict(sc), "seed": seed}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def _job(args):
    policy, sc, seed = args
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return asdict(ev.run_scenario(sc, Path(policy), seed))


class Runner:
    def __init__(self, workers: int, force: bool):
        self.workers, self.force = workers, force
        self.cache = OUT / "cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self._pool = None

    def run(self, jobs: list[tuple[Path, ev.Scenario, int]], what: str) -> list[dict]:
        keys = [_key(p, sc, s) for p, sc, s in jobs]
        results: list[dict | None] = [None] * len(jobs)
        todo = []
        for i, k in enumerate(keys):
            f = self.cache / f"{k}.json"
            if f.exists() and not self.force:
                results[i] = json.loads(f.read_text())
            else:
                todo.append(i)
        print(f"[{what}] {len(jobs)} runs, {len(jobs) - len(todo)} cached, {len(todo)} to simulate "
              f"on {self.workers} worker(s)", flush=True)
        if not todo:
            return results
        t0, done = time.perf_counter(), 0
        if self.workers <= 1:
            it = ((i, _job((str(jobs[i][0]), jobs[i][1], jobs[i][2]))) for i in todo)
        else:
            if self._pool is None:
                self._pool = cf.ProcessPoolExecutor(self.workers, mp_context=mp.get_context("spawn"))
            futs = {self._pool.submit(_job, (str(jobs[i][0]), jobs[i][1], jobs[i][2])): i for i in todo}
            it = ((futs[f], f.result()) for f in cf.as_completed(futs))
        for i, r in it:
            results[i] = r
            (self.cache / f"{keys[i]}.json").write_text(json.dumps(r))
            done += 1
            if done % max(1, len(todo) // 10) == 0 or done == len(todo):
                el = time.perf_counter() - t0
                print(f"  {done}/{len(todo)}  ({el / 60:.1f} min, about {el / done * (len(todo) - done) / 60:.0f} min left)",
                      flush=True)
        return results

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()


def summarize(runs: list[dict]) -> dict:
    total = sum(r["sim_seconds"] for r in runs)

    def mean(k):
        xs = [r[k] for r in runs if not math.isnan(r[k])]
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "falls_per_10min": sum(r["falls"] for r in runs) / (total / 600.0),
        "lin_vel_err_mps": mean("lin_vel_err_mps"),
        "yaw_rate_err_radps": mean("yaw_rate_err_radps"),
        "stall_fraction": mean("stall_fraction"),
        "upright_fraction": mean("upright_fraction"),
    }


# --------------------------------------------------------------------------
# Which policies
# --------------------------------------------------------------------------
def group_of(label: str) -> str:
    return re.split(r"_seed\d+$", label)[0]


def exported_policies(conditions: list[str] | None) -> list[tuple[str, Path]]:
    out = []
    vendor = POLICIES / "vendor" / "velstand.onnx"
    if vendor.exists() and (not conditions or "c1" in conditions):
        out.append(("c1", vendor))
    for f in sorted(POLICIES.glob("*.onnx")):
        g = group_of(f.stem)
        if g.startswith("steadycam") and not (conditions and g in conditions):
            continue  # side project: only when asked for by name
        if conditions and g not in conditions:
            continue
        out.append((f.stem, f))
    return sorted(out, key=lambda lp: (ORDER.get(group_of(lp[0]), 90), lp[0]))


def _scale(sc: ev.Scenario, quick: bool, seconds: float | None = None, seeds: int | None = None) -> ev.Scenario:
    sc = replace(sc, duration_s=seconds or sc.duration_s, seeds=seeds or sc.seeds)
    if quick:
        sc = replace(sc, duration_s=min(sc.duration_s, 20.0), seeds=1)
    return sc


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


COLORS = {"c1": "#7f7f7f", "c2": "#1f77b4", "c3": "#2ca02c", "c4": "#d62728", "c5": "#9467bd", "c3x": "#ff7f0e", "c6": "#17becf"}


# --------------------------------------------------------------------------
# 1. Budget curve
# --------------------------------------------------------------------------
def test_budget(runner: Runner, conditions: list[str] | None, quick: bool) -> None:
    keys = conditions or ["c3", "c5", "c3x", "c6"]
    found = [r for r in R.discover() if r.experiment.key in keys and r.seed is not None]
    if not found:
        print(f"[budget] no runs with checkpoints for {keys} under {R.LOGS}; skipped")
        return
    _, nominal = ev.load_site(REPO / "sites" / "nominal.toml")
    _, held = ev.load_site(REPO / "sites" / "held_out.toml")
    scen = [("nominal", _scale(s, quick, seeds=1)) for s in nominal] + [("held_out", _scale(s, quick, seeds=1)) for s in held]
    plan, jobs = [], []
    for run in found:
        for it in run.iterations:
            onnx = R.snapshot_onnx(run, it)
            for site, sc in scen:
                for seed in range(sc.seeds):
                    plan.append((run, it, site, sc.name))
                    jobs.append((onnx, sc, seed))
    res = runner.run(jobs, "budget")

    acc: dict = {}
    for (run, it, site, name), r in zip(plan, res):
        acc.setdefault((run.experiment.key, run.seed, run.experiment.start_iteration + it), {}).setdefault(
            (site, name), []).append(r)
    rows = []
    for (key, seed, it), by in sorted(acc.items(), key=lambda kv: (ORDER.get(kv[0][0], 90), kv[0][1], kv[0][2])):
        nom = summarize(by[("nominal", "nominal")])
        ho = {name: summarize(v) for (site, name), v in by.items() if site == "held_out"}
        rows.append({
            "condition": key, "seed": seed, "iteration": it,
            "nominal_speed_err": nom["lin_vel_err_mps"],
            "heldout_mean_speed_err": _mean([m["lin_vel_err_mps"] for m in ho.values()]),
            "heldout_combined_speed_err": ho.get("site_combined", {}).get("lin_vel_err_mps", float("nan")),
            "heldout_mean_falls_per_10min": _mean([m["falls_per_10min"] for m in ho.values()]),
            "heldout_bumped_falls_per_10min": ho.get("bumped", {}).get("falls_per_10min", float("nan")),
            "heldout_mean_stall": _mean([m["stall_fraction"] for m in ho.values()]),
        })
    _write_csv(OUT / "budget.csv", rows)

    plt = _plt()
    panels = [("nominal_speed_err", "Nominal speed error (m/s)"),
              ("heldout_mean_speed_err", "Held-out speed error, mean of 8 (m/s)"),
              ("heldout_mean_falls_per_10min", "Held-out falls / 10 min, mean of 8")]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (col, title) in zip(axs, panels):
        for key in sorted({r["condition"] for r in rows}, key=lambda k: ORDER.get(k, 90)):
            mine = [r for r in rows if r["condition"] == key]
            for seed in sorted({r["seed"] for r in mine}):
                s = [r for r in mine if r["seed"] == seed]
                ax.plot([r["iteration"] for r in s], [r[col] for r in s], color=COLORS.get(key), alpha=0.25, lw=1)
            its = sorted({r["iteration"] for r in mine})
            ax.plot(its, [_mean([r[col] for r in mine if r["iteration"] == i]) for i in its],
                    color=COLORS.get(key), lw=2.2, marker="o", ms=3, label=key.upper())
        ax.axvline(3000, color="0.6", ls="--", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training iteration (C5/C3x continue from C3 at 3000)")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend()
    fig.suptitle("Readiness vs. training budget (thin: each seed; thick: mean)")
    fig.tight_layout()
    fig.savefig(OUT / "budget.png", dpi=150)
    plt.close(fig)

    lines = ["# Readiness vs. training budget", "",
             "Every saved snapshot evaluated on the nominal and held-out sites (1 run per scenario, "
             f"{scen[0][1].duration_s:.0f} s). Mean over seeds. C5 and C3x start from C3 at iteration 3000.", "",
             "| Condition | Iteration | Seeds | Nominal speed err | Held-out speed err (mean) | Held-out combined | "
             "Held-out falls/10 min | Bumped falls/10 min | Held-out stall |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key in sorted({r["condition"] for r in rows}, key=lambda k: ORDER.get(k, 90)):
        mine = [r for r in rows if r["condition"] == key]
        for it in sorted({r["iteration"] for r in mine}):
            s = [r for r in mine if r["iteration"] == it]
            lines.append(
                f"| {key} | {it} | {len(s)} | {_mean([r['nominal_speed_err'] for r in s]):.3f} | "
                f"{_mean([r['heldout_mean_speed_err'] for r in s]):.3f} | {_mean([r['heldout_combined_speed_err'] for r in s]):.3f} | "
                f"{_mean([r['heldout_mean_falls_per_10min'] for r in s]):.1f} | {_mean([r['heldout_bumped_falls_per_10min'] for r in s]):.1f} | "
                f"{100 * _mean([r['heldout_mean_stall'] for r in s]):.0f}% |")
    (OUT / "budget.md").write_text("\n".join(lines) + "\n")
    print(f"[budget] wrote {OUT / 'budget.md'}, budget.csv, budget.png")


# --------------------------------------------------------------------------
# 2. Operating envelope
# --------------------------------------------------------------------------
def _fails(m: dict) -> bool:
    return m["falls_per_10min"] >= FAIL_FALLS_PER_10MIN or (m["stall_fraction"] or 0) >= FAIL_STALL


def test_envelope(runner: Runner, conditions: list[str] | None, quick: bool) -> None:
    pols = exported_policies(conditions)
    if not pols:
        print("[envelope] no exported policies; skipped")
        return
    _, nominal = ev.load_site(REPO / "sites" / "nominal.toml")
    base = _scale(nominal[0], quick, seconds=60.0, seeds=2)
    plan, jobs = [], []
    for label, path in pols:
        for axis, (_, _, levels, apply) in AXES.items():
            for x in levels:
                sc = replace(apply(base, x), name=f"{axis}_{x}")
                for seed in range(sc.seeds):
                    plan.append((label, axis, x))
                    jobs.append((path, sc, seed))
    res = runner.run(jobs, "envelope")
    acc: dict = {}
    for k, r in zip(plan, res):
        acc.setdefault(k, []).append(r)
    rows = []
    for (label, axis, x), rs in acc.items():
        m = summarize(rs)
        rows.append({"policy": label, "condition": group_of(label), "axis": axis, "level": x, **m, "fails": _fails(m)})
    _write_csv(OUT / "envelope.csv", rows)

    def rating(label, axis):
        levels = AXES[axis][2]
        last_ok = None
        for x in levels:
            r = next(r for r in rows if r["policy"] == label and r["axis"] == axis and r["level"] == x)
            if r["fails"]:
                return last_ok
            last_ok = x
        return f"{last_ok}+"  # never failed in the tested range

    lines = ["# Operating envelope", "",
             f"Nominal floor and commands, one stress ramped at a time ({base.duration_s:.0f} s x {base.seeds} runs per level). "
             f"Rating = last level before the policy fails (falls >= {FAIL_FALLS_PER_10MIN:.0f} per 10 min, or stalled "
             f">= {100 * FAIL_STALL:.0f}% of commanded time). \"+\" = never failed in the tested range; "
             "\"fails at start\" = already failing at the gentlest level.", "",
             "| Policy | " + " | ".join(f"{AXES[a][0]}{' (' + AXES[a][1] + ')' if AXES[a][1] else ''}" for a in AXES) + " |",
             "| --- | " + " | ".join("---" for _ in AXES) + " |"]
    for label, _ in pols:
        cells = []
        for a in AXES:
            v = rating(label, a)
            cells.append("fails at start" if v is None else str(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "Friction and battery ratings are the lowest value survived; the others are the highest."]
    (OUT / "envelope.md").write_text("\n".join(lines) + "\n")

    plt = _plt()
    fig, axs = plt.subplots(2, len(AXES), figsize=(3.2 * len(AXES), 6.5), squeeze=False)
    groups = sorted({r["condition"] for r in rows}, key=lambda k: ORDER.get(k, 90))
    for j, (axis, (title, unit, levels, _)) in enumerate(AXES.items()):
        for i, (col, ylab) in enumerate([("falls_per_10min", "Falls / 10 min"), ("lin_vel_err_mps", "Speed error (m/s)")]):
            ax = axs[i][j]
            for g in groups:
                ys = [_mean([r[col] for r in rows if r["condition"] == g and r["axis"] == axis and r["level"] == x])
                      for x in levels]
                ax.plot(levels, ys, marker="o", ms=3, color=COLORS.get(g), label=g.upper())
            if col == "falls_per_10min":
                ax.axhline(FAIL_FALLS_PER_10MIN, color="0.6", ls="--", lw=0.8)
                ax.set_title(title, fontsize=10)
            if axis in ("friction", "battery"):
                ax.invert_xaxis()
            ax.set_xlabel(unit or "coefficient")
            if j == 0:
                ax.set_ylabel(ylab)
            ax.spines[["top", "right"]].set_visible(False)
    axs[0][0].legend(fontsize=8)
    fig.suptitle("Operating envelope: one stress at a time, harsher to the right (mean over seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "envelope.png", dpi=150)
    plt.close(fig)
    print(f"[envelope] wrote {OUT / 'envelope.md'}, envelope.csv, envelope.png")


# --------------------------------------------------------------------------
# 3. Careful site
# --------------------------------------------------------------------------
def test_careful(runner: Runner, conditions: list[str] | None, quick: bool) -> None:
    pols = exported_policies(conditions)
    if not pols:
        print("[careful] no exported policies; skipped")
        return
    meta, scens = ev.load_site(REPO / "sites" / "careful.toml")
    scens = [_scale(s, quick) for s in scens]
    plan, jobs = [], []
    for label, path in pols:
        for sc in scens:
            for seed in range(sc.seeds):
                plan.append((label, sc.name))
                jobs.append((path, sc, seed))
    res = runner.run(jobs, "careful")
    acc: dict = {}
    for k, r in zip(plan, res):
        acc.setdefault(k, []).append(r)

    out_dir = OUT / "careful"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, path in pols:
        rep = {"label": label, "policy": str(path), "site": meta.get("name", "careful"),
               "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "scenarios": {}}
        nominal_file = REPO / "results" / f"{label}__nominal.json"
        nom_err = float("nan")
        if nominal_file.exists():
            nom_err = json.loads(nominal_file.read_text())["scenarios"]["nominal"]["lin_vel_err_mps"]
        for sc in scens:
            m = summarize(acc[(label, sc.name)])
            rep["scenarios"][sc.name] = {**m, "runs": acc[(label, sc.name)]}
            rows.append({"policy": label, "condition": group_of(label), "scenario": sc.name, **m,
                         "nominal_speed_err_at_0.3": nom_err})
        (out_dir / f"{label}__careful.json").write_text(json.dumps(rep, indent=2))
    # The tables cover every policy evaluated so far, not just this call's
    # (so `careful --conditions steadycam_v2` adds rows instead of replacing them).
    rows = []
    for f in sorted(out_dir.glob("*__careful.json"), key=lambda f: (ORDER.get(group_of(f.name.split("__")[0]), 90), f.name)):
        rep = json.loads(f.read_text())
        label = rep["label"]
        nominal_file = REPO / "results" / f"{label}__nominal.json"
        nom_err = float("nan")
        if nominal_file.exists():
            nom_err = json.loads(nominal_file.read_text())["scenarios"]["nominal"]["lin_vel_err_mps"]
        for name, m in rep["scenarios"].items():
            m = {k: v for k, v in m.items() if k != "runs"}
            rows.append({"policy": label, "condition": group_of(label), "scenario": name, **m,
                         "nominal_speed_err_at_0.3": nom_err})
    _write_csv(OUT / "careful.csv", rows)

    lines = ["# Slow-and-careful site", "",
             "Slow moves from a standstill (0.08-0.15 m/s, 0.3 rad/s); see sites/careful.toml. Stalled = share of "
             "commanded-motion time with under 20% of the commanded motion. Per policy, then mean per condition.", "",
             "| Policy | Scenario | Stalled | Speed error (m/s) | Yaw error (rad/s) | Falls / 10 min | Nominal speed err at 0.3 m/s |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['policy']} | {r['scenario']} | {100 * (r['stall_fraction'] or 0):.0f}% | "
                     f"{r['lin_vel_err_mps']:.3f} | {r['yaw_rate_err_radps']:.3f} | {r['falls_per_10min']:.1f} | "
                     f"{r['nominal_speed_err_at_0.3']:.3f} |")
    lines += ["", "| Condition | Scenario | Stalled (mean, min-max over its policies) | Speed error | Falls / 10 min |",
              "| --- | --- | --- | --- | --- |"]
    groups = sorted({r["condition"] for r in rows}, key=lambda k: ORDER.get(k, 90))
    for g in groups:
        for sc in scens:
            s = [r for r in rows if r["condition"] == g and r["scenario"] == sc.name]
            stalls = [100 * (r["stall_fraction"] or 0) for r in s]
            lines.append(f"| {g} | {sc.name} | {st.mean(stalls):.0f}% ({min(stalls):.0f}-{max(stalls):.0f}) | "
                         f"{_mean([r['lin_vel_err_mps'] for r in s]):.3f} | {_mean([r['falls_per_10min'] for r in s]):.1f} |")
    (OUT / "careful.md").write_text("\n".join(lines) + "\n")

    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / max(1, len(scens))
    for ax, (col, title, scale) in zip(axs, [("stall_fraction", "Stalled (% of commanded time)", 100),
                                              ("lin_vel_err_mps", "Speed error (m/s)", 1)]):
        for i, sc in enumerate(scens):
            xs = [k + (i - (len(scens) - 1) / 2) * width for k in range(len(groups))]
            ax.bar(xs, [scale * _mean([r[col] or 0 for r in rows if r["condition"] == g and r["scenario"] == sc.name])
                        for g in groups], width, label=sc.name)
            for k, g in enumerate(groups):  # individual seeds as dots
                ys = [scale * (r[col] or 0) for r in rows if r["condition"] == g and r["scenario"] == sc.name]
                ax.scatter([xs[k]] * len(ys), ys, s=10, color="k", zorder=3)
        ax.set_xticks(range(len(groups)), [g.upper() for g in groups])
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].legend(fontsize=8)
    fig.suptitle("Slow-and-careful site (bars: mean per condition; dots: each policy)")
    fig.tight_layout()
    fig.savefig(OUT / "careful.png", dpi=150)
    plt.close(fig)
    print(f"[careful] wrote {OUT / 'careful.md'}, careful.csv, careful.png")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("test", choices=["budget", "envelope", "careful", "all"])
    p.add_argument("--conditions", nargs="+", help="condition keys, e.g. c3 c5 c3x (default: budget c3 c5 c3x; others all)")
    p.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    p.add_argument("--quick", action="store_true", help="20 s x 1 run per scenario, to check everything works")
    p.add_argument("--force", action="store_true", help="ignore cached runs")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    conds = [c.lower() for c in args.conditions] if args.conditions else None
    runner = Runner(args.workers, args.force)
    t0 = time.perf_counter()
    try:
        if args.test in ("budget", "all"):
            test_budget(runner, conds, args.quick)
        if args.test in ("envelope", "all"):
            test_envelope(runner, conds, args.quick)
        if args.test in ("careful", "all"):
            test_careful(runner, conds, args.quick)
    finally:
        runner.close()
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min. Results in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
