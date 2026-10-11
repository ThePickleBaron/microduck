"""Two-weight response surface on C6v2: walking-speed vs turning emphasis.

    uv run python scripts/reward_surface.py design              # the 18-run table
    uv run python scripts/reward_surface.py train --dry-run     # the commands
    uv run python scripts/reward_surface.py train               # 18 short runs on the GPU (~2.7 h)
    uv run python scripts/reward_surface.py evaluate            # CPU, cached
    uv run python scripts/reward_surface.py analyze             # fit, contours, operating point
    uv run python scripts/reward_surface.py all

Design (src/microduck_pretrain/reward_surface.py): 3x3 full factorial in
log2 multipliers of the linear-tracking pair and the turning-tracking pair
(0.5x / 1x / 2x), 2 replicates (C6v2 seed 1 and seed 2 as parents), random
run order, 250 iterations per run. Scored like the screen (nominal, 8
held-out scenarios, careful site; 1 run x 120 s each).

Analysis: per response, a full quadratic in the coded factors
(a = linear, b = turning; -1/0/+1) fitted by least squares, with standard
errors, R^2 and a lack-of-fit test against the replicate (pure) error. The
operating point minimizes J = max(held-out speed error / 0.033,
held-out yaw error / 0.085) on the fitted surfaces inside the tested region,
subject to predicted careful-site stall <= 5% and held-out falls <= 1 per
10 min: 1.0 would mean C6v2's walking accuracy and C3x's turning accuracy at
once. Outputs: results/reward_surface/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import reward_screen as SCR  # noqa: E402  (shared response definitions and helpers)
from microduck_pretrain import reward_surface as RF  # noqa: E402

REPO = HERE.parent
OUT = REPO / "results" / "reward_surface"
RESPONSES = SCR.RESPONSES
SPEED_TARGET = 0.033   # C6v2 held-out mean speed error (m/s), 3 seeds
YAW_TARGET = 0.085     # C3x held-out mean yaw error (rad/s), 3 seeds
STALL_LIMIT = 5.0      # careful-site stall, %
FALLS_LIMIT = 1.0      # held-out falls per 10 min
TERMS = ["1", "a", "b", "ab", "a2", "b2"]


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
def write_design() -> list[dict]:
    rows = RF.runs()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "design.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "run", "seed", "linear_x", "turning_x", *RF.TERMS])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["run"], r["seed"], r["x"]["linear"], r["x"]["turning"],
                        *(r["scales"][t] for t in RF.TERMS)])
    return rows


def print_design(rows: list[dict]) -> None:
    print("order run        seed  linear  turning")
    for i, r in enumerate(rows, 1):
        print(f"{i:>5} {r['run']:<10} {r['seed']:>4}  {2.0 ** r['x']['linear']:>5}x  {2.0 ** r['x']['turning']:>6}x")
    print("\nlinear = track_linear_velocity + track_lin_relative; "
          "turning = track_angular_velocity + track_yaw_relative (multipliers on C6v2's weights)")


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def run_name(r: dict) -> str:
    return f"surface_{r['run']}"


def finished(iterations: int) -> dict:
    from microduck_pretrain import runs as R

    return {run.name: run for run in R.discover()
            if run.experiment.folder == RF.SURFACE_EXPERIMENT and run.last >= iterations - 1}


def parent(seed: int):
    from microduck_pretrain import runs as R

    run = R.find(R.discover(), RF.PARENT_KEY, seed)
    if run is None:
        raise SystemExit(f"No {RF.PARENT_KEY} seed {seed} run under logs/rsl_rl/. "
                         f"Train it first: uv run python scripts/train_followup.py c6v2 --seeds {seed}")
    return run


def train(rows: list[dict], args) -> int:
    from train_steadycam import stage_checkpoint

    done = finished(args.iterations)
    todo = [r for r in rows if run_name(r) not in done and (not args.only or r["run"] in args.only)]
    print(f"[surface] {len(rows) - len(todo)} of {len(rows)} runs finished, {len(todo)} to train "
          f"({args.iterations} iterations each)", flush=True)
    tags = {}
    for seed in sorted({r["parent_seed"] for r in todo}):
        src = parent(seed).checkpoints[parent(seed).last]
        tag = "warmstart_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{src.parent.name}_{src.stem}")
        tags[seed] = (src, tag)
        print(f"[surface] parent for seed {seed}: {src}", flush=True)
        if not args.dry_run:
            stage_checkpoint(src, tag, RF.SURFACE_EXPERIMENT)
    for k, r in enumerate(todo, 1):
        src, tag = tags[r["parent_seed"]]
        scales = {t: s for t, s in r["scales"].items() if s != 1.0}
        env = dict(os.environ, MICRODUCK_WARM_START="1")
        env[RF.SCALES_ENV] = json.dumps(scales) if scales else ""
        cmd = [
            "train", RF.SURFACE_TASK,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max-iterations", str(args.iterations),
            "--agent.seed", str(r["seed"]),
            "--agent.run-name", run_name(r),
            "--agent.logger", "tensorboard",
            "--agent.resume", "True",
            "--agent.load-run", re.escape(tag),
            "--agent.load-checkpoint", re.escape(src.name),
        ]
        print(f"\n[surface] {k}/{len(todo)}: {r['run']} (linear {2.0 ** r['x']['linear']}x, "
              f"turning {2.0 ** r['x']['turning']}x, seed {r['seed']})", flush=True)
        print(f"$ MICRODUCK_WARM_START=1 {RF.SCALES_ENV}='{env[RF.SCALES_ENV]}' uv run {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, env=env, cwd=REPO)
        if rc != 0:
            print(f"[surface] {r['run']} exited with code {rc}; stopping (re-run to continue).", file=sys.stderr)
            return rc
    return 0


# --------------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------------
def evaluate(rows: list[dict], args) -> int:
    import stress_tests as S
    from microduck_pretrain import evaluate as ev
    from microduck_pretrain import runs as R

    done = finished(args.iterations)
    policies = []
    for seed in RF.SEEDS:
        try:
            p = parent(seed)
            policies.append((f"parent_s{seed}", R.snapshot_onnx(p, p.last)))
        except SystemExit as e:
            print(f"[surface] {e}")
    for r in rows:
        run = done.get(run_name(r))
        if run is not None:
            policies.append((r["run"], R.snapshot_onnx(run, run.last)))
    missing = [r["run"] for r in rows if run_name(r) not in done]
    if missing:
        print(f"[surface] not trained yet, skipped: {', '.join(missing)}")
    if not any(label in {r['run'] for r in rows} for label, _ in policies):
        print("[surface] nothing to evaluate")
        return 1

    scen = []
    for site in ("nominal", "held_out", "careful"):
        _, scs = ev.load_site(REPO / "sites" / f"{site}.toml")
        scen += [(site, S._scale(s, args.quick, seeds=1)) for s in scs]
    plan, jobs = [], []
    for label, onnx in policies:
        for site, sc in scen:
            plan.append((label, site, sc.name))
            jobs.append((onnx, sc, 0))
    runner = S.Runner(args.workers, force=False)
    try:
        res = runner.run(jobs, "surface")
    finally:
        runner.close()

    by: dict[str, dict] = {}
    for (label, site, name), r in zip(plan, res):
        by.setdefault(label, {})[(site, name)] = r
    out = {label: responses(m) for label, m in by.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "responses.json").write_text(json.dumps(out, indent=2))
    print(f"[surface] wrote {OUT / 'responses.json'} ({len(out)} policies)")
    return 0


def responses(m: dict) -> dict:
    held = [v for (site, _), v in m.items() if site == "held_out"]
    careful = [v for (site, _), v in m.items() if site == "careful"]
    nom = next(v for (site, _), v in m.items() if site == "nominal")
    fpm = lambda r: r["falls"] / (r["sim_seconds"] / 600.0)  # noqa: E731
    return {
        "nom_speed": nom["lin_vel_err_mps"],
        "nom_yaw": nom["yaw_rate_err_radps"],
        "ho_speed": SCR._mean(r["lin_vel_err_mps"] for r in held),
        "ho_comb": m[("held_out", "site_combined")]["lin_vel_err_mps"],
        "ho_yaw": SCR._mean(r["yaw_rate_err_radps"] for r in held),
        "ho_falls": SCR._mean(fpm(r) for r in held),
        "careful_stall": 100 * SCR._mean(r["stall_fraction"] for r in careful),
    }


# --------------------------------------------------------------------------
# Analyze
# --------------------------------------------------------------------------
def features(a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.stack([np.ones_like(a), a, b, a * b, a * a, b * b], axis=-1)


def fit(points: list[tuple[int, int, float]]) -> dict:
    """Least-squares quadratic fit with standard errors, R^2 and lack of fit."""
    pts = [(a, b, y) for a, b, y in points if SCR._ok(y)]
    n = len(pts)
    if n < 7:
        return {"ok": False, "n": n}
    A = np.array([p[0] for p in pts], float)
    B = np.array([p[1] for p in pts], float)
    y = np.array([p[2] for p in pts], float)
    X = features(A, B)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = n - X.shape[1]
    sse = float(resid @ resid)
    s2 = sse / df if df > 0 else float("nan")
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    sst = float(((y - y.mean()) ** 2).sum())
    cells: dict[tuple, list[float]] = {}
    for a, b, v in pts:
        cells.setdefault((a, b), []).append(v)
    sspe = sum(sum((v - np.mean(vs)) ** 2 for v in vs) for vs in cells.values())
    df_pe = sum(len(vs) - 1 for vs in cells.values())
    df_lof = len(cells) - X.shape[1]
    lof = {"F": float("nan"), "p": float("nan"), "df": (df_lof, df_pe)}
    if df_lof > 0 and df_pe > 0 and sspe > 0:
        F = ((sse - sspe) / df_lof) / (sspe / df_pe)
        lof["F"] = F
        try:
            from scipy import stats

            lof["p"] = float(stats.f.sf(F, df_lof, df_pe))
        except Exception:
            pass
    return {"ok": True, "n": n, "beta": beta, "se": se, "df": df, "s": math.sqrt(s2) if s2 == s2 else float("nan"),
            "r2": 1 - sse / sst if sst > 0 else float("nan"), "lof": lof,
            "cells": {k: float(np.mean(v)) for k, v in cells.items()}}


def predict(f: dict, a, b):
    return features(a, b) @ f["beta"]


def t_crit(df: int) -> float:
    try:
        from scipy import stats

        return float(stats.t.ppf(0.975, df))
    except Exception:
        return SCR.T975.get(df, 2.0)


def analyze(rows: list[dict]) -> int:
    f = OUT / "responses.json"
    if not f.exists():
        print(f"[surface] {f} missing: run evaluate first")
        return 1
    resp = json.loads(f.read_text())
    have = [r for r in rows if r["run"] in resp]
    fits = {k: fit([(r["x"]["linear"], r["x"]["turning"], resp[r["run"]][k]) for r in have])
            for k, _, _ in RESPONSES}

    with (OUT / "runs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "seed", "linear_mult", "turning_mult", *(k for k, _, _ in RESPONSES)])
        for label in [lab for lab in resp if lab.startswith("parent")] + [r["run"] for r in rows]:
            if label not in resp:
                continue
            r = next((r for r in rows if r["run"] == label), None)
            w.writerow([label, r["seed"] if r else label[-1], 2.0 ** r["x"]["linear"] if r else 1.0,
                        2.0 ** r["x"]["turning"] if r else 1.0, *(resp[label][k] for k, _, _ in RESPONSES)])

    fmt = lambda k, v: f"{v:.1f}" if k in ("ho_falls", "careful_stall") else f"{v:.3f}"  # noqa: E731
    L = ["# Reward surface: walking-speed vs turning emphasis on C6v2", "",
         f"{len(have)} of {len(rows)} runs evaluated. Multipliers on C6v2's linear-tracking pair "
         "(track_linear_velocity + track_lin_relative) and turning-tracking pair "
         "(track_angular_velocity + track_yaw_relative); warm start from C6v2 seed 1 / 2, "
         f"{RF.SURFACE_ITERATIONS} iterations.", ""]

    # Cell means
    L += ["## Cell means (mean of 2 seeds)", ""]
    for k, h, _ in RESPONSES:
        fk = fits[k]
        if not fk.get("ok"):
            continue
        L += [f"**{h}**", "", "| turning \\ linear | 0.5x | 1x | 2x |", "| --- | --- | --- | --- |"]
        for b in (1, 0, -1):
            cells = [fk["cells"].get((a, b)) for a in (-1, 0, 1)]
            L.append(f"| {2.0 ** b}x | " + " | ".join("" if c is None else fmt(k, c) for c in cells) + " |")
        L.append("")
    parents = [lab for lab in resp if lab.startswith("parent")]
    if parents:
        L += ["**Parents (C6v2, before the burst)**", "",
              "| | " + " | ".join(h for _, h, _ in RESPONSES) + " |",
              "| --- | " + " | ".join("---" for _ in RESPONSES) + " |"]
        for lab in parents:
            L.append(f"| {lab} | " + " | ".join(fmt(k, resp[lab][k]) for k, _, _ in RESPONSES) + " |")
        L.append("")

    # Model fits
    L += ["## Fitted quadratic models", "",
          "y = b0 + b1 a + b2 b + b12 ab + b11 a^2 + b22 b^2, with a = log2(linear multiplier), "
          "b = log2(turning multiplier). `*` = |coefficient| > t(0.975, df) x SE. Lack of fit: F-test of "
          "the model's misfit against the seed-to-seed (replicate) error; p < 0.05 means the quadratic "
          "misses real structure.", "",
          "| Response | b0 | b1 (linear) | b2 (turning) | b12 | b11 | b22 | R^2 | s | lack of fit p |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, h, _ in RESPONSES:
        fk = fits[k]
        if not fk.get("ok"):
            L.append(f"| {h} | (n={fk.get('n')}, not enough data) |||||||||")
            continue
        tc = t_crit(fk["df"])
        cells = [f"{v:+.4f}{'*' if abs(v) > tc * s else ''}" for v, s in zip(fk["beta"], fk["se"])]
        p = fk["lof"]["p"]
        L.append(f"| {h} | " + " | ".join(cells) + f" | {fk['r2']:.2f} | {fk['s']:.4f} | "
                 f"{'n/a' if not SCR._ok(p) else f'{p:.2f}'} |")

    # Operating point
    op = operating_point(fits)
    L += ["", "## Operating point", "",
          f"Objective J = max(held-out speed error / {SPEED_TARGET}, held-out yaw error / {YAW_TARGET}): "
          "1.0 = C6v2's walking accuracy and C3x's turning accuracy at once. Searched on the fitted "
          f"surfaces inside the tested region, subject to predicted careful-site stall <= {STALL_LIMIT}% "
          f"and held-out falls <= {FALLS_LIMIT}/10 min.", ""]
    if op is None:
        L.append("Not enough data to fit the surfaces.")
    else:
        L += [f"- **Predicted best:** linear {op['lin_mult']:.2f}x, turning {op['turn_mult']:.2f}x "
              f"(ratio linear/turning {op['lin_mult'] / op['turn_mult']:.2f}). Predicted held-out speed "
              f"{op['ho_speed']:.3f} m/s, yaw {op['ho_yaw']:.3f} rad/s, stall {op['careful_stall']:.1f}%, "
              f"falls {op['ho_falls']:.2f}/10 min; J = {op['J']:.2f}.",
              f"- **Best tested cell:** linear {op['cell'][0]}x, turning {op['cell'][1]}x, J = {op['cell_J']:.2f} "
              f"(observed means: speed {op['cell_speed']:.3f}, yaw {op['cell_yaw']:.3f}).",
              f"- **C6v2 as is (1x, 1x), predicted:** J = {op['J_center']:.2f}.",
              "- A prediction is a hypothesis: confirm it with a full run (C3 parent, 1500 iterations, "
              "3 seeds) before trusting it."]
    L += ["", "Caveats: 250-iteration fine-tunes from C6v2, not training from scratch; two seeds per cell; "
          "three levels per factor, so the model cannot see structure finer than a quadratic."]
    (OUT / "surface.md").write_text("\n".join(L) + "\n")
    _plot(fits, resp, rows, op)
    print("\n".join(L))
    print(f"\n[surface] wrote {OUT}/surface.md, surface.png, runs.csv")
    return 0


def operating_point(fits: dict) -> dict | None:
    need = ("ho_speed", "ho_yaw", "careful_stall", "ho_falls")
    if not all(fits[k].get("ok") for k in need):
        return None
    g = np.linspace(-1, 1, 81)
    A, B = np.meshgrid(g, g)
    pr = {k: predict(fits[k], A, B) for k in need}
    J = np.maximum(pr["ho_speed"] / SPEED_TARGET, pr["ho_yaw"] / YAW_TARGET)
    feasible = (pr["careful_stall"] <= STALL_LIMIT) & (pr["ho_falls"] <= FALLS_LIMIT)
    Jm = np.where(feasible, J, np.inf)
    if not np.isfinite(Jm).any():
        Jm = J  # nothing feasible: report the unconstrained best, flagged by its stall/falls
    i = np.unravel_index(np.argmin(Jm), J.shape)
    cJ = {}
    for (a, b), sp in fits["ho_speed"]["cells"].items():
        yw = fits["ho_yaw"]["cells"].get((a, b))
        if yw is not None:
            cJ[(a, b)] = (max(sp / SPEED_TARGET, yw / YAW_TARGET), sp, yw)
    best = min(cJ, key=lambda c: cJ[c][0])
    center = {k: float(predict(fits[k], 0.0, 0.0)) for k in ("ho_speed", "ho_yaw")}
    return {"a": float(A[i]), "b": float(B[i]), "lin_mult": 2.0 ** float(A[i]), "turn_mult": 2.0 ** float(B[i]),
            **{k: float(pr[k][i]) for k in need}, "J": float(J[i]),
            "cell": (2.0 ** best[0], 2.0 ** best[1]), "cell_J": cJ[best][0], "cell_speed": cJ[best][1],
            "cell_yaw": cJ[best][2],
            "J_center": max(center["ho_speed"] / SPEED_TARGET, center["ho_yaw"] / YAW_TARGET),
            "grid": (A, B, J, feasible)}


def _plot(fits, resp, rows, op) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("ho_speed", "Held-out speed error (m/s)"), ("ho_yaw", "Held-out yaw error (rad/s)"),
              ("careful_stall", "Careful-site stall (%)"), ("ho_falls", "Held-out falls / 10 min")]
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
    g = np.linspace(-1, 1, 81)
    A, B = np.meshgrid(g, g)
    ticks = ([-1, 0, 1], ["0.5x", "1x", "2x"])
    pts = [(r["x"]["linear"], r["x"]["turning"]) for r in rows if r["run"] in resp]
    for ax, (k, title) in zip(axs.flat, panels):
        if not fits[k].get("ok"):
            ax.set_title(f"{title}: not enough data", fontsize=10)
            continue
        Z = predict(fits[k], A, B)
        cs = ax.contourf(A, B, Z, levels=14, cmap="viridis_r")
        ax.contour(A, B, Z, levels=7, colors="k", linewidths=0.4)
        fig.colorbar(cs, ax=ax, shrink=0.85)
        ax.scatter(*zip(*pts), s=18, c="w", edgecolors="k", zorder=3)
        if op:
            ax.plot(op["a"], op["b"], marker="*", ms=16, color="#d62728", mec="k", zorder=4)
        ax.set_title(f"{title}  (R² {fits[k]['r2']:.2f})", fontsize=10)
        ax.set_xticks(*ticks)
        ax.set_yticks(*ticks)
        ax.set_xlabel("linear-tracking weight")
        ax.set_ylabel("turning-tracking weight")
    ax = axs.flat[4]
    if op:
        A_, B_, J, feas = op["grid"]
        cs = ax.contourf(A_, B_, np.clip(J, 0, 4), levels=14, cmap="viridis_r")
        ax.contour(A_, B_, feas.astype(float), levels=[0.5], colors="#d62728", linewidths=1.5)
        fig.colorbar(cs, ax=ax, shrink=0.85)
        ax.plot(op["a"], op["b"], marker="*", ms=16, color="#d62728", mec="k")
        ax.set_title("Objective J (lower = better; red line = stall/falls limit)", fontsize=10)
        ax.set_xticks(*ticks)
        ax.set_yticks(*ticks)
        ax.set_xlabel("linear-tracking weight")
        ax.set_ylabel("turning-tracking weight")
    ax = axs.flat[5]
    for r in rows:
        if r["run"] not in resp:
            continue
        d = r["x"]["linear"] - r["x"]["turning"]
        ax.scatter(resp[r["run"]]["ho_speed"], resp[r["run"]]["ho_yaw"], s=30,
                   c=[plt.cm.coolwarm((d + 2) / 4)], edgecolors="k", zorder=3)
    for lab in resp:
        if lab.startswith("parent"):
            ax.scatter(resp[lab]["ho_speed"], resp[lab]["ho_yaw"], marker="s", s=50, c="w", edgecolors="k", zorder=4)
    ax.axvline(SPEED_TARGET, color="grey", ls=":", lw=1)
    ax.axhline(YAW_TARGET, color="grey", ls=":", lw=1)
    ax.set_xlabel("held-out speed error (m/s)")
    ax.set_ylabel("held-out yaw error (rad/s)")
    ax.set_title("Each run: red = more linear emphasis, blue = more turning\n"
                 "squares = C6v2 parents; dotted = C6v2 speed / C3x yaw targets", fontsize=9)
    fig.suptitle("Reward surface on C6v2: linear-tracking vs turning-tracking weight (red star = predicted operating point)")
    fig.tight_layout()
    fig.savefig(OUT / "surface.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("step", choices=["design", "train", "evaluate", "analyze", "all"])
    p.add_argument("--iterations", type=int, default=RF.SURFACE_ITERATIONS)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--only", nargs="+", help="train only these runs, e.g. L0T0_s1")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick", action="store_true", help="evaluate: 20 s runs, to check the plumbing")
    p.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    args = p.parse_args(argv)
    rows = write_design()
    if args.step == "design":
        print_design(rows)
        return 0
    if args.step in ("train", "all"):
        rc = train(rows, args)
        if rc or args.dry_run:
            return rc
    if args.step in ("evaluate", "all"):
        rc = evaluate(rows, args)
        if rc:
            return rc
    if args.step in ("analyze", "all"):
        return analyze(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
