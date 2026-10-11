"""Reward-weight screening experiment: which vendor reward weights matter?

    uv run python scripts/reward_screen.py design              # print the run table
    uv run python scripts/reward_screen.py train --dry-run     # show what would run
    uv run python scripts/reward_screen.py train               # 20 short runs on the GPU (~3 h)
    uv run python scripts/reward_screen.py evaluate            # CPU, cached, ~10 min
    uv run python scripts/reward_screen.py analyze             # effects table and heatmap
    uv run python scripts/reward_screen.py all                 # train, evaluate, analyze

Design (src/microduck_pretrain/reward_screen.py): Plackett-Burman, 16 runs x
15 reward weights at 0.5x / 2x the vendor value, plus 4 center runs at the
vendor weights (training seeds 1-4). Every run warm-starts from C3x seed 1 and
trains 250 iterations in C3x's world.

Each finished run is scored on the nominal site, the 8 held-out scenarios and
the careful site (1 run x 120 s each; the evaluator is deterministic). Main
effect of a weight = mean response of its 8 high runs - mean of its 8 low
runs. Noise: from the 4 center runs (t, 3 df) and, independently, Lenth's
pseudo standard error over the 15 effects.

Outputs: results/reward_screen/ (design.csv, runs.csv, effects.csv,
effects.md, effects.png). Training is resumable: finished runs are skipped.
Write-up: docs/reward_screen.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import statistics as st
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from microduck_pretrain import reward_screen as RS  # noqa: E402

REPO = HERE.parent
OUT = REPO / "results" / "reward_screen"
RESPONSES = [  # key, header, plain words (all: lower is better)
    ("nom_speed", "Nominal speed err (m/s)", "speed error at home"),
    ("nom_yaw", "Nominal yaw err (rad/s)", "turn-rate error at home"),
    ("ho_speed", "Held-out mean speed err (m/s)", "speed error on the held-out site"),
    ("ho_comb", "Held-out combined speed err (m/s)", "speed error with every site condition at once"),
    ("ho_yaw", "Held-out mean yaw err (rad/s)", "turn-rate error on the held-out site"),
    ("ho_falls", "Held-out falls/10 min", "falls on the held-out site"),
    ("careful_stall", "Careful-site stall (%)", "standing still on slow moves"),
]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------
def write_design() -> list[dict]:
    rows = RS.runs()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "design.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["order", "run", "kind", "seed", *RS.FACTORS])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["run"], r["kind"], r["seed"], *(r["scales"][k] for k in RS.FACTORS)])
    return rows


def print_design(rows: list[dict]) -> None:
    short = [re.sub(r"[aeiou_]", "", f)[:6] for f in RS.FACTORS]
    print("order run    seed  " + " ".join(f"{s:>6}" for s in short))
    for i, r in enumerate(rows, 1):
        cells = " ".join(f"{ {1: '+', -1: '-', 0: '0'}[r['levels'][f]]:>6}" for f in RS.FACTORS)
        print(f"{i:>5} {r['run']:<6} {r['seed']:>4}  {cells}")
    print(f"\n+ = {RS.HIGH}x vendor weight, - = {RS.LOW}x, 0 = vendor. Columns: " +
          ", ".join(f"{s}={f}" for s, f in zip(short, RS.FACTORS)))


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def run_name(r: dict) -> str:
    return f"screen_{r['run']}"


def finished(iterations: int) -> dict[str, object]:
    from microduck_pretrain import runs as R

    out = {}
    for run in R.discover():
        if run.experiment.folder == RS.SCREEN_EXPERIMENT and run.last >= iterations - 1:
            out[run.name] = run
    return out


def parent_checkpoint() -> Path:
    from microduck_pretrain import runs as R

    run = R.find(R.discover(), RS.PARENT_KEY, RS.PARENT_SEED)
    if run is None:
        raise SystemExit(f"No {RS.PARENT_KEY} seed {RS.PARENT_SEED} run found under logs/rsl_rl/. "
                         "Train it first: uv run python scripts/train_followup.py c3x --seeds 1")
    return run.checkpoints[run.last]


def train(rows: list[dict], args) -> int:
    from train_steadycam import stage_checkpoint

    done = finished(args.iterations)
    src = parent_checkpoint()
    tag = "warmstart_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{src.parent.name}_{src.stem}")
    todo = [r for r in rows if run_name(r) not in done and (not args.only or r["run"] in args.only)]
    print(f"[screen] parent {src}\n[screen] {len(rows) - len(todo)} of {len(rows)} runs finished, "
          f"{len(todo)} to train ({args.iterations} iterations each)", flush=True)
    if not args.dry_run and todo:
        stage_checkpoint(src, tag, RS.SCREEN_EXPERIMENT)
    for k, r in enumerate(todo, 1):
        scales = {f: s for f, s in r["scales"].items() if s != 1.0}
        env = dict(os.environ, MICRODUCK_WARM_START="1")
        env[RS.SCALES_ENV] = json.dumps(scales) if scales else ""
        cmd = [
            "train", RS.SCREEN_TASK,
            "--env.scene.num-envs", str(args.num_envs),
            "--agent.max-iterations", str(args.iterations),
            "--agent.seed", str(r["seed"]),
            "--agent.run-name", run_name(r),
            "--agent.logger", "tensorboard",
            "--agent.resume", "True",
            "--agent.load-run", re.escape(tag),
            "--agent.load-checkpoint", re.escape(src.name),
        ]
        print(f"\n[screen] {k}/{len(todo)}: {r['run']} ({r['kind']}, seed {r['seed']})", flush=True)
        print(f"$ MICRODUCK_WARM_START=1 {RS.SCALES_ENV}='{env[RS.SCALES_ENV]}' uv run {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, env=env, cwd=REPO)
        if rc != 0:
            print(f"[screen] {r['run']} exited with code {rc}; stopping (re-run to continue).", file=sys.stderr)
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
    parent = R.find(R.discover(), RS.PARENT_KEY, RS.PARENT_SEED)
    if parent is not None:
        policies.append(("parent", R.snapshot_onnx(parent, parent.last)))
    for r in rows:
        run = done.get(run_name(r))
        if run is not None:
            policies.append((r["run"], R.snapshot_onnx(run, run.last)))
    missing = [r["run"] for r in rows if run_name(r) not in done]
    if missing:
        print(f"[screen] not trained yet, skipped: {', '.join(missing)}")
    if not policies:
        print("[screen] nothing to evaluate")
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
        res = runner.run(jobs, "screen")
    finally:
        runner.close()

    by: dict[str, dict] = {}
    for (label, site, name), r in zip(plan, res):
        by.setdefault(label, {})[(site, name)] = r
    out = {}
    for label, m in by.items():
        held = [v for (site, _), v in m.items() if site == "held_out"]
        careful = [v for (site, _), v in m.items() if site == "careful"]
        nom = next(v for (site, _), v in m.items() if site == "nominal")
        fpm = lambda r: r["falls"] / (r["sim_seconds"] / 600.0)  # noqa: E731
        out[label] = {
            "nom_speed": nom["lin_vel_err_mps"],
            "nom_yaw": nom["yaw_rate_err_radps"],
            "ho_speed": _mean(r["lin_vel_err_mps"] for r in held),
            "ho_comb": m[("held_out", "site_combined")]["lin_vel_err_mps"],
            "ho_yaw": _mean(r["yaw_rate_err_radps"] for r in held),
            "ho_falls": _mean(fpm(r) for r in held),
            "careful_stall": 100 * _mean(r["stall_fraction"] for r in careful),
        }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "responses.json").write_text(json.dumps(out, indent=2))
    print(f"[screen] wrote {OUT / 'responses.json'} ({len(out)} policies)")
    return 0


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------
# Analyze
# --------------------------------------------------------------------------
def effects(rows: list[dict], resp: dict[str, dict]) -> dict:
    """Main effects per response, with two noise yardsticks."""
    screen = [r for r in rows if r["kind"] == "screen" and r["run"] in resp]
    center = [r for r in rows if r["kind"] == "center" and r["run"] in resp]
    out = {}
    for key, _, _ in RESPONSES:
        y = {r["run"]: resp[r["run"]][key] for r in screen}
        eff = {}
        for f in RS.FACTORS:
            hi = [y[r["run"]] for r in screen if r["levels"][f] > 0 and _ok(y[r["run"]])]
            lo = [y[r["run"]] for r in screen if r["levels"][f] < 0 and _ok(y[r["run"]])]
            eff[f] = (st.mean(hi) - st.mean(lo)) if hi and lo else float("nan")
        c = [resp[r["run"]][key] for r in center if _ok(resp[r["run"]][key])]
        center_thr = float("nan")
        if len(c) >= 2 and len(screen) >= 4:
            n_hi = n_lo = len(screen) / 2  # balanced columns
            se = st.stdev(c) * math.sqrt(1 / n_hi + 1 / n_lo)
            center_thr = T975.get(len(c) - 1, 2.0) * se
        a = sorted(abs(v) for v in eff.values() if not math.isnan(v))
        lenth_thr = float("nan")
        if len(a) >= 6:
            s0 = 1.5 * st.median(a)
            trimmed = [v for v in a if v < 2.5 * s0] or a
            lenth_thr = T975[max(1, min(10, round(len(a) / 3)))] * 1.5 * st.median(trimmed)
        out[key] = {"effects": eff, "center_threshold": center_thr, "lenth_threshold": lenth_thr,
                    "center_values": c, "center_mean": st.mean(c) if c else float("nan"),
                    "screen_mean": _mean(y.values())}
    return out


def _mark(v: float, e: dict) -> str:
    hits = sum(1 for t in (e["center_threshold"], e["lenth_threshold"]) if not math.isnan(t) and abs(v) > t)
    return "**" if hits == 2 else "*" if hits == 1 else ""


def analyze(rows: list[dict]) -> int:
    f = OUT / "responses.json"
    if not f.exists():
        print(f"[screen] {f} missing: run evaluate first")
        return 1
    resp = json.loads(f.read_text())
    E = effects(rows, resp)
    n_screen = sum(1 for r in rows if r["kind"] == "screen" and r["run"] in resp)
    n_center = sum(1 for r in rows if r["kind"] == "center" and r["run"] in resp)

    with (OUT / "runs.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "kind", "seed", *RS.FACTORS, *(k for k, _, _ in RESPONSES)])
        for label in ["parent"] + [r["run"] for r in rows]:
            if label not in resp:
                continue
            r = next((r for r in rows if r["run"] == label), None)
            w.writerow([label, r["kind"] if r else "parent", r["seed"] if r else RS.PARENT_SEED,
                        *((r["scales"][k] if r else 1.0) for k in RS.FACTORS),
                        *(resp[label][k] for k, _, _ in RESPONSES)])
    with (OUT / "effects.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["factor", *(k for k, _, _ in RESPONSES)])
        for fac in RS.FACTORS:
            w.writerow([fac, *(E[k]["effects"][fac] for k, _, _ in RESPONSES)])
        w.writerow(["center_threshold", *(E[k]["center_threshold"] for k, _, _ in RESPONSES)])
        w.writerow(["lenth_threshold", *(E[k]["lenth_threshold"] for k, _, _ in RESPONSES)])

    fmt = lambda k, v: f"{v:+.1f}" if k in ("ho_falls", "careful_stall") else f"{v:+.4f}"  # noqa: E731
    fmt0 = lambda k, v: f"{v:.1f}" if k in ("ho_falls", "careful_stall") else f"{v:.4f}"  # noqa: E731
    L = ["# Reward-weight screen", "",
         f"{n_screen} of 16 screening runs and {n_center} of {RS.N_CENTER} center runs evaluated. "
         f"Each weight at {RS.LOW}x or {RS.HIGH}x the vendor value; warm start from "
         f"{RS.PARENT_KEY} seed {RS.PARENT_SEED}, {RS.SCREEN_ITERATIONS} iterations.", "",
         "**Effect** = mean response with the weight at 2x minus mean with it at 0.5x. Every response is "
         "lower-is-better, so a **negative effect means raising that weight helps**. "
         "`**` = beyond both noise thresholds, `*` = beyond one.", "",
         "| Weight (vendor value) | " + " | ".join(h for _, h, _ in RESPONSES) + " |",
         "| --- | " + " | ".join("---" for _ in RESPONSES) + " |"]
    vendor = _vendor_weights()
    for fac in RS.FACTORS:
        cells = []
        for k, _, _ in RESPONSES:
            v = E[k]["effects"][fac]
            cells.append("" if math.isnan(v) else f"{fmt(k, v)}{_mark(v, E[k])}")
        L.append(f"| {fac} ({vendor.get(fac, '?')}) | " + " | ".join(cells) + " |")
    L.append("| *noise threshold, center runs (t, 95%)* | " +
             " | ".join(fmt0(k, E[k]["center_threshold"]) if not math.isnan(E[k]["center_threshold"]) else "n/a"
                        for k, _, _ in RESPONSES) + " |")
    L.append("| *noise threshold, Lenth (95%)* | " +
             " | ".join(fmt0(k, E[k]["lenth_threshold"]) if not math.isnan(E[k]["lenth_threshold"]) else "n/a"
                        for k, _, _ in RESPONSES) + " |")
    L += ["", "## Reference points", "",
          "| | " + " | ".join(h for _, h, _ in RESPONSES) + " |",
          "| --- | " + " | ".join("---" for _ in RESPONSES) + " |"]
    if "parent" in resp:
        L.append("| Parent (C3x seed 1, before the burst) | " +
                 " | ".join(fmt0(k, resp["parent"][k]) for k, _, _ in RESPONSES) + " |")
    for r in rows:
        if r["kind"] == "center" and r["run"] in resp:
            L.append(f"| Center {r['run']} (vendor weights, seed {r['seed']}) | " +
                     " | ".join(fmt0(k, resp[r['run']][k]) for k, _, _ in RESPONSES) + " |")
    L.append("| Mean of screening runs | " + " | ".join(fmt0(k, E[k]["screen_mean"]) for k, _, _ in RESPONSES) + " |")

    L += ["", "## In plain words", ""]
    any_hit = False
    for fac in RS.FACTORS:
        helps, hurts = [], []
        for k, _, words in RESPONSES:
            v = E[k]["effects"][fac]
            if _mark(v, E[k]):
                (helps if v < 0 else hurts).append(words)
        if helps or hurts:
            any_hit = True
            parts = []
            if helps:
                parts.append("lowers " + ", ".join(helps))
            if hurts:
                parts.append("raises " + ", ".join(hurts))
            L.append(f"- Raising **{fac}** " + "; ".join(parts) + ".")
    if not any_hit:
        L.append("- No weight's effect stands clear of run-to-run noise.")
    L += ["", "Caveats: resolution III (each main effect is aliased with two-weight interactions); "
          "local to the vendor recipe after a short fine-tune, not training from scratch; one training "
          "seed per screening run (the center runs measure that seed-to-seed noise)."]
    (OUT / "effects.md").write_text("\n".join(L) + "\n")
    _heatmap(E)
    print("\n".join(L))
    print(f"\n[screen] wrote {OUT}/effects.md, effects.csv, runs.csv, effects.png")
    return 0


def _vendor_weights() -> dict[str, float]:
    try:
        from microduck_pretrain.followups import make_c3x_env_cfg

        cfg = make_c3x_env_cfg()
        return {k: v.weight for k, v in cfg.rewards.items()}
    except Exception:  # pragma: no cover - analysis must still work without mjlab
        return {}


def _heatmap(E: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = [k for k, _, _ in RESPONSES]
    z = np.full((len(RS.FACTORS), len(keys)), np.nan)
    for j, k in enumerate(keys):
        thr = E[k]["center_threshold"]
        if math.isnan(thr) or thr == 0:
            thr = E[k]["lenth_threshold"]
        for i, f in enumerate(RS.FACTORS):
            v = E[k]["effects"][f]
            if not (math.isnan(v) or math.isnan(thr) or thr == 0):
                z[i, j] = v / thr
    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(np.clip(z, -3, 3), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if not np.isnan(z[i, j]):
                ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(z[i, j]) > 2 else "black")
    ax.set_xticks(range(len(keys)), [h for _, h, _ in RESPONSES], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(RS.FACTORS)), RS.FACTORS, fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("effect / noise threshold (|x| > 1 = clear of noise)")
    ax.set_title("Raising each reward weight 0.5x -> 2x: blue = better (lower error), red = worse", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "effects.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("step", choices=["design", "train", "evaluate", "analyze", "all"])
    p.add_argument("--iterations", type=int, default=RS.SCREEN_ITERATIONS)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--only", nargs="+", help="train only these runs, e.g. ctr1 pb01")
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
