"""Boom test: a thin carbon-fiber rod (up to 3 ft) with a small temperature
sensor on the end, clamped to the duck's back. No training, CPU only.

    uv run python scripts/boom_test.py                 # vendor policy + c3x, c6v2, c6v3 (whatever is exported)
    uv run python scripts/boom_test.py --conditions c1 c6v2 --quick

The boom is a reduced-order model (src/microduck_pretrain/evaluate.py,
add_boom): a rigid rod on two bending hinges whose spring reproduces the
rod's first bending frequency, with the rod's and sensor's real masses. A
0.91 m (3 ft), 3 mm solid pultruded rod with a 15 g sensor: ~10 g of rod,
first mode ~1.6 Hz, ~10 cm static sag, 0.18 N m at the clamp when
horizontal, on a 0.74 kg robot about 0.28 m tall.

Sweeps (flat floor, the nominal command schedule, 60 s x 2 runs each):
  length     forward / back / side / up / up45 at 0.30-0.91 m (15 g sensor, 3 mm rod)
  stiffness  each direction at 0.91 m with the rod made rigid (same masses):
             separates "it's heavy and far out" from "it whips"
  sensor     10 g and 20 g at 0.91 m
  rod        2 mm solid, 3 mm solid, 4/2 mm tube, 5/3 mm tube at 0.91 m (forward, side)

Outputs: results/boom/boom.md, boom.csv, boom.png. Details: docs/boom_test.md.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import stress_tests as S  # noqa: E402
from microduck_pretrain import evaluate as ev  # noqa: E402

REPO = HERE.parent
OUT = REPO / "results" / "boom"
DIRS = ["forward", "back", "side", "up", "up45"]
LENGTHS = [0.30, 0.45, 0.60, 0.75, 0.91]
RODS = {"2mm solid": (2.0, 0.0), "3mm solid": (3.0, 0.0), "4/2mm tube": (4.0, 2.0), "5/3mm tube": (5.0, 3.0)}
FAIL_FALLS = 10.0      # per 10 min, as the operating envelope
MOUNT_H = 0.167        # m: trunk top above the floor when standing (clamp height of a horizontal boom)


def configs() -> list[dict]:
    out = []
    base = {"tip": 15.0, "rod": "3mm solid", "e": 110.0}
    for d in DIRS:
        for L in LENGTHS:
            out.append({"sweep": "length", "dir": d, "L": L, **base, "rigid": False})
        out.append({"sweep": "stiffness", "dir": d, "L": 0.91, **base, "rigid": True})
        for tip in (10.0, 20.0):
            out.append({"sweep": "sensor", "dir": d, "L": 0.91, **base, "tip": tip, "rigid": False})
    for d in ("forward", "side"):
        for rod in RODS:
            if rod != "3mm solid":
                out.append({"sweep": "rod", "dir": d, "L": 0.91, **base, "rod": rod, "rigid": False})
    return out


def scenario(c: dict, quick: bool) -> ev.Scenario:
    od, bore = RODS[c["rod"]]
    sc = ev.Scenario(name=f"boom_{c['dir']}_{c['L']}", commands=list(ev.DEFAULT_COMMANDS), duration_s=60.0, seeds=2,
                     boom_length_m=c["L"], boom_dir=c["dir"], boom_tip_g=c["tip"], boom_rod_mm=od,
                     boom_bore_mm=bore, boom_e_gpa=c["e"], boom_rigid=c.get("rigid", False))
    return S._scale(sc, quick)



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--conditions", nargs="+", default=["c1", "c3x", "c6v2", "c6v3"])
    p.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    pols = S.exported_policies([c.lower() for c in args.conditions])
    if not pols:
        print("No exported policies for", args.conditions)
        return 1
    cfgs = configs()
    nominal = S._scale(replace(ev.Scenario(name="no_boom", commands=list(ev.DEFAULT_COMMANDS)), duration_s=60.0, seeds=2), args.quick)
    plan, jobs = [], []
    for label, path in pols:
        for seed in range(nominal.seeds):
            plan.append((label, None))
            jobs.append((path, nominal, seed))
        for i, c in enumerate(cfgs):
            sc = scenario(c, args.quick)
            for seed in range(sc.seeds):
                plan.append((label, i))
                jobs.append((path, sc, seed))
    runner = S.Runner(args.workers, args.force)
    t0 = time.perf_counter()
    try:
        res = runner.run(jobs, "boom")
    finally:
        runner.close()
    acc: dict = {}
    for k, r in zip(plan, res):
        acc.setdefault(k, []).append(r)

    rows = []
    for label, _ in pols:
        base = S.summarize(acc[(label, None)])
        for i, c in enumerate(cfgs):
            runs = acc[(label, i)]
            m = S.summarize(runs)
            bp = ev.boom_params(scenario(c, False))
            rows.append({"policy": label, "condition": S.group_of(label), **c,
                         "f1_hz": bp["f1_hz"], "static_sag_mm": bp["static_tip_mm_horizontal"],
                         "clamp_moment_nm": bp["moment_horizontal_nm"], **m,
                         "tip_rms_mm": S._mean([r.get("boom_tip_rms_mm", math.nan) for r in runs]),
                         "tip_max_mm": max((r.get("boom_tip_max_mm", math.nan) for r in runs
                                            if r.get("boom_tip_max_mm") == r.get("boom_tip_max_mm")), default=math.nan),
                         "no_boom_falls": base["falls_per_10min"], "no_boom_speed_err": base["lin_vel_err_mps"]})
    OUT.mkdir(parents=True, exist_ok=True)
    S._write_csv(OUT / "boom.csv", rows)
    write_report(rows, cfgs)
    plot(rows)
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min. Results in {OUT}")
    return 0


def _group(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def write_report(rows: list[dict], cfgs: list[dict]) -> None:
    conds = sorted({r["condition"] for r in rows}, key=lambda k: S.ORDER.get(k, 90))
    L = ["# Boom test: 3 ft carbon rod with a temperature sensor", "",
         "Flat floor, nominal commands, 60 s x 2 runs per setting, values pooled over a condition's seeds. "
         f"Fails = {FAIL_FALLS:.0f}+ falls per 10 min. Tip motion = how far the sensor moves about its static "
         "sag while the robot is upright (RMS / max).", "",
         "## The boom (3 mm solid rod, 15 g sensor)", "",
         "| Length (m) | Rod mass (g) | First bending mode (Hz) | Static tip sag, horizontal (mm) | Moment at the clamp, horizontal (N m) |",
         "| --- | --- | --- | --- | --- |"]
    for Lm in LENGTHS:
        bp = ev.boom_params(ev.Scenario(name="x", boom_length_m=Lm))
        L.append(f"| {Lm:.2f} | {1000 * bp['m_rod']:.1f} | {bp['f1_hz']:.1f} | {bp['static_tip_mm_horizontal']:.0f} | "
                 f"{bp['moment_horizontal_nm']:.3f} |")

    L += ["", "## Longest boom each policy carries without failing (falls per 10 min in brackets at 0.91 m)", "",
          "| Condition | " + " | ".join(DIRS) + " |", "| --- | " + " | ".join("---" for _ in DIRS) + " |"]
    for cnd in conds:
        cells = []
        for d in DIRS:
            ok = []
            for Lm in LENGTHS:
                g = _group(rows, condition=cnd, sweep="length", dir=d, L=Lm)
                if g and S._mean([r["falls_per_10min"] for r in g]) < FAIL_FALLS:
                    ok.append(Lm)
                else:
                    break
            at3 = _group(rows, condition=cnd, sweep="length", dir=d, L=0.91)
            f3 = S._mean([r["falls_per_10min"] for r in at3]) if at3 else math.nan
            best = f"{max(ok):.2f} m" if ok else "< 0.30 m"
            cells.append(f"{best}{'+' if ok and max(ok) == LENGTHS[-1] else ''} ({f3:.0f})")
        L.append(f"| {cnd} | " + " | ".join(cells) + " |")

    L += ["", "## Falls per 10 min by length", ""]
    for d in DIRS:
        L += [f"**{d}**", "", "| Condition | no boom | " + " | ".join(f"{x:.2f} m" for x in LENGTHS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in LENGTHS) + " |"]
        for cnd in conds:
            nb = S._mean([r["no_boom_falls"] for r in _group(rows, condition=cnd, sweep="length", dir=d)])
            vals = [S._mean([r["falls_per_10min"] for r in _group(rows, condition=cnd, sweep="length", dir=d, L=x)])
                    for x in LENGTHS]
            L.append(f"| {cnd} | {nb:.0f} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
        L.append("")

    L += ["## Is it the weight or the whip? (0.91 m, flexible 3 mm rod vs the same masses made rigid)", "",
          "| Condition | Direction | Falls: flexible | Falls: rigid | Speed err: flexible | Speed err: rigid | Tip RMS (mm): flexible |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for cnd in conds:
        for d in DIRS:
            fl = _group(rows, condition=cnd, sweep="length", dir=d, L=0.91)
            rg = _group(rows, condition=cnd, sweep="stiffness", dir=d)
            if fl and rg:
                f = lambda g, k: S._mean([r[k] for r in g])  # noqa: E731
                L.append(f"| {cnd} | {d} | {f(fl, 'falls_per_10min'):.0f} | {f(rg, 'falls_per_10min'):.0f} | "
                         f"{f(fl, 'lin_vel_err_mps'):.3f} | {f(rg, 'lin_vel_err_mps'):.3f} | {f(fl, 'tip_rms_mm'):.0f} |")

    L += ["", "## Sensor mass and rod choice (0.91 m): falls per 10 min", "",
          "| Condition | Direction | 10 g | 15 g | 20 g |", "| --- | --- | --- | --- | --- |"]
    for cnd in conds:
        for d in DIRS:
            vals = []
            for tip in (10.0, 15.0, 20.0):
                g = (_group(rows, condition=cnd, sweep="length", dir=d, L=0.91) if tip == 15.0
                     else _group(rows, condition=cnd, sweep="sensor", dir=d, tip=tip))
                vals.append(S._mean([r["falls_per_10min"] for r in g]))
            L.append(f"| {cnd} | {d} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
    L += ["", "| Condition | Direction | " + " | ".join(RODS) + " |", "| --- | --- | " + " | ".join("---" for _ in RODS) + " |"]
    for cnd in conds:
        for d in ("forward", "side"):
            vals = []
            for rod in RODS:
                g = (_group(rows, condition=cnd, sweep="length", dir=d, L=0.91) if rod == "3mm solid"
                     else _group(rows, condition=cnd, sweep="rod", dir=d, rod=rod))
                vals.append(S._mean([r["falls_per_10min"] for r in g]))
            L.append(f"| {cnd} | {d} | " + " | ".join(f"{v:.0f}" for v in vals) + " |")
    L += ["", "Rod properties at 0.91 m with a 15 g sensor (static sag from beam theory, small-deflection; "
          f"a horizontal boom is clamped about {1000 * MOUNT_H:.0f} mm above the floor):", "",
          "| Rod | Mass (g) | First mode (Hz) | Static sag (mm) | Note |", "| --- | --- | --- | --- | --- |"]
    for rod, (od, bore) in RODS.items():
        bp = ev.boom_params(ev.Scenario(name="x", boom_length_m=0.91, boom_rod_mm=od, boom_bore_mm=bore))
        note = ("sag exceeds the clamp height: a real horizontal boom would rest its tip on the floor "
                "(not simulated), so treat its numbers as invalid" if bp["static_tip_mm_horizontal"] > 1000 * MOUNT_H else "")
        L.append(f"| {rod} | {1000 * bp['m_rod']:.1f} | {bp['f1_hz']:.1f} | {bp['static_tip_mm_horizontal']:.0f} | {note} |")
    L += ["", "Model limits: one bending mode per axis (the rod's higher modes and torsion are left out), "
          "no collisions between the rod and the robot or floor, ideal clamp, damping ratio 2%."]
    (OUT / "boom.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def plot(rows: list[dict]) -> None:
    plt = S._plt()
    conds = sorted({r["condition"] for r in rows}, key=lambda k: S.ORDER.get(k, 90))
    fig, axs = plt.subplots(2, len(DIRS), figsize=(4 * len(DIRS), 7.5), sharey="row")
    for j, d in enumerate(DIRS):
        for cnd in conds:
            xs, fa, tip = [], [], []
            for Lm in LENGTHS:
                g = _group(rows, condition=cnd, sweep="length", dir=d, L=Lm)
                xs.append(Lm)
                fa.append(S._mean([r["falls_per_10min"] for r in g]))
                tip.append(S._mean([r["tip_rms_mm"] for r in g]))
            axs[0, j].plot(xs, fa, marker="o", ms=3, color=S.COLORS.get(cnd), label=cnd.upper())
            axs[1, j].plot(xs, tip, marker="o", ms=3, color=S.COLORS.get(cnd), label=cnd.upper())
        axs[0, j].axhline(FAIL_FALLS, color="grey", ls=":", lw=1)
        axs[0, j].set_title(f"boom {d}", fontsize=10)
        axs[1, j].set_xlabel("boom length (m)")
        for ax in axs[:, j]:
            ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].set_ylabel("falls / 10 min")
    axs[1, 0].set_ylabel("sensor motion, RMS (mm)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Carbon boom with a 15 g sensor (3 mm rod): falls and sensor motion vs boom length")
    fig.tight_layout()
    fig.savefig(OUT / "boom.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
