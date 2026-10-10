"""Camera-steadiness evaluation for the desktop-cameraman shots.

Runs an exported ONNX walking policy in CPU MuJoCo (same BAM servo model and
solver settings as `md-eval`) through the shot list in sites/cinema.toml and
measures what a viewer of the head camera would see:

    ang_rate_dps    RMS camera angular velocity after removing its own 1 s
                    moving average (deg/s): shake and wobble, independent of
                    whether the policy actually executes a commanded pan.
    lin_acc_mps2    RMS camera linear acceleration (m/s^2). Bob and jolt.
    roll_deg        RMS horizon tilt (deg).
    ang_rate_p95    95th percentile of the angular rate (deg/s): the worst jolts.
    falls           falls during the run (reset and counted, as in md-eval).
    lin_vel_err     |commanded - achieved| planar speed, 1 s average (m/s).
    stall_fraction  share of commanded-motion time spent standing still.

    uv run python scripts/eval_steadycam.py --policy policies/vendor/alpha_walking.onnx --label vendor
    uv run python scripts/eval_steadycam.py --policy policies/steadycam_seed1.onnx --label steadycam
    uv run python scripts/eval_steadycam.py --compare results/steadycam__vendor.json results/steadycam__steadycam.json
    uv run python scripts/eval_steadycam.py --sweep steadycam          # every saved snapshot of a run

--sweep replays each checkpoint of a training run (logs/, every 250
iterations) on a short version of the moving shots, to find where along the
run the policy got steadier and whether it kept walking.

Writes results/steadycam__<label>.json. Not part of the C1-C4 experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from microduck_pretrain.evaluate import (  # noqa: E402
    CONTROL_DT,
    DECIMATION,
    FALL_HEIGHT_M,
    FALL_TILT_GZ,
    SETTLE_S,
    SPAWN_HEIGHT_M,
    STALL_RATIO,
    Scenario,
    Segment,
    build_sim,
    load_site,
    make_policy,
)

CAMERA_SITE = "head_camera"
LATERAL_AXIS = 1  # site +Y is the image's left-right axis (see steadycam.py)


@dataclass
class CamResult:
    falls: int
    ang_rate_dps: float
    ang_rate_p95: float
    lin_acc_mps2: float
    roll_deg: float
    lin_vel_err: float
    stall_fraction: float
    wall_seconds: float


def run_shot(sc: Scenario, onnx_path: Path, seed: int) -> CamResult:
    rng = np.random.default_rng(seed)
    model, data, ctrl = build_sim(sc, rng)
    policy = make_policy(model, data, ctrl, onnx_path, sc.delay_steps)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CAMERA_SITE)
    if sid < 0:
        raise SystemExit(f"no '{CAMERA_SITE}' site in the {sc.scene} scene")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qadr, vadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

    def reset() -> None:
        mujoco.mj_resetData(model, data)
        data.qpos[qadr : qadr + 3] = [0.0, 0.0, SPAWN_HEIGHT_M]
        data.qpos[qadr + 3 : qadr + 7] = [1, 0, 0, 0]
        for i, idx in enumerate(policy.joint_qpos_indices):
            data.qpos[idx] = policy.default_pose[i]
        ctrl.reset(data.qpos)
        policy.last_action[:] = 0.0
        policy.set_position_targets(policy.default_pose)
        mujoco.mj_forward(model, data)

    schedule = sc.commands
    cycle = sum(s.seconds for s in schedule)

    def command_at(t: float) -> Segment:
        t = t % cycle
        for s in schedule:
            if t < s.seconds:
                return s
            t -= s.seconds
        return schedule[-1]

    reset()
    steps = int(round(sc.duration_s / CONTROL_DT))
    window = deque(maxlen=int(round(1.0 / CONTROL_DT)))
    vel6 = np.zeros(6)
    prev_lin = None
    w_hist = deque(maxlen=int(round(1.0 / CONTROL_DT)))
    since_reset = 0.0
    falls = 0
    ang, acc, roll, lin_errs = [], [], [], []
    moving = stalled = 0
    last = None
    wall0 = time.perf_counter()

    for k in range(steps):
        seg = command_at(k * CONTROL_DT)
        if seg is not last:
            with contextlib.redirect_stdout(io.StringIO()):
                policy.set_vel_cmd(seg.vx, seg.vy, seg.wz)
            last = seg
            window.clear()

        policy.apply_action(policy.infer())
        for _ in range(DECIMATION):
            ctrl.update()
            mujoco.mj_step(model, data)
        since_reset += CONTROL_DT

        gz = float(policy.get_projected_gravity()[2])
        if data.qpos[qadr + 2] < FALL_HEIGHT_M or gz > FALL_TILT_GZ:
            falls += 1
            reset()
            since_reset, prev_lin = 0.0, None
            window.clear()
            w_hist.clear()
            continue

        # Camera motion, world-aligned frame at the site: [ang(3), lin(3)].
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, sid, vel6, 0)
        w = vel6[:3].copy()
        w_hist.append(w)
        w_hp = w - np.mean(w_hist, axis=0)  # high-pass: wobble, not the intended pan
        lin = vel6[3:].copy()
        lateral_z = float(data.site_xmat[sid].reshape(3, 3)[2, LATERAL_AXIS])

        quat = data.qpos[qadr + 3 : qadr + 7].astype(np.float32)
        v_body = policy.quat_rotate_inverse(quat, data.qvel[vadr : vadr + 3].astype(np.float32))
        window.append((float(v_body[0]), float(v_body[1]), float(data.qvel[vadr + 5])))

        if since_reset >= SETTLE_S:
            ang.append(float(np.linalg.norm(w_hp)))
            roll.append(math.degrees(math.asin(max(-1.0, min(1.0, lateral_z)))))
            if prev_lin is not None:
                acc.append(float(np.linalg.norm(lin - prev_lin)) / CONTROL_DT)
            if len(window) == window.maxlen:
                vx, vy, wz = (sum(x[i] for x in window) / len(window) for i in range(3))
                lin_errs.append(math.hypot(vx - seg.vx, vy - seg.vy))
                cmd_lin, cmd_yaw = math.hypot(seg.vx, seg.vy), abs(seg.wz)
                if cmd_lin > 0 or cmd_yaw > 0:
                    moving += 1
                    lin_ok = cmd_lin > 0 and math.hypot(vx, vy) >= STALL_RATIO * cmd_lin
                    yaw_ok = cmd_yaw > 0 and abs(wz) >= STALL_RATIO * cmd_yaw
                    stalled += 0 if (lin_ok or yaw_ok) else 1
        prev_lin = lin

    def rms(xs):
        return float(np.sqrt(np.mean(np.square(xs)))) if xs else float("nan")

    return CamResult(
        falls=falls,
        ang_rate_dps=math.degrees(rms(ang)),
        ang_rate_p95=math.degrees(float(np.percentile(ang, 95))) if ang else float("nan"),
        lin_acc_mps2=rms(acc),
        roll_deg=rms(roll),
        lin_vel_err=float(np.mean(lin_errs)) if lin_errs else float("nan"),
        stall_fraction=stalled / moving if moving else float("nan"),
        wall_seconds=time.perf_counter() - wall0,
    )


METRICS = ("ang_rate_dps", "ang_rate_p95", "lin_acc_mps2", "roll_deg", "lin_vel_err", "stall_fraction")


def summarize(runs: list[CamResult]) -> dict:
    out = {"falls": sum(r.falls for r in runs)}
    for m in METRICS:
        vals = [getattr(r, m) for r in runs if not math.isnan(getattr(r, m))]
        out[m] = float(np.mean(vals)) if vals else float("nan")
    out["runs"] = [asdict(r) for r in runs]
    return out


def compare(paths: list[Path]) -> int:
    reports = [json.loads(p.read_text()) for p in paths]
    shots = list(dict.fromkeys(k for rep in reports for k in rep["shots"]))
    cols = ("ang_rate_dps", "lin_acc_mps2", "roll_deg", "falls", "stall_fraction")
    print(f"{'shot':<12} {'policy':<16} " + " ".join(f"{c:>14}" for c in cols))
    for shot in shots:
        for rep in reports:
            s = rep["shots"].get(shot)
            if s is None:
                continue
            vals = " ".join(f"{s[c]:>14.3f}" if isinstance(s[c], float) else f"{s[c]:>14}" for c in cols)
            print(f"{shot:<12} {rep['label']:<16} {vals}")
        print()
    return 0


def sweep(key: str, seed: int, only: list[str] | None, duration: float, site: Path, out_dir: Path) -> int:
    sys.path.insert(0, str(REPO / "src"))
    from microduck_pretrain import runs as R

    run = R.find(R.discover(), key, seed)
    if run is None:
        print(f"No {key} seed {seed} run with checkpoints under {R.LOGS}")
        return 1
    _, shots = load_site(site)
    want = set(only or ["dolly_in", "pan", "orbit"])
    shots = [s for s in shots if s.name in want]
    rows, report = [], {"run": str(run.path), "key": key, "seed": seed, "duration_s": duration, "snapshots": {}}
    print(f"Sweeping {run.path.name}: snapshots {run.iterations}, shots {[s.name for s in shots]}, {duration:.0f} s each")
    for it in run.iterations:
        onnx = R.snapshot_onnx(run, it)
        res = {}
        for sc in shots:
            sc.duration_s = duration
            r = run_shot(sc, onnx, 0)
            res[sc.name] = asdict(r)
            rows.append((it, sc.name, r))
            print(f"  snapshot {it:>5}  {sc.name:<9} shake={r.ang_rate_dps:6.1f} deg/s  bob={r.lin_acc_mps2:5.2f} m/s2  "
                  f"roll={r.roll_deg:4.1f} deg  falls={r.falls}  stalled={r.stall_fraction:.0%}", flush=True)
        report["snapshots"][str(it)] = res
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"steadycam_sweep__{key}_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", type=Path, help="exported walking policy (.onnx)")
    p.add_argument("--label", help="name for this policy, e.g. vendor or steadycam")
    p.add_argument("--site", type=Path, default=REPO / "sites" / "cinema.toml")
    p.add_argument("--only", nargs="*", help="run only these shots")
    p.add_argument("--seeds", type=int)
    p.add_argument("--duration", type=float)
    p.add_argument("--out", type=Path, default=REPO / "results")
    p.add_argument("--compare", nargs="+", type=Path, help="print a table from result files instead")
    p.add_argument("--sweep", metavar="KEY", help="evaluate every snapshot of a run, e.g. steadycam")
    p.add_argument("--seed", type=int, default=1, help="with --sweep: which run seed (default 1)")
    args = p.parse_args(argv)

    if args.sweep:
        return sweep(args.sweep, args.seed, args.only, args.duration or 20.0, args.site, args.out)

    if args.compare:
        return compare(args.compare)
    if not args.policy or not args.label:
        p.error("--policy and --label are required (or use --compare)")
    if not args.policy.exists():
        p.error(f"policy not found: {args.policy}")

    meta, shots = load_site(args.site)
    if args.only:
        shots = [s for s in shots if s.name in set(args.only)]
    report = {
        "label": args.label,
        "policy": str(args.policy),
        "site": meta.get("name", args.site.stem),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shots": {},
    }
    for sc in shots:
        if args.seeds:
            sc.seeds = args.seeds
        if args.duration:
            sc.duration_s = args.duration
        runs = []
        for seed in range(sc.seeds):
            r = run_shot(sc, args.policy, seed)
            runs.append(r)
            print(
                f"[{args.label}] {sc.name:<11} seed {seed}: shake={r.ang_rate_dps:5.1f} deg/s "
                f"(p95 {r.ang_rate_p95:5.1f})  bob={r.lin_acc_mps2:5.2f} m/s2  "
                f"roll={r.roll_deg:4.1f} deg  falls={r.falls}  stalled={r.stall_fraction:.0%}  "
                f"({r.wall_seconds:.0f}s)",
                flush=True,
            )
        report["shots"][sc.name] = summarize(runs)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"steadycam__{args.label}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
