"""`md-eval`: headless held-out-site evaluation of a walking policy.

Runs an exported ONNX walking policy in CPU MuJoCo, using the same BAM servo
model the policies were trained against, under the scenarios defined in a site
file (sites/*.toml). Training uses GPU MuJoCo Warp; this evaluator is a
different simulator, which makes it a miniature version of the sim-to-real gap.

    uv run md-eval --policy policies/c3_seed1.onnx --label c3 --site sites/held_out.toml
    uv run md-eval --policy policies/vendor/velstand.onnx --label c1 --site sites/nominal.toml

Writes results/<label>__<site>.json. Combine results with `md-report`.

Metrics per scenario (averaged over seeds):
    falls_per_10min     trunk below 6 cm or tilted past 60 deg, counted then reset
    lin_vel_err_mps     |commanded - achieved| planar speed, 1 s moving average
    yaw_rate_err_radps  |commanded - achieved| yaw rate, 1 s moving average
    upright_fraction    share of time spent walking (not in the 1 s after a reset)
    stall_fraction      share of commanded-motion time spent standing still
                        (achieved < 20% of the command): catches a policy that
                        will not break away from standstill
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
import sys
import time
import tomllib
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
VENDOR = REPO / "third_party" / "microduck_rl"
ROBOT_DIR = VENDOR / "src" / "mjlab_microduck" / "robot" / "microduck"
# The walk-model scenes: the collision model every velocity task trains on
# (feet are the only parts expected to touch the floor while walking).
SCENES = {
    "flat": ROBOT_DIR / "scene_walk.xml",
    "backlash": ROBOT_DIR / "scene_walk_backlash.xml",
}
FOOT_GEOMS = ("left_foot_collision", "right_foot_collision")

SIM_DT = 0.005          # physics step (s), as in training and infer_policy.py
DECIMATION = 4          # policy runs at 50 Hz
CONTROL_DT = SIM_DT * DECIMATION
FALL_HEIGHT_M = 0.06    # trunk origin below this = fallen
FALL_TILT_GZ = -0.5     # body-frame gravity z above this = tilted past 60 deg
SETTLE_S = 1.0          # ignore tracking error for this long after a reset
SPAWN_HEIGHT_M = 0.125  # as in infer_policy.py


# --------------------------------------------------------------------------
# Site description
# --------------------------------------------------------------------------
@dataclass
class Segment:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    seconds: float = 3.0


@dataclass
class Scenario:
    name: str
    description: str = ""
    scene: str = "flat"                      # "flat" | "backlash"
    foot_friction: float = 1.0               # tangential mu of both foot geoms
    payload_kg: float = 0.0                  # point mass added to the trunk
    vin: float = 7.4                         # battery voltage (V)
    vin_drop_gain: float = 0.1               # load-dependent sag (V per N*m)
    delay_steps: int = 0                     # policy-to-motor delay, control steps
    rough_height_mm: float = 0.0             # 0 = flat floor; else max bump height
    rough_cell_m: float = 0.10               # bump spacing
    push_speed_mps: float = 0.0              # 0 = no pushes
    push_interval_s: tuple[float, float] = (4.0, 8.0)
    duration_s: float = 120.0
    seeds: int = 3
    commands: list[Segment] = field(default_factory=list)


DEFAULT_COMMANDS = [
    Segment(vx=0.30, seconds=6.0),
    Segment(wz=0.8, seconds=3.0),
    Segment(vx=0.25, vy=0.15, seconds=4.0),
    Segment(seconds=2.0),
    Segment(vx=0.30, seconds=4.0),
    Segment(vx=-0.25, seconds=3.0),
    Segment(wz=-0.8, seconds=3.0),
]
STALL_RATIO = 0.2       # achieved < 20% of the command while commanded = stalled


def load_site(path: Path) -> tuple[dict, list[Scenario]]:
    raw = tomllib.loads(path.read_text())
    meta = raw.get("site", {})
    defaults = raw.get("defaults", {})
    shared_cmds = [Segment(**s) for s in raw.get("commands", [])] or DEFAULT_COMMANDS
    scenarios = []
    for entry in raw.get("scenario", []):
        merged = {**defaults, **entry}
        cmds = [Segment(**s) for s in merged.pop("commands", [])] or shared_cmds
        if "push_interval_s" in merged:
            merged["push_interval_s"] = tuple(merged["push_interval_s"])
        scenarios.append(Scenario(**merged, commands=cmds))
    if not scenarios:
        raise SystemExit(f"{path}: no [[scenario]] entries")
    return meta, scenarios


# --------------------------------------------------------------------------
# Vendored helpers (BAM actuator setup + observation building)
# --------------------------------------------------------------------------
def _load_infer_module():
    path = VENDOR / "scripts" / "infer_policy.py"
    spec = importlib.util.spec_from_file_location("md_vendor_infer_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_INFER = None


def infer_module():
    global _INFER
    if _INFER is None:
        _INFER = _load_infer_module()
    return _INFER


def _add_rough_floor(spec: mujoco.MjSpec, height_mm: float, cell_m: float, rng) -> None:
    """Lay a random heightfield (0..height_mm bumps) over a 20 m x 20 m area."""
    half = 10.0
    n = int(round(2 * half / cell_m)) + 1
    elevation = rng.random((n, n))
    elevation[n // 2 - 2 : n // 2 + 3, n // 2 - 2 : n // 2 + 3] = 0.0  # flat spawn pad
    hf = spec.add_hfield()
    hf.name = "md_rough_floor"
    hf.size = [half, half, max(height_mm, 1e-3) / 1000.0, 0.01]
    hf.nrow = n
    hf.ncol = n
    hf.userdata = elevation.astype(np.float32).flatten().tolist()
    g = spec.worldbody.add_geom()
    g.name = "md_rough_floor"
    g.type = mujoco.mjtGeom.mjGEOM_HFIELD
    g.hfieldname = "md_rough_floor"
    g.pos = [0.0, 0.0, 0.0]


def _apply_training_solver(model: mujoco.MjModel) -> None:
    """Match the solver settings the velocity tasks train with (mjlab
    MujocoCfg for Mjlab-Velocity-*): the XML defaults differ (Euler
    integrator, 100 iterations), and the integrator alone changes how a
    policy breaks away from standstill."""
    model.opt.timestep = SIM_DT
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.impratio = 1.0
    model.opt.iterations = 10
    model.opt.tolerance = 1e-8
    model.opt.ls_iterations = 20
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 50


def build_sim(sc: Scenario, rng):
    """Compile the scenario's MuJoCo model with BAM actuators. Mirrors
    infer_policy.load_mujoco_with_bam, plus the scenario's floor and payload."""
    inf = infer_module()
    from bam.mujoco import MujocoController

    bam_model = inf.load_bam_model(inf.BAM_KP_FW, sc.vin, None)
    kt, R = bam_model.kt.value, bam_model.R.value
    force_limit = bam_model.actuator.vin * kt / R

    spec = mujoco.MjSpec.from_file(str(SCENES[sc.scene]))
    if sc.rough_height_mm > 0:
        _add_rough_floor(spec, sc.rough_height_mm, sc.rough_cell_m, rng)
    if sc.payload_kg > 0:
        trunk = spec.body("trunk_base")
        trunk.mass = trunk.mass + sc.payload_kg

    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        joint = spec.joint(tgt_name)
        joint.damping = np.zeros((3, 1))
        joint.frictionloss = 0.0
        joint.solref_friction = inf.BAM_STIFF_SOLREF_FRICTION
        joint.solimp_friction = inf.BAM_STIFF_SOLIMP_FRICTION

    # Contact settings applied in training by microduck_constants.FULL_COLLISION:
    # feet condim 3 with priority, every other collision geom frictionless.
    for g in spec.geoms:
        if not g.name.endswith("_collision"):
            continue
        if g.name in FOOT_GEOMS:
            g.condim = 3
            g.priority = 1
            g.friction = [sc.foot_friction, g.friction[1], g.friction[2]]
        else:
            g.condim = 1

    model = spec.compile()
    _apply_training_solver(model)
    data = mujoco.MjData(model)
    gain = sc.vin_drop_gain if sc.vin_drop_gain > 0 else None
    ctrl = MujocoController(bam_model, names, model, data, vin_drop_gain=gain, vin_min=inf.BAM_VIN_MIN)
    return model, data, ctrl


def make_policy(model, data, ctrl, onnx_path: Path, delay_steps: int):
    inf = infer_module()
    import onnxruntime as ort

    obs_len = ort.InferenceSession(str(onnx_path)).get_inputs()[0].shape[-1]
    with contextlib.redirect_stdout(io.StringIO()):
        policy = inf.PolicyInference(
            model,
            data,
            bam_ctrl=ctrl,
            walking_onnx_path=str(onnx_path),
            delay_min_lag=delay_steps,
            delay_max_lag=delay_steps,
            use_projected_gravity=True,
            new_cmd_obs=(obs_len == 61),
        )
    _read_through_backlash(policy, model, data)
    return policy


def _read_through_backlash(policy, model, data) -> None:
    """On the backlash model the real encoder sits on the OUTPUT side of the
    gear play, so joint observations must add each passive_*_backlash hinge
    (as tasks/backlash.py does in training)."""
    q_extra, v_extra = [], []
    for i in range(model.nu):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[i, 0])
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"passive_{jname}_backlash")
        if bid < 0:
            return
        q_extra.append(int(model.jnt_qposadr[bid]))
        v_extra.append(int(model.jnt_dofadr[bid]))
    qi, vi = np.array(policy.joint_qpos_indices), np.array(policy.joint_qvel_indices)
    qe, ve = np.array(q_extra), np.array(v_extra)
    policy.get_joint_pos_relative = lambda: (data.qpos[qi] + data.qpos[qe]).astype(np.float32) - policy.default_pose
    policy.get_joint_vel = lambda: (data.qvel[vi] + data.qvel[ve]).astype(np.float32)


# --------------------------------------------------------------------------
# Episode runner
# --------------------------------------------------------------------------
@dataclass
class RunResult:
    sim_seconds: float
    falls: int
    lin_vel_err_mps: float
    yaw_rate_err_radps: float
    upright_fraction: float
    stall_fraction: float
    wall_seconds: float


def run_scenario(sc: Scenario, onnx_path: Path, seed: int) -> RunResult:
    rng = np.random.default_rng(seed)
    model, data, ctrl = build_sim(sc, rng)
    policy = make_policy(model, data, ctrl, onnx_path, sc.delay_steps)

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
    since_reset = 0.0
    falls = 0
    lin_errs, yaw_errs = [], []
    upright_steps = 0
    moving_steps = stalled_steps = 0
    next_push = rng.uniform(*sc.push_interval_s) if sc.push_speed_mps > 0 else math.inf
    last_cmd = None
    wall0 = time.perf_counter()

    for k in range(steps):
        t = k * CONTROL_DT
        seg = command_at(t)
        if seg is not last_cmd:
            with contextlib.redirect_stdout(io.StringIO()):
                policy.set_vel_cmd(seg.vx, seg.vy, seg.wz)
            last_cmd = seg
            window.clear()

        if t >= next_push:
            ang = rng.uniform(0, 2 * math.pi)
            data.qvel[vadr] = sc.push_speed_mps * math.cos(ang)
            data.qvel[vadr + 1] = sc.push_speed_mps * math.sin(ang)
            next_push = t + rng.uniform(*sc.push_interval_s)

        policy.apply_action(policy.infer())
        for _ in range(DECIMATION):
            ctrl.update()
            mujoco.mj_step(model, data)
        since_reset += CONTROL_DT

        # Fall check
        gz = float(policy.get_projected_gravity()[2])
        if data.qpos[qadr + 2] < FALL_HEIGHT_M or gz > FALL_TILT_GZ:
            falls += 1
            reset()
            since_reset = 0.0
            window.clear()
            continue

        # Tracking error on a 1 s moving average of body-frame velocity
        quat = data.qpos[qadr + 3 : qadr + 7].astype(np.float32)
        v_body = policy.quat_rotate_inverse(quat, data.qvel[vadr : vadr + 3].astype(np.float32))
        window.append((float(v_body[0]), float(v_body[1]), float(data.qvel[vadr + 5])))
        if since_reset >= SETTLE_S:
            upright_steps += 1
            if len(window) == window.maxlen:
                vx, vy, wz = (sum(w[i] for w in window) / len(window) for i in range(3))
                lin_errs.append(math.hypot(vx - seg.vx, vy - seg.vy))
                yaw_errs.append(abs(wz - seg.wz))
                cmd_lin, cmd_yaw = math.hypot(seg.vx, seg.vy), abs(seg.wz)
                if cmd_lin > 0 or cmd_yaw > 0:
                    moving_steps += 1
                    lin_ok = cmd_lin > 0 and math.hypot(vx, vy) >= STALL_RATIO * cmd_lin
                    yaw_ok = cmd_yaw > 0 and abs(wz) >= STALL_RATIO * cmd_yaw
                    if not (lin_ok or yaw_ok):
                        stalled_steps += 1

    return RunResult(
        sim_seconds=sc.duration_s,
        falls=falls,
        lin_vel_err_mps=float(np.mean(lin_errs)) if lin_errs else float("nan"),
        yaw_rate_err_radps=float(np.mean(yaw_errs)) if yaw_errs else float("nan"),
        upright_fraction=upright_steps / steps,
        stall_fraction=stalled_steps / moving_steps if moving_steps else float("nan"),
        wall_seconds=time.perf_counter() - wall0,
    )


def summarize(runs: list[RunResult]) -> dict:
    total_s = sum(r.sim_seconds for r in runs)

    def mean(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return float(np.mean(xs)) if xs else float("nan")

    def std(xs):
        xs = [x for x in xs if not math.isnan(x)]
        return float(np.std(xs)) if len(xs) > 1 else 0.0

    per_run_fpm = [r.falls / (r.sim_seconds / 600.0) for r in runs]
    return {
        "falls_per_10min": sum(r.falls for r in runs) / (total_s / 600.0),
        "falls_per_10min_std": std(per_run_fpm),
        "lin_vel_err_mps": mean([r.lin_vel_err_mps for r in runs]),
        "lin_vel_err_std": std([r.lin_vel_err_mps for r in runs]),
        "yaw_rate_err_radps": mean([r.yaw_rate_err_radps for r in runs]),
        "upright_fraction": mean([r.upright_fraction for r in runs]),
        "stall_fraction": mean([r.stall_fraction for r in runs]),
        "sim_seconds": total_s,
        "runs": [asdict(r) for r in runs],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="md-eval", description=__doc__.split("\n\n")[0])
    p.add_argument("--policy", type=Path, required=True, help="exported walking policy (.onnx)")
    p.add_argument("--label", required=True, help="name for this policy in reports, e.g. c3 or c3_seed2")
    p.add_argument("--site", type=Path, default=REPO / "sites" / "held_out.toml")
    p.add_argument("--only", nargs="*", help="run only these scenario names")
    p.add_argument("--seeds", type=int, help="override seeds per scenario")
    p.add_argument("--duration", type=float, help="override seconds per run")
    p.add_argument("--out", type=Path, default=REPO / "results")
    args = p.parse_args(argv)

    if not args.policy.exists():
        p.error(f"policy not found: {args.policy}")
    meta, scenarios = load_site(args.site)
    if args.only:
        scenarios = [s for s in scenarios if s.name in set(args.only)]

    site_name = meta.get("name", args.site.stem)
    report = {
        "label": args.label,
        "policy": str(args.policy),
        "site": site_name,
        "site_file": str(args.site),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": {},
    }
    for sc in scenarios:
        if args.seeds:
            sc.seeds = args.seeds
        if args.duration:
            sc.duration_s = args.duration
        runs = []
        for seed in range(sc.seeds):
            r = run_scenario(sc, args.policy, seed)
            runs.append(r)
            print(
                f"[{args.label}] {sc.name:<16} seed {seed}: falls={r.falls:<3} "
                f"lin_err={r.lin_vel_err_mps:.3f} m/s  yaw_err={r.yaw_rate_err_radps:.3f} rad/s  "
                f"upright={r.upright_fraction:.0%}  stalled={r.stall_fraction:.0%}  ({r.wall_seconds:.0f}s)",
                flush=True,
            )
        summary = summarize(runs)
        summary["config"] = {k: v for k, v in asdict(sc).items() if k != "commands"}
        report["scenarios"][sc.name] = summary

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.label}__{site_name}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
