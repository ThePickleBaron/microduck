"""Deployment check: can each exported policy be installed on a Microduck as is?

    uv run python scripts/check_policies.py                 # every policies/*.onnx
    uv run python scripts/check_policies.py policies/c6_seed1.onnx
    uv run python scripts/check_policies.py --manifest      # also write policies/manifest.json

Pollen's roadmap (milestone M8, "model channel") will let robots install
policies published outside the shipped set, and says every model must pass a
shape check, obs[1,61] -> actions[1,14], before it goes live. This script runs
that check and the rest of what the robot's runtime relies on, using the
shipped walk policy (policies/vendor/velstand.onnx) as the reference:

  contract   input "obs" float[1,61], output "actions" float[1,14]; ONNX IR
             version and opset no newer than the reference; only operators the
             reference uses; the same metadata keys, with identical joint
             order, default pose, gains, observation and command layout and
             action scale (the runtime reads these from the file).
  numerics   finite actions for zero, typical and extreme observations; action
             magnitudes in the same range as the reference on the same inputs.
  cost       single-thread inference time and file size, as a ratio to the
             reference. The reference already runs at 50 Hz on the robot, so a
             ratio near 1 means the same budget.
  behavior   in the CPU simulator: stands still on a zero command from rest,
             and walks a 0.3 m/s command without falling (20 s each).

Writes results/policy_check.md. Exit code 1 if any policy fails. With
--manifest, also writes policies/manifest.json in the format of Pollen's
shipped manifest (schema 2), with training provenance for each policy that
passed. The robot does not read our manifest yet (M8 is not shipped); it is
the record a model channel would need.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

REFERENCE = REPO / "policies" / "vendor" / "velstand.onnx"
OBS_LEN, ACT_LEN = 61, 14
MUST_MATCH = ("joint_names", "default_joint_pos", "joint_stiffness", "joint_damping",
              "command_names", "observation_names", "action_scale")
ACTION_RANGE_FACTOR = 3.0      # max |action| may be up to 3x the reference's on the same inputs
COST_RATIO_MAX = 2.0           # inference time and file size vs. the reference (timing at ~0.02 ms is noisy)
STAND_DRIFT_MAX = 0.05         # m/s mean speed while asked to stand
WALK_ERR_MAX = 0.20            # m/s mean speed error while asked to walk 0.3 m/s

TASKS = {  # condition key -> training task id (for the manifest)
    "c2": "Pretrain-C2-Idealized-Flat-MicroDuck",
    "c3": "Pretrain-C3-Standard-Flat-MicroDuck",
    "c4": "Pretrain-C4-Site-Rough-Backlash-MicroDuck",
    "c5": "Pretrain-C5-SiteFinetune-Rough-Backlash-MicroDuck",
    "c3x": "Pretrain-C3X-Standard-Extended-Flat-MicroDuck",
    "c6": "Pretrain-C6-Careful-Flat-MicroDuck",
    "steadycam": "Steadycam-Walk-Flat-MicroDuck",
    "steadycam_v2": "Steadycam-Walk-v2-Flat-MicroDuck",
}


@dataclass
class Report:
    path: Path
    checks: list[tuple[str, str, bool, str]] = field(default_factory=list)  # (group, name, ok, detail)
    info: dict = field(default_factory=dict)

    def add(self, group: str, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((group, name, bool(ok), detail))

    @property
    def ok(self) -> bool:
        return all(c[2] for c in self.checks)

    def failed(self) -> list[str]:
        return [f"{g}/{n}: {d}" for g, n, ok, d in self.checks if not ok]


def _meta(model) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def _session(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def _probe_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Zero, typical (unit-scale noise around 0) and extreme observations."""
    typical = rng.normal(0.0, 1.0, size=(32, OBS_LEN)).astype(np.float32)
    typical[:, 3:6] = [0.0, 0.0, -1.0]  # projected gravity: upright
    extreme = rng.uniform(-10.0, 10.0, size=(32, OBS_LEN)).astype(np.float32)
    return {"zero": np.zeros((1, OBS_LEN), np.float32), "typical": typical, "extreme": extreme}


def _run(sess, x: np.ndarray) -> np.ndarray:
    name = sess.get_inputs()[0].name
    return np.concatenate([sess.run(None, {name: row[None, :]})[0] for row in x], axis=0)


def _time_ms(sess, n: int = 300) -> float:
    name = sess.get_inputs()[0].name
    x = np.zeros((1, OBS_LEN), np.float32)
    for _ in range(20):
        sess.run(None, {name: x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return st.median(ts)


def check(path: Path, ref: Path = REFERENCE, behavior: bool = True) -> Report:
    import onnx

    rep = Report(path)
    try:
        model = onnx.load(str(path))
        sess = _session(path)
    except Exception as e:  # noqa: BLE001
        rep.add("contract", "loads", False, f"{type(e).__name__}: {e}")
        return rep
    rep.add("contract", "loads", True)
    refm, refs = onnx.load(str(ref)), _session(ref)

    # ---- contract
    ins, outs = sess.get_inputs(), sess.get_outputs()
    rep.add("contract", "one input 'obs' float[1,61]",
            len(ins) == 1 and ins[0].name == "obs" and list(ins[0].shape) == [1, OBS_LEN] and ins[0].type == "tensor(float)",
            ", ".join(f"{i.name} {i.type} {i.shape}" for i in ins))
    rep.add("contract", "one output 'actions' float[1,14]",
            len(outs) == 1 and outs[0].name == "actions" and list(outs[0].shape) == [1, ACT_LEN] and outs[0].type == "tensor(float)",
            ", ".join(f"{o.name} {o.type} {o.shape}" for o in outs))
    rep.add("contract", "IR version", model.ir_version <= refm.ir_version, f"{model.ir_version} (reference {refm.ir_version})")
    opset = {o.domain or "ai.onnx": o.version for o in model.opset_import}
    ref_opset = {o.domain or "ai.onnx": o.version for o in refm.opset_import}
    rep.add("contract", "opset", all(d in ref_opset and v <= ref_opset[d] for d, v in opset.items()),
            f"{opset} (reference {ref_opset})")
    ops = {n.op_type for n in model.graph.node}
    ref_ops = {n.op_type for n in refm.graph.node}
    rep.add("contract", "operators", ops <= ref_ops,
            "extra: " + ", ".join(sorted(ops - ref_ops)) if ops - ref_ops else ", ".join(sorted(ops)))
    meta, ref_meta = _meta(model), _meta(refm)
    missing = sorted(set(ref_meta) - set(meta))
    rep.add("contract", "metadata keys", not missing, "missing: " + ", ".join(missing) if missing else f"{len(meta)} keys")
    for k in MUST_MATCH:
        if k in ref_meta:
            rep.add("contract", f"metadata {k}", meta.get(k) == ref_meta[k],
                    "identical" if meta.get(k) == ref_meta[k] else f"differs: {str(meta.get(k))[:60]}...")
    rep.info["run_path"] = meta.get("run_path", "")

    # ---- numerics
    rng = np.random.default_rng(0)
    probes = _probe_inputs(rng)
    worst = 0.0
    for name, x in probes.items():
        try:
            y, yr = _run(sess, x), _run(refs, x)
        except Exception as e:  # noqa: BLE001
            rep.add("numerics", f"{name} observations", False, f"{type(e).__name__}: {e}")
            continue
        finite = bool(np.isfinite(y).all())
        ratio = float(np.abs(y).max() / max(1e-6, np.abs(yr).max()))
        worst = max(worst, ratio)
        ok = finite and (name == "extreme" or ratio <= ACTION_RANGE_FACTOR)
        rep.add("numerics", f"{name} observations", ok,
                f"finite={finite}, max|action| {np.abs(y).max():.2f} vs reference {np.abs(yr).max():.2f}")
    rep.info["action_ratio"] = worst

    # ---- cost
    t, tr = _time_ms(sess), _time_ms(refs)
    size, size_ref = path.stat().st_size, ref.stat().st_size
    rep.add("cost", "inference time", t <= COST_RATIO_MAX * tr, f"{t:.3f} ms (reference {tr:.3f} ms, x{t / tr:.2f})")
    rep.add("cost", "file size", size <= COST_RATIO_MAX * size_ref, f"{size / 1e3:.0f} kB (reference {size_ref / 1e3:.0f} kB)")
    rep.info["ms"], rep.info["ms_ref"] = t, tr

    # ---- behavior
    if behavior:
        import contextlib
        import io

        from microduck_pretrain import evaluate as ev

        base = ev.Scenario(name="check", delay_steps=1, duration_s=20.0, seeds=1)
        stand = ev.Scenario(**{**base.__dict__, "name": "stand", "commands": [ev.Segment(seconds=20.0)]})
        walk = ev.Scenario(**{**base.__dict__, "name": "walk", "commands": [ev.Segment(vx=0.30, seconds=20.0)]})
        with contextlib.redirect_stdout(io.StringIO()):
            rs = ev.run_scenario(stand, path, 0)
            rw = ev.run_scenario(walk, path, 0)
        rep.add("behavior", "stands still on a zero command", rs.falls == 0 and rs.lin_vel_err_mps <= STAND_DRIFT_MAX,
                f"falls {rs.falls}, drift {rs.lin_vel_err_mps:.3f} m/s")
        rep.add("behavior", "walks 0.3 m/s without falling",
                rw.falls == 0 and rw.lin_vel_err_mps <= WALK_ERR_MAX and (rw.stall_fraction or 0) < 0.5,
                f"falls {rw.falls}, speed error {rw.lin_vel_err_mps:.3f} m/s, stalled {100 * (rw.stall_fraction or 0):.0f}%")
    return rep


def _group(label: str) -> str:
    return re.split(r"_seed\d+$", label)[0]


def _git_commit() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip())
        return sha, dirty
    except Exception:  # noqa: BLE001
        return "unknown", True


def write_manifest(reports: list[Report], out: Path) -> None:
    sha, dirty = _git_commit()
    entries = []
    for r in reports:
        if not r.ok:
            continue
        label = r.path.stem
        run_path = r.info.get("run_path", "")
        m = re.search(r"model_(\d+)\.pt$", run_path)
        entries.append({
            "file": r.path.name,
            "kind": "perpetual",
            "slot": "walk",
            "entry_pose": "standing",
            "description": f"{label}: Microduck pre-training project policy (condition {_group(label)}).",
            "training": {
                "task_id": TASKS.get(_group(label), "unknown"),
                "repo": "ThePickleBaron/microduck",
                "commit": sha,
                "dirty": dirty,
                "run": str(Path(run_path).parent.name) if run_path else "unknown",
                "checkpoint": int(m.group(1)) if m else None,
                "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(r.path.stat().st_mtime)),
            },
            "checked": {"reference": REFERENCE.name, "date": time.strftime("%Y-%m-%d"), "passed": True},
        })
    manifest = {
        "schema_version": 2,
        "model_api": 1,
        "obs_len": OBS_LEN,
        "action_len": ACT_LEN,
        "robot": {"model": "microduck", "hw_rev": 1, "servos": "xl330", "control_hz": 50},
        "description": "Walking policies pre-trained in simulation for the IDAI 600 Microduck project. "
                       "Each passed scripts/check_policies.py against the shipped velstand.onnx.",
        "policies": entries,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n")


def write_markdown(reports: list[Report], out: Path) -> None:
    groups = ["contract", "numerics", "cost", "behavior"]
    lines = ["# Policy deployment check", "",
             f"Reference: `policies/vendor/{REFERENCE.name}` (the shipped walk policy). "
             "Contract = what the robot's runtime and Pollen's planned model channel (M8) require. "
             "See scripts/check_policies.py for every check.", "",
             "| Policy | Result | " + " | ".join(g.capitalize() for g in groups) + " | Inference (ms, x ref) |",
             "| --- | --- | " + " | ".join("---" for _ in groups) + " | --- |"]
    for r in reports:
        cells = []
        for g in groups:
            cs = [c for c in r.checks if c[0] == g]
            cells.append("-" if not cs else ("pass" if all(c[2] for c in cs) else f"**fail** ({sum(not c[2] for c in cs)})"))
        ms = r.info.get("ms")
        t = f"{ms:.3f} (x{ms / r.info['ms_ref']:.2f})" if ms else "-"
        lines.append(f"| {r.path.name} | {'PASS' if r.ok else '**FAIL**'} | " + " | ".join(cells) + f" | {t} |")
    bad = [r for r in reports if not r.ok]
    if bad:
        lines += ["", "## Failures", ""]
        for r in bad:
            lines += [f"- **{r.path.name}**"] + [f"  - {f}" for f in r.failed()]
    lines += ["", "## Details", ""]
    for r in reports:
        lines += [f"### {r.path.name}", "", "| Check | Result | Detail |", "| --- | --- | --- |"]
        lines += [f"| {g}: {n} | {'ok' if ok else '**FAIL**'} | {d} |" for g, n, ok, d in r.checks]
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("policies", nargs="*", type=Path, help="ONNX files (default: every policies/*.onnx)")
    p.add_argument("--no-behavior", action="store_true", help="skip the two 20 s simulator checks")
    p.add_argument("--manifest", action="store_true", help="also write policies/manifest.json for the passing policies")
    p.add_argument("--out", type=Path, default=REPO / "results" / "policy_check.md")
    args = p.parse_args(argv)
    if not REFERENCE.exists():
        p.error(f"reference policy missing: {REFERENCE} (run scripts/get_vendor_policy.py)")
    paths = args.policies or sorted((REPO / "policies").glob("*.onnx"))
    if not paths:
        print("No policies to check.")
        return 1
    reports = []
    for path in paths:
        r = check(path, behavior=not args.no_behavior)
        reports.append(r)
        print(f"{'PASS' if r.ok else 'FAIL'}  {path.name}" + ("" if r.ok else "\n  " + "\n  ".join(r.failed())), flush=True)
    write_markdown(reports, args.out)
    print(f"\nWrote {args.out}")
    if args.manifest:
        out = REPO / "policies" / "manifest.json"
        write_manifest(reports, out)
        print(f"Wrote {out} ({sum(r.ok for r in reports)} policies)")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
