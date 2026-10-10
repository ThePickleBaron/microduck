"""Catalog of every training run on disk, and policies rebuilt from any snapshot.

Training keeps a checkpoint every 250 iterations (`model_<N>.pt` in each run
folder under logs/rsl_rl/<experiment>/). Anything we want to show after the
fact (timelapse videos, late evaluations of early snapshots) starts here:

    from microduck_pretrain import runs
    for run in runs.discover():           # every run, newest/longest per name
        print(run.label, run.iterations)  # e.g. c3_seed1 [0, 250, ..., 2999]
    onnx = runs.snapshot_onnx(run, 1250)  # a playable policy for that snapshot

To make a new kind of training show up with a friendly name, add one line to
EXPERIMENTS. Runs from folders not listed there are still found, just labeled
with their folder name. Never delete logs/: the checkpoints are the only record
of how each policy learned.

`uv run python -m microduck_pretrain.runs` prints the inventory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOGS = REPO / "logs" / "rsl_rl"
CACHE = REPO / "videos" / ".cache"
VENDOR_ONNX = REPO / "policies" / "vendor" / "alpha_walking.onnx"

NORMALIZER_EPS = 1e-2  # rsl_rl EmpiricalNormalization: (x - mean) / (std + eps)
CKPT_RE = re.compile(r"model_(\d+)\.pt$")
RUN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_(?P<name>.+)$")


@dataclass(frozen=True)
class Experiment:
    folder: str          # logs/rsl_rl/<folder>
    key: str             # short id used in labels: c3 -> c3_seed1
    title: str           # panel title, e.g. "C3 · Standard"
    caption: str         # one plain-language line for non-technical viewers
    start_iteration: int = 0   # warm-started runs continue from a C3 policy at 3000
    order: int = 50            # left-to-right order in videos


EXPERIMENTS: dict[str, Experiment] = {
    e.folder: e
    for e in (
        Experiment("c2_idealized", "c2", "C2 · Idealized", "Trained in a perfect world", order=20),
        Experiment("c3_standard", "c3", "C3 · Standard", "Trained with Pollen's randomization", order=30),
        Experiment("c4_site", "c4", "C4 · Naive site", "Trained for the site, from scratch", order=40),
        Experiment("c5_site_finetune", "c5", "C5 · Site fine-tune", "C3, then tuned for the site",
                   start_iteration=3000, order=50),
        Experiment("c3x_extended", "c3x", "C3x · Control", "C3, simply trained longer",
                   start_iteration=3000, order=60),
        Experiment("steadycam_walk", "steadycam", "Steady-cam", "C3, tuned to keep the camera steady",
                   start_iteration=3000, order=70),
    )
}
VENDOR = Experiment("vendor", "c1", "C1 · Vendor", "Pollen's shipped policy, as delivered", order=10)


@dataclass
class Run:
    experiment: Experiment
    path: Path
    name: str                      # run name, e.g. c3_seed1
    seed: int | None
    checkpoints: dict[int, Path] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name

    @property
    def iterations(self) -> list[int]:
        return sorted(self.checkpoints)

    @property
    def last(self) -> int:
        return max(self.checkpoints)

    def nearest(self, iteration: int) -> int:
        """Latest snapshot at or before `iteration` (first one if none)."""
        before = [i for i in self.iterations if i <= iteration]
        return max(before) if before else self.iterations[0]


def _experiment_for(folder: str) -> Experiment:
    if folder in EXPERIMENTS:
        return EXPERIMENTS[folder]
    return Experiment(folder, folder, folder.replace("_", " ").title(), f"Runs in logs/rsl_rl/{folder}", order=90)


def discover(logs: Path = LOGS) -> list[Run]:
    """Every run with at least one checkpoint, one per (experiment, run name).

    When several folders share a run name (a short test run and the real run),
    the one with the most training wins, then the newest. Warm-start staging
    folders are skipped.
    """
    best: dict[tuple[str, str], Run] = {}
    if not logs.exists():
        return []
    for exp_dir in sorted(p for p in logs.iterdir() if p.is_dir()):
        exp = _experiment_for(exp_dir.name)
        for run_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            if run_dir.name.startswith("warmstart_") or run_dir.name == "wandb_checkpoints":
                continue
            ckpts = {int(m.group(1)): p for p in run_dir.glob("model_*.pt") if (m := CKPT_RE.search(p.name))}
            if not ckpts:
                continue
            m = RUN_RE.match(run_dir.name)
            name = m.group("name") if m else run_dir.name
            sm = re.search(r"seed(\d+)$", name)
            run = Run(exp, run_dir, name, int(sm.group(1)) if sm else None, ckpts)
            key = (exp.folder, name)
            old = best.get(key)
            if old is None or (run.last, run_dir.name) > (old.last, old.path.name):
                best[key] = run
    return sorted(best.values(), key=lambda r: (r.experiment.order, r.name))


def find(runs: list[Run], key: str, seed: int) -> Run | None:
    for r in runs:
        if r.experiment.key == key and r.seed == seed:
            return r
    return None


def snapshot_onnx(run: Run, iteration: int, cache: Path = CACHE) -> Path:
    """A 61 -> 14 ONNX policy rebuilt from one checkpoint (cached).

    Same math as the official exporter: the observation normalizer, then the
    actor network's mean action. Verified against scripts/export.py in
    tests/test_runs.py.
    """
    import numpy as np
    import onnx
    import torch
    from onnx import TensorProto, helper, numpy_helper

    it = run.nearest(iteration)
    out = cache / run.experiment.folder / run.path.name / f"model_{it}.onnx"
    src = run.checkpoints[it]
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    sd = torch.load(src, map_location="cpu", weights_only=False)["actor_state_dict"]
    arr = lambda k: sd[k].detach().cpu().numpy().astype(np.float32)  # noqa: E731

    inits = [
        numpy_helper.from_array(arr("obs_normalizer._mean").reshape(1, -1), "mean"),
        numpy_helper.from_array((arr("obs_normalizer._std") + NORMALIZER_EPS).reshape(1, -1), "scale"),
    ]
    nodes = [helper.make_node("Sub", ["obs", "mean"], ["c"]), helper.make_node("Div", ["c", "scale"], ["h0"])]
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    h = "h0"
    for n, idx in enumerate(layers):
        inits += [
            numpy_helper.from_array(arr(f"mlp.{idx}.weight").T.copy(), f"W{idx}"),
            numpy_helper.from_array(arr(f"mlp.{idx}.bias"), f"b{idx}"),
        ]
        last = n == len(layers) - 1
        z = "actions" if last else f"z{idx}"
        nodes += [helper.make_node("MatMul", [h, f"W{idx}"], [f"m{idx}"]),
                  helper.make_node("Add", [f"m{idx}", f"b{idx}"], [z])]
        if not last:
            nodes.append(helper.make_node("Elu", [z], [f"a{idx}"], alpha=1.0))
            h = f"a{idx}"
    n_in, n_out = sd["mlp.0.weight"].shape[1], sd[f"mlp.{layers[-1]}.weight"].shape[0]
    graph = helper.make_graph(
        nodes, "snapshot_policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_in])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n_out])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out)
    return out


def main() -> None:
    found = discover()
    if not found:
        print(f"No runs with checkpoints under {LOGS}")
        return
    for r in found:
        its = r.iterations
        print(f"{r.experiment.title:<22} {r.name:<16} {len(its):>3} snapshots  "
              f"iterations {its[0]}..{its[-1]}  ({r.path.relative_to(REPO)})")
    print(f"\nVendor policy (C1): {'found' if VENDOR_ONNX.exists() else 'missing'} at {VENDOR_ONNX.relative_to(REPO)}")


if __name__ == "__main__":
    main()
