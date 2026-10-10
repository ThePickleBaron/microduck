"""The run catalog behind the timelapse videos: which runs it finds, and that a
policy rebuilt from any snapshot computes exactly what the official exporter
would."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from microduck_pretrain import runs as R


def _touch_run(logs: Path, folder: str, run_dir: str, iterations: list[int]) -> Path:
    d = logs / folder / run_dir
    d.mkdir(parents=True)
    for i in iterations:
        (d / f"model_{i}.pt").write_bytes(b"")
    return d


def test_discover_keeps_longest_run_and_skips_staging(tmp_path):
    logs = tmp_path / "rsl_rl"
    _touch_run(logs, "c3_standard", "2026-10-08_10-00-00_c3_seed1", [0, 9])            # short test run
    real = _touch_run(logs, "c3_standard", "2026-10-09_10-00-00_c3_seed1", [0, 250, 2999])
    _touch_run(logs, "c3_standard", "2026-10-09_12-00-00_c3_seed2", [0, 250])
    _touch_run(logs, "c5_site_finetune", "warmstart_2026-10-09_10-00-00_c3_seed1_model_2999", [2999])
    _touch_run(logs, "c5_site_finetune", "2026-10-10_09-00-00_c5_seed1", [3000, 3250])
    _touch_run(logs, "my_new_task", "2026-10-11_09-00-00_try_seed4", [0])
    (logs / "c3_standard" / "2026-10-12_00-00-00_c3_seed3").mkdir()                      # no checkpoints yet

    found = R.discover(logs)
    names = [(r.experiment.key, r.name) for r in found]
    assert names == [("c3", "c3_seed1"), ("c3", "c3_seed2"), ("c5", "c5_seed1"), ("my_new_task", "try_seed4")]

    c3 = R.find(found, "c3", 1)
    assert c3.path == real and c3.iterations == [0, 250, 2999] and c3.last == 2999
    assert c3.nearest(1000) == 250 and c3.nearest(5000) == 2999

    c5 = R.find(found, "c5", 1)
    assert c5.experiment.start_iteration == 3000 and c5.nearest(0) == 3000  # before the first snapshot -> first

    new = R.find(found, "my_new_task", 4)
    assert new.experiment.title == "My New Task"  # unknown folders still show up


def test_every_condition_has_a_catalog_entry():
    from microduck_pretrain.conditions import TRAINABLE

    keys = {e.key for e in R.EXPERIMENTS.values()}
    assert {c.key for c in TRAINABLE} <= keys
    assert {"c5", "c3x", "steadycam"} <= keys


def _fake_checkpoint(path: Path, sizes=(61, 512, 256, 128, 14), seed=0) -> dict:
    g = torch.Generator().manual_seed(seed)
    sd = {
        "obs_normalizer._mean": torch.randn(1, sizes[0], generator=g),
        "obs_normalizer._std": torch.rand(1, sizes[0], generator=g) + 0.1,
    }
    for n, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        sd[f"mlp.{2 * n}.weight"] = torch.randn(b, a, generator=g) / a**0.5
        sd[f"mlp.{2 * n}.bias"] = torch.randn(b, generator=g) * 0.1
    torch.save({"actor_state_dict": sd}, path)
    return sd


def _reference(sd: dict, obs: np.ndarray) -> np.ndarray:
    x = (torch.from_numpy(obs) - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + R.NORMALIZER_EPS)
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    for n, i in enumerate(idx):
        x = x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]
        if n < len(idx) - 1:
            x = torch.nn.functional.elu(x)
    return x.numpy()


def test_snapshot_onnx_matches_actor_math(tmp_path):
    import onnxruntime as ort

    logs = tmp_path / "rsl_rl"
    d = _touch_run(logs, "c3_standard", "2026-10-09_10-00-00_c3_seed1", [])
    sd = _fake_checkpoint(d / "model_250.pt")
    run = R.discover(logs)[0]

    onnx_path = R.snapshot_onnx(run, 300, cache=tmp_path / "cache")
    assert onnx_path.name == "model_250.onnx"
    sess = ort.InferenceSession(str(onnx_path))
    obs = np.random.default_rng(1).normal(size=(1, 61)).astype(np.float32)
    got = sess.run(["actions"], {"obs": obs})[0]
    np.testing.assert_allclose(got, _reference(sd, obs), atol=1e-5)

    mtime = onnx_path.stat().st_mtime_ns
    assert R.snapshot_onnx(run, 250, cache=tmp_path / "cache").stat().st_mtime_ns == mtime  # cached


@pytest.mark.slow
def test_snapshot_onnx_matches_official_exporter(tmp_path):
    """Same answer as scripts/export.py on a real checkpoint (skipped if none)."""
    import onnxruntime as ort

    from microduck_pretrain.conditions import TRAINABLE

    task = {c.key: c.task_id for c in TRAINABLE}
    run = next((r for r in R.discover() if r.experiment.key in task), None)
    if run is None:
        pytest.skip("no trained run under logs/rsl_rl")
    official = tmp_path / "official.onnx"
    subprocess.run(
        [sys.executable, str(R.REPO / "scripts" / "export.py"), task[run.experiment.key],
         "--checkpoint-file", str(run.checkpoints[run.last]), "--onnx-file", str(official), "--num-envs", "1"],
        check=True, cwd=R.REPO, capture_output=True,
    )
    mine = R.snapshot_onnx(run, run.last, cache=tmp_path / "cache")
    obs = np.random.default_rng(2).normal(size=(1, 61)).astype(np.float32)
    a = ort.InferenceSession(str(official))
    b = ort.InferenceSession(str(mine))
    out_a = a.run(None, {a.get_inputs()[0].name: obs})[0]
    out_b = b.run(None, {"obs": obs})[0]
    np.testing.assert_allclose(out_b, out_a, atol=1e-5)
