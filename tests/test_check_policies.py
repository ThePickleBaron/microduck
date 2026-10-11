"""scripts/check_policies.py: the shipped policy passes its own contract; the
simplified snapshot ONNX used for videos does not (it is not deployable)."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_policies", REPO / "scripts" / "check_policies.py")
cp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cp  # dataclasses look the module up while it loads
spec.loader.exec_module(cp)

pytestmark = pytest.mark.skipif(not cp.REFERENCE.exists(), reason="vendor policy not downloaded")


def test_reference_passes_contract_and_numerics():
    r = cp.check(cp.REFERENCE, behavior=False)
    assert r.ok, r.failed()


def test_snapshot_onnx_is_flagged_as_not_deployable(tmp_path):
    import torch

    from microduck_pretrain import runs as R

    d = tmp_path / "rsl_rl" / "c3_standard" / "2026-10-09_10-00-00_c3_seed1"
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(0)
    sizes = (61, 512, 256, 128, 14)
    sd = {"obs_normalizer._mean": torch.zeros(1, 61), "obs_normalizer._std": torch.ones(1, 61)}
    for n, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        sd[f"mlp.{2 * n}.weight"] = torch.randn(b, a, generator=g) * 0.01
        sd[f"mlp.{2 * n}.bias"] = torch.zeros(b)
    torch.save({"actor_state_dict": sd}, d / "model_0.pt")
    run = R.discover(tmp_path / "rsl_rl")[0]
    onnx_path = R.snapshot_onnx(run, 0, cache=tmp_path / "cache")
    r = cp.check(onnx_path, behavior=False)
    names = {n for g_, n, ok, _ in r.checks if not ok}
    assert not r.ok and "metadata keys" in names and "operators" in names
    # shape contract itself is right: only packaging differs
    assert {n for g_, n, ok, _ in r.checks if ok} >= {"one input 'obs' float[1,61]", "one output 'actions' float[1,14]"}
