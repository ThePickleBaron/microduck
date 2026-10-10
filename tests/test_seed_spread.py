"""scripts/seed_spread.py: per-seed tables and the paired seed-by-seed verdicts."""

import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("seed_spread", REPO / "scripts" / "seed_spread.py")
ss = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ss)


def _write(d: Path, label: str, err: float, falls: float = 0.0):
    sc = {"lin_vel_err_mps": err, "yaw_rate_err_radps": err, "falls_per_10min": falls, "stall_fraction": 0.0}
    held = {k: dict(sc) for k in ("slippery_floor", "bumped", "site_combined")}
    (d / f"{label}__held_out.json").write_text(json.dumps({"scenarios": held}))
    (d / f"{label}__nominal.json").write_text(json.dumps({"scenarios": {"nominal": dict(sc)}}))


def test_paired_verdicts(tmp_path):
    for s, (a, b) in enumerate([(0.12, 0.10), (0.11, 0.10), (0.09, 0.10)], start=1):
        _write(tmp_path, f"c5_seed{s}", a)
        _write(tmp_path, f"c3x_seed{s}", b)
    _write(tmp_path, "c1", 0.13)  # unseeded policies are listed but not paired
    text = ss.build(ss.load(tmp_path))
    assert "| c1 |" in text
    assert "c5 better on 1/3, worse on 2/3" in text
    assert "c5 (n=3)" in text and "c1 (n=" not in text
