"""reward_terms.py reads TensorBoard curves and lines them up with the
measured speed error."""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("reward_terms_driver", REPO / "scripts" / "reward_terms.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_terms_lined_up_with_speed_error(tmp_path, monkeypatch):
    from torch.utils.tensorboard import SummaryWriter

    from microduck_pretrain import runs as R

    rt = _mod()
    runs = []
    for seed in (1, 2):
        d = tmp_path / f"2026-10-01_00-00-00_c3_seed{seed}"
        w = SummaryWriter(str(d))
        for it in range(0, 1001, 5):
            w.add_scalar("Episode_Reward/air_time", 0.5 + it / 1000, it)          # keeps rising
            w.add_scalar("Episode_Reward/track_linear_velocity", 1.0 - it / 4000, it)
            w.add_scalar("Episode_Reward/foot_slip", -0.01, it)
            w.add_scalar("Train/mean_reward", 10 + it / 100, it)
        w.close()
        runs.append(R.Run(R.EXPERIMENTS["c3_standard"], d, f"c3_seed{seed}", seed, {1000: d / "model_1000.pt"}))
    speed = {0: 0.20, 250: 0.08, 500: 0.09, 750: 0.10, 1000: 0.11}
    monkeypatch.setattr(rt, "OUT", tmp_path / "out")
    monkeypatch.setattr(rt, "budget_curve", lambda key: speed)

    assert rt.analyze("c3", runs)
    md = (tmp_path / "out" / "c3.md").read_text()
    assert "lowest at iteration 250" in md
    row = next(line for line in md.splitlines() if line.startswith("| air_time"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert float(cells[1]) == pytest.approx(0.75, abs=0.01)    # at 250
    assert float(cells[2]) == pytest.approx(1.49, abs=0.02)    # at 1000 (window clipped at the end)
    assert float(cells[4]) > 0.5                                # rose while accuracy fell
    assert (tmp_path / "out" / "c3.png").exists() and (tmp_path / "out" / "c3.csv").exists()


def test_pearson():
    rt = _mod()
    assert rt.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert rt.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert rt.pearson([1, 2], [1, 2]) != rt.pearson([1, 2], [1, 2])   # nan: too few points
