"""``model_type="xgb"`` end to end: train and register, backtest, and walk-forward CV."""

from __future__ import annotations

import pytest

pytest.importorskip("xgboost")

from trader.config import Settings  # noqa: E402
from trader.service.training import CVConfig, TrainingSpec, run_cv, run_training  # noqa: E402

pytestmark = pytest.mark.slow


def _spec(**over):
    base = dict(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        horizon="5m",
        feature_set="price_v1",
        model_type="xgb",
        stop=0.01,
        take=0.01,
        max_bars=6,
        window=8,
        n_estimators=30,
        num_leaves=7,
        lr=0.1,
        device="cpu",
        seed=0,
    )
    base.update(over)
    return TrainingSpec(**base)


def test_xgb_spec_uses_the_tabular_layout():
    assert _spec().layout == "tabular"


async def test_trains_registers_and_backtests(tmp_path, seed_trainable_lake):
    from trader.service.backtest import BacktestSpec, run_backtest

    seed_trainable_lake(tmp_path, bars=1600)
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    result = await run_training(settings, _spec(train_end="2024-01-08", val_end="2024-01-10"))

    model_id = result["model_id"]
    assert (settings.models_dir / model_id / "model.json").is_file()

    backtest = await run_backtest(
        settings,
        BacktestSpec(
            symbols=["X"],
            uics=[211],
            asset_type="Stock",
            strategy="xgb",
            params={"model": model_id, "threshold": 0.0},
            horizon="5m",
        ),
    )
    assert len(backtest.equity) > 0


async def test_run_cv_with_xgb(tmp_path, seed_trainable_lake):
    seed_trainable_lake(tmp_path, bars=4000)
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    cv = CVConfig(folds=2, train_days=8, val_days=2, test_days=1.5, fee_bps=0.2, spread_bps=1.0)
    out = await run_cv(settings, _spec(), cv)

    assert len(out["folds"]) == 2
    assert out["aggregate"]["n_folds"] == 2
