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


async def test_report_carries_direction_skill_for_val_and_test(tmp_path, seed_trainable_lake):
    seed_trainable_lake(tmp_path, bars=1600)
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    result = await run_training(settings, _spec(train_end="2024-01-06", val_end="2024-01-08"))

    metrics, report = result["metrics"], result["report"]
    for split in ("val", "test"):
        assert 0.0 <= metrics[f"{split}_dir_auc"] <= 1.0
        assert f"{split}_long_top10" in metrics
        assert len(report["confidence"][split]) == 5
    # trees now report test accuracy and a test confusion like the LSTM
    assert "test_macro_f1" in metrics
    assert report["test_confusion"] is not None


@pytest.mark.parametrize(("model_type", "expected"), [("lstm", 1e-3), ("gbm", 0.05), ("xgb", 0.05)])
def test_unset_lr_uses_the_model_familys_default(model_type, expected):
    assert _spec(model_type=model_type, lr=None).learning_rate == expected
    assert _spec(model_type=model_type, lr=0.2).learning_rate == 0.2


def test_a_tree_sweep_does_not_inherit_the_lstm_learning_rate():
    """Tuning rebuilds the spec with model_type = the strategy; an unset lr must follow it."""
    from trader.service.tuning import TuningSpec

    base = _spec(model_type="lstm", lr=None)
    cv = CVConfig(folds=2, train_days=8, val_days=2, test_days=1.5)
    spec = TuningSpec(strategy="xgb", base=base, cv=cv, grid={"num_leaves": [7, 15]})
    train, _ = spec.specs_for({"num_leaves": 7})
    assert train.learning_rate == 0.05


async def test_primary_trains_only_on_gated_bars_and_records_it(tmp_path, seed_trainable_lake):
    import json

    seed_trainable_lake(tmp_path, bars=4000)
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    dates = dict(train_end="2024-01-10", val_end="2024-01-15")  # gate on 8th and 14th
    plain = await run_training(settings, _spec(**dates))
    gated = await run_training(
        settings, _spec(primary="tsmom", primary_lookbacks="1/2", name="gated", **dates)
    )

    assert 0 < gated["report"]["n_train"] < plain["report"]["n_train"]
    manifest = json.loads((settings.models_dir / gated["model_id"] / "manifest.json").read_text())
    assert manifest["primary"] == {"name": "tsmom", "lookbacks": [1, 2]}
    assert gated["model_id"].split("-")[-1] != plain["model_id"].split("-")[-1]
