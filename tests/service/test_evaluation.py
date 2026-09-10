"""``run_strategy_cv`` + the classical branch of ``TuningSpec``: one leaderboard."""

from __future__ import annotations

import pytest

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.evaluation import StrategyCVSpec, run_strategy_cv
from trader.service.tuning import TuningSpec


@pytest.fixture
def cv_settings(tmp_path, seed_trainable_lake) -> Settings:
    seed_trainable_lake(tmp_path, bars=4000)  # ~13.9 days of 5m bars
    return Settings(data_dir=tmp_path, state_dir=tmp_path / "state")


def _spec(**over) -> StrategyCVSpec:
    base = dict(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        strategy="ma_cross",
        params={"fast": 3, "slow": 10},
        horizon="5m",
        folds=2,
        train_days=8,
        val_days=2,
        test_days=1.5,
        fee_bps=0.2,
        spread_bps=1.0,
    )
    base.update(over)
    return StrategyCVSpec(**base)


async def test_walk_forward_scores_every_fold(cv_settings):
    out = await run_strategy_cv(cv_settings, _spec())

    assert len(out["folds"]) == 2
    assert out["aggregate"]["n_folds"] == 2
    for key in ("sharpe", "turnover", "total_return", "hit_rate", "profit_factor"):
        assert key in out["aggregate"]
    for fold in out["folds"]:
        assert {"fold", "train_end", "val_end", "test_end", "metrics"} <= set(fold)
        assert "sharpe" in fold["metrics"]


async def test_orb_also_runs_through_the_same_path(cv_settings):
    out = await run_strategy_cv(
        cv_settings,
        _spec(strategy="orb", params={"open_minutes": 15, "stop": 0.004, "take": 0.008}),
    )
    assert len(out["folds"]) == 2


async def test_a_model_strategy_is_rejected(cv_settings):
    with pytest.raises(InvalidRequest, match="model strategy"):
        await run_strategy_cv(cv_settings, _spec(strategy="lstm", params={}))


async def test_bad_params_surface_as_invalid_request(cv_settings):
    with pytest.raises(InvalidRequest, match="bad params"):
        await run_strategy_cv(cv_settings, _spec(params={"fast": 50, "slow": 10}))


def test_tuning_spec_accepts_classical_grid_keys():
    spec = TuningSpec(
        base=dict(symbols=["X"], uics=[211], asset_type="Stock", horizon="5m"),
        cv=dict(train_days=8, val_days=2, test_days=1.5),
        strategy="orb",
        params={"long_only": True},
        grid={"open_minutes": [5, 15], "stop": [0.002, 0.004], "leverage": [1.0, 2.0]},
    )
    assert not spec.is_model_sweep
    assert len(spec.combinations()) == 8
    job = spec.specs_for({"open_minutes": 15, "stop": 0.004, "leverage": 2.0})
    assert isinstance(job, StrategyCVSpec)
    assert job.params == {"long_only": True, "open_minutes": 15, "stop": 0.004}
    assert job.leverage == 2.0
    assert job.fee_bps == 0.0


def test_tuning_spec_rejects_a_non_ctor_grid_key():
    with pytest.raises(ValueError, match="not valid for strategy 'ma_cross'"):
        TuningSpec(
            base=dict(symbols=["X"]),
            cv=dict(train_days=8, val_days=2, test_days=1.5),
            strategy="ma_cross",
            grid={"hidden": [64, 128]},
        )


def test_model_sweep_still_validates_against_training_fields():
    spec = TuningSpec(
        base=dict(symbols=["X"]),
        cv=dict(train_days=8, val_days=2, test_days=1.5),
        strategy="gbm",
        grid={"num_leaves": [15, 31], "lr": [0.05, 0.1]},
    )
    assert spec.is_model_sweep
    train, cv = spec.specs_for({"num_leaves": 31, "lr": 0.1})
    assert train.model_type == "gbm"
    assert train.layout == "tabular"
    assert train.num_leaves == 31
