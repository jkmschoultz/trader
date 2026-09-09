"""``run_training``: trains from the lake, registers a model, structured errors."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trader.config import Settings  # noqa: E402
from trader.service.errors import InvalidRequest, SeriesNotStored  # noqa: E402
from trader.service.training import TrainingSpec, run_training  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def trainable_settings(tmp_path, seed_trainable_lake) -> Settings:
    seed_trainable_lake(tmp_path, bars=1600)
    return Settings(data_dir=tmp_path, state_dir=tmp_path / "state")


def _spec(**over):
    base = dict(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        horizon="5m",
        feature_set="price_v1",
        stop=0.01,
        take=0.01,
        max_bars=6,
        window=8,
        train_end="2024-01-08",
        val_end="2024-01-10",
        hidden=8,
        layers=1,
        epochs=2,
        batch_size=16,
        seed=0,
    )
    base.update(over)
    return TrainingSpec(**base)


async def test_trains_and_registers_a_model(trainable_settings):
    seen: list[float] = []
    result = await run_training(trainable_settings, _spec(), progress=seen.append)

    model_id = result["model_id"]
    assert model_id.startswith("lstm-")
    assert "val_macro_f1" in result["metrics"]
    assert set(result["class_distribution"]) >= {"train", "val"}
    assert seen and seen[-1] == pytest.approx(1.0)

    directory = trainable_settings.models_dir / model_id
    for name in ("manifest.json", "weights.pt", "scaler.json", "feature_spec.json"):
        assert (directory / name).is_file(), name


async def test_registered_model_backtests_through_the_unchanged_path(trainable_settings):
    from trader.service.backtest import BacktestSpec, run_backtest

    model_id = (await run_training(trainable_settings, _spec()))["model_id"]

    spec = BacktestSpec(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        strategy="lstm",
        params={
            "model": model_id,
            "models_dir": str(trainable_settings.models_dir),
            "threshold": 0.0,
        },
        horizon="5m",
    )
    result = await run_backtest(trainable_settings, spec)
    assert len(result.equity) > 0


async def test_unknown_feature_set_is_invalid(trainable_settings):
    with pytest.raises(InvalidRequest, match="unknown feature set"):
        await run_training(trainable_settings, _spec(feature_set="nope"))


async def test_too_few_samples_after_the_split_is_invalid(trainable_settings):
    # a train window shorter than the feature warmup leaves nothing to fit
    with pytest.raises(InvalidRequest, match="too few samples"):
        await run_training(
            trainable_settings,
            _spec(train_end="2024-01-02T16:30", val_end="2024-01-02T18:00"),
        )


async def test_empty_lake_points_at_backfill(trainable_settings):
    with pytest.raises(SeriesNotStored):
        await run_training(trainable_settings, _spec(symbols=["Y"], uics=[999]))


async def test_torch_missing_is_an_invalid_request(trainable_settings, monkeypatch):
    import trader.models._optional as opt

    def no_torch():
        raise ValueError('needs the optional model dependencies (x); pip install -e ".[model]"')

    monkeypatch.setattr(opt, "require_torch", no_torch)
    with pytest.raises(InvalidRequest, match=r"\[model\]"):
        await run_training(trainable_settings, _spec())


def test_spec_rejects_val_end_before_train_end():
    with pytest.raises(ValueError, match="val_end"):
        _spec(train_end="2024-02-01", val_end="2024-01-01")


@pytest.fixture
def cv_settings(tmp_path, seed_trainable_lake) -> Settings:
    seed_trainable_lake(tmp_path, bars=4000)  # ~13.9 days of 5m bars
    return Settings(data_dir=tmp_path, state_dir=tmp_path / "state")


async def test_run_cv_walks_forward_and_backtests_each_fold(cv_settings):
    from trader.service.training import CVConfig, run_cv

    seen: list[float] = []
    cv = CVConfig(folds=2, train_days=8, val_days=2, test_days=1.5, fee_bps=0.2, spread_bps=1.0)
    out = await run_cv(cv_settings, _spec(train_end=None, val_end=None), cv, progress=seen.append)

    assert len(out["folds"]) == 2
    for f in out["folds"]:
        assert f["test_end"] and f["fold"] in (1, 2)
        assert {"sharpe", "total_return", "turnover"} <= set(f["metrics"])
    agg = out["aggregate"]
    assert agg["n_folds"] == 2
    assert isinstance(agg["folds_sharpe_gt_0_5"], int)
    assert set(agg["sharpe"]) == {"median", "mean", "std"}
    assert seen and seen[-1] == pytest.approx(1.0)


async def test_run_cv_rejects_a_span_too_short_for_the_folds(cv_settings):
    from trader.service.training import CVConfig, run_cv

    cv = CVConfig(folds=20, train_days=8, val_days=2, test_days=2)
    with pytest.raises(InvalidRequest, match="not enough"):
        await run_cv(cv_settings, _spec(train_end=None, val_end=None), cv)
