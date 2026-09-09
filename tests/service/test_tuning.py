"""``run_tuning``: grid sweep, per-config CV, ranked report, resilient to a bad config."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trader.config import Settings  # noqa: E402
from trader.service.tuning import TuningSpec, render, run_tuning  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture
def cv_settings(tmp_path, seed_trainable_lake) -> Settings:
    seed_trainable_lake(tmp_path, bars=4000)  # ~13.9 days of 5m bars
    return Settings(data_dir=tmp_path, state_dir=tmp_path / "state")


def _spec(grid: dict) -> TuningSpec:
    return TuningSpec(
        base=dict(
            symbols=["X"],
            uics=[211],
            asset_type="Stock",
            horizon="5m",
            feature_set="price_v1",
            stop=0.01,
            take=0.01,
            max_bars=6,
            window=8,
            hidden=8,
            layers=1,
            epochs=2,
            batch_size=16,
        ),
        cv=dict(folds=2, train_days=8, val_days=2, test_days=1.5, fee_bps=0.2, spread_bps=1.0),
        grid=grid,
    )


async def test_sweep_ranks_every_config(cv_settings):
    seen: list[float] = []
    report = await run_tuning(
        cv_settings, _spec({"window": [8, 12], "threshold": [0.0, 0.3]}), progress=seen.append
    )

    assert report["n_configs"] == 4
    assert report["n_errored"] == 0
    ranks = [r["rank"] for r in report["results"]]
    assert ranks == [1, 2, 3, 4]
    # ranked by descending median Sharpe
    sharpes = [r["aggregate"]["sharpe"]["median"] for r in report["results"]]
    assert sharpes == sorted(sharpes, reverse=True)
    assert len(report["top"]) == 4
    assert seen[-1] == pytest.approx(1.0)

    text = render(report)
    assert "Tuning sweep: 4 configs" in text
    assert "window=8 threshold=0.0" in text


async def test_a_broken_config_is_recorded_not_fatal(cv_settings):
    report = await run_tuning(cv_settings, _spec({"window": [8, 4000]}))

    assert report["n_configs"] == 2
    assert report["n_errored"] == 1
    good = [r for r in report["results"] if "aggregate" in r]
    bad = [r for r in report["results"] if "error" in r]
    assert good[0]["config"] == {"window": 8} and good[0]["rank"] == 1
    assert bad[0]["config"] == {"window": 4000} and bad[0]["rank"] is None
    assert "n/a" not in render(report).splitlines()[0]


def test_grid_key_must_be_a_real_field():
    with pytest.raises(ValueError, match="not a TrainingSpec or CVConfig field"):
        _spec({"nonsense": [1, 2]})


def test_base_split_dates_are_rejected():
    with pytest.raises(ValueError, match="run_cv derives folds"):
        TuningSpec(
            base=dict(symbols=["X"], train_end="2024-01-08", val_end="2024-01-10"),
            cv=dict(train_days=8, val_days=2, test_days=2),
            grid={"window": [8, 16]},
        )
