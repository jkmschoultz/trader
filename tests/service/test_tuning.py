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
        max_workers=1,  # in-process: no torch subprocesses under pytest
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
    assert "Tuning sweep [lstm]: 4 configs" in text
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
    with pytest.raises(ValueError, match="not valid for strategy 'lstm'"):
        _spec({"nonsense": [1, 2]})


def test_resolved_workers_caps_at_configs_and_cpu():
    spec = _spec({"window": [8, 12, 16]})
    spec = spec.model_copy(update={"max_workers": 0})
    assert 1 <= spec.resolved_workers(3) <= 3
    assert spec.model_copy(update={"max_workers": 8}).resolved_workers(3) == 3
    assert spec.model_copy(update={"max_workers": 2}).resolved_workers(3) == 2


async def test_sweep_runs_across_processes(cv_settings):
    """max_workers=2 fans the configs out to a process pool and still ranks them."""
    spec = _spec({"threshold": [0.0, 0.3]}).model_copy(update={"max_workers": 2})
    report = await run_tuning(cv_settings, spec)

    assert report["n_configs"] == 2 and report["n_errored"] == 0
    assert [r["rank"] for r in report["results"]] == [1, 2]


def test_base_split_dates_are_rejected():
    with pytest.raises(ValueError, match="folds are derived"):
        TuningSpec(
            base=dict(symbols=["X"], train_end="2024-01-08", val_end="2024-01-10"),
            cv=dict(train_days=8, val_days=2, test_days=2),
            grid={"window": [8, 16]},
        )


def test_finish_line_reports_the_config_best_and_eta():
    from datetime import UTC, datetime, timedelta

    from trader.service.tuning import _finish_line

    done = {"aggregate": {"sharpe": {"median": 0.42}}}
    other = {"config": {"window": 16}, "aggregate": {"sharpe": {"median": 1.5}}}
    line = _finish_line(
        2,
        8,
        {"window": 4},
        done,
        [other, {"config": {"window": 4}, **done}, None],
        datetime.now(UTC) - timedelta(minutes=10),
    )
    assert line.startswith("config 2/8 done (window=4)")
    assert "median Sharpe +0.42" in line
    assert "best +1.50" in line
    assert "elapsed 10m00s" in line and "eta 30m00s" in line


def test_finish_line_reports_an_errored_config():
    from datetime import UTC, datetime

    from trader.service.tuning import _finish_line

    line = _finish_line(
        1, 3, {"window": 64}, {"error": "fold 1/5: too few samples"}, [None], datetime.now(UTC)
    )
    assert "error: fold 1/5: too few samples" in line
    assert "none scored yet" in line
