"""``run_backtest``: same result as a direct engine call, structured errors."""

from __future__ import annotations

import pytest

from trader.service.backtest import BacktestSpec, run_backtest
from trader.service.errors import InvalidRequest, SeriesNotStored


async def test_runs_from_the_lake_without_touching_the_network(seeded_settings):
    spec = BacktestSpec(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        strategy="ma_cross",
        params={"fast": 3, "slow": 8},
        horizon="5m",
    )
    result = await run_backtest(seeded_settings, spec)

    assert result.config["labels"] == ["X"]
    assert len(result.equity) > 0
    assert "sharpe" in result.metrics.as_dict()


async def test_matches_a_direct_engine_run(seeded_settings):
    from trader import backtest as bt
    from trader.data.lake import BarLake, SeriesKey
    from trader.strategies import get_strategy

    spec = BacktestSpec(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        strategy="ma_cross",
        params={"fast": 3, "slow": 8},
        horizon=5,
    )
    via_service = await run_backtest(seeded_settings, spec)

    frame = BarLake(seeded_settings.data_dir).read(SeriesKey("Stock", 211, 5))
    direct = bt.run(
        {"X": frame},
        get_strategy("ma_cross")(fast=3, slow=8),
        horizon=5,
        allocator=bt.get_allocator("equal-weight"),
    )
    assert via_service.final_equity == pytest.approx(direct.final_equity)


async def test_progress_callback_is_invoked(seeded_settings):
    seen: list[float] = []
    spec = BacktestSpec(symbols=["X"], uics=[211], asset_type="Stock", strategy="ma_cross")
    await run_backtest(seeded_settings, spec, progress=seen.append)

    assert seen and seen[-1] == pytest.approx(1.0)
    assert all(0.0 < p <= 1.0 for p in seen)


async def test_unknown_strategy_is_an_invalid_request(seeded_settings):
    spec = BacktestSpec(symbols=["X"], uics=[211], asset_type="Stock", strategy="nope")
    with pytest.raises(InvalidRequest, match="ma_cross"):
        await run_backtest(seeded_settings, spec)


async def test_bad_params_are_an_invalid_request(seeded_settings):
    spec = BacktestSpec(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        strategy="ma_cross",
        params={"fast": 50, "slow": 10},  # fast must be shorter than slow
    )
    with pytest.raises(InvalidRequest, match="bad params"):
        await run_backtest(seeded_settings, spec)


async def test_empty_lake_points_at_backfill(seeded_settings):
    spec = BacktestSpec(
        symbols=["Y"],
        uics=[999],
        asset_type="Stock",
        strategy="ma_cross",
        horizon="5m",
    )
    with pytest.raises(SeriesNotStored, match="trader data backfill") as exc:
        await run_backtest(seeded_settings, spec)

    # the structured payload a UI needs to offer a one-click backfill
    assert exc.value.data == {
        "kind": "series_not_stored",
        "symbol": "Y",
        "horizon": "5m",
        "asset_type": "Stock",
        "uic": 999,
        "since": None,
    }


def test_spec_rejects_an_unserved_horizon():
    with pytest.raises(ValueError, match="not served by Saxo"):
        BacktestSpec(symbols=["X"], strategy="ma_cross", horizon="7m")


def test_spec_rejects_more_uics_than_symbols():
    with pytest.raises(ValueError, match="more uics than symbols"):
        BacktestSpec(symbols=["X"], strategy="ma_cross", uics=[1, 2])
