"""Metrics: annualisation factor, and the equity/trade statistics by hand."""

from __future__ import annotations

import math
import statistics

import pandas as pd
import pytest

from trader.backtest.metrics import compute, periods_per_year


def _equity(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-03-01 14:30", periods=len(values), freq="5min", tz="UTC")
    return pd.Series(values, index=idx, name="equity")


def test_periods_per_year_uses_the_session_length_when_known():
    # A 6h25m session at 5-minute bars is 78 bars/day.
    assert periods_per_year(5, 385) == pytest.approx(78 * 252)


def test_periods_per_year_falls_back_to_a_24h_day():
    assert periods_per_year(5, None) == pytest.approx((1440 / 5) * 252)


def test_total_return_and_drawdown_by_hand():
    metrics = compute(_equity([100, 110, 99, 108]), pd.DataFrame(), periods_per_year=19_656)
    assert metrics.total_return == pytest.approx(0.08)
    assert metrics.max_drawdown == pytest.approx(99 / 110 - 1)


def test_sharpe_matches_the_plain_formula():
    values = [100, 110, 105, 115]
    rets = [b / a - 1 for a, b in zip(values, values[1:], strict=False)]
    ppy = 19_656
    expected = statistics.mean(rets) / statistics.stdev(rets) * math.sqrt(ppy)

    metrics = compute(_equity(values), pd.DataFrame(), periods_per_year=ppy)
    assert metrics.sharpe == pytest.approx(expected)


def test_trade_statistics_by_hand():
    trades = pd.DataFrame(
        {
            "pnl": [10.0, -5.0, 20.0, -2.0],
            "return": [0.10, -0.05, 0.20, -0.02],
        }
    )
    metrics = compute(
        _equity([100, 101, 102, 103]),
        trades,
        periods_per_year=19_656,
        exposure=0.5,
        turnover=3.0,
    )
    assert metrics.n_trades == 4
    assert metrics.hit_rate == pytest.approx(0.5)
    assert metrics.avg_win == pytest.approx(0.15)
    assert metrics.avg_loss == pytest.approx(-0.035)
    assert metrics.profit_factor == pytest.approx(30 / 7)
    assert metrics.exposure == 0.5
    assert metrics.turnover == 3.0


def test_cagr_is_not_reported_for_a_sub_week_sample():
    metrics = compute(_equity([100, 120]), pd.DataFrame(), periods_per_year=19_656)
    assert math.isnan(metrics.cagr)


def test_empty_trades_give_zeroed_trade_stats():
    metrics = compute(_equity([100, 101, 100]), pd.DataFrame(), periods_per_year=19_656)
    assert metrics.n_trades == 0
    assert metrics.hit_rate == 0.0
    assert metrics.profit_factor == 0.0
