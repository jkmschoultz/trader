"""Metrics: annualisation factor, and the equity/trade statistics by hand."""

from __future__ import annotations

import math
import statistics

import numpy as np
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


# --- annualisation from the calendar --------------------------------------------


def _daily_curve(days_per_week: int, years: float, growth: float) -> pd.Series:
    """Equity compounding smoothly to ``growth`` over ``years`` of daily bars."""
    all_days = pd.date_range("2020-01-01", periods=int(round(365.25 * years)) + 1, tz="UTC")
    times = all_days[all_days.dayofweek < days_per_week]
    n = len(times)
    rng = np.random.default_rng(0)
    steps = np.log(growth) / (n - 1) + rng.normal(0, 0.01, n - 1)
    steps -= steps.mean() - np.log(growth) / (n - 1)  # exact total growth
    return pd.Series(100.0 * np.exp(np.r_[0.0, np.cumsum(steps)]), index=times)


@pytest.mark.parametrize("days_per_week", [7, 5])  # crypto trades weekends, FX does not
def test_cagr_uses_calendar_years_whatever_the_bar_count(days_per_week):
    curve = _daily_curve(days_per_week, years=4.0, growth=4.0)
    m = compute(curve, pd.DataFrame(), periods_per_year=252)  # 252 is wrong for 7-day
    assert m.cagr == pytest.approx(4.0 ** (1 / 4.0) - 1, rel=2e-3)


def test_sharpe_scales_by_observed_bars_per_year():
    curve = _daily_curve(7, years=4.0, growth=2.0)
    rets = curve.pct_change().dropna()
    m = compute(curve, pd.DataFrame(), periods_per_year=252)
    per_year = len(rets) / 4.0  # ~365
    assert m.sharpe == pytest.approx(rets.mean() / rets.std() * np.sqrt(per_year), rel=2e-3)
    assert m.ann_vol == pytest.approx(rets.std() * np.sqrt(per_year), rel=2e-3)


def test_short_runs_keep_the_assumed_periods_per_year():
    curve = _daily_curve(7, years=0.1, growth=1.05)
    rets = curve.pct_change().dropna()
    m = compute(curve, pd.DataFrame(), periods_per_year=252)
    assert m.ann_vol == pytest.approx(rets.std() * np.sqrt(252))
