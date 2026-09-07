"""The cost model: spread source, slippage direction, and commission floor."""

from __future__ import annotations

import math

from trader.backtest import CostModel
from trader.strategies.base import Side


def test_spread_comes_from_the_bar_ask_when_present():
    model = CostModel(half_spread_bps=1.0)  # would be 0.0001; the ask should win
    bar = {"close": 100.0, "close_ask": 100.2}
    # (100.2 - 100) / 100 / 2 = 0.001
    assert math.isclose(model.half_spread_frac(bar), 0.001)


def test_spread_falls_back_to_bps_without_an_ask():
    model = CostModel(half_spread_bps=2.0)
    assert math.isclose(model.half_spread_frac({"close": 100.0, "close_ask": math.nan}), 0.0002)
    assert math.isclose(model.half_spread_frac(None), 0.0002)


def test_slippage_and_spread_push_the_fill_against_the_trade():
    model = CostModel(half_spread_bps=10.0, slippage_bps=5.0)  # 0.0015 total
    buy = model.fill_price(Side.BUY, 100.0, None)
    sell = model.fill_price(Side.SELL, 100.0, None)
    assert math.isclose(buy, 100.15)
    assert math.isclose(sell, 99.85)


def test_commission_takes_the_per_fill_minimum_when_it_bites():
    model = CostModel(commission_bps=1.0, commission_min=5.0)
    # 10 units * $2 * 1bp = $0.002 -> floored to the $5 minimum.
    assert model.commission(10, 2.0) == 5.0
    # 10_000 units * $50 * 1bp = $50 -> above the floor, so unchanged.
    assert math.isclose(model.commission(10_000, 50.0), 50.0)


def test_a_zero_size_fill_costs_nothing():
    assert CostModel(commission_min=5.0).commission(0, 100.0) == 0.0
