"""Allocators: sign handling, the leverage cap, and CLI-name construction."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from trader.backtest.allocator import (
    AllocatorContext,
    EqualWeight,
    FixedFraction,
    PassThrough,
    get_allocator,
)
from trader.strategies.base import PortfolioView


def _ctx(leverage_cap: float = 1.0) -> AllocatorContext:
    return AllocatorContext(
        histories={}, portfolio=PortfolioView(0.0, 0.0, {}), leverage_cap=leverage_cap
    )


def test_passthrough_uses_convictions_directly_within_the_cap():
    weights = PassThrough().weights({"A": 0.3, "B": -0.5}, _ctx(leverage_cap=1.0))
    assert weights == {"A": 0.3, "B": -0.5}


def test_passthrough_clamps_a_single_conviction_to_the_cap():
    assert PassThrough().weights({"A": -1.5}, _ctx(leverage_cap=1.0)) == {"A": -1.0}


def test_passthrough_rescales_when_gross_exposure_exceeds_the_cap():
    weights = PassThrough().weights({"A": 0.8, "B": 0.8}, _ctx(leverage_cap=1.0))
    assert weights["A"] == pytest.approx(0.5)
    assert weights["B"] == pytest.approx(0.5)


def test_equal_weight_splits_the_cap_across_active_names_keeping_sign():
    weights = EqualWeight().weights({"A": 0.8, "B": -0.2, "C": 0.0}, _ctx(leverage_cap=1.0))
    assert math.isclose(weights["A"], 0.5)
    assert math.isclose(weights["B"], -0.5)
    assert weights["C"] == 0.0


def test_equal_weight_with_no_active_convictions_is_all_zero():
    assert EqualWeight().weights({"A": 0.0, "B": 0.0}, _ctx()) == {"A": 0.0, "B": 0.0}


def test_fixed_fraction_scales_down_when_the_basket_exceeds_the_cap():
    weights = FixedFraction(0.5).weights({"A": 1.0, "B": 1.0, "C": 1.0}, _ctx(leverage_cap=1.0))
    # Three names at 0.5 gross = 1.5 -> scaled to 1.0 total.
    assert math.isclose(sum(abs(w) for w in weights.values()), 1.0)


def test_get_allocator_by_name_and_the_unknown_case():
    assert isinstance(get_allocator("equal-weight"), EqualWeight)
    with pytest.raises(KeyError, match="unknown allocator"):
        get_allocator("nope")


# --- vol-target -------------------------------------------------------------------


def _history(daily_vol: float, *, days: int = 60, weekends: bool = True, seed: int = 0):
    import numpy as np

    times = pd.date_range("2024-01-01", periods=days * 2, freq="D", tz="UTC")
    if not weekends:
        times = times[times.dayofweek < 5]
    times = times[:days]
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, daily_vol, days)
    rets = (rets - rets.mean()) / rets.std(ddof=1) * daily_vol  # exact sample vol
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"time": times, "close": close})


def _vctx(histories, cap=1.0):
    from trader.backtest.allocator import AllocatorContext

    return AllocatorContext(
        histories=histories,
        portfolio=PortfolioView(cash=1.0, equity=1.0, positions={}),
        leverage_cap=cap,
    )


def test_vol_target_sizes_inversely_to_volatility_and_splits_the_target():
    from trader.backtest.allocator import VolTarget

    calm, wild = _history(0.01), _history(0.04, seed=1)  # 365-day calendar: ~19% / ~76%
    alloc = VolTarget(target_ann_vol=0.2, lookback=30)
    w = alloc.weights({"calm": 1.0, "wild": -1.0}, _vctx({"calm": calm, "wild": wild}))
    sig_calm = calm["close"].iloc[-31:].pct_change().std() * (30 / (30 / 365.25)) ** 0.5
    assert w["calm"] == pytest.approx(0.2 / sig_calm / 2, rel=1e-6)  # half the target each
    assert w["wild"] < 0 and abs(w["wild"]) < w["calm"] / 3


def test_vol_target_counts_bars_per_year_from_the_calendar():
    """The same daily vol reads higher annualised for 24/7 crypto than weekday FX."""
    from trader.backtest.allocator import VolTarget

    alloc = VolTarget(target_ann_vol=0.1, lookback=20)
    crypto = alloc.weights({"x": 1.0}, _vctx({"x": _history(0.01, weekends=True)}))["x"]
    fx = alloc.weights({"x": 1.0}, _vctx({"x": _history(0.01, weekends=False)}))["x"]
    assert crypto < fx  # 365 vs ~260 bars/year: more annual vol, smaller weight


def test_vol_target_scales_with_conviction_and_caps_each_name():
    from trader.backtest.allocator import VolTarget

    calm = _history(0.001)  # tiny vol would ask for huge leverage
    alloc = VolTarget(target_ann_vol=0.2, lookback=20, max_weight=1.0)
    full = alloc.weights({"a": 1.0}, _vctx({"a": calm}, cap=5.0))["a"]
    assert full == 1.0
    half = VolTarget(target_ann_vol=0.02, lookback=20).weights(
        {"a": 0.5}, _vctx({"a": calm}, cap=5.0)
    )["a"]
    whole = VolTarget(target_ann_vol=0.02, lookback=20).weights(
        {"a": 1.0}, _vctx({"a": calm}, cap=5.0)
    )["a"]
    assert half == pytest.approx(whole / 2)
