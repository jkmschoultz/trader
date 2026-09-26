"""Time-series momentum: blended lookback signs, the rebalance schedule, long-only."""

from __future__ import annotations

import numpy as np
import pytest

from trader.strategies.base import Flat, Hold, Target
from trader.strategies.tsmom import TimeSeriesMomentum


def test_warmup_covers_the_longest_lookback_plus_skip():
    assert TimeSeriesMomentum(lookbacks="5,20", skip=3).warmup == 24


def test_lookbacks_accept_a_string_list_or_int():
    assert TimeSeriesMomentum(lookbacks="21, 63").lookbacks == (21, 63)
    assert TimeSeriesMomentum(lookbacks=[5, 10]).lookbacks == (5, 10)
    assert TimeSeriesMomentum(lookbacks=10).lookbacks == (10,)
    assert TimeSeriesMomentum(lookbacks="21/63/126").lookbacks == (21, 63, 126)  # grid-safe
    with pytest.raises(ValueError):
        TimeSeriesMomentum(lookbacks="0,5")


def test_uptrend_goes_fully_long(make_bars, make_context):
    strat = TimeSeriesMomentum(lookbacks="3,6", rebalance=1)
    decision = strat.on_bar(make_context(make_bars(list(np.arange(10.0, 20.0)))))
    assert isinstance(decision, Target) and decision.weight == 1.0


def test_mixed_lookbacks_blend_to_a_partial_conviction(make_bars, make_context):
    # up over the last 3 bars, down over the last 6
    closes = [20, 19, 18, 17, 10, 11, 12, 13]
    strat = TimeSeriesMomentum(lookbacks="3,6", rebalance=1)
    decision = strat.on_bar(make_context(make_bars(closes)))
    assert isinstance(decision, Flat)  # +1 and -1 average to 0
    strat = TimeSeriesMomentum(lookbacks="2,3,6", rebalance=1)
    decision = strat.on_bar(make_context(make_bars(closes)))
    assert decision.weight == pytest.approx(1 / 3)


def test_long_only_goes_flat_in_a_downtrend(make_bars, make_context):
    closes = list(np.arange(20.0, 10.0, -1.0))
    assert (
        TimeSeriesMomentum(lookbacks="3", rebalance=1)
        .on_bar(make_context(make_bars(closes)))
        .weight
        == -1.0
    )
    assert isinstance(
        TimeSeriesMomentum(lookbacks="3", rebalance=1, long_only=True).on_bar(
            make_context(make_bars(closes))
        ),
        Flat,
    )


def test_trades_only_on_the_rebalance_schedule(make_bars, make_context):
    strat = TimeSeriesMomentum(lookbacks="3", rebalance=5)  # warmup 4
    closes = list(np.arange(10.0, 30.0))
    acted = [
        not isinstance(strat.on_bar(make_context(make_bars(closes[:n]))), Hold)
        for n in range(4, 16)
    ]
    assert [i + 4 for i, a in enumerate(acted) if a] == [4, 9, 14]


def test_skip_measures_the_trend_before_the_latest_bars(make_bars, make_context):
    closes = [10, 11, 12, 13, 14, 5]  # a crash on the last bar
    assert (
        TimeSeriesMomentum(lookbacks="3", rebalance=1)
        .on_bar(make_context(make_bars(closes)))
        .weight
        == -1.0
    )
    assert (
        TimeSeriesMomentum(lookbacks="3", rebalance=1, skip=1)
        .on_bar(make_context(make_bars(closes)))
        .weight
        == 1.0
    )
