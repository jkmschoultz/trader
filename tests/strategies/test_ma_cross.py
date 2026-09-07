"""MA crossover: warmup gate, the long/short flip, and long-only behaviour."""

from __future__ import annotations

import pytest

from trader.strategies.base import Flat, Hold, Target
from trader.strategies.ma_cross import MACrossStrategy


def test_fast_must_be_shorter_than_slow():
    with pytest.raises(ValueError, match="shorter than slow"):
        MACrossStrategy(fast=30, slow=10)


def test_holds_until_it_has_slow_bars(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5)
    assert strat.warmup == 5
    assert isinstance(strat.on_bar(make_context(make_bars([10, 11, 12]))), Hold)


def test_goes_long_when_fast_is_above_slow(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5)
    decision = strat.on_bar(make_context(make_bars([10, 10, 10, 12, 14])))
    assert isinstance(decision, Target)
    assert decision.weight == 1.0


def test_goes_short_when_fast_is_below_slow(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5)
    decision = strat.on_bar(make_context(make_bars([14, 12, 10, 10, 9])))
    assert isinstance(decision, Target)
    assert decision.weight == -1.0


def test_long_only_goes_flat_instead_of_short(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5, long_only=True)
    decision = strat.on_bar(make_context(make_bars([14, 12, 10, 10, 9])))
    assert isinstance(decision, Flat)
