"""Allocators: sign handling, the leverage cap, and CLI-name construction."""

from __future__ import annotations

import math

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
