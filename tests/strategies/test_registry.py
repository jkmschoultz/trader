"""The strategy name registry: lookup, listing, and its two failure modes."""

from __future__ import annotations

import pytest

from trader.strategies import InstrumentStrategy, available, get_strategy, register
from trader.strategies.base import Flat
from trader.strategies.registry import UnknownStrategy


def test_the_builtins_are_registered():
    assert {"ma_cross", "orb"} <= set(available())


def test_get_strategy_returns_the_class():
    cls = get_strategy("ma_cross")
    assert cls.name == "ma_cross"
    assert issubclass(cls, InstrumentStrategy)


def test_unknown_name_lists_what_is_available():
    with pytest.raises(UnknownStrategy) as excinfo:
        get_strategy("does_not_exist")
    assert "ma_cross" in str(excinfo.value)


def test_registering_a_taken_name_is_an_error():
    with pytest.raises(ValueError, match="already registered"):

        @register("orb")
        class _Clashing(InstrumentStrategy):
            def on_bar(self, ctx):
                return Flat()


def test_register_sets_the_name_and_is_retrievable():
    @register("unit_test_dummy")
    class _Dummy(InstrumentStrategy):
        def on_bar(self, ctx):
            return Flat()

    try:
        assert _Dummy.name == "unit_test_dummy"
        assert get_strategy("unit_test_dummy") is _Dummy
    finally:
        from trader.strategies.registry import _STRATEGIES  # noqa: PLC0415

        _STRATEGIES.pop("unit_test_dummy", None)
