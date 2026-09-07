"""The strategy / allocator catalogue the UI builds its forms from."""

from __future__ import annotations

from trader.service import list_allocators, list_strategies


def test_lists_the_builtin_strategies_with_their_params():
    by_name = {s.name: s for s in list_strategies()}
    assert {"ma_cross", "orb"} <= by_name.keys()

    ma = by_name["ma_cross"]
    assert ma.summary
    params = {p.name: p for p in ma.params}
    assert params["fast"].default == 10
    assert params["fast"].type == "int"
    assert params["long_only"].default is False
    assert all(not p.required for p in ma.params)


def test_orb_exposes_an_optional_max_bars():
    orb = next(s for s in list_strategies() if s.name == "orb")
    max_bars = next(p for p in orb.params if p.name == "max_bars")
    assert max_bars.default is None


def test_lists_allocators_by_name():
    names = {a.name for a in list_allocators()}
    assert {"equal-weight", "passthrough", "vol-target", "fixed-fraction"} == names
