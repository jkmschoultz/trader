"""The strategy / allocator / feature-set / model catalogue the UI builds forms from."""

from __future__ import annotations

import json

from trader.config import Settings
from trader.service import (
    list_allocators,
    list_feature_sets,
    list_models,
    list_strategies,
)


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


def test_lists_feature_sets_with_their_columns():
    by_name = {f.name: f for f in list_feature_sets()}
    assert {"price_v1", "mtf_v1"} <= by_name.keys()

    price = by_name["price_v1"]
    assert price.summary
    assert "rsi" in price.columns
    assert price.context_capable is False
    assert by_name["mtf_v1"].context_capable is True


def test_list_models_is_empty_then_populated(tmp_path):
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    assert list_models(settings) == []

    model_dir = settings.models_dir / "lstm-20240101-000000-abcd1234"
    model_dir.mkdir(parents=True)
    (model_dir / "manifest.json").write_text(
        json.dumps(
            {
                "id": "lstm-20240101-000000-abcd1234",
                "name": "lstm",
                "created_at": "2024-01-01T00:00:00+00:00",
                "base_horizon": 5,
                "context_horizons": [15],
                "feature_set": "mtf_v1",
                "feature_digest": "abc",
                "window": 32,
                "barriers": {"stop": 0.005, "take": 0.01, "max_bars": 24},
                "metrics": {"val_macro_f1": 0.5},
                "data": {"symbols": ["AAPL"], "asset_type": "Stock"},
            }
        )
    )

    (summary,) = list_models(settings)
    assert summary.id == "lstm-20240101-000000-abcd1234"
    assert summary.feature_set == "mtf_v1"
    assert summary.context_horizons == [15]
    assert summary.symbols == ["AAPL"]
