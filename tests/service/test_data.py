"""``lake_series`` and ``read_bars``: coverage listing and chart-ready slices."""

from __future__ import annotations

import pytest

from trader.service.data import lake_series, read_bars
from trader.service.errors import SeriesNotStored


def test_lists_stored_series_with_coverage(seeded_settings):
    series = lake_series(seeded_settings)
    assert len(series) == 1
    info = series[0]
    assert (info.asset_type, info.uic, info.horizon) == ("Stock", 211, 5)
    assert info.horizon_label == "5m"
    assert info.rows == 80
    assert info.first < info.last


def test_read_bars_returns_unix_seconds_and_ohlcv(seeded_settings):
    out = read_bars(seeded_settings, asset_type="Stock", uic=211, horizon=5)
    assert out["rows"] == 80
    assert out["decimated"] is False
    first = out["bars"][0]
    assert set(first) == {"time", "open", "high", "low", "close", "volume"}
    assert isinstance(first["time"], int)
    assert out["bars"] == sorted(out["bars"], key=lambda b: b["time"])


def test_read_bars_decimates_to_the_point_cap(seeded_settings):
    out = read_bars(seeded_settings, asset_type="Stock", uic=211, horizon=5, max_points=10)
    assert out["decimated"] is True
    assert out["returned"] <= 11  # cap plus the always-kept last bar
    assert out["rows"] == 80
    assert out["bars"][-1]["time"] > out["bars"][0]["time"]


def test_read_bars_on_a_missing_series_raises(seeded_settings):
    with pytest.raises(SeriesNotStored, match="trader data backfill"):
        read_bars(seeded_settings, asset_type="Stock", uic=404, horizon=5)
