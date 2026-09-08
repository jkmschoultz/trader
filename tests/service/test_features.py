"""``run_feature_build``: summaries off the lake, structured errors, persistence."""

from __future__ import annotations

import pytest

from trader.service.errors import InvalidRequest, SeriesNotStored
from trader.service.features import FeatureBuildSpec, run_feature_build


async def test_builds_price_features_from_the_lake(seeded_settings):
    spec = FeatureBuildSpec(symbols=["X"], uics=[211], asset_type="Stock", horizon="5m")
    (summary,) = await run_feature_build(seeded_settings, spec)

    assert summary["label"] == "X"
    assert summary["feature_set"] == "price_v1"
    assert summary["rows"] == 80
    assert summary["columns"] and "rsi" in summary["columns"]
    assert summary["warmup"] > 0
    assert 0 <= summary["nan_rows"] <= 80
    assert summary["persisted_files"] == 0


async def test_mtf_feature_set_pulls_in_context_columns(seeded_settings):
    spec = FeatureBuildSpec(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        horizon="5m",
        context_horizons=["15m"],
        feature_set="mtf_v1",
    )
    (summary,) = await run_feature_build(seeded_settings, spec)
    assert any(c.startswith("h15_") for c in summary["columns"])


async def test_progress_is_reported(seeded_settings):
    seen: list[float] = []
    spec = FeatureBuildSpec(symbols=["X"], uics=[211], asset_type="Stock", horizon="5m")
    await run_feature_build(seeded_settings, spec, progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0)


async def test_unknown_feature_set_is_an_invalid_request(seeded_settings):
    spec = FeatureBuildSpec(
        symbols=["X"], uics=[211], asset_type="Stock", horizon="5m", feature_set="nope"
    )
    with pytest.raises(InvalidRequest, match="unknown feature set"):
        await run_feature_build(seeded_settings, spec)


async def test_empty_lake_points_at_backfill(seeded_settings):
    spec = FeatureBuildSpec(symbols=["Y"], uics=[999], asset_type="Stock", horizon="5m")
    with pytest.raises(SeriesNotStored, match="trader data backfill"):
        await run_feature_build(seeded_settings, spec)


async def test_persist_writes_a_cache_that_round_trips(seeded_settings):
    from trader.data.lake import BarLake, SeriesKey
    from trader.features import compute_feature_frame
    from trader.features.cache import FeatureLake

    key = SeriesKey("Stock", 211, 5)
    spec = FeatureBuildSpec(
        symbols=["X"], uics=[211], asset_type="Stock", horizon="5m", persist=True
    )
    (summary,) = await run_feature_build(seeded_settings, spec)
    assert summary["persisted_files"] >= 1

    frame = BarLake(seeded_settings.data_dir).read(key)
    expected, fspec = compute_feature_frame(frame, feature_set="price_v1", base_horizon=5)
    cached = FeatureLake(seeded_settings.data_dir).read(key, fspec)

    assert list(cached.columns) == list(expected.columns)
    assert len(cached) == len(expected)


def test_spec_rejects_an_unserved_context_horizon():
    with pytest.raises(ValueError, match="not served by Saxo"):
        FeatureBuildSpec(symbols=["X"], horizon="5m", context_horizons=["7m"])
