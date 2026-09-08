"""``run_labelling``: class-balance summary off the lake, structured errors."""

from __future__ import annotations

import pytest

from trader.service.errors import SeriesNotStored
from trader.service.labels import LabelSpec, run_labelling


async def test_summarises_the_class_balance(seeded_settings):
    spec = LabelSpec(
        symbols=["X"],
        uics=[211],
        asset_type="Stock",
        horizon="5m",
        stop=0.01,
        take=0.01,
        max_bars=6,
    )
    summary = await run_labelling(seeded_settings, spec)

    assert summary["n_events"] > 0
    assert set(summary["class_counts"]) == {-1, 0, 1}
    assert sum(summary["class_counts"].values()) == summary["n_events"]
    assert set(summary["barrier_breakdown"]) == {"stop", "take", "time"}
    assert summary["barriers"] == {"stop": 0.01, "take": 0.01, "max_bars": 6}
    assert "X" in summary["per_symbol"]
    assert summary["per_symbol"]["X"]["key"] == "Stock:211@5m"


async def test_progress_is_reported(seeded_settings):
    seen: list[float] = []
    spec = LabelSpec(symbols=["X"], uics=[211], asset_type="Stock", horizon="5m")
    await run_labelling(seeded_settings, spec, progress=seen.append)
    assert seen and seen[-1] == pytest.approx(1.0)


async def test_empty_lake_points_at_backfill(seeded_settings):
    spec = LabelSpec(symbols=["Y"], uics=[999], asset_type="Stock", horizon="5m")
    with pytest.raises(SeriesNotStored, match="trader data backfill"):
        await run_labelling(seeded_settings, spec)


def test_spec_rejects_a_bad_horizon():
    with pytest.raises(ValueError, match="not served by Saxo"):
        LabelSpec(symbols=["X"], horizon="7m")


def test_spec_rejects_a_non_positive_max_bars():
    with pytest.raises(ValueError, match="max_bars"):
        LabelSpec(symbols=["X"], max_bars=0)


def test_spec_rejects_a_non_positive_stop():
    with pytest.raises(ValueError, match="stop"):
        LabelSpec(symbols=["X"], stop=0.0)
