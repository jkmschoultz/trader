"""Grid alignment, resampling, and gap detection on bar frames."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trader.data.bars import align_to_grid, clip, describe, gaps, is_aligned, off_grid, resample
from trader.data.lake import empty_frame, normalise


def _frame(start: datetime, count: int, horizon: int = 1, *, volume: float | None = 100.0):
    return normalise(
        pd.DataFrame(
            {
                "time": [start + timedelta(minutes=horizon * i) for i in range(count)],
                "open": [10.0 + i for i in range(count)],
                "high": [11.0 + i for i in range(count)],
                "low": [9.0 + i for i in range(count)],
                "close": [10.5 + i for i in range(count)],
                "volume": [volume] * count,
                "interest": [None] * count,
                "close_ask": [None] * count,
            }
        )
    )


# ------------------------------------------------------------------ alignment


def test_bars_on_the_grid_are_aligned():
    assert is_aligned(_frame(datetime(2024, 3, 1, 14, 30, tzinfo=UTC), 10, 5), 5)


def test_off_grid_finds_jittered_timestamps():
    frame = _frame(datetime(2024, 3, 1, 14, 30, tzinfo=UTC), 5, 5)
    frame.loc[2, "time"] = frame.loc[2, "time"] + pd.Timedelta(seconds=7)

    offenders = off_grid(frame, 5)
    assert len(offenders) == 1
    assert not is_aligned(frame, 5)


def test_align_to_grid_snaps_jitter_down():
    frame = _frame(datetime(2024, 3, 1, 14, 30, tzinfo=UTC), 5, 5)
    frame.loc[2, "time"] = frame.loc[2, "time"] + pd.Timedelta(seconds=7)

    snapped = align_to_grid(frame, 5)
    assert is_aligned(snapped, 5)
    assert len(snapped) == 5


def test_weekly_and_monthly_bars_are_exempt_from_the_grid_check():
    """Calendar boundaries do not sit on an epoch-anchored minute grid."""
    frame = _frame(datetime(2024, 3, 4, tzinfo=UTC), 4, 10080)
    assert off_grid(frame, 10080).empty


# ----------------------------------------------------------------- resampling


def test_resample_aggregates_ohlcv_correctly():
    frame = _frame(datetime(2024, 3, 1, 14, 0, tzinfo=UTC), 10, 1)
    out = resample(frame, source=1, target=5)

    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == frame.iloc[0]["open"]
    assert first["close"] == frame.iloc[4]["close"]
    assert first["high"] == frame.iloc[:5]["high"].max()
    assert first["low"] == frame.iloc[:5]["low"].min()
    assert first["volume"] == frame.iloc[:5]["volume"].sum()


def test_resample_drops_bins_with_no_source_bars():
    """An empty bin means the market was shut; a flat invented bar would lie."""
    early = _frame(datetime(2024, 3, 1, 14, 0, tzinfo=UTC), 5, 1)
    late = _frame(datetime(2024, 3, 1, 15, 0, tzinfo=UTC), 5, 1)
    frame = normalise(pd.concat([early, late], ignore_index=True))

    out = resample(frame, source=1, target=5)
    assert len(out) == 2
    assert (out["time"].diff().iloc[1]) == pd.Timedelta(hours=1)


def test_resample_keeps_missing_volume_missing():
    """sum() of an all-NaN bin is 0.0; 'no volume reported' is not 'zero traded'."""
    frame = _frame(datetime(2024, 3, 1, 14, 0, tzinfo=UTC), 10, 1, volume=None)
    out = resample(frame, source=1, target=5)
    assert out["volume"].isna().all()


def test_resample_to_the_same_horizon_is_a_no_op():
    frame = _frame(datetime(2024, 3, 1, 14, 0, tzinfo=UTC), 5, 5)
    assert resample(frame, source=5, target=5).equals(frame)


@pytest.mark.parametrize(("source", "target"), [(5, 7), (5, 1), (15, 20)])
def test_resample_rejects_targets_that_are_not_whole_multiples(source, target):
    with pytest.raises(ValueError, match="whole multiple"):
        resample(_frame(datetime(2024, 3, 1, tzinfo=UTC), 5, source), source=source, target=target)


def test_resample_refuses_weekly_and_monthly_bins():
    frame = _frame(datetime(2024, 3, 1, tzinfo=UTC), 20, 1440)
    with pytest.raises(ValueError, match="fixed-minute bins"):
        resample(frame, source=1440, target=10080)


def test_resample_of_an_empty_frame_is_empty():
    assert resample(empty_frame(), source=1, target=5).empty


# ----------------------------------------------------------------------- gaps


def test_gaps_finds_a_hole_and_counts_the_missing_bars():
    early = _frame(datetime(2024, 3, 1, 14, 0, tzinfo=UTC), 5, 1)
    late = _frame(datetime(2024, 3, 1, 14, 10, tzinfo=UTC), 5, 1)
    frame = normalise(pd.concat([early, late], ignore_index=True))

    found = gaps(frame, 1)
    assert len(found) == 1
    assert found[0].missing == 5
    assert found[0].after == datetime(2024, 3, 1, 14, 4, tzinfo=UTC)
    assert found[0].before == datetime(2024, 3, 1, 14, 10, tzinfo=UTC)


def test_a_continuous_series_has_no_gaps():
    assert gaps(_frame(datetime(2024, 3, 1, tzinfo=UTC), 100, 1), 1) == []


def test_gaps_can_ignore_short_holes():
    early = _frame(datetime(2024, 3, 1, 14, 0, tzinfo=UTC), 3, 1)
    late = _frame(datetime(2024, 3, 1, 14, 4, tzinfo=UTC), 3, 1)
    frame = normalise(pd.concat([early, late], ignore_index=True))

    assert len(gaps(frame, 1)) == 1
    assert gaps(frame, 1, min_missing=5) == []


def test_gaps_needs_at_least_two_bars():
    assert gaps(_frame(datetime(2024, 3, 1, tzinfo=UTC), 1), 1) == []
    assert gaps(empty_frame(), 1) == []


# ------------------------------------------------------------ clip & describe


def test_clip_is_inclusive_at_both_ends():
    start = datetime(2024, 3, 1, tzinfo=UTC)
    frame = _frame(start, 10, 1)
    out = clip(frame, start=start + timedelta(minutes=2), end=start + timedelta(minutes=5))
    assert len(out) == 4


def test_describe_summarises_a_series():
    frame = _frame(datetime(2024, 3, 1, tzinfo=UTC), 60, 1)
    summary = describe(frame, 1)

    assert summary["rows"] == 60
    assert summary["gaps"] == 0
    assert summary["density"] == 1.0
    assert summary["off_grid"] == 0
    assert summary["has_volume"] is True
    assert summary["has_spread"] is False


def test_describe_of_an_empty_frame_reports_no_rows():
    assert describe(empty_frame(), 1) == {"rows": 0}
