"""The Parquet bar lake: schema, partitioning, idempotent merges, coverage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trader.data.lake import (
    BAR_COLUMNS,
    TIME_DTYPE,
    BarLake,
    SeriesKey,
    bars_to_frame,
    empty_frame,
    normalise,
    period_key,
)
from trader.saxo.charts import Bar

KEY = SeriesKey("Stock", 211, 1)


def _frame(start: datetime, count: int, horizon: int = 1, close_from: float = 100.0):
    return pd.DataFrame(
        {
            "time": [start + timedelta(minutes=horizon * i) for i in range(count)],
            "open": [close_from + i for i in range(count)],
            "high": [close_from + 1 + i for i in range(count)],
            "low": [close_from - 1 + i for i in range(count)],
            "close": [close_from + 0.5 + i for i in range(count)],
            "volume": [1000 + i for i in range(count)],
            "interest": [None] * count,
            "close_ask": [None] * count,
        }
    )


# ------------------------------------------------------------------ schema


def test_empty_frame_has_the_canonical_schema():
    frame = empty_frame()
    assert tuple(frame.columns) == BAR_COLUMNS
    assert frame.empty


def test_normalise_fills_columns_a_partial_frame_is_missing():
    """FX bars have no volume; exchange bars have no close_ask. Both must merge."""
    partial = pd.DataFrame(
        {
            "time": [datetime(2024, 3, 1, tzinfo=UTC)],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
        }
    )
    out = normalise(partial)
    assert tuple(out.columns) == BAR_COLUMNS
    assert out["volume"].isna().all()
    assert out["close_ask"].isna().all()
    assert out["trading_state"].isna().all()


def test_trading_state_dtype_is_the_same_whether_or_not_the_column_was_present():
    """A frame missing the column entirely must normalise identically to one
    that has it, filled with nulls -- otherwise concatenating the two upcasts
    silently, the same failure mode TIME_DTYPE guards against."""
    without_column = normalise(
        pd.DataFrame(
            {
                "time": [datetime(2024, 3, 1, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
            }
        )
    )
    with_column_all_null = normalise(
        pd.DataFrame(
            {
                "time": [datetime(2024, 3, 1, tzinfo=UTC)],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "trading_state": [None],
            }
        )
    )
    assert without_column["trading_state"].dtype == with_column_all_null["trading_state"].dtype


def test_normalise_sorts_and_deduplicates_keeping_the_later_row():
    """A refetched bar is a revision, so the incoming value must win."""
    stamp = datetime(2024, 3, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "time": [stamp + timedelta(minutes=1), stamp, stamp],
            "open": [2.0, 1.0, 1.0],
            "high": [2.0, 1.0, 1.0],
            "low": [2.0, 1.0, 1.0],
            "close": [2.0, 1.0, 9.9],
            "volume": [1, 1, 1],
            "interest": [None] * 3,
            "close_ask": [None] * 3,
        }
    )
    out = normalise(frame)
    assert len(out) == 2
    assert list(out["time"]) == sorted(out["time"])
    assert out.iloc[0]["close"] == 9.9


def test_normalise_makes_naive_timestamps_utc():
    out = normalise(_frame(datetime(2024, 3, 1), 3))
    assert str(out["time"].dtype) == TIME_DTYPE


def test_time_dtype_is_pinned_across_every_entry_point(tmp_path):
    """Frames that disagree on the timestamp unit upcast silently when concatenated."""
    lake = BarLake(tmp_path)
    from_python = normalise(_frame(datetime(2024, 3, 1, tzinfo=UTC), 3))
    from_bars = bars_to_frame(
        [Bar(Time=datetime(2024, 3, 1, tzinfo=UTC), open=1.0, high=1.0, low=1.0, close=1.0)]
    )
    lake.write(KEY, from_python)
    from_parquet = lake.read(KEY)

    units = {str(f["time"].dtype) for f in (empty_frame(), from_python, from_bars, from_parquet)}
    assert units == {TIME_DTYPE}


def test_bars_to_frame_carries_the_spread_column():
    bars = [
        Bar(
            Time=datetime(2024, 3, 1, tzinfo=UTC),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0855,
            close_ask=1.0856,
        )
    ]
    frame = bars_to_frame(bars)
    assert frame.iloc[0]["close_ask"] == pytest.approx(1.0856)
    assert pd.isna(frame.iloc[0]["volume"])


def test_bars_to_frame_of_nothing_is_an_empty_canonical_frame():
    assert tuple(bars_to_frame([]).columns) == BAR_COLUMNS


def test_bars_to_frame_carries_the_trading_state():
    bars = [
        Bar(
            Time=datetime(2024, 3, 1, tzinfo=UTC),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            trading_state="Automated",
        )
    ]
    frame = bars_to_frame(bars)
    assert frame.iloc[0]["trading_state"] == "Automated"


def test_trading_state_survives_a_parquet_round_trip(tmp_path):
    lake = BarLake(tmp_path)
    bars = [
        Bar(
            Time=datetime(2024, 3, 1, tzinfo=UTC),
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.0,
            trading_state="Automated",
        ),
        Bar(Time=datetime(2024, 3, 1, 0, 1, tzinfo=UTC), open=1.0, high=1.0, low=1.0, close=1.0),
    ]
    lake.write(KEY, bars)

    out = lake.read(KEY)
    assert out.iloc[0]["trading_state"] == "Automated"
    assert pd.isna(out.iloc[1]["trading_state"])


# ------------------------------------------------------------- partitioning


@pytest.mark.parametrize(
    ("horizon", "expected"),
    [
        (1, "2024-03"),
        (5, "2024-03"),
        (30, "2024-03"),
        (60, "2024"),
        (240, "2024"),
        (1440, "all"),
        (10080, "all"),
    ],
)
def test_partition_width_scales_with_the_horizon(horizon, expected):
    assert period_key(datetime(2024, 3, 15, tzinfo=UTC), horizon) == expected


def test_a_page_straddling_a_month_boundary_writes_both_partitions(tmp_path):
    lake = BarLake(tmp_path)
    # Ten minutes either side of midnight on 1 April.
    lake.write(KEY, _frame(datetime(2024, 3, 31, 23, 55, tzinfo=UTC), 20))

    names = sorted(p.name for p in lake.files(KEY))
    assert names == ["2024-03.parquet", "2024-04.parquet"]
    assert len(lake.read(KEY)) == 20


# ------------------------------------------------------------------- writes


def test_write_then_read_round_trips_through_parquet(tmp_path):
    lake = BarLake(tmp_path)
    written = lake.write(KEY, _frame(datetime(2024, 3, 1, tzinfo=UTC), 100))

    assert written.rows_added == 100
    assert written.rows_updated == 0

    out = lake.read(KEY)
    assert len(out) == 100
    assert tuple(out.columns) == BAR_COLUMNS
    assert str(out["time"].dtype) == TIME_DTYPE


def test_rewriting_the_same_bars_adds_nothing(tmp_path):
    """Idempotence is what makes an interrupted backfill safe to just re-run."""
    lake = BarLake(tmp_path)
    frame = _frame(datetime(2024, 3, 1, tzinfo=UTC), 50)

    lake.write(KEY, frame)
    second = lake.write(KEY, frame)

    assert second.rows_added == 0
    assert second.rows_updated == 50
    assert len(lake.read(KEY)) == 50


def test_overlapping_writes_merge_rather_than_replace(tmp_path):
    lake = BarLake(tmp_path)
    start = datetime(2024, 3, 1, tzinfo=UTC)

    lake.write(KEY, _frame(start, 30))
    result = lake.write(KEY, _frame(start + timedelta(minutes=20), 30))

    assert result.rows_added == 20
    assert result.rows_updated == 10
    out = lake.read(KEY)
    assert len(out) == 50
    assert list(out["time"]) == sorted(out["time"])


def test_a_refetched_bar_overwrites_the_stored_one(tmp_path):
    lake = BarLake(tmp_path)
    start = datetime(2024, 3, 1, tzinfo=UTC)
    lake.write(KEY, _frame(start, 5))

    revised = _frame(start, 5)
    revised.loc[2, "close"] = 999.0
    lake.write(KEY, revised)

    assert lake.read(KEY).iloc[2]["close"] == 999.0


def test_writing_nothing_touches_no_files(tmp_path):
    lake = BarLake(tmp_path)
    result = lake.write(KEY, empty_frame())
    assert result == type(result)(0, 0, 0, 0)
    assert lake.files(KEY) == []


def test_write_accepts_bar_objects_directly(tmp_path):
    lake = BarLake(tmp_path)
    bars = [
        Bar(
            Time=datetime(2024, 3, 1, tzinfo=UTC) + timedelta(minutes=i),
            open=1.0,
            high=2.0,
            low=0.5,
            close=1.5,
            volume=10,
        )
        for i in range(5)
    ]
    assert lake.write(KEY, bars).rows_added == 5


# ------------------------------------------------------- reading and coverage


def test_read_clips_to_a_window(tmp_path):
    lake = BarLake(tmp_path)
    start = datetime(2024, 3, 1, tzinfo=UTC)
    lake.write(KEY, _frame(start, 60))

    out = lake.read(KEY, start=start + timedelta(minutes=10), end=start + timedelta(minutes=19))
    assert len(out) == 10
    assert out["time"].iloc[0] == pd.Timestamp(start + timedelta(minutes=10))


def test_read_of_an_unknown_series_is_empty_not_an_error(tmp_path):
    out = BarLake(tmp_path).read(SeriesKey("Stock", 999, 1))
    assert out.empty
    assert tuple(out.columns) == BAR_COLUMNS


def test_coverage_spans_every_partition(tmp_path):
    lake = BarLake(tmp_path)
    lake.write(KEY, _frame(datetime(2024, 3, 31, 23, 50, tzinfo=UTC), 30))

    coverage = lake.coverage(KEY)
    assert coverage.rows == 30
    assert coverage.files == 2
    assert coverage.first == datetime(2024, 3, 31, 23, 50, tzinfo=UTC)
    assert coverage.last == datetime(2024, 4, 1, 0, 19, tzinfo=UTC)


def test_coverage_of_an_unknown_series_is_none(tmp_path):
    assert BarLake(tmp_path).coverage(SeriesKey("Stock", 999, 1)) is None


def test_series_lists_every_stored_triple(tmp_path):
    lake = BarLake(tmp_path)
    start = datetime(2024, 3, 1, tzinfo=UTC)
    lake.write(SeriesKey("Stock", 211, 1), _frame(start, 5))
    lake.write(SeriesKey("Stock", 211, 60), _frame(start, 5, horizon=60))
    lake.write(SeriesKey("FxSpot", 21, 5), _frame(start, 5, horizon=5))

    assert lake.series() == [
        SeriesKey("FxSpot", 21, 5),
        SeriesKey("Stock", 211, 1),
        SeriesKey("Stock", 211, 60),
    ]


def test_drop_removes_a_series(tmp_path):
    lake = BarLake(tmp_path)
    lake.write(KEY, _frame(datetime(2024, 3, 1, tzinfo=UTC), 10))

    assert lake.drop(KEY) == 1
    assert lake.coverage(KEY) is None
    assert lake.series() == []


def test_a_failed_write_strands_no_temp_file_and_keeps_the_old_one(tmp_path, monkeypatch):
    """A write that dies mid-Parquet must leave the previous partition readable."""
    lake = BarLake(tmp_path)
    start = datetime(2024, 3, 1, tzinfo=UTC)
    lake.write(KEY, _frame(start, 5))

    def explode(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", explode)
    with pytest.raises(OSError):
        lake.write(KEY, _frame(start + timedelta(minutes=5), 5))

    monkeypatch.undo()
    assert list(lake.directory(KEY).glob("*.tmp")) == []
    assert len(lake.read(KEY)) == 5, "the previous file survived intact"
