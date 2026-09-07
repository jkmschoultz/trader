"""Session calendars: Saxo's live view, and hours inferred from bar data."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trader.data.bars import gaps
from trader.data.calendars import (
    SessionCalendar,
    drop_session_breaks,
    filter_regular_hours,
    infer_regular_hours,
    resolve_timezone,
    session_table,
    trading_days,
)
from trader.data.lake import normalise
from trader.saxo.exchanges import Exchange

NEW_YORK = ZoneInfo("America/New_York")


def _exchange(**overrides) -> Exchange:
    # Shaped like the real SIM response: a small integer TimeZone code plus an
    # abbreviation and the zone's current offset -- Saxo has no zone-name field.
    payload = {
        "ExchangeId": "NASDAQ",
        "Name": "NASDAQ",
        "CountryCode": "US",
        "Currency": "USD",
        "TimeZone": 3,
        "TimeZoneAbbreviation": "EDT",
        "TimeZoneOffset": "-04:00:00",
        "AllDay": False,
        "ExchangeSessions": [
            {
                "State": "Closed",
                "StartTime": "2024-03-01T00:00:00Z",
                "EndTime": "2024-03-01T14:30:00Z",
            },
            {
                "State": "AutomatedTrading",
                "StartTime": "2024-03-01T14:30:00Z",
                "EndTime": "2024-03-01T21:00:00Z",
            },
            {
                "State": "Closed",
                "StartTime": "2024-03-01T21:00:00Z",
                "EndTime": "2024-03-04T14:30:00Z",
            },
            {
                "State": "AutomatedTrading",
                "StartTime": "2024-03-04T14:30:00Z",
                "EndTime": "2024-03-04T21:00:00Z",
            },
        ],
    }
    payload.update(overrides)
    return Exchange.model_validate(payload)


def _session_frame(days: int = 10, *, horizon: int = 30) -> pd.DataFrame:
    """A US-equities-shaped series: 09:30-16:00 New York, weekdays only."""
    rows = []
    day = datetime(2024, 3, 4, tzinfo=NEW_YORK)  # a Monday
    added = 0
    while added < days:
        if day.weekday() < 5:
            open_at = day.replace(hour=9, minute=30)
            bars = (6 * 60 + 30) // horizon
            for i in range(bars):
                rows.append(open_at + timedelta(minutes=horizon * i))
            added += 1
        day += timedelta(days=1)

    return normalise(
        pd.DataFrame(
            {
                "time": [t.astimezone(UTC) for t in rows],
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 100.0,
                "interest": None,
                "close_ask": None,
            }
        )
    )


# ------------------------------------------------------------------ timezones


def test_a_known_saxo_timezone_code_maps_to_iana():
    assert resolve_timezone(_exchange()) == NEW_YORK


def test_a_different_known_code_maps_to_its_own_zone():
    assert resolve_timezone(_exchange(TimeZone=10, TimeZoneAbbreviation="JST")) == ZoneInfo(
        "Asia/Tokyo"
    )


def test_an_unmapped_code_falls_back_to_the_reported_offset():
    """Wrong across DST, but better than pretending the venue is in UTC."""
    zone = resolve_timezone(
        _exchange(TimeZone=9999, TimeZoneAbbreviation="XYZ", TimeZoneOffset="-03:30:00")
    )
    assert zone.utcoffset(None) == timedelta(hours=-3, minutes=-30)


def test_no_zone_information_at_all_falls_back_to_utc():
    assert resolve_timezone(_exchange(TimeZone=None, TimeZoneOffset="")) is UTC


# ----------------------------------------------------------- SessionCalendar


def test_is_open_during_continuous_trading():
    calendar = SessionCalendar.from_exchange(_exchange())
    assert calendar.is_open(datetime(2024, 3, 1, 15, 0, tzinfo=UTC)) is True


def test_is_closed_outside_the_trading_session():
    calendar = SessionCalendar.from_exchange(_exchange())
    assert calendar.is_open(datetime(2024, 3, 1, 22, 0, tzinfo=UTC)) is False


def test_unknown_rather_than_a_guess_outside_the_published_window():
    """Saxo publishes a rolling window; beyond it, 'closed' would be a fabrication."""
    calendar = SessionCalendar.from_exchange(_exchange())
    assert calendar.is_open(datetime(2025, 6, 1, 15, 0, tzinfo=UTC)) is None
    assert calendar.state_at(datetime(2025, 6, 1, 15, 0, tzinfo=UTC)) is None


def test_an_all_day_venue_with_no_sessions_is_open():
    calendar = SessionCalendar.from_exchange(_exchange(AllDay=True, ExchangeSessions=[]))
    assert calendar.is_open(datetime(2024, 3, 1, 3, 0, tzinfo=UTC)) is True


def test_next_open_skips_non_trading_states():
    calendar = SessionCalendar.from_exchange(_exchange())
    assert calendar.next_open(datetime(2024, 3, 1, 22, 0, tzinfo=UTC)) == datetime(
        2024, 3, 4, 14, 30, tzinfo=UTC
    )


def test_next_transition_reports_the_state_it_begins():
    calendar = SessionCalendar.from_exchange(_exchange())
    moment, state = calendar.next_transition(datetime(2024, 3, 1, 15, 0, tzinfo=UTC))
    assert moment == datetime(2024, 3, 1, 21, 0, tzinfo=UTC)
    assert state == "Closed"


def test_covers_reports_the_published_span():
    first, last = SessionCalendar.from_exchange(_exchange()).covers
    assert first == datetime(2024, 3, 1, tzinfo=UTC)
    assert last == datetime(2024, 3, 4, 21, 0, tzinfo=UTC)


# --------------------------------------------------------------- session_table


def test_session_table_has_one_row_per_local_trading_day():
    table = session_table(_session_frame(days=10), NEW_YORK)
    assert len(table) == 10
    assert table["first"].iloc[0] == time(9, 30)
    assert (table["bars"] == 13).all()


def test_session_table_of_an_empty_frame_has_the_right_columns():
    from trader.data.lake import empty_frame

    table = session_table(empty_frame(), NEW_YORK)
    assert table.empty
    assert "bars" in table.columns


# ---------------------------------------------------------- inferred sessions


def test_infer_regular_hours_recovers_the_us_equity_session():
    hours = infer_regular_hours(_session_frame(days=10, horizon=30), NEW_YORK)
    assert hours.open_time == time(9, 30)
    # The last *bar* opens at 15:30 and covers the half hour to the 16:00 close.
    assert hours.close_time == time(15, 30)
    assert hours.weekdays == frozenset({0, 1, 2, 3, 4})
    assert hours.days_observed == 10


def test_a_single_half_day_does_not_move_the_inferred_close():
    """The mode ignores an early close; the extremes would not."""
    frame = _session_frame(days=10, horizon=30)
    local = frame["time"].dt.tz_convert(NEW_YORK)
    half_day = local.dt.date == datetime(2024, 3, 8).date()
    trimmed = frame[~(half_day & (local.dt.hour >= 13))]

    hours = infer_regular_hours(trimmed, NEW_YORK)
    assert hours.close_time == time(15, 30)


def test_infer_refuses_with_too_little_data():
    """A calendar guessed from two days is worse than no calendar."""
    assert infer_regular_hours(_session_frame(days=2), NEW_YORK) is None


def test_regular_hours_contains_rejects_weekends_and_out_of_hours():
    hours = infer_regular_hours(_session_frame(days=10), NEW_YORK)

    assert hours.contains(datetime(2024, 3, 5, 15, 0, tzinfo=UTC))  # Tue 10:00 ET
    assert not hours.contains(datetime(2024, 3, 5, 12, 0, tzinfo=UTC))  # Tue 07:00 ET
    assert not hours.contains(datetime(2024, 3, 5, 22, 0, tzinfo=UTC))  # Tue 17:00 ET
    assert not hours.contains(datetime(2024, 3, 9, 15, 0, tzinfo=UTC))  # Saturday


def test_filter_regular_hours_drops_extended_hours_bars():
    frame = _session_frame(days=10, horizon=30)
    hours = infer_regular_hours(frame, NEW_YORK)

    premarket = frame.head(1).copy()
    premarket["time"] = premarket["time"] - pd.Timedelta(hours=3)
    with_extended = normalise(pd.concat([frame, premarket], ignore_index=True))

    assert len(with_extended) == len(frame) + 1
    assert len(filter_regular_hours(with_extended, hours)) == len(frame)


def test_trading_days_lists_local_dates():
    days = trading_days(_session_frame(days=5), NEW_YORK)
    assert len(days) == 5
    assert days[0].date() == datetime(2024, 3, 4).date()


# ------------------------------------------------------------ gap attribution


def test_overnight_and_weekend_breaks_are_not_data_gaps():
    frame = _session_frame(days=10, horizon=30)
    hours = infer_regular_hours(frame, NEW_YORK)

    raw = gaps(frame, 30)
    assert raw, "the raw scan sees every overnight break"
    assert drop_session_breaks(raw, hours) == []


def test_a_hole_inside_trading_hours_is_a_real_gap():
    frame = _session_frame(days=10, horizon=30)
    hours = infer_regular_hours(frame, NEW_YORK)

    # Remove the four bars opening 11:00, 11:30, 12:00 and 12:30 ET on day two.
    local = frame["time"].dt.tz_convert(NEW_YORK)
    hole = (local.dt.date == datetime(2024, 3, 5).date()) & local.dt.hour.isin([11, 12])
    punctured = frame[~hole].reset_index(drop=True)

    real = drop_session_breaks(gaps(punctured, 30), hours)
    assert len(real) == 1
    assert real[0].missing == 4
    assert real[0].after == datetime(2024, 3, 5, 15, 30, tzinfo=UTC)  # 10:30 ET


@pytest.mark.parametrize("timezone", [UTC, NEW_YORK])
def test_gap_attribution_is_stable_across_timezones(timezone):
    """Whether a gap is real must not depend on the zone used to describe it."""
    frame = _session_frame(days=10, horizon=30)
    hours = infer_regular_hours(frame, timezone)
    assert hours is not None
    assert drop_session_breaks(gaps(frame, 30), hours) == []
