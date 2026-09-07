"""Opening-range breakout: range formation, both breakout directions, one entry
per day, and the session-close flatten."""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from trader.data.calendars import RegularHours
from trader.strategies.base import Flat, Hold, Target
from trader.strategies.orb import OpeningRangeBreakout

NY = ZoneInfo("America/New_York")
SESSION = RegularHours(
    timezone=NY,
    open_time=time(9, 30),
    close_time=time(15, 55),
    weekdays=frozenset({0, 1, 2, 3, 4}),
    days_observed=20,
)
# 2024-03-01 is before US DST: 09:30 ET == 14:30 UTC.
OPEN_UTC = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)


def test_flat_while_the_range_is_still_forming(make_bars, replay):
    frame = make_bars([100, 100, 100], start=OPEN_UTC, highs=[100.5] * 3, lows=[99.5] * 3)
    decisions = replay(OpeningRangeBreakout(open_minutes=15), frame, session=SESSION)
    assert all(isinstance(d, Flat) for d in decisions)


def test_upside_breakout_goes_long_with_a_bracket(make_bars, replay):
    frame = make_bars(
        [100, 100, 100, 101, 101],
        start=OPEN_UTC,
        highs=[100.5, 100.5, 100.5, 101.2, 101.2],
        lows=[99.5, 99.5, 99.5, 100.8, 100.8],
    )
    decisions = replay(
        OpeningRangeBreakout(open_minutes=15, stop=0.004, take=0.009), frame, session=SESSION
    )

    breakout = decisions[3]
    assert isinstance(breakout, Target)
    assert breakout.weight == 1.0
    assert breakout.stop == 0.004
    assert breakout.take == 0.009
    assert isinstance(decisions[4], Hold)  # one entry per day


def test_downside_breakout_goes_short_unless_long_only(make_bars, replay):
    frame = make_bars(
        [100, 100, 100, 99],
        start=OPEN_UTC,
        highs=[100.5, 100.5, 100.5, 99.2],
        lows=[99.5, 99.5, 99.5, 98.8],
    )
    short = replay(OpeningRangeBreakout(open_minutes=15), frame, session=SESSION)[3]
    assert isinstance(short, Target)
    assert short.weight == -1.0

    flat = replay(OpeningRangeBreakout(open_minutes=15, long_only=True), frame, session=SESSION)[3]
    assert isinstance(flat, Flat)


def test_no_breakout_day_stays_flat(make_bars, replay):
    frame = make_bars(
        [100, 100, 100, 100, 100.1, 99.9],
        start=OPEN_UTC,
        highs=[100.4] * 6,
        lows=[99.6] * 6,
    )
    decisions = replay(OpeningRangeBreakout(open_minutes=15), frame, session=SESSION)
    assert all(isinstance(d, Flat) for d in decisions)


def test_flattens_at_the_session_close(make_bars, replay):
    frame = make_bars(
        [100, 100, 100, 105],
        start=OPEN_UTC,
        highs=[100.5, 100.5, 100.5, 105],
        lows=[99.5, 99.5, 99.5, 105],
    )
    frame.loc[3, "time"] = pd.Timestamp("2024-03-01 21:00", tz="UTC")  # 16:00 ET, past close
    decisions = replay(OpeningRangeBreakout(open_minutes=15), frame, session=SESSION)
    assert isinstance(decisions[3], Flat)
