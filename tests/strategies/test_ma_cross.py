"""MA crossover: warmup gate, the long/short flip, long-only, and flat-at-close."""

from __future__ import annotations

from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from trader.data.calendars import RegularHours
from trader.strategies.base import Flat, Hold, Target
from trader.strategies.ma_cross import MACrossStrategy

NY = ZoneInfo("America/New_York")
SESSION = RegularHours(
    timezone=NY,
    open_time=time(9, 30),
    close_time=time(15, 55),
    weekdays=frozenset({0, 1, 2, 3, 4}),
    days_observed=20,
)
OPEN_UTC = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)  # 09:30 ET, pre-DST


def test_fast_must_be_shorter_than_slow():
    with pytest.raises(ValueError, match="shorter than slow"):
        MACrossStrategy(fast=30, slow=10)


def test_holds_until_it_has_slow_bars(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5)
    assert strat.warmup == 5
    assert isinstance(strat.on_bar(make_context(make_bars([10, 11, 12]))), Hold)


def test_goes_long_when_fast_is_above_slow(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5)
    decision = strat.on_bar(make_context(make_bars([10, 10, 10, 12, 14])))
    assert isinstance(decision, Target)
    assert decision.weight == 1.0


def test_goes_short_when_fast_is_below_slow(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5)
    decision = strat.on_bar(make_context(make_bars([14, 12, 10, 10, 9])))
    assert isinstance(decision, Target)
    assert decision.weight == -1.0


def test_long_only_goes_flat_instead_of_short(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5, long_only=True)
    decision = strat.on_bar(make_context(make_bars([14, 12, 10, 10, 9])))
    assert isinstance(decision, Flat)


def test_flat_eod_flattens_at_the_session_close(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5, flat_eod=True)
    # a clean long signal, but the last bar prints at 21:00 UTC == 16:00 ET, past close
    frame = make_bars([10, 10, 10, 12, 14], start=OPEN_UTC)
    frame.loc[4, "time"] = pd.Timestamp("2024-03-01 21:00", tz="UTC")
    assert isinstance(strat.on_bar(make_context(frame, session=SESSION)), Flat)


def test_flat_eod_still_trades_intraday(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5, flat_eod=True)
    frame = make_bars([10, 10, 10, 12, 14], start=OPEN_UTC)  # all within the session
    decision = strat.on_bar(make_context(frame, session=SESSION))
    assert isinstance(decision, Target)
    assert decision.weight == 1.0


def test_flat_eod_is_a_noop_without_a_session(make_bars, make_context):
    strat = MACrossStrategy(fast=2, slow=5, flat_eod=True)
    frame = make_bars([10, 10, 10, 12, 14])
    decision = strat.on_bar(make_context(frame))  # session=None
    assert isinstance(decision, Target)
