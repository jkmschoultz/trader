"""Opening-range breakout -- a genuinely intraday strategy with a bracket.

Each trading day, the high and low of the first ``open_minutes`` define a range.
The first bar to *close* beyond that range triggers a position in the breakout
direction, bracketed by a stop and a take-profit. One entry per day; the
position is flattened at the session close if neither barrier is hit first.

This one exercises the parts ``ma_cross`` does not: per-instrument day state, the
inferred session calendar, and the engine's bracket handling.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target
from trader.strategies.registry import register


@register("orb")
class OpeningRangeBreakout(InstrumentStrategy):
    """First-``open_minutes`` range breakout, long/short, bracketed.

    Args:
        open_minutes: minutes from the session open that define the range.
        stop: bracket stop, a positive fraction of the entry price.
        take: bracket take-profit, same units.
        max_bars: optional vertical barrier, in bars.
        long_only: ignore downside breakouts.

    Without an inferred session calendar (``ctx.session``) the range is anchored
    to the first bar of each local day and the position is not force-flattened
    intraday -- so a session calendar is worth having for this one.
    """

    def __init__(
        self,
        *,
        open_minutes: int = 15,
        stop: float = 0.005,
        take: float = 0.01,
        max_bars: int | None = None,
        long_only: bool = False,
    ) -> None:
        if open_minutes <= 0:
            raise ValueError("open_minutes must be positive")
        self.open_minutes = open_minutes
        self.stop = stop
        self.take = take
        self.max_bars = max_bars
        self.long_only = long_only
        self.warmup = 0

        self._day: object = None
        self._hi: float | None = None
        self._lo: float | None = None
        self._first_clock: datetime | None = None
        self._done = False

    def _reset(self, day: object) -> None:
        self._day = day
        self._hi = self._lo = None
        self._first_clock = None
        self._done = False

    def on_bar(self, ctx: BarContext) -> Decision:
        bar = ctx.bar
        tz = ctx.session.timezone if ctx.session is not None else None
        local = bar["time"].tz_convert(tz) if tz is not None else bar["time"]
        day, clock = local.date(), local.time()

        if day != self._day:
            self._reset(day)

        if self._first_clock is None:
            self._first_clock = local

        if ctx.session is not None:
            open_at = datetime.combine(day, ctx.session.open_time)
            range_end = (open_at + timedelta(minutes=self.open_minutes)).time()
            if clock < ctx.session.open_time:
                return Flat()  # pre-market print: not part of the opening range
            if clock >= ctx.session.close_time:
                self._done = True
                return Flat()
        else:
            range_end = (self._first_clock + timedelta(minutes=self.open_minutes)).time()

        if clock < range_end:
            high, low = float(bar["high"]), float(bar["low"])
            self._hi = high if self._hi is None else max(self._hi, high)
            self._lo = low if self._lo is None else min(self._lo, low)
            return Flat()

        if self._done or self._hi is None:
            return Hold() if self._done else Flat()

        close = float(bar["close"])
        bracket = {"stop": self.stop, "take": self.take, "max_bars": self.max_bars}
        if close > self._hi:
            self._done = True
            return Target(1.0, **bracket)
        if close < self._lo:
            self._done = True
            return Flat() if self.long_only else Target(-1.0, **bracket)
        return Flat()
