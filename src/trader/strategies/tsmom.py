"""Time-series momentum: follow each instrument's own trend.

The best-documented systematic edge in FX, commodities and crypto (Moskowitz,
Ooi & Pedersen 2012, "Time Series Momentum"): an asset that rose over the past
months tends to keep rising for a while, and one that fell tends to keep
falling. Go long what is up over the lookback, short what is down.

Several lookbacks are blended -- by default about 1, 3, 6 and 12 months of
trading days -- and the conviction is the average of their signs, in ``[-1, 1]``.
Blending is the usual guard against betting everything on one arbitrary window.
With the ``equal-weight`` allocator only the sign matters; ``passthrough``
sizes by the blended conviction.

It trades on a fixed schedule (every ``rebalance`` bars) and holds in between,
so turnover and costs stay small. Designed for daily bars: on intraday bars the
lookbacks are bar counts and mean something much shorter.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target
from trader.strategies.registry import register


def _ints(value: str | int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        # "/" or "+" as well as ",": a sweep grid already splits values on commas
        return tuple(int(v) for v in re.split(r"[,/+\s]+", value) if v)
    return tuple(int(v) for v in value)


@register("tsmom")
class TimeSeriesMomentum(InstrumentStrategy):
    """Long if the instrument rose over each lookback, short if it fell; blended.

    Args:
        lookbacks: bars to measure the trend over: a list, or a string separated
            by ``,`` ``/`` or ``+`` (use ``/`` in a sweep grid: ``21/63/126/252``).
            Default ``21,63,126,252``: about 1, 3, 6 and 12 months of trading days.
        rebalance: re-read the signal every this many bars and hold in between
            (5 = weekly on daily bars).
        skip: ignore the most recent ``skip`` bars when measuring the trend (some
            studies skip a month to step around short-term reversal).
        long_only: go flat instead of short on a downtrend.
    """

    def __init__(
        self,
        *,
        lookbacks: str | Sequence[int] = "21,63,126,252",
        rebalance: int = 5,
        skip: int = 0,
        long_only: bool = False,
    ) -> None:
        self.lookbacks = _ints(lookbacks)
        if not self.lookbacks or min(self.lookbacks) < 1:
            raise ValueError(f"lookbacks must be positive bar counts, got {lookbacks!r}")
        if rebalance < 1:
            raise ValueError(f"rebalance must be >= 1, got {rebalance}")
        if skip < 0:
            raise ValueError(f"skip must be >= 0, got {skip}")
        self.rebalance = int(rebalance)
        self.skip = int(skip)
        self.long_only = bool(long_only)
        self.warmup = max(self.lookbacks) + self.skip + 1

    def conviction(self, close) -> float:
        """Mean of the lookback-return signs, measured ``skip`` bars back."""
        end = float(close.iloc[-1 - self.skip])
        signs = []
        for k in self.lookbacks:
            start = float(close.iloc[-1 - self.skip - k])
            signs.append((end > start) - (end < start))
        return sum(signs) / len(signs)

    def on_bar(self, ctx: BarContext) -> Decision:
        if ctx.bars_seen < self.warmup:
            return Hold()
        if (ctx.bars_seen - self.warmup) % self.rebalance:
            return Hold()
        conviction = self.conviction(ctx.history["close"])
        if self.long_only:
            conviction = max(conviction, 0.0)
        return Target(conviction) if conviction else Flat()
