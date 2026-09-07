"""Moving-average crossover -- the simplest thing that exercises the interface.

Long while the fast SMA is above the slow one, short (or flat, if ``long_only``)
while it is below. Always in the market once warmed up. It is here to prove the
plumbing end to end, and as a baseline every more elaborate strategy should be
able to beat after costs.
"""

from __future__ import annotations

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target
from trader.strategies.registry import register


@register("ma_cross")
class MACrossStrategy(InstrumentStrategy):
    """SMA(fast) vs SMA(slow) on the bar close.

    Args:
        fast: window of the fast simple moving average, in bars.
        slow: window of the slow one; must be longer than ``fast``.
        long_only: go :class:`Flat` instead of short when fast is below slow.
    """

    def __init__(self, *, fast: int = 10, slow: int = 30, long_only: bool = False) -> None:
        if fast < 1 or slow < 1:
            raise ValueError("fast and slow windows must be positive")
        if fast >= slow:
            raise ValueError(f"fast window ({fast}) must be shorter than slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.long_only = long_only
        self.warmup = slow

    def on_bar(self, ctx: BarContext) -> Decision:
        close = ctx.history["close"]
        if len(close) < self.slow:
            return Hold()
        fast_ma = close.iloc[-self.fast :].mean()
        slow_ma = close.iloc[-self.slow :].mean()
        if fast_ma > slow_ma:
            return Target(1.0)
        if fast_ma < slow_ma:
            return Flat() if self.long_only else Target(-1.0)
        return Hold()
