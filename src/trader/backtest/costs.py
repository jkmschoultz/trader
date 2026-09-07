"""The cost model: what a fill actually costs versus its reference price.

Three components, all optional and additive:

* **Commission** -- basis points of notional, a per-unit fee, or a per-fill
  minimum, whichever bites.
* **Spread** -- half the bid/ask spread, crossed on entry and again on exit.
  When the reference bar carries ``close_ask`` (quote-driven instruments, FX)
  the real spread is used; otherwise ``half_spread_bps`` stands in. Taking it
  from ``close_ask`` rather than a mid is the whole reason the lake keeps the
  ask column -- for intraday FX the spread is the dominant cost.
* **Slippage** -- a flat basis-point haircut for market impact and the gap
  between a bar's open print and what you'd actually get.

Every component pushes the fill *against* the trade: a buy fills higher, a sell
lower. The reference bar passed to :meth:`CostModel.fill_price` must be a
*completed* bar (its ``close``/``close_ask`` are known); the engine passes the
last closed bar, never the one being entered on.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from trader.strategies.base import Side

_BPS = 1e-4


@dataclass(frozen=True)
class CostModel:
    """Commission, spread, and slippage assumptions for one instrument."""

    commission_bps: float = 0.0
    commission_per_unit: float = 0.0
    commission_min: float = 0.0
    half_spread_bps: float = 0.0
    slippage_bps: float = 0.0

    def half_spread_frac(self, ref_bar: Mapping[str, float] | None) -> float:
        """Half-spread as a fraction of price, from the bar's ask if it has one."""
        if ref_bar is not None:
            ask = ref_bar.get("close_ask")
            close = ref_bar.get("close")
            if (
                ask is not None
                and close
                and not math.isnan(ask)
                and not math.isnan(close)
                and ask > close
            ):
                return (ask - close) / close / 2.0
        return self.half_spread_bps * _BPS

    def fill_price(
        self, side: Side, reference: float, ref_bar: Mapping[str, float] | None = None
    ) -> float:
        """Reference price nudged adverse by half-spread + slippage."""
        adverse = self.half_spread_frac(ref_bar) + self.slippage_bps * _BPS
        sign = 1.0 if side is Side.BUY else -1.0
        return reference * (1.0 + sign * adverse)

    def commission(self, units: float, price: float) -> float:
        """Commission on a fill of ``units`` (unsigned is fine) at ``price``."""
        qty = abs(units)
        if qty == 0:
            return 0.0
        fee = qty * price * self.commission_bps * _BPS + qty * self.commission_per_unit
        return max(fee, self.commission_min)
