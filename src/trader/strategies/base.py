"""The strategy interface every algorithm plugs into.

One interface has to serve three consumers -- the backtest engine now, paper
trading in Phase 5, live trading in Phase 6 -- so it is deliberately narrow. A
strategy is called once per completed bar and returns a :data:`Decision`; it
never sees an order book, a fill, or a clock it can advance. The engine owns all
of that.

Decisions, from least to most explicit
--------------------------------------
* :class:`Hold` -- leave the position exactly as it is. Also what a strategy
  returns before it has enough history to have an opinion.
* :class:`Flat` -- close to zero.
* :class:`Target` -- a signed *conviction* in ``[-1, 1]``, not a position. An
  :class:`~trader.backtest.allocator.Allocator` turns the convictions of every
  instrument into portfolio weights, and the engine sizes those to whole units
  at the next bar's open. Optional ``stop``/``take``/``max_bars`` attach a
  bracket, expressed as fractions of the entry price and a bar count -- the same
  three barriers Phase 3's triple-barrier labelling uses.
* :class:`Orders` -- an escape hatch carrying explicit unit counts. It bypasses
  the allocator and the equity-fraction sizing entirely, for a strategy that
  needs to place its own orders (a grid, a scale-in ladder).

Causality
---------
``ctx.history`` holds only bars that have *fully closed*. The bar whose open the
resulting decision will be filled at is not in it and cannot be seen. See
``docs/backtest.md`` for the timing model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from trader.data.calendars import RegularHours


class Side(StrEnum):
    """Which way an explicit :class:`Order` goes."""

    BUY = "buy"
    SELL = "sell"


# --------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class Hold:
    """Keep the current position unchanged this bar."""


@dataclass(frozen=True)
class Flat:
    """Close the position to zero."""


@dataclass(frozen=True)
class Target:
    """A signed conviction, optionally bracketed.

    Args:
        weight: signed conviction in ``[-1, 1]``. ``+1`` is "as long as this
            strategy ever gets", ``-1`` the same short; the allocator decides
            what fraction of equity that becomes.
        stop: bracket stop as a positive fraction of the entry price -- ``0.01``
            exits a long at 1% below the fill. ``None`` leaves the exit to a
            later signal.
        take: bracket take-profit, same units as ``stop``.
        max_bars: vertical barrier -- force an exit after this many of the
            instrument's bars, at that bar's open.
    """

    weight: float
    stop: float | None = None
    take: float | None = None
    max_bars: int | None = None

    def __post_init__(self) -> None:
        if not -1.0 <= self.weight <= 1.0:
            raise ValueError(f"Target.weight must be in [-1, 1], got {self.weight}")
        for name in ("stop", "take"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"Target.{name} must be a positive fraction, got {value}")
        if self.max_bars is not None and self.max_bars <= 0:
            raise ValueError(f"Target.max_bars must be positive, got {self.max_bars}")

    @property
    def direction(self) -> int:
        return int(math.copysign(1.0, self.weight)) if self.weight else 0


@dataclass(frozen=True)
class Order:
    """One explicit order, in units, carried by :class:`Orders`."""

    side: Side
    units: float
    stop: float | None = None
    take: float | None = None
    max_bars: int | None = None

    def __post_init__(self) -> None:
        if self.units <= 0:
            raise ValueError(f"Order.units must be positive, got {self.units}")

    @property
    def signed_units(self) -> float:
        return self.units if self.side is Side.BUY else -self.units


@dataclass(frozen=True)
class Orders:
    """Explicit orders that bypass allocation and equity-fraction sizing."""

    orders: tuple[Order, ...]

    def __init__(self, orders: Sequence[Order]) -> None:
        object.__setattr__(self, "orders", tuple(orders))


Decision = Hold | Flat | Target | Orders


# ----------------------------------------------------------------- context views


@dataclass(frozen=True)
class Position:
    """A read-only snapshot of one instrument's position."""

    label: str
    units: float = 0.0
    avg_price: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.units != 0.0

    @property
    def direction(self) -> int:
        return int(math.copysign(1.0, self.units)) if self.units else 0


@dataclass(frozen=True)
class PortfolioView:
    """A read-only snapshot of the whole book at decision time."""

    cash: float
    equity: float
    positions: Mapping[str, Position]

    def position(self, label: str) -> Position:
        return self.positions.get(label, Position(label))


@dataclass(frozen=True)
class BarContext:
    """What an :class:`InstrumentStrategy` sees for one instrument, one bar.

    ``history`` is every fully-closed bar so far in the canonical lake schema,
    oldest first, and is a view -- treat it as read-only. ``bar`` is its last
    row, the bar the decision is made on; the fill happens at the *next* bar's
    open, which is not visible here.
    """

    label: str
    now: datetime
    horizon: int
    history: pd.DataFrame
    position: Position
    portfolio: PortfolioView
    session: RegularHours | None = None

    @property
    def bar(self) -> pd.Series:
        return self.history.iloc[-1]

    @property
    def price(self) -> float:
        """Close of the most recent completed bar."""
        return float(self.history["close"].iloc[-1])

    @property
    def bars_seen(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class PanelContext:
    """What a :class:`PortfolioStrategy` sees: the whole cross-section at once.

    ``histories`` holds a completed-bar view per instrument that has any history
    yet. ``active`` is the subset that just closed a bar at this event -- only
    those can be traded now; the rest hold their position.
    """

    now: datetime
    horizon: int
    histories: Mapping[str, pd.DataFrame]
    active: frozenset[str]
    portfolio: PortfolioView
    sessions: Mapping[str, RegularHours | None]


# ----------------------------------------------------------------- strategy ABCs


class Strategy(ABC):  # noqa: B024 - the two subclasses add the abstract on_bar
    """Common base. Concrete strategies subclass one of the two below."""

    #: Bars of history required before :meth:`on_bar` should be trusted. The
    #: engine will not call the strategy until this many closed bars exist.
    warmup: int = 0

    #: Registry name, set by :func:`trader.strategies.registry.register`.
    name: str = ""

    def on_start(self, ctx: BarContext | PanelContext) -> None:  # noqa: B027 - optional hook
        """Called once, with the first context the strategy will see."""

    def on_finish(self) -> None:  # noqa: B027 - optional hook
        """Called once after the last bar."""


class InstrumentStrategy(Strategy):
    """A strategy that reasons about one instrument at a time.

    The engine runs an independent instance per instrument, so instance state
    (an intraday range, a bar counter) is private to that instrument.
    """

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> Decision: ...


class PortfolioStrategy(Strategy):
    """A strategy that reasons across every instrument at once.

    Returns a decision per instrument; a missing entry means :class:`Hold`. For
    fully explicit control, map a label to an :class:`Orders`.
    """

    @abstractmethod
    def on_bar(self, ctx: PanelContext) -> Mapping[str, Decision]: ...
