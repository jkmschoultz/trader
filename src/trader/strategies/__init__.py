"""Strategies: the :class:`~trader.strategies.base.Strategy` interface, a
name registry, and the built-in classical algorithms.

Importing this package registers the built-ins (:mod:`ma_cross`, :mod:`orb`), so
``registry.get_strategy("orb")`` works without importing the module directly.

The interface has no third-party dependency; the built-in strategies read the
canonical bar frame (pandas), which comes with the ``[data]`` extra.
"""

from __future__ import annotations

from trader.strategies import lstm, ma_cross, orb  # noqa: F401 - registration side effect
from trader.strategies.base import (
    BarContext,
    Decision,
    Flat,
    Hold,
    InstrumentStrategy,
    Order,
    Orders,
    PanelContext,
    PortfolioStrategy,
    PortfolioView,
    Position,
    Side,
    Strategy,
    Target,
)
from trader.strategies.registry import UnknownStrategy, available, get_strategy, register

__all__ = [
    "BarContext",
    "Decision",
    "Flat",
    "Hold",
    "InstrumentStrategy",
    "Order",
    "Orders",
    "PanelContext",
    "PortfolioStrategy",
    "PortfolioView",
    "Position",
    "Side",
    "Strategy",
    "Target",
    "UnknownStrategy",
    "available",
    "get_strategy",
    "register",
]
