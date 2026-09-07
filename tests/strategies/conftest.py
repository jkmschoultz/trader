"""Fixtures for exercising a strategy's ``on_bar`` in isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trader.data.lake import normalise
from trader.strategies.base import BarContext, PortfolioView, Position

_START = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)


def _make_bars(
    closes: list[float],
    *,
    start: datetime = _START,
    horizon: int = 5,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    return normalise(
        pd.DataFrame(
            {
                "time": [start + timedelta(minutes=horizon * i) for i in range(n)],
                "open": closes,
                "high": highs if highs is not None else [c + 0.01 for c in closes],
                "low": lows if lows is not None else [c - 0.01 for c in closes],
                "close": closes,
                "volume": [1_000.0] * n,
                "interest": [None] * n,
                "close_ask": [None] * n,
            }
        )
    )


def _context(history: pd.DataFrame, *, session=None, units: float = 0.0, horizon: int = 5):
    return BarContext(
        label="TEST",
        now=history["time"].iloc[-1] + pd.Timedelta(minutes=horizon),
        horizon=horizon,
        history=history,
        position=Position("TEST", units=units, avg_price=float(history["close"].iloc[-1])),
        portfolio=PortfolioView(cash=100_000.0, equity=100_000.0, positions={}),
        session=session,
    )


@pytest.fixture
def make_bars():
    """A canonical bar frame from a list of closes; O=H=L=C unless overridden."""
    return _make_bars


@pytest.fixture
def make_context():
    """One :class:`BarContext` from a history frame."""
    return _context


@pytest.fixture
def replay():
    """Feed a strategy each growing prefix of a frame; return every decision."""

    def _replay(strat, frame: pd.DataFrame, *, session=None) -> list:
        return [
            strat.on_bar(_context(frame.iloc[:i], session=session))
            for i in range(1, len(frame) + 1)
        ]

    return _replay
