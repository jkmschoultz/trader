"""Event-driven backtesting: engine, cost model, allocators, metrics.

Like :mod:`trader.data`, this package needs the ``[data]`` extra (pandas, numpy)::

    pip install -e ".[data]"

The one entry point is :func:`run`. Everything it needs -- a
:class:`~trader.backtest.costs.CostModel`, an
:class:`~trader.backtest.allocator.Allocator`, per-instrument
:class:`~trader.backtest.engine.Instrument` metadata -- has a sensible default,
so a single-instrument run is ``run({label: frame}, strategy, horizon=5)``.
"""

from __future__ import annotations

from trader.backtest.allocator import (
    Allocator,
    EqualWeight,
    FixedFraction,
    PassThrough,
    VolTarget,
    get_allocator,
)
from trader.backtest.costs import CostModel
from trader.backtest.engine import Instrument, run
from trader.backtest.metrics import InstrumentStats, Metrics, compute, periods_per_year
from trader.backtest.result import BacktestResult

__all__ = [
    "Allocator",
    "BacktestResult",
    "CostModel",
    "EqualWeight",
    "FixedFraction",
    "Instrument",
    "InstrumentStats",
    "Metrics",
    "PassThrough",
    "VolTarget",
    "compute",
    "get_allocator",
    "periods_per_year",
    "run",
]
