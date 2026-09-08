"""Helpers for building small, hand-checked bar frames."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest


def _bar_frame(rows, *, horizon: int = 5, start=datetime(2024, 3, 1, 14, 30, tzinfo=UTC)):
    """``rows`` is a list of ``(open, high, low, close)`` tuples."""
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    return bars_to_frame(
        [
            Bar(
                Time=start + timedelta(minutes=horizon * i),
                open=float(o),
                high=float(h),
                low=float(low),
                close=float(c),
                volume=1_000.0,
            )
            for i, (o, h, low, c) in enumerate(rows)
        ]
    )


def _flat(price, n):
    return [(price, price, price, price)] * n


@pytest.fixture
def bar_frame():
    return _bar_frame


@pytest.fixture
def flat_rows():
    return _flat


@pytest.fixture
def random_walk_frame():
    def build(n=400, *, seed=0, horizon=5):
        rng = np.random.default_rng(seed)
        close = 100.0 + np.cumsum(rng.normal(0.0, 0.25, n))
        rows = [(p, p + 0.4, p - 0.4, p) for p in close]
        return _bar_frame(rows, horizon=horizon)

    return build
