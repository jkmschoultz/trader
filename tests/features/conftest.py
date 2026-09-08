"""A synthetic canonical bar frame for the feature tests.

Long enough (default 500 bars) to clear the warmup of every indicator and leave
a healthy stretch of non-NaN rows to assert on.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest


def _make_bars(n: int = 500, *, seed: int = 0, horizon: int = 5):
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    rng = np.random.default_rng(seed)
    start = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.2, n))
    return bars_to_frame(
        [
            Bar(
                Time=start + timedelta(minutes=horizon * i),
                open=float(p),
                high=float(p) + 0.3,
                low=float(p) - 0.3,
                close=float(p),
                volume=1_000.0 + i,
            )
            for i, p in enumerate(close)
        ]
    )


@pytest.fixture
def make_bars():
    """Call ``make_bars(n=..., seed=..., horizon=...)`` for a canonical bar frame."""
    return _make_bars
