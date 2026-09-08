"""Fixtures for the model-layer tests.

Torch tests use ``importorskip`` at module scope, so this file must import
cleanly without the ``[model]`` extra.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest


def _bars(n=1500, *, seed=0, horizon=5):
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
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


def _separable_sequences(n_per_class=300, *, window=8, features=4, seed=0):
    """Three classes whose sequences sit at different means -- trivially learnable."""
    rng = np.random.default_rng(seed)
    blocks_x, blocks_y = [], []
    for cls in range(3):
        centre = (cls - 1) * 1.5
        block = rng.normal(centre, 0.25, size=(n_per_class, window, features))
        blocks_x.append(block.astype("float32"))
        blocks_y += [cls] * n_per_class
    x = np.concatenate(blocks_x)
    y = np.array(blocks_y, dtype="int64")
    order = rng.permutation(len(y))
    return x[order], y[order]


@pytest.fixture
def bars():
    return _bars


@pytest.fixture
def separable_sequences():
    return _separable_sequences
