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


@pytest.fixture
def trained_artifacts():
    """Everything ``ModelRegistry.save`` needs, from a tiny 3-epoch run.

    Torch-gated: the body imports the model layer, so call it only from a test
    that has already ``importorskip``-ed torch.
    """

    def build(*, feature_set="price_v1", context=(), window=16, n=2200, seed=1):
        import pandas as pd

        from trader.labels.triple_barrier import LabelConfig
        from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split
        from trader.models.lstm import LSTMConfig
        from trader.models.scaler import StandardScaler
        from trader.models.training import train_model

        label = LabelConfig(stop=0.01, take=0.01, max_bars=12)
        spec = WindowSpec(
            window=window,
            feature_set=feature_set,
            base_horizon=5,
            context_horizons=tuple(context),
            label=label,
        )
        bundle = build_bundle(_bars(n, seed=seed), spec=spec)
        t = bundle.t
        split = SplitSpec(
            train_end=pd.Timestamp(t[int(len(t) * 0.7)]).to_pydatetime(),
            val_end=pd.Timestamp(t[int(len(t) * 0.85)]).to_pydatetime(),
            embargo_bars=20,
        )
        tr, va, te = time_split(bundle, split, base_horizon=5, max_bars=12)
        scaler = StandardScaler().fit(bundle.X[tr])
        config = LSTMConfig(n_features=bundle.X.shape[-1], hidden=8, layers=1, dropout=0.0)
        model, report = train_model(
            scaler.transform(bundle.X[tr]),
            bundle.y[tr],
            bundle.w[tr],
            scaler.transform(bundle.X[va]),
            bundle.y[va],
            config=config,
            epochs=3,
            Xte=scaler.transform(bundle.X[te]),
            yte=bundle.y[te],
            batch_size=128,
            seed=0,
        )
        return {
            "name": "lstm",
            "model": model,
            "scaler": scaler,
            "feature_spec": bundle.feature_spec,
            "label": label,
            "window": window,
            "split": split,
            "config": config,
            "report": report,
            "data_spec": {"symbols": ["X"], "uics": [211], "asset_type": "Stock"},
            "optimiser": {"lr": 1e-3, "epochs": 3, "seed": 0},
        }

    return build
