"""A ``structure_v1`` model sees the same features live as it was trained on.

Prior-week levels need ~2 weeks of bars, far more than the strategy's bar-count
recompute tail. ``ModelStrategy`` extends the tail by ``spec.level_days`` of
wall-clock history; without that, live inference would see NaN levels (and
hold forever) where training saw values.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

pytestmark = pytest.mark.slow

WINDOW = 4
HORIZON = 15


def _bars(n, *, seed):
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    rng = np.random.default_rng(seed)
    start = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.15, n))
    return bars_to_frame(
        [
            Bar(
                Time=start + timedelta(minutes=HORIZON * i),
                open=float(p),
                high=float(p) + 0.2,
                low=float(p) - 0.2,
                close=float(p),
            )
            for i, p in enumerate(close)
        ]
    )


@pytest.fixture(scope="module")
def structure_model(tmp_path_factory):
    """A tiny GBM on structure_v1 (no context, so the bar-count tail is short)."""
    from trader.labels.triple_barrier import LabelConfig
    from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split
    from trader.models.gbm import GBMConfig, train_gbm
    from trader.models.registry import ModelRegistry
    from trader.models.scaler import StandardScaler

    bars = _bars(2900, seed=3)  # ~30 days of 15m bars
    label = LabelConfig(stop=0.01, take=0.01, max_bars=8)
    spec = WindowSpec(
        window=WINDOW,
        feature_set="structure_v1",
        base_horizon=HORIZON,
        context_horizons=(),
        label=label,
        layout="tabular",
    )
    bundle = build_bundle(bars, spec=spec)
    t = bundle.t
    split = SplitSpec(
        train_end=pd.Timestamp(t[int(len(t) * 0.7)]).to_pydatetime(),
        val_end=pd.Timestamp(t[int(len(t) * 0.85)]).to_pydatetime(),
        embargo_bars=12,
    )
    tr, va, _ = time_split(bundle, split, base_horizon=HORIZON, max_bars=8)
    scaler = StandardScaler().fit(bundle.X[tr])
    config = GBMConfig(n_estimators=20, num_leaves=7)
    model, report = train_gbm(
        scaler.transform(bundle.X[tr]),
        bundle.y[tr],
        bundle.w[tr],
        scaler.transform(bundle.X[va]),
        bundle.y[va],
        config=config,
        seed=0,
    )
    root = tmp_path_factory.mktemp("models")
    info = ModelRegistry(root).save(
        name="gbm",
        model=model,
        scaler=scaler,
        feature_spec=bundle.feature_spec,
        label=label,
        window=WINDOW,
        split=split,
        config=config,
        report=report,
        model_type="gbm",
        layout="tabular",
        data_spec={"symbols": ["X"], "asset_type": "FxSpot"},
    )
    return str(root), info.id, bars


def test_live_recompute_matches_full_history_features(structure_model):
    from trader.features.base import FeatureSpec
    from trader.features.pipeline import compute_feature_frame
    from trader.strategies import get_strategy
    from trader.strategies.base import BarContext

    root, model_id, bars = structure_model
    strat = get_strategy("gbm")(model=model_id, models_dir=root)
    fspec = FeatureSpec.from_file(f"{root}/{model_id}/feature_spec.json")
    full, _ = compute_feature_frame(bars, spec=fspec)
    full_rows = full.to_numpy(dtype="float32")

    # the bar-count tail alone is ~4 days -- the prior week would be missing
    assert strat._tail < 7 * 24 * 60 // HORIZON

    for i in (2600, 2750, 2899):
        ctx = BarContext(
            label="X",
            now=bars["time"].iloc[i],
            horizon=HORIZON,
            history=bars.iloc[: i + 1],
            position=None,
            portfolio=None,
        )
        live = strat._window_rows(ctx)
        expected = full_rows[i + 1 - WINDOW : i + 1]
        assert not np.isnan(live).any(), "live recompute left level features NaN"
        np.testing.assert_allclose(live, expected, rtol=1e-4, atol=1e-4)
