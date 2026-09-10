"""``GBMStrategy``: warmup, decision mapping, brackets, and end-to-end causality."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from trader.strategies import get_strategy  # noqa: E402
from trader.strategies.base import Flat, Hold, Target  # noqa: E402

pytestmark = pytest.mark.slow

WINDOW = 8
MAX_BARS = 10


def _bars(n, *, seed):
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.25, n))
    return bars_to_frame(
        [
            Bar(
                Time=start + timedelta(minutes=5 * i),
                open=float(p),
                high=float(p) + 0.4,
                low=float(p) - 0.4,
                close=float(p),
                volume=1_000.0 + i,
            )
            for i, p in enumerate(close)
        ]
    )


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory):
    """Train a tiny GBM on synthetic bars and register it; return (dir, id)."""
    from trader.labels.triple_barrier import LabelConfig
    from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split
    from trader.models.gbm import GBMConfig, train_gbm
    from trader.models.registry import ModelRegistry
    from trader.models.scaler import StandardScaler

    bars = _bars(2600, seed=1)
    label = LabelConfig(stop=0.01, take=0.01, max_bars=MAX_BARS)
    spec = WindowSpec(
        window=WINDOW,
        feature_set="price_v1",
        base_horizon=5,
        context_horizons=(),
        label=label,
        layout="tabular",
    )
    bundle = build_bundle(bars, spec=spec)
    t = bundle.t
    split = SplitSpec(
        train_end=pd.Timestamp(t[int(len(t) * 0.7)]).to_pydatetime(),
        val_end=pd.Timestamp(t[int(len(t) * 0.85)]).to_pydatetime(),
        embargo_bars=15,
    )
    tr, va, _ = time_split(bundle, split, base_horizon=5, max_bars=MAX_BARS)
    scaler = StandardScaler().fit(bundle.X[tr])
    config = GBMConfig(n_estimators=40, num_leaves=15)
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
        data_spec={"symbols": ["X"], "asset_type": "Stock"},
    )
    return str(root), info.id


def test_warmup_is_window_plus_feature_warmup(saved_model):
    root, model_id = saved_model
    strat = get_strategy("gbm")(model=model_id, models_dir=root)

    from trader.features.base import FeatureSpec

    fspec = FeatureSpec.from_file(f"{root}/{model_id}/feature_spec.json")
    assert strat.warmup == WINDOW + fspec.warmup
    assert strat._layout == "tabular"


def test_holds_until_warmed_up_then_decides(saved_model, make_context):
    root, model_id = saved_model
    strat = get_strategy("gbm")(model=model_id, models_dir=root, threshold=0.0)
    bars = _bars(strat.warmup + 60, seed=5)

    early = strat.on_bar(make_context(bars.iloc[: strat.warmup - 1]))
    assert isinstance(early, Hold)

    late = strat.on_bar(make_context(bars.iloc[: strat.warmup + 40]))
    assert isinstance(late, Target | Hold | Flat)


def test_on_no_signal_is_configurable(saved_model, make_context):
    root, model_id = saved_model
    bars = _bars(400, seed=9)

    hold_strat = get_strategy("gbm")(model=model_id, models_dir=root, threshold=0.99)
    flat_strat = get_strategy("gbm")(
        model=model_id, models_dir=root, threshold=0.99, on_no_signal="flat"
    )
    ctx = make_context(bars.iloc[: hold_strat.warmup + 30])
    assert isinstance(hold_strat.on_bar(ctx), Hold)
    assert isinstance(flat_strat.on_bar(ctx), Flat)


def test_bracket_carries_the_trained_barriers(saved_model, make_context):
    root, model_id = saved_model
    strat = get_strategy("gbm")(model=model_id, models_dir=root, threshold=0.0)
    bars = _bars(strat.warmup + 120, seed=2)

    targets = [
        d
        for i in range(strat.warmup + 1, len(bars))
        if isinstance(d := strat.on_bar(make_context(bars.iloc[:i])), Target)
    ]
    assert targets, "expected at least one Target with threshold=0"
    assert all(t.max_bars == MAX_BARS for t in targets)
    assert all(t.take == 0.01 and t.stop == 0.01 for t in targets)


def test_decisions_are_causal_in_a_backtest(saved_model):
    """Perturbing bars after t leaves every decision at/before t byte-identical."""
    from trader import backtest as bt

    root, model_id = saved_model
    bars = _bars(700, seed=4)
    cut = 500

    def run(frame):
        strat = get_strategy("gbm")(model=model_id, models_dir=root, threshold=0.0)
        return bt.run({"X": frame}, strat, horizon=5, allocator=bt.get_allocator("equal-weight"))

    ref = run(bars)
    tampered = bars.copy()
    tampered.loc[tampered.index > cut, ["open", "high", "low", "close"]] *= 1.5
    per = run(tampered)

    ref_fills = ref.fills[ref.fills["time"] <= bars["time"].iloc[cut]].reset_index(drop=True)
    per_fills = per.fills[per.fills["time"] <= bars["time"].iloc[cut]].reset_index(drop=True)
    pd.testing.assert_frame_equal(ref_fills, per_fills)


def test_requires_lightgbm(monkeypatch, saved_model):
    import builtins

    root, model_id = saved_model
    real_import = builtins.__import__

    def no_lightgbm(name, *args, **kwargs):
        if name == "lightgbm" or name.startswith("lightgbm."):
            raise ImportError("no lightgbm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_lightgbm)
    with pytest.raises(ValueError, match=r"\[model\]"):
        get_strategy("gbm")(model=model_id, models_dir=root)
