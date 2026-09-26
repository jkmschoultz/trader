"""A model trained on ATR-scaled labels trades an ATR-scaled bracket -- the same one.

The labels convert ``stop`` / ``take`` ATR multiples into price fractions with
the decision bar's ATR; ``ModelStrategy`` must do the identical conversion from
the history it holds, or the model is taught one exit and trades another.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

WINDOW = 4
HORIZON = 15


def _bars(n, *, seed):
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    rng = np.random.default_rng(seed)
    start = datetime(2024, 3, 4, 0, 0, tzinfo=UTC)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    spread = 0.1 + 0.1 * np.abs(np.sin(np.arange(n) / 50))  # volatility that changes
    return bars_to_frame(
        [
            Bar(
                Time=start + timedelta(minutes=HORIZON * i),
                open=float(p),
                high=float(p + s),
                low=float(p - s),
                close=float(p),
            )
            for i, (p, s) in enumerate(zip(close, spread, strict=True))
        ]
    )


@pytest.fixture(scope="module")
def atr_model(tmp_path_factory):
    from trader.labels.triple_barrier import LabelConfig
    from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split
    from trader.models.gbm import GBMConfig, train_gbm
    from trader.models.registry import ModelRegistry
    from trader.models.scaler import StandardScaler

    bars = _bars(1500, seed=5)
    label = LabelConfig(stop=2.0, take=3.0, max_bars=8, scale="atr")
    spec = WindowSpec(
        window=WINDOW,
        feature_set="price_v1",
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
    config = GBMConfig(n_estimators=10, num_leaves=7)
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
    return str(root), info, bars


def _ctx(bars, i):
    from trader.strategies.base import BarContext

    return BarContext(
        label="X",
        now=bars["time"].iloc[i],
        horizon=HORIZON,
        history=bars.iloc[: i + 1],
        position=None,
        portfolio=None,
    )


def test_manifest_records_the_scale(atr_model):
    _, info, _ = atr_model
    assert info.barriers["scale"] == "atr"
    assert info.barriers["stop"] == 2.0


def test_bracket_is_the_labels_atr_multiple_at_the_decision_bar(atr_model):
    from trader.labels.triple_barrier import barrier_unit
    from trader.strategies import get_strategy

    root, info, bars = atr_model
    strat = get_strategy("gbm")(model=info.id, models_dir=root)
    unit = barrier_unit(bars, "atr")  # the full-history ATR the labels used

    for i in (900, 1200, 1499):
        bracket = strat._bracket(_ctx(bars, i))
        assert bracket["stop"] == pytest.approx(2.0 * unit[i], rel=1e-9)
        assert bracket["take"] == pytest.approx(3.0 * unit[i], rel=1e-9)
        assert bracket["max_bars"] == 8


def test_signals_carry_the_scaled_bracket(atr_model):
    from trader.labels.triple_barrier import barrier_unit
    from trader.strategies import get_strategy
    from trader.strategies.base import Target

    root, info, bars = atr_model
    strat = get_strategy("gbm")(model=info.id, models_dir=root, threshold=0.0)
    unit = barrier_unit(bars, "atr")
    i = 1300
    ctx = _ctx(bars, i)
    decision = strat.on_bar(ctx)
    assert isinstance(decision, Target)
    assert decision.stop == pytest.approx(2.0 * unit[i], rel=1e-9)


# --- meta-labelling primary -----------------------------------------------------------


@pytest.fixture(scope="module")
def gated_model(atr_model, tmp_path_factory):
    """The same trees, re-registered as trained under a 1/2-day momentum gate."""
    from trader.models.registry import ModelRegistry

    root, info, bars = atr_model
    src = ModelRegistry(root)
    predictor, scaler, _ = src.load_model(info.id)
    import json

    manifest = json.loads((src.path(info.id) / "manifest.json").read_text())
    manifest["primary"] = {"name": "tsmom", "lookbacks": [1, 2]}
    (src.path(info.id) / "manifest.json").write_text(json.dumps(manifest))
    return root, info.id, bars


def test_gated_model_is_flat_when_the_primary_is_off_and_never_short(gated_model):
    from trader.labels.primary import PrimarySpec, primary_gate
    from trader.strategies import get_strategy
    from trader.strategies.base import Flat, Target

    root, model_id, bars = gated_model
    strat = get_strategy("gbm")(model=model_id, models_dir=root, threshold=0.0)
    gate = primary_gate(bars, HORIZON, PrimarySpec(lookbacks=(1, 2)))
    assert gate[400:].any() and not gate[400:].all()  # both states occur

    for i in range(400, len(bars), 13):
        decision = strat.on_bar(_ctx(bars, i))
        if not gate[i]:
            assert isinstance(decision, Flat), i
        if isinstance(decision, Target):
            assert decision.weight > 0, i  # long-only under a long-only primary
