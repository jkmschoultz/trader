"""``compute_feature_frame``: shape, columns, MTF alignment, spec round-trip."""

from __future__ import annotations

import pandas as pd
import pytest

from trader.features import FeatureSpec, compute_feature_frame
from trader.features.registry import UnknownFeatureSet


def test_price_v1_shape_and_index(make_bars):
    bars = make_bars(400)
    feats, spec = compute_feature_frame(bars, feature_set="price_v1", base_horizon=5)

    assert spec.name == "price_v1"
    assert len(feats) == len(bars)
    assert list(feats.columns) == list(spec.columns)
    assert feats.index.name == "time"
    assert feats.index.equals(pd.DatetimeIndex(bars["time"], name="time"))
    # warmup NaNs at the head, real values once warmed up
    assert feats.iloc[0].isna().any()
    assert not feats.iloc[-1].isna().any()


def test_mtf_v1_adds_prefixed_context_columns(make_bars):
    bars = make_bars(500)
    feats, spec = compute_feature_frame(
        bars, feature_set="mtf_v1", base_horizon=5, context_horizons=[15, 60]
    )

    assert spec.context_horizons == (15, 60)
    for horizon in (15, 60):
        cols = [c for c in feats.columns if c.startswith(f"h{horizon}_")]
        assert cols, f"no h{horizon}_ columns"
        # present (non-NaN) well before the end of the series
        assert feats[cols].notna().any(axis=1).iloc[-1]


def test_context_horizon_must_divide_the_base(make_bars):
    bars = make_bars(200)
    with pytest.raises(ValueError, match="not a multiple"):
        compute_feature_frame(bars, feature_set="mtf_v1", base_horizon=5, context_horizons=[7])


def test_a_frozen_spec_reproduces_the_frame_exactly(make_bars):
    bars = make_bars(400)
    first, spec = compute_feature_frame(
        bars, feature_set="mtf_v1", base_horizon=5, context_horizons=[15]
    )
    round_tripped = FeatureSpec.from_dict(spec.to_dict())
    assert round_tripped == spec

    again, _ = compute_feature_frame(bars, spec=round_tripped)
    pd.testing.assert_frame_equal(first, again)


def test_unknown_feature_set_raises(make_bars):
    with pytest.raises(UnknownFeatureSet, match="price_v1"):
        compute_feature_frame(make_bars(50), feature_set="nope", base_horizon=5)


def test_volume_less_bars_do_not_produce_all_nan_columns(make_bars):
    """FX / quote-driven CFDs carry no volume; features must still be usable."""
    bars = make_bars(400)
    bars["volume"] = float("nan")

    feats, spec = compute_feature_frame(bars, feature_set="price_v1", base_horizon=5)
    warm = feats.iloc[spec.warmup + 5 :]
    assert not warm.isna().all().any(), "an all-NaN feature column would drop every window"
    assert (warm["volz"] == 0.0).all()  # neutral, not NaN
    assert warm["vwap_dist"].notna().all()  # unweighted fallback


def test_tail_slice_reproduces_the_last_rows(make_bars):
    """``LSTMStrategy.on_bar`` computes features on a bounded tail of history, not
    the whole thing (that is the O(N^2) fix). The retained rows must match a
    full-history compute to well under float tolerance -- the tail keeps
    ``warmup + _TAIL_PAD * context_ratio`` bars, enough EMA burn-in that the
    residual is numerical noise."""
    from trader.strategies.lstm import _TAIL_PAD

    bars = make_bars(6000, seed=7)
    window = 32

    for ctx in ([], [15, 60]):
        fs = "mtf_v1" if ctx else "price_v1"
        full, spec = compute_feature_frame(
            bars, feature_set=fs, base_horizon=5, context_horizons=ctx
        )
        ratio = max((-(-h // 5) for h in spec.context_horizons), default=1)
        tail = bars.iloc[-(window + spec.warmup + _TAIL_PAD * ratio) :]
        sliced, _ = compute_feature_frame(tail, spec=spec)

        a = full.to_numpy("float64")[-window:]
        b = sliced.to_numpy("float64")[-window:]
        assert pd.notna(a).all() and pd.notna(b).all()
        assert abs(b - a).max() < 1e-5


def test_digest_changes_with_the_spec(make_bars):
    _, price = compute_feature_frame(make_bars(120), feature_set="price_v1", base_horizon=5)
    _, mtf = compute_feature_frame(
        make_bars(120), feature_set="mtf_v1", base_horizon=5, context_horizons=[15]
    )
    assert price.digest() != mtf.digest()
