"""``build_bundle``: window alignment, label mapping, no-leak boundary."""

from __future__ import annotations

import numpy as np
import pytest

from trader.features import compute_feature_frame
from trader.labels.triple_barrier import LabelConfig
from trader.models.dataset import WindowSpec, build_bundle

LABEL = LabelConfig(stop=0.01, take=0.01, max_bars=12)


def _spec(window=16, feature_set="price_v1", context=()):
    return WindowSpec(
        window=window,
        feature_set=feature_set,
        base_horizon=5,
        context_horizons=tuple(context),
        label=LABEL,
    )


def test_shapes_and_dtypes(bars):
    bundle = build_bundle(bars(1500), spec=_spec())
    assert bundle.X.ndim == 3
    n, window, n_feat = bundle.X.shape
    assert window == 16
    assert n_feat == len(bundle.feature_names)
    assert bundle.X.dtype == np.float32
    assert bundle.y.shape == (n,) and bundle.y.dtype == np.int64
    assert bundle.w.shape == (n,) and np.all(bundle.w == 1.0)
    assert bundle.t.shape == (n,) and np.all(np.diff(bundle.t) > 0)


def test_labels_are_mapped_to_zero_one_two(bars):
    bundle = build_bundle(bars(1500), spec=_spec())
    assert set(np.unique(bundle.y)) <= {0, 1, 2}


def test_window_ends_at_the_decision_bar_and_starts_window_minus_one_back(bars):
    frame = bars(1200)
    spec = _spec(window=16)
    bundle = build_bundle(frame, spec=spec)

    feats, _ = compute_feature_frame(frame, feature_set="price_v1", base_horizon=5)
    feat_ns = (
        feats.index.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]").astype("int64")
    )

    row = int(np.where(feat_ns == bundle.t[0])[0][0])
    fv = feats.to_numpy(np.float32)
    assert np.allclose(bundle.X[0, -1], fv[row], equal_nan=True)
    assert np.allclose(bundle.X[0, 0], fv[row - 15], equal_nan=True)


def test_a_label_only_reads_bars_after_its_decision_bar(bars):
    """Mangle the bars up to and including a decision bar; its label must not move."""
    frame = bars(1200)
    spec = _spec(window=16)
    reference = build_bundle(frame, spec=spec)

    feats, _ = compute_feature_frame(frame, feature_set="price_v1", base_horizon=5)
    feat_ns = (
        feats.index.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]").astype("int64")
    )
    decision_row = int(np.where(feat_ns == reference.t[100])[0][0])

    tampered = frame.copy()
    upto = tampered.index <= decision_row
    tampered.loc[upto, ["open", "high", "low", "close"]] *= 1.5
    tampered.loc[upto, "volume"] *= 3.0
    perturbed = build_bundle(tampered, spec=spec)

    # sample 100's window is now different, but its forward-looking label is not
    j = int(np.where(perturbed.t == reference.t[100])[0][0])
    assert perturbed.y[j] == reference.y[100]


def test_too_few_bars_raises(bars):
    with pytest.raises(ValueError, match="window"):
        build_bundle(bars(20), spec=_spec(window=64))


def test_context_horizons_add_feature_columns(bars):
    plain = build_bundle(bars(1500), spec=_spec(feature_set="mtf_v1"))
    withctx = build_bundle(bars(1500), spec=_spec(feature_set="mtf_v1", context=(15, 60)))
    assert withctx.X.shape[-1] > plain.X.shape[-1]
    assert any(name.startswith("h15_") for name in withctx.feature_names)


def test_sample_weights_are_carried_through(bars):
    import pandas as pd

    frame = bars(1200)
    spec = _spec()
    base = build_bundle(frame, spec=spec)
    weights = pd.Series(2.0, index=pd.DatetimeIndex(frame["time"]))
    weighted = build_bundle(frame, spec=spec, weights=weights)
    assert np.allclose(weighted.w, 2.0)
    assert len(weighted) == len(base)
