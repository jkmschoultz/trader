"""``time_split``: chronological masks with purge and embargo."""

from __future__ import annotations

import pandas as pd
import pytest

from trader.labels.triple_barrier import LabelConfig
from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split

MAX_BARS = 12
LABEL = LabelConfig(stop=0.01, take=0.01, max_bars=MAX_BARS)
SPEC = WindowSpec(
    window=16, feature_set="price_v1", base_horizon=5, context_horizons=(), label=LABEL
)


def _bundle(bars):
    return build_bundle(bars(2000), spec=SPEC)


def _dates(bundle):
    t = bundle.t
    return (
        pd.Timestamp(t[int(len(t) * 0.6)]).to_pydatetime(),
        pd.Timestamp(t[int(len(t) * 0.8)]).to_pydatetime(),
    )


def test_masks_are_disjoint_and_ordered(bars):
    bundle = _bundle(bars)
    train_end, val_end = _dates(bundle)
    tr, va, te = time_split(
        bundle, SplitSpec(train_end, val_end), base_horizon=5, max_bars=MAX_BARS
    )

    assert not (tr & va).any()
    assert not (va & te).any()
    assert not (tr & te).any()
    assert bundle.t[tr].max() < bundle.t[va].min()
    assert bundle.t[va].max() < bundle.t[te].min()


def test_purge_drops_training_labels_that_reach_past_the_boundary(bars):
    bundle = _bundle(bars)
    train_end, val_end = _dates(bundle)
    tr, _, _ = time_split(
        bundle, SplitSpec(train_end, val_end, embargo_bars=0), base_horizon=5, max_bars=MAX_BARS
    )

    purge_ns = (1 + MAX_BARS) * 5 * 60 * 1_000_000_000
    train_end_ns = pd.Timestamp(train_end).value
    assert (bundle.t[tr] + purge_ns < train_end_ns).all()


def test_embargo_removes_a_band_before_each_boundary(bars):
    bundle = _bundle(bars)
    train_end, val_end = _dates(bundle)
    embargo = 20
    tr, va, _ = time_split(
        bundle,
        SplitSpec(train_end, val_end, embargo_bars=embargo),
        base_horizon=5,
        max_bars=MAX_BARS,
    )

    band_ns = embargo * 5 * 60 * 1_000_000_000
    assert (bundle.t[tr] < pd.Timestamp(train_end).value - band_ns).all()
    assert (bundle.t[va] < pd.Timestamp(val_end).value - band_ns).all()


def test_test_set_is_everything_after_val_end(bars):
    bundle = _bundle(bars)
    train_end, val_end = _dates(bundle)
    _, _, te = time_split(bundle, SplitSpec(train_end, val_end), base_horizon=5, max_bars=MAX_BARS)
    assert (bundle.t[te] >= pd.Timestamp(val_end).value).all()


def test_val_end_must_be_after_train_end():
    with pytest.raises(ValueError, match="val_end"):
        SplitSpec(
            train_end=pd.Timestamp("2024-06-01").to_pydatetime(),
            val_end=pd.Timestamp("2024-05-01").to_pydatetime(),
        )
