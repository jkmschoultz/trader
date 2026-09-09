"""``time_split``: chronological masks with purge and embargo."""

from __future__ import annotations

import pandas as pd
import pytest

from trader.labels.triple_barrier import LabelConfig
from trader.models.dataset import (
    SplitSpec,
    WindowSpec,
    build_bundle,
    time_split,
    walk_forward_splits,
)

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


def test_test_end_bounds_and_right_purges_the_test_set(bars):
    bundle = _bundle(bars)
    t = bundle.t
    train_end = pd.Timestamp(t[int(len(t) * 0.5)]).to_pydatetime()
    val_end = pd.Timestamp(t[int(len(t) * 0.65)]).to_pydatetime()
    test_end = pd.Timestamp(t[int(len(t) * 0.85)]).to_pydatetime()

    _, _, te = time_split(
        bundle,
        SplitSpec(train_end, val_end, embargo_bars=0, test_end=test_end),
        base_horizon=5,
        max_bars=MAX_BARS,
    )
    purge_ns = (1 + MAX_BARS) * 5 * 60 * 1_000_000_000
    assert (bundle.t[te] >= pd.Timestamp(val_end).value).all()
    assert (bundle.t[te] + purge_ns < pd.Timestamp(test_end).value).all()


def test_train_start_makes_a_rolling_window(bars):
    bundle = _bundle(bars)
    t = bundle.t
    train_start = pd.Timestamp(t[int(len(t) * 0.2)]).to_pydatetime()
    train_end = pd.Timestamp(t[int(len(t) * 0.6)]).to_pydatetime()
    val_end = pd.Timestamp(t[int(len(t) * 0.8)]).to_pydatetime()

    tr, _, _ = time_split(
        bundle,
        SplitSpec(train_end, val_end, embargo_bars=0, train_start=train_start),
        base_horizon=5,
        max_bars=MAX_BARS,
    )
    assert (bundle.t[tr] >= pd.Timestamp(train_start).value).all()


def test_walk_forward_splits_tile_the_tail(bars):
    bundle = _bundle(bars)
    span_days = (bundle.t.max() - bundle.t.min()) / 1e9 / 86400
    each = span_days / 8  # 3 folds: 3*each train + each val + each test, stepped each

    splits = walk_forward_splits(
        bundle.t,
        n_folds=3,
        train_days=each * 3,
        val_days=each,
        test_days=each,
        embargo_bars=10,
        mode="rolling",
    )
    assert len(splits) == 3
    # ordered, non-overlapping test windows, each a fixed-width rolling train
    for a, b in zip(splits, splits[1:], strict=False):
        assert b.train_end > a.train_end
        assert a.test_end <= b.val_end
    for s in splits:
        assert s.train_start is not None and s.test_end is not None
        width = s.train_end - s.train_start
        assert abs(width - (splits[0].train_end - splits[0].train_start)) < pd.Timedelta(minutes=1)


def test_walk_forward_splits_raises_when_span_too_short(bars):
    bundle = _bundle(bars)
    with pytest.raises(ValueError, match="not enough"):
        walk_forward_splits(bundle.t, n_folds=50, train_days=365, val_days=60, test_days=60)
