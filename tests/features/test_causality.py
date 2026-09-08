"""Causality gate: no future bar may change a past feature row.

If this fails, a model trained on these features is reading its own answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from trader.features import compute_feature_frame


def _columns_that_changed(before, after):
    both_nan = before.isna() & after.isna()
    equal = both_nan | np.isclose(before.fillna(0.0), after.fillna(0.0), equal_nan=True)
    return [col for col in before.columns if not equal[col].all()]


def _assert_unchanged_before(bars, cut, *, columns=None, **build):
    """Compute features, then again with every bar after ``cut`` mangled."""
    reference, _ = compute_feature_frame(bars, **build)

    tampered = bars.copy()
    future = tampered.index > cut
    tampered.loc[future, ["open", "high", "low", "close"]] *= 3.0
    tampered.loc[future, "volume"] *= 7.0
    perturbed, _ = compute_feature_frame(tampered, **build)

    cols = list(columns) if columns is not None else list(reference.columns)
    before = reference.iloc[: cut + 1][cols]
    after = perturbed.iloc[: cut + 1][cols]
    bad = _columns_that_changed(before, after)
    assert not bad, f"future bars leaked into past rows via: {bad}"


@pytest.mark.parametrize("cut", [200, 350])
def test_price_v1_is_causal(make_bars, cut):
    _assert_unchanged_before(make_bars(500), cut, feature_set="price_v1", base_horizon=5)


@pytest.mark.parametrize("cut", [200, 350])
def test_mtf_v1_is_causal_including_context(make_bars, cut):
    _assert_unchanged_before(
        make_bars(500),
        cut,
        feature_set="mtf_v1",
        base_horizon=5,
        context_horizons=[15, 60],
    )


def test_context_columns_specifically_are_causal(make_bars):
    """Zero in on the h*_ context columns: no future bar may move a past one."""
    bars = make_bars(600)
    _, spec = compute_feature_frame(
        bars, feature_set="mtf_v1", base_horizon=5, context_horizons=[15, 60]
    )
    ctx_cols = [c for c in spec.columns if c.startswith("h")]
    _assert_unchanged_before(
        bars,
        400,
        columns=ctx_cols,
        feature_set="mtf_v1",
        base_horizon=5,
        context_horizons=[15, 60],
    )
