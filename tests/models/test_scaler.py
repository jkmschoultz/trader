"""The hand-rolled StandardScaler: fit stats, transform, JSON round-trip."""

from __future__ import annotations

import numpy as np

from trader.models.scaler import StandardScaler


def test_fit_transform_centres_and_scales():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(400, 6, 4)).astype("float32")

    scaler = StandardScaler().fit(x)
    out = scaler.transform(x)

    assert out.dtype == np.float32
    flat = out.reshape(-1, 4)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-3)


def test_matches_numpy_by_hand():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(200, 4)).astype("float32")
    scaler = StandardScaler().fit(x)

    expected = (x - x.mean(axis=0)) / x.std(axis=0)
    assert np.allclose(scaler.transform(x), expected, atol=1e-4)


def test_constant_column_is_left_alone_not_divided_by_zero():
    x = np.c_[np.ones(50), np.arange(50.0)].astype("float32")
    out = StandardScaler().fit(x).transform(x)
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 0], out[0, 0])  # constant stays constant


def test_dict_round_trip():
    rng = np.random.default_rng(2)
    x = rng.normal(3.0, 2.0, size=(100, 5)).astype("float32")
    scaler = StandardScaler().fit(x)

    restored = StandardScaler.from_dict(scaler.to_dict())
    assert np.allclose(scaler.transform(x), restored.transform(x))


def test_nan_rows_do_not_poison_the_fit():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, size=(100, 3)).astype("float32")
    x[:10] = np.nan
    scaler = StandardScaler().fit(x)
    assert np.isfinite(scaler.mean_).all()
    assert np.isfinite(scaler.scale_).all()
