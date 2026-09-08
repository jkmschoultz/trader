"""The technical primitives: bounds, known values, and no lookahead."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trader.features import indicators as ind


def test_log_return_of_a_constant_series_is_zero():
    close = pd.Series([100.0] * 20)
    assert ind.log_return(close, 1).dropna().abs().max() == 0.0


def test_log_return_matches_the_definition():
    close = pd.Series([100.0, 110.0, 121.0])
    got = ind.log_return(close, 1)
    assert got.iloc[1] == np.log(1.1)
    assert got.iloc[2] == np.log(1.1)


def test_rsi_stays_in_range_and_reads_100_on_a_pure_uptrend():
    close = pd.Series(np.arange(1, 60, dtype=float))
    rsi = ind.rsi(close, 14).dropna()
    assert ((rsi >= 0.0) & (rsi <= 100.0)).all()
    assert rsi.iloc[-1] == 100.0


def test_macd_of_a_constant_series_is_zero():
    close = pd.Series([50.0] * 80)
    line, signal, hist = ind.macd(close)
    assert line.abs().max() == 0.0
    assert signal.abs().max() == 0.0
    assert hist.abs().max() == 0.0


def test_atr_is_non_negative():
    frame = pd.DataFrame(
        {
            "high": np.arange(10, 40, dtype=float),
            "low": np.arange(9, 39, dtype=float),
            "close": np.arange(9.5, 39.5, dtype=float),
        }
    )
    assert (ind.atr(frame, 14).dropna() >= 0.0).all()


def test_volume_zscore_flags_a_spike():
    rng = np.random.default_rng(0)
    volume = pd.Series(np.r_[1_000.0 + rng.normal(0, 20, 40), [10_000.0]])
    z = ind.volume_zscore(volume, 20)
    assert abs(z.iloc[39]) < 3.0  # ordinary bar sits within a few sigma
    assert z.iloc[40] > 3.0  # the spike stands well clear of the noise


def test_every_primitive_is_causal():
    rng = np.random.default_rng(1)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 200)))
    frame = pd.DataFrame(
        {"high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1_000.0}
    )
    cut = 120
    tampered_close = close.copy()
    tampered_close.iloc[cut + 1 :] *= 3.0
    tampered_frame = frame.copy()
    tampered_frame.iloc[cut + 1 :, :] *= 3.0

    pairs = [
        (ind.log_return(close, 5), ind.log_return(tampered_close, 5)),
        (ind.rsi(close, 14), ind.rsi(tampered_close, 14)),
        (ind.macd(close)[2], ind.macd(tampered_close)[2]),
        (ind.atr(frame, 14), ind.atr(tampered_frame, 14)),
        (ind.volume_zscore(frame["volume"], 20), ind.volume_zscore(tampered_frame["volume"], 20)),
    ]
    for original, perturbed in pairs:
        a, b = original.iloc[: cut + 1], perturbed.iloc[: cut + 1]
        assert ((a.isna() & b.isna()) | (a == b)).all()
