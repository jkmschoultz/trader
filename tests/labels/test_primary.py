"""The meta-labelling primary: the tsmom gate from completed days, vectorised == live."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trader.labels.primary import PrimarySpec, daily_conviction, gate_now, primary_gate

SPEC = PrimarySpec(lookbacks=(3, 10))


def _hourly(days=40, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(times))))
    return pd.DataFrame({"time": times, "open": close, "high": close, "low": close, "close": close})


def test_an_hourly_bar_sees_only_days_complete_by_its_close():
    frame = _hourly()
    conv = daily_conviction(frame, 60, SPEC)
    daily = frame.groupby(frame["time"].dt.floor("D"))["close"].last()
    expected_day = {
        "2024-01-20 22:00": "2024-01-19",  # closes 23:00 on the 20th: the 20th is not done
        "2024-01-20 23:00": "2024-01-20",  # closes at midnight: the 20th is done
    }
    for bar, day in expected_day.items():
        i = int(np.flatnonzero(frame["time"] == pd.Timestamp(bar, tz="UTC"))[0])
        d = daily.index.get_loc(pd.Timestamp(day, tz="UTC"))
        want = np.mean([np.sign(daily.iloc[d] - daily.iloc[d - k]) for k in SPEC.lookbacks])
        assert conv[i] == pytest.approx(want)


def test_nan_until_the_longest_lookback_has_history():
    conv = daily_conviction(_hourly(), 60, SPEC)
    assert np.isnan(conv[: 10 * 24]).all()
    assert not np.isnan(conv[12 * 24 :]).any()


def test_live_gate_matches_the_vectorised_gate():
    frame = _hourly(days=60, seed=3)
    full = primary_gate(frame, 60, SPEC)
    for i in range(15 * 24, len(frame), 7):
        assert gate_now(frame.iloc[: i + 1], 60, SPEC) == full[i], i


def test_future_bars_do_not_move_the_gate():
    frame = _hourly(days=50, seed=5)
    cut = 30 * 24 + 5
    tampered = frame.copy()
    tampered.loc[tampered.index > cut, ["open", "high", "low", "close"]] *= 3
    a = daily_conviction(frame, 60, SPEC)[: cut + 1]
    b = daily_conviction(tampered, 60, SPEC)[: cut + 1]
    np.testing.assert_array_equal(np.nan_to_num(a, nan=9), np.nan_to_num(b, nan=9))


def test_on_daily_bars_it_is_the_tsmom_conviction():
    from trader.strategies.tsmom import TimeSeriesMomentum

    rng = np.random.default_rng(1)
    times = pd.date_range("2024-01-01", periods=80, freq="D", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 80)))
    frame = pd.DataFrame({"time": times, "close": close})
    conv = daily_conviction(frame, 1440, SPEC)
    strat = TimeSeriesMomentum(lookbacks=SPEC.lookbacks)
    for i in range(20, 80):
        assert conv[i] == pytest.approx(strat.conviction(frame["close"].iloc[: i + 1]))


def test_spec_round_trips_and_parses():
    spec = PrimarySpec.parse("tsmom", "21/63")
    assert spec.lookbacks == (21, 63)
    assert PrimarySpec.from_dict(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        PrimarySpec(name="macd")
