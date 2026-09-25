"""``regime_v1``: causality, known-answer gap cases, ETF coverage, and the live recompute tail."""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
import pytest

from trader.features import compute_feature_frame, get_feature_set
from trader.features.base import FeatureSpec
from trader.features.regime import REGIME_COLUMNS, _regime_block


def _frame(times, opens, closes, *, spread=0.0):
    """A canonical bar frame; high / low wrap open and close by ``spread``."""
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    return bars_to_frame(
        [
            Bar(
                Time=t,
                open=float(o),
                high=max(o, c) + spread,
                low=min(o, c) - spread,
                close=float(c),
            )
            for t, o, c in zip(times, opens, closes, strict=True)
        ]
    )


def _session_times(days: int, *, horizon: int):
    """Bar open times 09:30-16:00 New York on weekdays, like an ETF."""
    per_day = 390 // horizon
    times = []
    day = pd.Timestamp("2024-01-02")
    while len(times) < days * per_day:
        if day.dayofweek < 5:
            open_ = (day + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York")
            times += [
                (open_ + pd.Timedelta(minutes=horizon * i)).tz_convert(UTC).to_pydatetime()
                for i in range(per_day)
            ]
        day += pd.Timedelta(days=1)
    return times, per_day


def _session_bars(days: int, *, horizon: int = 15, seed: int = 0):
    times, per_day = _session_times(days, horizon=horizon)
    rng = np.random.default_rng(seed)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(times))))
    opens = np.concatenate([[close[0]], close[:-1]])
    opens[::per_day] *= np.exp(rng.normal(0.0, 0.01, len(opens[::per_day])))  # overnight gaps
    return _frame(times, opens, close, spread=0.02)


# --- causality ----------------------------------------------------------------


@pytest.mark.parametrize("cut", [9000, 10400])
def test_regime_v1_is_causal(make_bars, cut):
    """Past rows must not move when every later bar is mangled -- daily stats included."""
    bars = make_bars(10560, horizon=15)  # 110 days of 24h 15m bars
    build = {"feature_set": "regime_v1", "base_horizon": 15, "context_horizons": [60]}
    reference, _ = compute_feature_frame(bars, **build)

    tampered = bars.copy()
    future = tampered.index > cut
    tampered.loc[future, ["open", "high", "low", "close"]] *= 3.0
    perturbed, _ = compute_feature_frame(tampered, **build)

    before = reference.iloc[: cut + 1]
    after = perturbed.iloc[: cut + 1]
    same = (before.isna() & after.isna()) | np.isclose(before.fillna(0), after.fillna(0))
    leaked = [c for c in before.columns if not same[c].all()]
    assert not leaked, f"future bars leaked into past rows via: {leaked}"
    assert before[list(REGIME_COLUMNS)].iloc[-1].notna().all()


# --- gaps ----------------------------------------------------------------------


def _gap_day_bars():
    """25 quiet hourly-ish days alternating closes 100 / 101, then a day that
    opens at 104 (a gap up from 100) and trades down to 102: half the gap filled."""
    times, per_day = _session_times(26, horizon=65)  # 6 bars a day
    closes, opens = [], []
    for d in range(25):
        level = 100.0 if d % 2 == 0 else 101.0
        opens += [level] * per_day
        closes += [level] * per_day
    # day 24 closes at 100; day 25 gaps to 104 and slides
    opens += [104.0, 104.0, 103.0, 102.0, 102.0, 102.5]
    closes += [104.0, 103.0, 102.0, 102.0, 102.5, 103.0]
    return _frame(times, opens, closes)


def test_gap_size_and_fill():
    bars = _gap_day_bars()
    block = _regime_block(bars, get_feature_set("regime_v1").resolve(base_horizon=65))
    last_day = block.iloc[-6:]

    assert (last_day["gap_z"] > 0).all()
    # the gap is ln(104/100) in units of a 20-day vol of alternating +/- 1% moves
    assert last_day["gap_z"].iloc[0] == pytest.approx(
        np.log(1.04) / pd.Series(np.log([101 / 100, 100 / 101] * 10)).std(), rel=1e-9
    )
    assert last_day["gap_fill"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.5, 0.5, 0.5])
    # five days before the gap closed at 101, the day before at 100: a gap up against the trend
    assert last_day["gap_trend"].iloc[0] == -1.0
    assert last_day["day_ret_z"].iloc[-1] < 0  # closed below the open


def test_prior_day_stats_ignore_the_current_day():
    """Moving today's bars must not move anything but today's intraday columns."""
    bars = _gap_day_bars()
    spec = get_feature_set("regime_v1").resolve(base_horizon=65)
    reference = _regime_block(bars, spec)
    tampered = bars.copy()
    tampered.loc[tampered.index[-3:], ["open", "high", "low", "close"]] *= 1.5
    perturbed = _regime_block(tampered, spec)

    daily_cols = ["dvol_20", "dvol_ratio", "dvol_pct", "dvolvol", "gap_z", "gap_trend"]
    pd.testing.assert_frame_equal(reference[daily_cols], perturbed[daily_cols])


# --- coverage and the live tail ----------------------------------------------------


def test_session_bound_instrument_has_no_nan_after_history():
    bars = _session_bars(110)
    feats, _ = compute_feature_frame(
        bars, feature_set="regime_v1", base_horizon=15, context_horizons=[60]
    )
    settled = feats.iloc[-26 * 20 :]  # the last 20 trading days

    assert not settled.isna().any().any(), settled.columns[settled.isna().any()].tolist()
    assert settled["gap_z"].abs().gt(0).any()


def test_level_days_tail_reproduces_the_full_history():
    """A live recompute over the last ``level_days`` matches the full-history row."""
    bars = _session_bars(150)
    spec = get_feature_set("regime_v1").resolve(base_horizon=15, context_horizons=(60,))
    full, _ = compute_feature_frame(bars, spec=spec)

    cutoff = bars["time"].iloc[-1] - pd.Timedelta(days=spec.level_days)
    tail = bars.iloc[int(bars["time"].searchsorted(cutoff, side="left")) :]
    live, _ = compute_feature_frame(tail, spec=spec)

    np.testing.assert_allclose(
        live.iloc[-1].to_numpy(), full.iloc[-1].to_numpy(), rtol=1e-6, atol=1e-9
    )


# --- spec ------------------------------------------------------------------------


def test_regime_spec_round_trips():
    spec = get_feature_set("regime_v1").resolve(base_horizon=15, context_horizons=(60,))
    structure = get_feature_set("structure_v1").resolve(base_horizon=15, context_horizons=(60,))
    assert spec.level_days == 120
    assert spec.columns == structure.columns + REGIME_COLUMNS
    again = FeatureSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.digest() == spec.digest()
