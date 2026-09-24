"""``structure_v1``: causality, known-answer cases, and coverage on a session-bound instrument."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trader.features import compute_feature_frame, get_feature_set
from trader.features.base import FeatureSpec
from trader.features.structure import _prior_period, _swing_block, trading_day


def _frame(times, close, *, spread=0.0):
    """A canonical bar frame from explicit times and closes (high/low = close +/- spread)."""
    from trader.data.lake import bars_to_frame
    from trader.saxo.charts import Bar

    return bars_to_frame(
        [
            Bar(
                Time=t, open=float(c), high=float(c) + spread, low=float(c) - spread, close=float(c)
            )
            for t, c in zip(times, close, strict=True)
        ]
    )


def _flat_atr(bars):
    return pd.Series(1.0, index=bars.index)


# --- causality ----------------------------------------------------------------


@pytest.mark.parametrize("cut", [5200, 6600])
def test_structure_v1_is_causal(make_bars, cut):
    """Past rows must not move when every later bar is mangled -- swings included."""
    bars = make_bars(7200)  # 25 days of 5m bars: prior-week levels exist before the cut
    build = {"feature_set": "structure_v1", "base_horizon": 5, "context_horizons": [15, 60]}
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


# --- swings and Fibonacci --------------------------------------------------------


def _leg_bars():
    """Down to a swing low of 100 at bar 20, up to a swing high of 110 at bar 40,
    then a pullback to 106.18 (a 0.382 retracement) at bar 50. Strictly monotone
    between the turns, so no other swing point is ever confirmed."""
    close = np.concatenate(
        [
            np.linspace(105.0, 100.0, 21),  # bars 0..20
            np.linspace(100.0, 110.0, 21)[1:],  # bars 21..40
            np.linspace(110.0, 106.18, 11)[1:],  # bars 41..50
        ]
    )
    start = datetime(2024, 3, 4, 12, 0, tzinfo=UTC)
    return _frame([start + timedelta(minutes=5 * i) for i in range(len(close))], close)


def test_swing_high_is_only_known_k_bars_later():
    bars = _leg_bars()
    cols = _swing_block(bars, _flat_atr(bars), k=3)
    last_hi = bars["close"] - cols["sw3_hi_dist"]  # atr == 1

    # the high prints at bar 40 but is unknowable until bars 41..43 fail to beat it
    assert not np.isclose(last_hi.iloc[42], 110.0)
    assert np.isclose(last_hi.iloc[43], 110.0)
    assert cols["sw3_hi_age"].iloc[43] == pytest.approx(3 / 100)


def test_fib_retracement_of_the_last_leg():
    bars = _leg_bars()
    cols = _swing_block(bars, _flat_atr(bars), k=3)
    last = bars.index[-1]

    assert cols["sw3_leg_dir"][last] == 1.0  # the newest swing point is the high
    assert cols["sw3_leg_atr"][last] == pytest.approx(10.0)
    assert cols["sw3_fib_retr"][last] == pytest.approx(0.382, abs=1e-9)
    assert cols["sw3_fib_382"][last] == pytest.approx(0.0, abs=1e-9)
    # 0.618 sits 0.236 of a 10-point leg further on: price is 2.36 ATR short of it
    assert cols["sw3_fib_618"][last] == pytest.approx(-2.36, abs=1e-9)
    assert cols["sw3_fib_near"][last] == pytest.approx(0.0, abs=1e-9)


# --- clocks and levels -------------------------------------------------------------


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (datetime(2024, 7, 1, 20, 55, tzinfo=UTC), datetime(2024, 7, 1, 21, 0, tzinfo=UTC)),  # EDT
        (datetime(2024, 1, 2, 21, 55, tzinfo=UTC), datetime(2024, 1, 2, 22, 0, tzinfo=UTC)),  # EST
    ],
)
def test_trading_day_rolls_at_five_pm_new_york_across_dst(before, after):
    days = trading_day(pd.DatetimeIndex([before, after]))
    assert days[1] - days[0] == pd.Timedelta(days=1)


def test_sunday_evening_belongs_to_monday():
    sunday_open = datetime(2024, 3, 10, 22, 0, tzinfo=UTC)  # 18:00 New York, Sunday
    assert trading_day(pd.DatetimeIndex([sunday_open]))[0].dayofweek == 0


def test_prior_period_levels():
    times = [datetime(2024, 3, 4, 14, 0, tzinfo=UTC) + timedelta(hours=i) for i in range(6)]
    bars = _frame(times, [10, 12, 11, 20, 18, 19], spread=0.5)
    period = np.array([1, 1, 1, 2, 2, 2])
    prior = _prior_period(bars, period)

    assert prior.iloc[:3].isna().all().all()  # nothing before the first period
    assert prior.iloc[3:]["high"].tolist() == [12.5] * 3
    assert prior.iloc[3:]["low"].tolist() == [9.5] * 3
    assert prior.iloc[3:]["close"].tolist() == [11.0] * 3


# --- coverage ------------------------------------------------------------------


def _us_session_bars(days: int, *, seed: int = 0):
    """5m bars 09:30-16:00 New York on weekdays only, like an ETF."""
    rng = np.random.default_rng(seed)
    times = []
    day = pd.Timestamp("2024-03-04")  # naive: step wall-clock days, spanning the DST change
    while len(times) < days * 78:
        if day.dayofweek < 5:
            open_ = (day + pd.Timedelta(hours=9, minutes=30)).tz_localize("America/New_York")
            times += [(open_ + pd.Timedelta(minutes=5 * i)).tz_convert(UTC) for i in range(78)]
        day += pd.Timedelta(days=1)
    close = 50.0 + np.cumsum(rng.normal(0.0, 0.05, len(times)))
    return _frame([t.to_pydatetime() for t in times], close, spread=0.03)


def test_session_bound_instrument_has_no_all_nan_columns():
    """An ETF never trades Asian hours: those columns are 0 with a flag, not NaN."""
    bars = _us_session_bars(25)
    feats, _ = compute_feature_frame(
        bars, feature_set="structure_v1", base_horizon=5, context_horizons=[15, 60]
    )
    settled = feats.iloc[len(feats) // 2 :]  # well past the first prior week

    assert not settled.isna().any().any(), settled.columns[settled.isna().any()].tolist()
    assert (settled["asia_ok"] == 0).all()
    assert (settled[["asia_hi_dist", "asia_lo_dist", "asia_range_atr"]] == 0).all().all()
    assert (settled["or_ok"] == 1).all()  # the NY opening range always exists


# --- spec ------------------------------------------------------------------------


def test_existing_specs_serialise_exactly_as_before():
    """The new spec fields are omitted at their defaults, so old digests are unchanged."""
    spec = get_feature_set("mtf_v1").resolve(base_horizon=5, context_horizons=(15,))
    assert "swing_orders" not in spec.to_dict()
    assert "level_days" not in spec.to_dict()


def test_structure_spec_round_trips():
    spec = get_feature_set("structure_v1").resolve(base_horizon=15, context_horizons=(60,))
    assert spec.swing_orders == (3, 12)
    assert spec.level_days == 15
    again = FeatureSpec.from_dict(spec.to_dict())
    assert again == spec
    assert again.digest() == spec.digest()
