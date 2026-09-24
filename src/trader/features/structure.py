"""``structure_v1``: market-session and price-structure features on top of ``mtf_v1``.

Two groups, both built from OHLC alone (no volume, so FX works):

**Sessions and time** -- where in the trading day a bar sits. FX behaviour is
session-driven (Asia ranges, London breakouts, the New York open, the London
4pm fix), and equity-hours instruments key off the cash open.

**Levels and structure** -- where price sits relative to levels traders watch:
prior day / week high, low and close, floor pivots, round numbers, the day's
running range, confirmed swing highs and lows, market structure (higher highs /
higher lows), and the Fibonacci retracement of the most recent swing leg.

Conventions that make one set work for a 24-hour FX pair and a 6.5-hour ETF:

- Clocks are fixed market clocks, not the instrument's inferred session:
  ``America/New_York`` and ``Europe/London``, both DST-aware.
- The **trading day** rolls at 17:00 New York (the FX convention), so Sunday
  evening belongs to Monday and a US cash session sits inside one day.
- Distances are in **ATR units** (``(close - level) / ATR``), so a 20-pip
  EURUSD move and a $600 BTC move read on the same scale.
- A column that cannot exist for an instrument (the Asian range of an ETF that
  only trades New York hours) is **0 with a ``*_ok`` flag**, never NaN -- an
  all-NaN column would drop every training row. A level that merely needs more
  history (prior week, a first swing) *is* NaN until it exists, like warmup.

Causality: every column at bar ``i`` uses bars ``0..i`` only. A swing point is
the trap -- a high at bar ``j`` is only known to be a swing high once ``k``
later bars have failed to exceed it, so it is recorded at ``j + k``, never at
``j``. ``tests/features/test_structure.py`` pins that and the rest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from trader.features import indicators as ind
from trader.features.base import FeatureSpec
from trader.features.registry import register_feature_set
from trader.features.sets import MtfV1, _context_columns, _price_columns

_NY = ZoneInfo("America/New_York")
_LDN = ZoneInfo("Europe/London")

#: swing orders used by ``structure_v1`` unless overridden: short and medium swings
DEFAULT_SWING_ORDERS: tuple[int, ...] = (3, 12)
#: calendar days the level features need -- covers a prior week plus a weekend
DEFAULT_LEVEL_DAYS = 15
#: Fibonacci retracement levels measured from the end of the leg
FIB_LEVELS: tuple[float, ...] = (0.382, 0.5, 0.618, 0.786)
#: level distances are clipped to +/- this many ATRs
_DIST_CLIP = 25.0

# minutes-of-day anchors (local wall clock)
_LDN_OPEN, _LDN_CLOSE, _LDN_FIX = 8 * 60, 16 * 60 + 30, 16 * 60
_NY_SESSION_OPEN, _NY_SESSION_CLOSE = 8 * 60, 17 * 60
_NY_CASH_OPEN, _NY_CASH_CLOSE = 9 * 60 + 30, 16 * 60
_OPENING_RANGE_END = 10 * 60  # New York opening range: 09:30-10:00
_ROLL = 17 * 60  # trading-day roll, New York

SESSION_COLUMNS: tuple[str, ...] = (
    "sess_asia",
    "sess_london",
    "sess_ny",
    "sess_overlap",
    "london_elapsed",
    "ny_elapsed",
    "fix_hours",
    "dow_sin",
    "dow_cos",
    "asia_ok",
    "asia_hi_dist",
    "asia_lo_dist",
    "asia_range_atr",
    "or_ok",
    "or_hi_dist",
    "or_lo_dist",
    "or_range_atr",
)

LEVEL_COLUMNS: tuple[str, ...] = (
    "pdh_dist",
    "pdl_dist",
    "pdc_dist",
    "pd_range_atr",
    "pwh_dist",
    "pwl_dist",
    "dh_dist",
    "dl_dist",
    "piv_p_dist",
    "piv_r1_dist",
    "piv_s1_dist",
    "round_fine_dist",
    "round_coarse_dist",
)

_SWING_SUFFIXES: tuple[str, ...] = (
    "hi_dist",
    "lo_dist",
    "hi_age",
    "lo_age",
    "hh",
    "hl",
    "leg_dir",
    "leg_atr",
    "fib_retr",
    *(f"fib_{round(level * 1000)}" for level in FIB_LEVELS),
    "fib_near",
)


def swing_columns(orders: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"sw{k}_{s}" for k in orders for s in _SWING_SUFFIXES)


def structure_columns(spec: FeatureSpec) -> tuple[str, ...]:
    return SESSION_COLUMNS + LEVEL_COLUMNS + swing_columns(spec.swing_orders)


# --- clocks ------------------------------------------------------------------


def _minutes(local: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(local.hour * 60 + local.minute)


def trading_day(times: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The trading day each bar belongs to: New York date, rolling at 17:00."""
    ny = pd.DatetimeIndex(times).tz_convert(_NY)
    return (ny.tz_localize(None) + pd.Timedelta(minutes=24 * 60 - _ROLL)).normalize()


def _trading_week(day: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monday of each trading day's week."""
    return day - pd.to_timedelta(day.dayofweek, unit="D")


# --- building blocks -----------------------------------------------------------


def _running_range(
    high: pd.Series, low: pd.Series, member: np.ndarray, day: np.ndarray
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Running high / low of the ``member`` bars so far in each day, held after.

    Returns ``(hi, lo, ok)``: NaN / NaN / 0 until the day's first member bar.
    """
    frame = pd.DataFrame(
        {
            "h": high.where(member).to_numpy(),
            "l": low.where(member).to_numpy(),
            "day": day,
        }
    )
    grouped = frame.groupby("day")
    hi = grouped["h"].cummax()
    lo = grouped["l"].cummin()
    # cummax skips NaN but leaves NaN rows NaN: hold the last value within the day
    hi = hi.groupby(frame["day"]).ffill()
    lo = lo.groupby(frame["day"]).ffill()
    ok = hi.notna().astype(float)
    return hi.set_axis(high.index), lo.set_axis(high.index), ok.set_axis(high.index)


def _prior_period(bars: pd.DataFrame, period: np.ndarray) -> pd.DataFrame:
    """Each bar's *previous* period's high / low / close (NaN for the first period)."""
    frame = pd.DataFrame(
        {
            "high": bars["high"].to_numpy(),
            "low": bars["low"].to_numpy(),
            "close": bars["close"].to_numpy(),
            "period": period,
        }
    )
    agg = frame.groupby("period", sort=True).agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    prior = agg.shift(1)
    out = prior.reindex(frame["period"]).reset_index(drop=True)
    out.index = bars.index
    return out


def _swing_block(bars: pd.DataFrame, atr: pd.Series, k: int) -> dict[str, pd.Series]:
    """Swing-point, structure and Fibonacci columns for one swing order ``k``."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    n = len(bars)
    pos = pd.Series(np.arange(n, dtype=float), index=bars.index)

    # a swing high at j = i - k is confirmed at i when high[j] is the max of [j-k, j+k]
    cand_hi = high.shift(k)
    cand_lo = low.shift(k)
    hi_conf = cand_hi.notna() & (cand_hi >= high.rolling(2 * k + 1).max())
    lo_conf = cand_lo.notna() & (cand_lo <= low.rolling(2 * k + 1).min())

    def last_and_prev(values: pd.Series, conf: pd.Series):
        events = values[conf]
        last = events.reindex(bars.index).ffill()
        prev = events.shift(1).reindex(bars.index).ffill()
        where = (pos - k)[conf].reindex(bars.index).ffill()
        return last, prev, where

    last_hi, prev_hi, hi_pos = last_and_prev(cand_hi, hi_conf)
    last_lo, prev_lo, lo_pos = last_and_prev(cand_lo, lo_conf)

    # the most recent leg runs into whichever swing point is newer
    up_leg = hi_pos >= lo_pos
    # a gap can leave the older swing high below the newer swing low: measure
    # the leg as a size either way rather than dropping the row
    leg = (last_hi - last_lo).abs().where(lambda s: s > 0)
    retr = pd.Series(
        np.where(up_leg, (last_hi - close) / leg, (close - last_lo) / leg), index=bars.index
    ).clip(-1.0, 2.0)
    known = last_hi.notna() & last_lo.notna()

    out: dict[str, pd.Series] = {
        "hi_dist": (close - last_hi) / atr,
        "lo_dist": (close - last_lo) / atr,
        "hi_age": (pos - hi_pos) / 100.0,
        "lo_age": (pos - lo_pos) / 100.0,
        "hh": np.sign(last_hi - prev_hi),
        "hl": np.sign(last_lo - prev_lo),
        "leg_dir": pd.Series(np.where(up_leg, 1.0, -1.0), index=bars.index).where(known),
        "leg_atr": leg / atr,
        "fib_retr": retr,
    }
    fib_dists = []
    for level in FIB_LEVELS:
        # signed ATR distance of price from the level, along the retracement axis
        dist = (retr - level) * leg / atr
        out[f"fib_{round(level * 1000)}"] = dist
        fib_dists.append(dist.abs())
    out["fib_near"] = pd.concat(fib_dists, axis=1).min(axis=1, skipna=False)
    return {f"sw{k}_{name}": series for name, series in out.items()}


def _structure_block(bars: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Every ``structure_v1``-only column, indexed like ``bars``."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    times = pd.DatetimeIndex(bars["time"])
    horizon = pd.Timedelta(minutes=spec.base_horizon)
    atr = ind.atr(bars, spec.atr_window)

    ny = times.tz_convert(_NY)
    ldn = times.tz_convert(_LDN)
    ny_m, ldn_m = _minutes(ny), _minutes(ldn)
    day = np.asarray(trading_day(times))
    week = np.asarray(_trading_week(pd.DatetimeIndex(day)))

    out: dict[str, pd.Series | np.ndarray] = {}

    # --- sessions and time ---------------------------------------------------
    asia = (ny_m >= _ROLL) | (ldn_m < _LDN_OPEN)
    london = (ldn_m >= _LDN_OPEN) & (ldn_m < _LDN_CLOSE)
    new_york = (ny_m >= _NY_SESSION_OPEN) & (ny_m < _NY_SESSION_CLOSE)
    out["sess_asia"] = asia.astype(float)
    out["sess_london"] = london.astype(float)
    out["sess_ny"] = new_york.astype(float)
    out["sess_overlap"] = (london & new_york).astype(float)
    out["london_elapsed"] = np.clip((ldn_m - _LDN_OPEN) / (_LDN_CLOSE - _LDN_OPEN), 0.0, 1.0)
    out["ny_elapsed"] = np.clip((ny_m - _NY_CASH_OPEN) / (_NY_CASH_CLOSE - _NY_CASH_OPEN), 0, 1)
    # the fix window only exists on London trading hours; hold at the clip outside
    out["fix_hours"] = np.clip((ldn_m - _LDN_FIX) / 60.0, -3.0, 3.0)
    dow = pd.DatetimeIndex(day).dayofweek.to_numpy()
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # Asian range: bars from the roll to the London open, running then held
    a_hi, a_lo, a_ok = _running_range(high, low, asia, day)
    out["asia_ok"] = a_ok
    out["asia_hi_dist"] = ((close - a_hi) / atr).where(a_ok > 0, 0.0)
    out["asia_lo_dist"] = ((close - a_lo) / atr).where(a_ok > 0, 0.0)
    out["asia_range_atr"] = ((a_hi - a_lo) / atr).where(a_ok > 0, 0.0)

    # New York opening range 09:30-10:00: any bar overlapping the window counts,
    # so a 1h base bar starting 09:00 still contributes
    start_m = ny_m
    end_m = start_m + int(horizon.total_seconds() // 60)
    in_or = (start_m < _OPENING_RANGE_END) & (end_m > _NY_CASH_OPEN)
    o_hi, o_lo, o_ok = _running_range(high, low, in_or, day)
    out["or_ok"] = o_ok
    out["or_hi_dist"] = ((close - o_hi) / atr).where(o_ok > 0, 0.0)
    out["or_lo_dist"] = ((close - o_lo) / atr).where(o_ok > 0, 0.0)
    out["or_range_atr"] = ((o_hi - o_lo) / atr).where(o_ok > 0, 0.0)

    # --- levels ---------------------------------------------------------------
    pd_ = _prior_period(bars, day)
    out["pdh_dist"] = (close - pd_["high"]) / atr
    out["pdl_dist"] = (close - pd_["low"]) / atr
    out["pdc_dist"] = (close - pd_["close"]) / atr
    out["pd_range_atr"] = (pd_["high"] - pd_["low"]) / atr
    pw = _prior_period(bars, week)
    out["pwh_dist"] = (close - pw["high"]) / atr
    out["pwl_dist"] = (close - pw["low"]) / atr

    frame = pd.DataFrame({"h": high.to_numpy(), "l": low.to_numpy(), "day": day})
    day_hi = frame.groupby("day")["h"].cummax().set_axis(bars.index)
    day_lo = frame.groupby("day")["l"].cummin().set_axis(bars.index)
    out["dh_dist"] = (close - day_hi) / atr
    out["dl_dist"] = (close - day_lo) / atr

    pivot = (pd_["high"] + pd_["low"] + pd_["close"]) / 3.0
    out["piv_p_dist"] = (close - pivot) / atr
    out["piv_r1_dist"] = (close - (2 * pivot - pd_["low"])) / atr
    out["piv_s1_dist"] = (close - (2 * pivot - pd_["high"])) / atr

    # round numbers: a price-magnitude grid (EURUSD 1.0800 -> 0.01, the "big
    # figure"; $60 -> 0.1; $100k -> 1000) and a 10x coarser one
    fine = 10.0 ** (np.floor(np.log10(close.where(close > 0))) - 2)
    for name, step in (("round_fine_dist", fine), ("round_coarse_dist", fine * 10)):
        out[name] = (close - (close / step).round() * step) / atr

    for k in spec.swing_orders:
        out.update(_swing_block(bars, atr, k))

    block = pd.DataFrame({name: np.asarray(col, dtype=float) for name, col in out.items()})
    # a level 80 ATR away carries no more information than one 25 away, and the
    # long tail would dominate a standardised input
    dist_cols = [c for c in block.columns if c.endswith(("_dist", "_atr")) or "_fib_" in c]
    block[dist_cols] = block[dist_cols].clip(-_DIST_CLIP, _DIST_CLIP)
    block.index = pd.DatetimeIndex(bars["time"], name="time")
    return block


@register_feature_set("structure_v1")
class StructureV1(MtfV1):
    """mtf_v1 plus session / key-level / swing-structure / Fibonacci features."""

    context_capable = True

    def resolve(
        self,
        *,
        base_horizon: int,
        context_horizons: tuple[int, ...] = (),
        **overrides: object,
    ) -> FeatureSpec:
        overrides.setdefault("swing_orders", DEFAULT_SWING_ORDERS)
        overrides.setdefault("level_days", DEFAULT_LEVEL_DAYS)
        ctx = tuple(sorted(int(h) for h in context_horizons))
        spec = FeatureSpec(
            name="structure_v1",
            base_horizon=base_horizon,
            context_horizons=ctx,
            **overrides,  # type: ignore[arg-type]
        )
        columns = _price_columns(spec) + _context_columns(spec) + structure_columns(spec)
        return replace(spec, columns=columns)

    def compute(
        self,
        bars: pd.DataFrame,
        *,
        spec: FeatureSpec,
        context: Mapping[int, pd.DataFrame],
        session,
    ) -> pd.DataFrame:
        mtf_spec = replace(spec, columns=_price_columns(spec) + _context_columns(spec))
        base = super().compute(bars, spec=mtf_spec, context=context, session=session)
        structure = _structure_block(bars, spec)
        merged = pd.concat([base, structure], axis=1)
        return merged.reindex(columns=list(spec.columns))
