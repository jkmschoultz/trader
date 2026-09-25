"""``regime_v1``: volatility-regime, trend-quality and gap features on top of ``structure_v1``.

Three groups, all from OHLC alone (no volume, so FX works):

**Volatility regime** -- how much the market is moving and whether that is
changing. Volatility clusters, so this is the most predictable thing in a candle
series. Intraday estimators that use the whole candle (Parkinson,
Garman-Klass, Rogers-Satchell) are more accurate than close-to-close for the
same number of bars; daily close-to-close volatility includes overnight gaps.

**Trend quality** -- is price trending cleanly or chopping? Momentum over 1 / 5
/ 20 trading days scaled by daily volatility, the efficiency ratio (net move /
path length), ADX, the R^2 and slope of a rolling straight-line fit, lag-1
return autocorrelation, and distance to / slope of moving averages.

**Gaps** -- the trading day's opening gap from the prior close (in daily-vol
units), how much of it has been filled so far, and whether it runs with or
against the prior five-day trend. IBIT/ETHA gap most mornings; for 24-hour FX
the gap is ~0 and these columns stay near 0.

Conventions shared with ``structure_v1``: the trading day rolls at 17:00 New
York (:func:`~trader.features.structure.trading_day`); distances are in ATR or
daily-vol units and clipped to +/-25. Bar-count windows (``*_20``, ``*_60``)
mean different wall-clock spans on 24h FX and a 6.5h ETF session; daily
features (``dvol_*``, ``mom_*``, ``dma20_dist``, gaps) are built from whole
trading days so they read the same on both.

Causality: daily statistics come only from *completed* trading days (shifted one
day), joined back onto each bar of the current day. The current day's open is
known from its first bar. ``tests/features/test_regime.py`` pins this.

The daily columns need ~80 trading days of history (a 60-day percentile of a
20-day volatility), so ``level_days`` defaults to 120 calendar days. Rows before
that are NaN and dropped at the dataset boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from trader.features import indicators as ind
from trader.features.base import FeatureSpec
from trader.features.registry import register_feature_set
from trader.features.sets import _context_columns, _price_columns
from trader.features.structure import (
    DEFAULT_SWING_ORDERS,
    StructureV1,
    structure_columns,
    trading_day,
)

#: calendar days of history: 20-day vol + a 60-trading-day percentile, plus weekends
DEFAULT_LEVEL_DAYS = 120
#: gaps smaller than this many daily-vol units count as no gap
_GAP_MIN_Z = 0.1
#: gaps at least this large get a with/against-trend reading
_GAP_TREND_Z = 0.25
_CLIP = 25.0
_LN2 = np.log(2.0)

VOL_COLUMNS: tuple[str, ...] = (
    "pk_vol_20",
    "gk_vol_20",
    "rs_vol_20",
    "gk_ratio",
    "dvol_20",
    "dvol_ratio",
    "dvol_pct",
    "dvolvol",
)

TREND_COLUMNS: tuple[str, ...] = (
    "mom_1",
    "mom_5",
    "mom_20",
    "dma20_dist",
    "er_20",
    "er_60",
    "adx",
    "di_diff",
    "r2_60",
    "slope_60",
    "ac1_60",
    "ema50_dist",
    "ema200_dist",
    "ema50_slope",
    "upfrac_20",
)

GAP_COLUMNS: tuple[str, ...] = (
    "gap_z",
    "gap_fill",
    "gap_trend",
    "day_ret_z",
)

REGIME_COLUMNS: tuple[str, ...] = VOL_COLUMNS + TREND_COLUMNS + GAP_COLUMNS

#: unbounded columns clipped to +/- _CLIP
_CLIPPED: tuple[str, ...] = (
    "mom_1",
    "mom_5",
    "mom_20",
    "dma20_dist",
    "slope_60",
    "ema50_dist",
    "ema200_dist",
    "ema50_slope",
    "gap_z",
    "day_ret_z",
)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """``num / den``, 0 where ``den`` is 0 (a flat window), NaN where ``den`` is NaN."""
    return (num / den).where(den != 0, 0.0).where(den.notna())


def _daily_stats(bars: pd.DataFrame, day: np.ndarray) -> pd.DataFrame:
    """Per-bar statistics of *completed* trading days, plus the current day's open.

    Every column but ``day_open`` is shifted one trading day before being joined
    back, so a bar only ever sees days that have already closed.
    """
    frame = pd.DataFrame(
        {
            "open": bars["open"].to_numpy(),
            "high": bars["high"].to_numpy(),
            "low": bars["low"].to_numpy(),
            "close": bars["close"].to_numpy(),
            "day": day,
        }
    )
    d = frame.groupby("day", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    )
    ret = np.log(d["close"] / d["close"].shift(1))
    pk = np.log(d["high"] / d["low"]) ** 2 / (4 * _LN2)

    stats = pd.DataFrame(index=d.index)
    stats["dvol_5"] = ret.rolling(5).std()
    stats["dvol_20"] = ret.rolling(20).std()
    stats["dvol_pct"] = stats["dvol_20"].rolling(60).rank(pct=True)
    stats["dvolvol"] = np.log(np.sqrt(pk.where(pk > 0))).rolling(20, min_periods=15).std()
    stats["dma20"] = d["close"].rolling(20).mean()
    # after the one-day shift below, close_k is the close k completed days back
    for k in (1, 5, 20):
        stats[f"close_{k}"] = d["close"].shift(k - 1)
    stats["trend_5"] = np.log(d["close"] / d["close"].shift(5))

    prior = stats.shift(1)
    prior["day_open"] = d["open"]
    out = prior.reindex(day).reset_index(drop=True)
    out.index = bars.index
    return out


def _regime_block(bars: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Every ``regime_v1``-only column, indexed like ``bars``."""
    open_, high, low, close = bars["open"], bars["high"], bars["low"], bars["close"]
    atr = ind.atr(bars, spec.atr_window)
    day = np.asarray(trading_day(bars["time"]))
    daily = _daily_stats(bars, day)
    dvol = daily["dvol_20"]

    out: dict[str, pd.Series] = {}

    # --- volatility regime -----------------------------------------------------
    ln_hl = np.log(high / low)
    ln_co = np.log(close / open_)
    pk = ln_hl**2 / (4 * _LN2)
    gk = 0.5 * ln_hl**2 - (2 * _LN2 - 1) * ln_co**2
    rs = np.log(high / close) * np.log(high / open_) + np.log(low / close) * np.log(low / open_)
    out["pk_vol_20"] = np.sqrt(pk.rolling(20).mean())
    out["gk_vol_20"] = np.sqrt(gk.rolling(20).mean().clip(lower=0.0))
    out["rs_vol_20"] = np.sqrt(rs.rolling(20).mean().clip(lower=0.0))
    out["gk_ratio"] = _safe_div(out["gk_vol_20"], np.sqrt(gk.rolling(100).mean().clip(lower=0.0)))
    out["dvol_20"] = dvol
    out["dvol_ratio"] = _safe_div(daily["dvol_5"], dvol).clip(0.0, 10.0)
    out["dvol_pct"] = daily["dvol_pct"]
    out["dvolvol"] = daily["dvolvol"]

    # --- trend quality ---------------------------------------------------------
    for k in (1, 5, 20):
        out[f"mom_{k}"] = np.log(close / daily[f"close_{k}"]) / (dvol * np.sqrt(k))
    out["dma20_dist"] = np.log(close / daily["dma20"]) / dvol

    step = close.diff().abs()
    for n in (20, 60):
        out[f"er_{n}"] = _safe_div((close - close.shift(n)).abs(), step.rolling(n).sum())

    adx, plus_di, minus_di = ind.adx(bars, spec.atr_window)
    out["adx"] = adx / 100.0
    out["di_diff"] = (plus_di - minus_di) / 100.0

    t = pd.Series(np.arange(len(bars), dtype=float), index=bars.index)
    close_std = close.rolling(60).std()
    corr = close.rolling(60).corr(t).where(close_std != 0, 0.0).clip(-1.0, 1.0)
    out["r2_60"] = corr**2
    # fitted move across the window, in ATRs: slope * 60 bars
    out["slope_60"] = corr * close_std / t.rolling(60).std() * 60 / atr

    ret = ind.log_return(close, 1)
    ac1 = ret.rolling(60).corr(ret.shift(1))
    out["ac1_60"] = ac1.where(ret.rolling(60).std() != 0, 0.0).clip(-1.0, 1.0)

    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
    out["ema50_dist"] = (close - ema50) / atr
    out["ema200_dist"] = (close - ema200) / atr
    out["ema50_slope"] = (ema50 - ema50.shift(10)) / atr
    move = close.diff()
    out["upfrac_20"] = (move > 0).astype(float).where(move.notna()).rolling(20).mean()

    # --- gaps --------------------------------------------------------------------
    day_open = daily["day_open"]
    prior_close = daily["close_1"]
    gap = np.log(day_open / prior_close)
    gap_z = gap / dvol
    out["gap_z"] = gap_z

    frame = pd.DataFrame({"h": high.to_numpy(), "l": low.to_numpy(), "day": day})
    day_hi = frame.groupby("day")["h"].cummax().set_axis(bars.index)
    day_lo = frame.groupby("day")["l"].cummin().set_axis(bars.index)
    fill = pd.Series(
        np.where(
            gap > 0,
            (day_open - day_lo) / (day_open - prior_close),
            (day_hi - day_open) / (prior_close - day_open),
        ),
        index=bars.index,
    ).clip(0.0, 1.0)
    material = gap_z.abs() >= _GAP_MIN_Z
    out["gap_fill"] = fill.where(material, 0.0).where(gap_z.notna())
    against = np.sign(gap) * np.sign(daily["trend_5"])
    out["gap_trend"] = against.where(gap_z.abs() >= _GAP_TREND_Z, 0.0).where(
        gap_z.notna() & daily["trend_5"].notna()
    )
    out["day_ret_z"] = np.log(close / day_open) / dvol

    block = pd.DataFrame({name: np.asarray(col, dtype=float) for name, col in out.items()})
    block[list(_CLIPPED)] = block[list(_CLIPPED)].clip(-_CLIP, _CLIP)
    block.index = pd.DatetimeIndex(bars["time"], name="time")
    return block


@register_feature_set("regime_v1")
class RegimeV1(StructureV1):
    """structure_v1 plus volatility-regime, trend-quality and gap features."""

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
            name="regime_v1",
            base_horizon=base_horizon,
            context_horizons=ctx,
            **overrides,  # type: ignore[arg-type]
        )
        columns = (
            _price_columns(spec) + _context_columns(spec) + structure_columns(spec) + REGIME_COLUMNS
        )
        return replace(spec, columns=columns)

    def compute(
        self,
        bars: pd.DataFrame,
        *,
        spec: FeatureSpec,
        context: Mapping[int, pd.DataFrame],
        session,
    ) -> pd.DataFrame:
        structure_spec = replace(
            spec, columns=_price_columns(spec) + _context_columns(spec) + structure_columns(spec)
        )
        base = super().compute(bars, spec=structure_spec, context=context, session=session)
        regime = _regime_block(bars, spec)
        merged = pd.concat([base, regime], axis=1)
        return merged.reindex(columns=list(spec.columns))
