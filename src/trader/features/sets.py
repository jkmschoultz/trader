"""The built-in feature sets: ``price_v1`` and ``mtf_v1``.

``price_v1`` is single-timeframe price / momentum / volatility / session
features. ``mtf_v1`` is ``price_v1`` on the base horizon plus a handful of
coarse features from each context horizon, joined causally onto the base
timeline (a context bar is only visible once it has closed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, tzinfo

import numpy as np
import pandas as pd

from trader.features import indicators as ind
from trader.features.base import FeatureSet, FeatureSpec, attach_context
from trader.features.registry import register_feature_set

# Coarse features taken from each context horizon in mtf_v1.
_CONTEXT_COLUMNS: tuple[str, ...] = ("ret_1", "rsi", "vol", "macd_hist", "atr_pct")


def _price_columns(spec: FeatureSpec) -> tuple[str, ...]:
    cols: list[str] = []
    cols += [f"ret_{n}" for n in spec.returns]
    cols += [f"vol_{w}" for w in spec.vol_windows]
    cols += [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_pct",
        "range_pct",
        "volz",
        "vwap_dist",
        "ovn_gap",
        "tod_sin",
        "tod_cos",
        "session_phase",
    ]
    return tuple(cols)


def _context_columns(spec: FeatureSpec) -> tuple[str, ...]:
    return tuple(f"h{h}_{c}" for h in spec.context_horizons for c in _CONTEXT_COLUMNS)


def _session_timezone(session) -> tzinfo:
    return session.timezone if session is not None else UTC


def _price_block(bars: pd.DataFrame, spec: FeatureSpec, session) -> pd.DataFrame:
    """The single-timeframe feature columns, indexed by bar ``time``."""
    close = bars["close"]
    volume = bars["volume"]
    out: dict[str, pd.Series] = {}

    for n in spec.returns:
        out[f"ret_{n}"] = ind.log_return(close, n)
    for w in spec.vol_windows:
        out[f"vol_{w}"] = ind.rolling_vol(close, w)

    out["rsi"] = ind.rsi(close, spec.rsi_window)
    line, signal, hist = ind.macd(close, *spec.macd)
    out["macd"] = line
    out["macd_signal"] = signal
    out["macd_hist"] = hist
    out["atr_pct"] = ind.atr(bars, spec.atr_window) / close
    out["range_pct"] = ind.range_pct(bars)
    out["volz"] = ind.volume_zscore(volume, spec.volz_window)

    tz = _session_timezone(session)
    local = pd.DatetimeIndex(bars["time"]).tz_convert(tz)
    day = local.normalize()
    minutes = local.hour * 60 + local.minute

    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    frame = pd.DataFrame(
        {"tpv": (typical * volume).to_numpy(), "vol": volume.to_numpy(), "day": day}
    )
    cum_tpv = frame.groupby("day")["tpv"].cumsum()
    cum_vol = frame.groupby("day")["vol"].cumsum()
    vwap = pd.Series((cum_tpv / cum_vol).to_numpy(), index=close.index)
    out["vwap_dist"] = close / vwap - 1.0

    prev_close = close.shift(1)
    new_day = np.asarray(day) != np.roll(np.asarray(day), 1)
    new_day[0] = False
    out["ovn_gap"] = pd.Series(
        np.where(new_day, (bars["open"] / prev_close - 1.0).to_numpy(), 0.0),
        index=close.index,
    )

    angle = 2.0 * np.pi * minutes / 1440.0
    out["tod_sin"] = pd.Series(np.sin(angle), index=close.index)
    out["tod_cos"] = pd.Series(np.cos(angle), index=close.index)

    if session is not None:
        open_m = session.open_time.hour * 60 + session.open_time.minute
        close_m = session.close_time.hour * 60 + session.close_time.minute
        span = max(close_m - open_m, 1)
        phase = (minutes - open_m) / span
    else:
        phase = minutes / 1440.0
    out["session_phase"] = pd.Series(np.clip(phase, 0.0, 1.0), index=close.index)

    block = pd.DataFrame(out)
    block.index = pd.DatetimeIndex(bars["time"], name="time")
    return block


def _context_block(ctx_bars: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """The coarse per-context-horizon columns, with a ``time`` column kept."""
    close = ctx_bars["close"]
    _, _, hist = ind.macd(close, *spec.macd)
    out = pd.DataFrame(
        {
            "time": ctx_bars["time"].to_numpy(),
            "ret_1": ind.log_return(close, 1).to_numpy(),
            "rsi": ind.rsi(close, spec.rsi_window).to_numpy(),
            "vol": ind.rolling_vol(close, spec.vol_windows[0]).to_numpy(),
            "macd_hist": hist.to_numpy(),
            "atr_pct": (ind.atr(ctx_bars, spec.atr_window) / close).to_numpy(),
        }
    )
    return out


@register_feature_set("price_v1")
class PriceV1(FeatureSet):
    """Single-timeframe price, momentum, volatility, and session features."""

    context_capable = False

    def resolve(
        self,
        *,
        base_horizon: int,
        context_horizons: tuple[int, ...] = (),
        **overrides: object,
    ) -> FeatureSpec:
        spec = FeatureSpec(
            name="price_v1",
            base_horizon=base_horizon,
            context_horizons=(),
            **overrides,  # type: ignore[arg-type]
        )
        return replace(spec, columns=_price_columns(spec))

    def compute(
        self,
        bars: pd.DataFrame,
        *,
        spec: FeatureSpec,
        context: Mapping[int, pd.DataFrame],
        session,
    ) -> pd.DataFrame:
        return _price_block(bars, spec, session).reindex(columns=list(spec.columns))


@register_feature_set("mtf_v1")
class MtfV1(FeatureSet):
    """price_v1 on the base horizon plus coarse features from each context horizon."""

    context_capable = True

    def resolve(
        self,
        *,
        base_horizon: int,
        context_horizons: tuple[int, ...] = (),
        **overrides: object,
    ) -> FeatureSpec:
        ctx = tuple(sorted(int(h) for h in context_horizons))
        spec = FeatureSpec(
            name="mtf_v1",
            base_horizon=base_horizon,
            context_horizons=ctx,
            **overrides,  # type: ignore[arg-type]
        )
        return replace(spec, columns=_price_columns(spec) + _context_columns(spec))

    def compute(
        self,
        bars: pd.DataFrame,
        *,
        spec: FeatureSpec,
        context: Mapping[int, pd.DataFrame],
        session,
    ) -> pd.DataFrame:
        base = _price_block(bars, spec, session)
        base_time = bars["time"].reset_index(drop=True)
        pieces = [base.reset_index(drop=True)]
        for h in spec.context_horizons:
            ctx_bars = context[h]
            if ctx_bars.empty:
                aligned = pd.DataFrame(
                    np.nan,
                    index=range(len(base_time)),
                    columns=[f"h{h}_{c}" for c in _CONTEXT_COLUMNS],
                    dtype="float64",
                )
            else:
                ctx_feats = _context_block(ctx_bars, spec)
                aligned = attach_context(
                    base_time,
                    ctx_feats,
                    context_horizon=h,
                    base_horizon=spec.base_horizon,
                )
                aligned = aligned.add_prefix(f"h{h}_")
            pieces.append(aligned)

        merged = pd.concat(pieces, axis=1)
        merged.index = pd.DatetimeIndex(base_time, name="time")
        return merged.reindex(columns=list(spec.columns))
