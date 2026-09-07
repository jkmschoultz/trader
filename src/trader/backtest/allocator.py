"""Allocators: per-instrument convictions -> portfolio weights.

An :class:`~trader.strategies.base.InstrumentStrategy` emits a conviction in
``[-1, 1]`` for its own instrument and knows nothing about the rest of the book.
The allocator is what turns the set of convictions into an actual weight vector,
under a gross-exposure (leverage) cap. It runs every time any instrument
rebalances, so it must be cheap and deterministic.

A weight is a signed fraction of *current portfolio equity*; the engine sizes it
to whole units at the next bar's open. ``sum(abs(weights)) <= leverage_cap`` is
enforced by every allocator here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from trader.strategies.base import PortfolioView


@dataclass(frozen=True)
class AllocatorContext:
    """Everything an allocator may look at."""

    histories: Mapping[str, object]  # label -> completed-bar DataFrame view
    portfolio: PortfolioView
    leverage_cap: float


def _cap(weights: dict[str, float], leverage_cap: float) -> dict[str, float]:
    """Scale ``weights`` down proportionally if gross exposure exceeds the cap."""
    gross = sum(abs(w) for w in weights.values())
    if gross > leverage_cap and gross > 0:
        scale = leverage_cap / gross
        return {k: w * scale for k, w in weights.items()}
    return weights


class Allocator(ABC):
    """Combine convictions into capped portfolio weights."""

    @abstractmethod
    def weights(
        self, convictions: Mapping[str, float], ctx: AllocatorContext
    ) -> dict[str, float]: ...


class PassThrough(Allocator):
    """Use each conviction directly as its weight. The single-instrument default."""

    def weights(self, convictions: Mapping[str, float], ctx: AllocatorContext) -> dict[str, float]:
        cap = ctx.leverage_cap
        capped = {k: max(-cap, min(cap, v)) for k, v in convictions.items()}
        return _cap(capped, cap)


class EqualWeight(Allocator):
    """Split the leverage cap equally across every non-flat conviction, keeping sign."""

    def weights(self, convictions: Mapping[str, float], ctx: AllocatorContext) -> dict[str, float]:
        active = [k for k, v in convictions.items() if v]
        if not active:
            return {k: 0.0 for k in convictions}
        each = ctx.leverage_cap / len(active)
        return {k: math.copysign(each, v) if v else 0.0 for k, v in convictions.items()}


class FixedFraction(Allocator):
    """Give every active conviction the same fixed weight ``fraction``, sign kept.

    Gross exposure is still capped, so a basket larger than
    ``leverage_cap / fraction`` names is scaled down proportionally.
    """

    def __init__(self, fraction: float = 0.1) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction

    def weights(self, convictions: Mapping[str, float], ctx: AllocatorContext) -> dict[str, float]:
        raw = {k: math.copysign(self.fraction, v) if v else 0.0 for k, v in convictions.items()}
        return _cap(raw, ctx.leverage_cap)


class VolTarget(Allocator):
    """Weight each name inversely to its recent volatility, then scale to a target.

    ``weight_i ∝ conviction_i / sigma_i``, where ``sigma_i`` is the annualised
    standard deviation of the instrument's recent bar returns. The vector is
    scaled so the equity-weighted portfolio volatility is roughly
    ``target_ann_vol``, then capped. Names without enough history fall back to
    an equal share.
    """

    def __init__(
        self,
        target_ann_vol: float = 0.15,
        *,
        lookback: int = 20,
        periods_per_year: float = 252.0,
    ) -> None:
        if target_ann_vol <= 0:
            raise ValueError("target_ann_vol must be positive")
        self.target_ann_vol = target_ann_vol
        self.lookback = lookback
        self.periods_per_year = periods_per_year

    def _sigma(self, history: object) -> float | None:
        try:
            close = history["close"]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive
            return None
        if len(close) < self.lookback + 1:
            return None
        rets = close.pct_change().dropna().iloc[-self.lookback :]
        sd = float(rets.std(ddof=1))
        if not sd or math.isnan(sd):
            return None
        return sd * math.sqrt(self.periods_per_year)

    def weights(self, convictions: Mapping[str, float], ctx: AllocatorContext) -> dict[str, float]:
        raw: dict[str, float] = {}
        for label, conviction in convictions.items():
            if not conviction:
                raw[label] = 0.0
                continue
            sigma = self._sigma(ctx.histories.get(label))
            scale = (self.target_ann_vol / sigma) if sigma else 1.0
            raw[label] = math.copysign(min(scale, 1.0), conviction)
        return _cap(raw, ctx.leverage_cap)


_ALLOCATORS: dict[str, type[Allocator]] = {
    "passthrough": PassThrough,
    "equal-weight": EqualWeight,
    "fixed-fraction": FixedFraction,
    "vol-target": VolTarget,
}


def get_allocator(name: str, **kwargs: object) -> Allocator:
    """Build an allocator by CLI name (``equal-weight``, ``vol-target``, ...)."""
    try:
        cls = _ALLOCATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown allocator {name!r}; choose one of {', '.join(sorted(_ALLOCATORS))}"
        ) from None
    return cls(**kwargs)  # type: ignore[arg-type]
