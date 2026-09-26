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
    """Size each position so the whole book runs at roughly ``target_ann_vol``.

    ``weight_i = conviction_i * (target_ann_vol / sigma_i) / n_active``, where
    ``sigma_i`` is the annualised standard deviation of the instrument's last
    ``lookback`` bar returns. So a calm market gets a bigger position and a wild
    one a smaller one, and a half-strength conviction gets half the size.
    Splitting the target across the ``n_active`` names makes the book hit it
    when they move together (BTC and ETH) and undershoot when they are
    unrelated -- the cautious side.

    Each weight is capped at ``max_weight`` (1.0 = no leverage in one name),
    then gross exposure at the engine's leverage cap. Names without enough
    history get an equal share.

    ``periods_per_year=None`` counts bars per calendar year from the history's
    own timestamps: 365 for daily crypto, ~260 for daily FX, and the right
    number for intraday sessions.
    """

    def __init__(
        self,
        target_ann_vol: float = 0.15,
        *,
        lookback: int = 20,
        periods_per_year: float | None = None,
        max_weight: float = 1.0,
    ) -> None:
        if target_ann_vol <= 0:
            raise ValueError("target_ann_vol must be positive")
        if lookback < 2:
            raise ValueError("lookback must be at least 2 bars")
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        self.target_ann_vol = float(target_ann_vol)
        self.lookback = int(lookback)
        self.periods_per_year = periods_per_year
        self.max_weight = float(max_weight)

    def _sigma(self, history: object) -> float | None:
        try:
            close = history["close"]  # type: ignore[index]
            times = history["time"]  # type: ignore[index]
        except Exception:  # pragma: no cover - defensive
            return None
        if len(close) < self.lookback + 1:
            return None
        window = close.iloc[-self.lookback - 1 :]
        rets = window.pct_change().dropna()
        sd = float(rets.std(ddof=1))
        if not sd or math.isnan(sd):
            return None
        per_year = self.periods_per_year
        if per_year is None:
            span = times.iloc[-1] - times.iloc[-self.lookback - 1]
            years = span.total_seconds() / (365.25 * 86400)
            if years <= 0:
                return None
            per_year = self.lookback / years
        return sd * math.sqrt(per_year)

    def weights(self, convictions: Mapping[str, float], ctx: AllocatorContext) -> dict[str, float]:
        active = [label for label, c in convictions.items() if c]
        raw: dict[str, float] = {label: 0.0 for label in convictions}
        if not active:
            return raw
        share = 1.0 / len(active)
        for label in active:
            conviction = convictions[label]
            sigma = self._sigma(ctx.histories.get(label))
            size = (self.target_ann_vol / sigma) * abs(conviction) if sigma else 1.0
            raw[label] = math.copysign(min(size * share, self.max_weight), conviction)
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


def available_allocators() -> dict[str, type[Allocator]]:
    """Every registered allocator name -> class, for a catalogue. Sorted by name."""
    return dict(sorted(_ALLOCATORS.items()))
