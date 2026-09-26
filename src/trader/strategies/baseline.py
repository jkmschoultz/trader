"""No-skill reference strategies: what a result has to beat.

A strategy's backtest means little on its own -- costs, exits and the market's
own drift all move the number. ``baseline`` trades the same way as the strategy
under test but picks its direction with no information:

- ``direction="random"``: a coin flip per decision.
- ``direction="long_flat"``: a coin flip between long and flat -- the fair
  baseline for a long-only strategy, which is either long or out.
- ``direction="long"`` / ``"short"``: always the same side (buy-and-hold when
  there is no bracket).

Two modes, matching the two kinds of strategy here:

- **Scheduled** (no ``stop`` / ``take`` / ``max_bars``): re-draw the direction
  every ``hold_bars`` bars and hold in between -- the baseline for ``tsmom``.
- **Bracketed**: whenever flat, enter (with probability ``entry_prob``) with
  the given bracket -- the baseline for a model strategy with the same exits.
  ``barrier_scale="atr"`` reads stop / take as ATR multiples, exactly as the
  model labels do.

``primary="tsmom"`` only trades while daily momentum is long (flat otherwise),
the same gate a meta-labelled model trades under -- so "primary + the model's
exits, no model" is the baseline that isolates what the model adds.

The coin flips are a pure function of ``(seed, instrument, bar time)``, so a run
is reproducible and one seed means the same sequence everywhere. Run several
seeds to see the spread luck alone produces.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target
from trader.strategies.registry import register

# bars of history for an ATR-scaled bracket, as in ModelStrategy
_ATR_TAIL = 700


@register("baseline")
class Baseline(InstrumentStrategy):
    """Random-direction or one-sided entries, on a schedule or with a bracket.

    Args:
        direction: ``"random"``, ``"long_flat"``, ``"long"`` or ``"short"``.
        seed: which random sequence (only matters for ``"random"``).
        hold_bars: scheduled mode: re-draw the direction every this many bars.
        stop: bracketed mode: stop distance (fraction, or ATRs with ``"atr"``).
        take: bracketed mode: take-profit distance, same units.
        max_bars: bracketed mode: exit after this many bars.
        barrier_scale: ``"fraction"`` or ``"atr"`` -- see
            :func:`trader.labels.triple_barrier.barrier_unit`.
        entry_prob: bracketed mode: chance of entering on a bar where flat, to
            match a model's trade frequency (1.0 = always in a trade).
        warmup: bars to wait before the first decision (to start where the
            strategy under test starts).
        primary: ``"tsmom"`` to trade only while the momentum gate is long.
        primary_lookbacks: the gate's lookbacks in days, ``"21/63/126/252"``.
    """

    def __init__(
        self,
        *,
        direction: Literal["random", "long_flat", "long", "short"] = "random",
        seed: int = 0,
        hold_bars: int = 5,
        stop: float | None = None,
        take: float | None = None,
        max_bars: int | None = None,
        barrier_scale: Literal["fraction", "atr"] = "fraction",
        entry_prob: float = 1.0,
        warmup: int = 0,
        primary: str | None = None,
        primary_lookbacks: str = "21/63/126/252",
    ) -> None:
        if direction not in ("random", "long_flat", "long", "short"):
            raise ValueError(
                f"direction must be random, long_flat, long or short, got {direction!r}"
            )
        if barrier_scale not in ("fraction", "atr"):
            raise ValueError(f"barrier_scale must be fraction or atr, got {barrier_scale!r}")
        if hold_bars < 1:
            raise ValueError(f"hold_bars must be >= 1, got {hold_bars}")
        if not 0.0 < entry_prob <= 1.0:
            raise ValueError(f"entry_prob must be in (0, 1], got {entry_prob}")
        self.direction = direction
        self.seed = int(seed)
        self.hold_bars = int(hold_bars)
        self.stop = stop
        self.take = take
        self.max_bars = max_bars
        self.barrier_scale = barrier_scale
        self.entry_prob = float(entry_prob)
        self.bracketed = any(v is not None for v in (stop, take, max_bars))
        if self.bracketed and barrier_scale == "atr":
            from trader.labels.triple_barrier import ATR_WINDOW

            warmup = max(warmup, ATR_WINDOW + 1)
        self.warmup = max(int(warmup), 1)  # the first decision needs one closed bar
        self.primary = None
        if primary:
            from trader.labels.primary import LiveGate, PrimarySpec

            self.primary = LiveGate(PrimarySpec.parse(primary, primary_lookbacks))
            self.history_days = self.primary.spec.span_days

    def _uniforms(self, ctx: BarContext) -> tuple[float, float]:
        """Two reproducible uniforms in [0, 1) for this instrument and bar."""
        stamp = ctx.history["time"].iloc[-1]
        digest = hashlib.blake2b(
            f"{self.seed}|{ctx.label}|{stamp.isoformat()}".encode(), digest_size=16
        ).digest()
        a = int.from_bytes(digest[:8], "big") / 2**64
        b = int.from_bytes(digest[8:], "big") / 2**64
        return a, b

    def _side(self, draw: float) -> float:
        if self.direction == "long":
            return 1.0
        if self.direction == "short":
            return -1.0
        if self.direction == "long_flat":
            return 1.0 if draw < 0.5 else 0.0
        return 1.0 if draw < 0.5 else -1.0

    def _bracket(self, ctx: BarContext) -> dict | None:
        unit = 1.0
        if self.barrier_scale == "atr":
            from trader.labels.triple_barrier import barrier_unit

            unit = float(barrier_unit(ctx.history.iloc[-_ATR_TAIL:], "atr")[-1])
            if not math.isfinite(unit) or unit <= 0:
                return None
        return {
            "stop": self.stop * unit if self.stop is not None else None,
            "take": self.take * unit if self.take is not None else None,
            "max_bars": self.max_bars,
        }

    def on_bar(self, ctx: BarContext) -> Decision:
        if ctx.bars_seen < self.warmup:
            return Hold()
        if self.primary is not None and not self.primary(ctx.history, ctx.horizon, ctx.label):
            return Flat()
        enter_draw, side_draw = self._uniforms(ctx)

        if not self.bracketed:
            if (ctx.bars_seen - self.warmup) % self.hold_bars:
                return Hold()
            side = self._side(side_draw)
            return Target(side) if side else Flat()

        if ctx.position.units:
            return Hold()  # the bracket manages the open trade
        if enter_draw >= self.entry_prob:
            return Hold()
        bracket = self._bracket(ctx)
        if bracket is None:
            return Hold()
        side = self._side(side_draw)
        return Target(side, **bracket) if side else Hold()
