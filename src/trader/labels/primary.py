"""Primary signals for meta-labelling: a simple rule picks the side, a model filters it.

Meta-labelling (López de Prado, ch. 3) splits a trading decision in two. A
*primary* rule says when to be long -- here daily time-series momentum -- and a
model, trained only on the bars where that rule is on, answers the narrower
question "will *this* entry work?". That is usually easier to learn than
direction from scratch, and it inherits the primary rule's edge rather than
having to find one.

The primary here is the ``tsmom`` strategy's signal, computed from **completed
UTC days** built out of the bars themselves (the last close of each day), so it
works on any intraday horizon and needs no separate daily series. A bar sees the
days that finished at or before its own close: an hourly bar opening at 23:00
sees that day; one opening at 22:00 does not.

Two entry points compute the same thing:

- :func:`daily_conviction` / :func:`primary_gate` -- vectorised over a whole
  frame, for choosing training samples;
- :func:`gate_now` -- for the latest bar of a history, used by strategies at
  decision time. It reads only a tail, so a backtest stays linear.

``tests/labels/test_primary.py`` checks they agree bar for bar.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_LOOKBACKS",
    "LiveGate",
    "PrimarySpec",
    "daily_conviction",
    "gate_now",
    "primary_gate",
]

#: tsmom's default: ~1, 3, 6 and 12 months of daily bars
DEFAULT_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)


@dataclass(frozen=True)
class PrimarySpec:
    """Which primary rule, with which settings. JSON round-trips via to/from_dict."""

    name: str = "tsmom"
    lookbacks: tuple[int, ...] = DEFAULT_LOOKBACKS

    def __post_init__(self) -> None:
        if self.name != "tsmom":
            raise ValueError(f"unknown primary {self.name!r}; available: tsmom")
        if not self.lookbacks or min(self.lookbacks) < 1:
            raise ValueError(f"lookbacks must be positive day counts, got {self.lookbacks}")

    @property
    def span_days(self) -> int:
        """Days of history :func:`gate_now` reads: the longest lookback plus slack."""
        return max(self.lookbacks) + 3

    def to_dict(self) -> dict:
        return {"name": self.name, "lookbacks": list(self.lookbacks)}

    @classmethod
    def from_dict(cls, data: Mapping) -> PrimarySpec:
        return cls(
            name=str(data.get("name", "tsmom")),
            lookbacks=tuple(int(k) for k in data.get("lookbacks", DEFAULT_LOOKBACKS)),
        )

    @classmethod
    def parse(cls, name: str, lookbacks: str | Sequence[int] | None = None) -> PrimarySpec:
        """From CLI-style inputs: ``lookbacks`` as ``"21/63/126/252"`` or a list."""
        from trader.strategies.tsmom import _ints

        return cls(name=name, lookbacks=_ints(lookbacks) if lookbacks else DEFAULT_LOOKBACKS)


def daily_conviction(frame: pd.DataFrame, horizon: int, spec: PrimarySpec) -> np.ndarray:
    """tsmom conviction in ``[-1, 1]`` known at each bar's close; NaN until enough days.

    The conviction is the mean sign of ``close[d] - close[d - k]`` over the
    lookbacks, on the series of daily closes (days counted by position, so
    weekend-less FX counts trading days, like ``tsmom`` on daily bars).
    """
    times = pd.DatetimeIndex(frame["time"])
    day = times.floor("D")
    closes = pd.Series(frame["close"].to_numpy(dtype=float), index=day).groupby(level=0).last()
    conv = sum(np.sign(closes - closes.shift(k)) for k in spec.lookbacks) / len(spec.lookbacks)

    # the newest day complete by each bar's close: the day before that close's date
    known_day = (times + pd.Timedelta(minutes=horizon)).floor("D") - pd.Timedelta(days=1)
    pos = conv.index.searchsorted(known_day, side="right") - 1
    values = conv.to_numpy()
    out = np.full(len(frame), np.nan)
    ok = pos >= 0
    out[ok] = values[pos[ok]]
    return out


def primary_gate(frame: pd.DataFrame, horizon: int, spec: PrimarySpec) -> np.ndarray:
    """True where the primary rule is long (conviction > 0) at each bar's close."""
    conv = daily_conviction(frame, horizon, spec)
    return np.nan_to_num(conv, nan=0.0) > 0


def gate_now(history: pd.DataFrame, horizon: int, spec: PrimarySpec) -> bool:
    """Is the primary long at the latest bar of ``history``? Reads a bounded tail."""
    bars_per_day = max(1, 1440 // horizon)
    tail = history.iloc[-(spec.span_days + 2) * bars_per_day :]
    conv = daily_conviction(tail, horizon, spec)
    return bool(len(conv) and np.nan_to_num(conv[-1], nan=0.0) > 0)


class LiveGate:
    """:func:`gate_now` for a strategy, cached per instrument until a day completes.

    The gate can only change when a new day finishes, so it is recomputed at
    most once per instrument per day instead of on every intraday bar.
    """

    def __init__(self, spec: PrimarySpec) -> None:
        self.spec = spec
        self._cache: dict[str, tuple[pd.Timestamp, bool]] = {}

    def __call__(self, history: pd.DataFrame, horizon: int, label: str) -> bool:
        day = (history["time"].iloc[-1] + pd.Timedelta(minutes=horizon)).floor("D")
        cached = self._cache.get(label)
        if cached is not None and cached[0] == day:
            return cached[1]
        on = gate_now(history, horizon, self.spec)
        self._cache[label] = (day, on)
        return on
