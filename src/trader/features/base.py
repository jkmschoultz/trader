"""The feature-set interface and the frozen spec that pins one computation.

A :class:`FeatureSet` is a named, stateless recipe. Calling
:meth:`FeatureSet.resolve` turns a base horizon and a choice of context horizons
into a :class:`FeatureSpec`: a fully-resolved, JSON-serialisable, hashable
description of exactly which columns come out and how. Training freezes that spec
into the model's manifest; inference reloads it and calls
:meth:`FeatureSet.compute` again, so the live features match the trained ones bar
for bar even if the registered set later changes.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trader.data.calendars import RegularHours

__all__ = ["FeatureSet", "FeatureSpec", "attach_context"]


@dataclass(frozen=True)
class FeatureSpec:
    """A resolved, hashable description of one feature computation.

    Every field is a primitive or a tuple of primitives, so the spec serialises
    to JSON with :meth:`to_dict` and back with :meth:`from_dict`, and
    :meth:`digest` is stable across processes. ``columns`` is filled in by
    :meth:`FeatureSet.resolve` and is the exact output column order.
    """

    name: str
    base_horizon: int
    context_horizons: tuple[int, ...] = ()
    returns: tuple[int, ...] = (1, 5, 15)
    vol_windows: tuple[int, ...] = (20, 60)
    rsi_window: int = 14
    macd: tuple[int, int, int] = (12, 26, 9)
    atr_window: int = 14
    volz_window: int = 20
    regular_hours_only: bool = True
    columns: tuple[str, ...] = field(default=())

    @property
    def warmup(self) -> int:
        """Bars of history before a row's features should be trusted.

        A lower bound, not a guarantee: EWM-based columns never fully shed their
        seed. Rows inside the warmup are kept as NaN and dropped only at the
        dataset boundary.
        """
        ewm_reach = 3 * max(self.rsi_window, self.macd[1], self.atr_window)
        roll_reach = max(self.volz_window, *self.vol_windows, max(self.returns) + 1)
        warmup = max(ewm_reach, roll_reach)
        if self.context_horizons and self.base_horizon:
            ratio = max(ceil(h / self.base_horizon) for h in self.context_horizons)
            warmup = max(warmup, ewm_reach * ratio + ratio)
        return int(warmup)

    def to_dict(self) -> dict:
        """A plain JSON-safe dict; tuples become lists."""
        out = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in out.items()}

    @classmethod
    def from_dict(cls, data: Mapping) -> FeatureSpec:
        """Inverse of :meth:`to_dict`."""
        fields = {
            "name": data["name"],
            "base_horizon": int(data["base_horizon"]),
            "context_horizons": tuple(data.get("context_horizons", ())),
            "returns": tuple(data.get("returns", (1, 5, 15))),
            "vol_windows": tuple(data.get("vol_windows", (20, 60))),
            "rsi_window": int(data.get("rsi_window", 14)),
            "macd": tuple(data.get("macd", (12, 26, 9))),
            "atr_window": int(data.get("atr_window", 14)),
            "volz_window": int(data.get("volz_window", 20)),
            "regular_hours_only": bool(data.get("regular_hours_only", True)),
            "columns": tuple(data.get("columns", ())),
        }
        return cls(**fields)

    def digest(self) -> str:
        """A stable sha1 over the spec -- the cache key and manifest fingerprint."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()

    def write(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path) -> FeatureSpec:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FeatureSet(ABC):
    """A named feature recipe. Stateless: one instance is registered and reused."""

    #: Registry name, set by :func:`trader.features.registry.register_feature_set`.
    name: str = ""

    #: Whether :meth:`resolve` does anything useful with ``context_horizons``.
    context_capable: bool = False

    @abstractmethod
    def resolve(
        self,
        *,
        base_horizon: int,
        context_horizons: tuple[int, ...] = (),
        **overrides: object,
    ) -> FeatureSpec:
        """Build the frozen :class:`FeatureSpec` this set produces."""

    @abstractmethod
    def compute(
        self,
        bars: pd.DataFrame,
        *,
        spec: FeatureSpec,
        context: Mapping[int, pd.DataFrame],
        session: RegularHours | None,
    ) -> pd.DataFrame:
        """Return a frame indexed by bar ``time``, columns exactly ``spec.columns``.

        ``bars`` is a canonical base-horizon frame, oldest first. ``context``
        maps each context horizon to its resampled canonical frame. Rows inside
        the warmup are left as NaN.
        """


def attach_context(
    base_time: pd.Series,
    context_features: pd.DataFrame,
    *,
    context_horizon: int,
    base_horizon: int,
) -> pd.DataFrame:
    """Align context-bar features onto the base timeline with no lookahead.

    A context bar opening at ``T`` covers ``[T, T + context_horizon)`` and is
    only known once it closes. A base row opening at ``s`` is itself only known
    at ``s + base_horizon`` (its own close). So each base row takes the newest
    context row whose close instant is *strictly before* the base row's close --
    the conservative choice, one context bar of latency rather than risking a
    simultaneous close counting as visible.

    Returns a frame indexed like ``base_time`` (a plain RangeIndex), with the
    same columns as ``context_features`` and NaN where no context row qualifies.
    """
    known_at = pd.DatetimeIndex(context_features["time"]) + pd.Timedelta(minutes=context_horizon)
    order = np.argsort(known_at.values, kind="stable")
    known_sorted = known_at.values[order]

    base_close = pd.DatetimeIndex(base_time) + pd.Timedelta(minutes=base_horizon)
    # Index of the last context row known strictly before each base close.
    pos = np.searchsorted(known_sorted, base_close.values, side="left") - 1

    value_cols = [c for c in context_features.columns if c != "time"]
    out = pd.DataFrame(np.nan, index=range(len(base_time)), columns=value_cols, dtype="float64")
    valid = pos >= 0
    if valid.any():
        src_rows = order[pos[valid]]
        out.loc[valid, value_cols] = context_features.iloc[src_rows][value_cols].to_numpy()
    return out
