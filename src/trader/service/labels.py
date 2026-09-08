"""Triple-barrier labelling over the lake, without a CLI or an HTTP layer.

Mirrors :func:`trader.service.features.run_feature_build`: a pydantic
:class:`LabelSpec` in, a summary of the class balance and barrier breakdown out.
Labels are a cheap pure function of bars and parameters, so nothing is persisted.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trader.config import Settings
from trader.service.inputs import parse_since

__all__ = ["LabelSpec", "run_labelling"]


class LabelSpec(BaseModel):
    """What to label, for which instruments."""

    symbols: list[str] = Field(min_length=1)
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    exchange: str | None = None
    horizon: int = 5
    since: datetime | None = None
    stop: float | None = Field(default=0.005, gt=0)
    take: float | None = Field(default=0.01, gt=0)
    max_bars: int = Field(default=24, ge=1)
    min_return: float = Field(default=0.0, ge=0)

    @field_validator("horizon", mode="before")
    @classmethod
    def _horizon(cls, value: object) -> int:
        from trader.saxo.charts import ChartError, parse_horizon

        try:
            return parse_horizon(value)  # type: ignore[arg-type]
        except ChartError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("since", mode="before")
    @classmethod
    def _since(cls, value: object) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value  # type: ignore[return-value]
        return parse_since(str(value))

    @field_validator("uics")
    @classmethod
    def _uics_fit(cls, value: list[int], info: Any) -> list[int]:
        symbols = info.data.get("symbols") or []
        if len(value) > len(symbols):
            raise ValueError("more uics than symbols")
        return value


def _summarise(events) -> dict[str, Any]:
    resolved = events.dropna(subset=["label"])
    n = len(resolved)
    counts = {
        int(k): int(v)
        for k, v in resolved["label"].value_counts().reindex([-1.0, 0.0, 1.0], fill_value=0).items()
    }
    barrier = {
        k: int(v)
        for k, v in resolved["barrier"]
        .value_counts()
        .reindex(["stop", "take", "time"], fill_value=0)
        .items()
    }
    return {
        "n_events": n,
        "class_counts": counts,
        "class_pct": {k: round(v / n, 4) if n else 0.0 for k, v in counts.items()},
        "barrier_breakdown": barrier,
        "mean_abs_ret": round(float(resolved["ret"].abs().mean()), 6) if n else None,
        "median_bars_held": float(resolved["bars_held"].median()) if n else None,
    }


async def run_labelling(
    settings: Settings,
    spec: LabelSpec,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Label every symbol in ``spec`` and return an aggregate + per-symbol summary.

    Raises:
        SeriesNotStored: the lake holds nothing for one of the symbols.
    """
    import pandas as pd

    from trader.labels.triple_barrier import triple_barrier
    from trader.service._panel import load_panel

    series = await load_panel(
        settings,
        symbols=spec.symbols,
        uics=spec.uics,
        asset_type=spec.asset_type,
        exchange=spec.exchange,
        horizon=spec.horizon,
        since=spec.since,
    )

    per_symbol: dict[str, Any] = {}
    frames = []
    for index, item in enumerate(series):
        events = triple_barrier(
            item.frame,
            stop=spec.stop,
            take=spec.take,
            max_bars=spec.max_bars,
            min_return=spec.min_return,
        )
        per_symbol[item.label] = {"key": str(item.key), **_summarise(events)}
        frames.append(events.dropna(subset=["label"]))
        if progress is not None:
            progress((index + 1) / len(series))

    combined = pd.concat(frames) if frames else _empty_events()
    return {
        **_summarise(combined),
        "barriers": {"stop": spec.stop, "take": spec.take, "max_bars": spec.max_bars},
        "per_symbol": per_symbol,
    }


def _empty_events():
    import pandas as pd

    cols = ("label", "barrier", "ret", "bars_held")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})
