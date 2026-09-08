"""Build causal model features from the lake, without a CLI or an HTTP layer.

Mirrors :func:`trader.service.backtest.run_backtest`: a pydantic
:class:`FeatureBuildSpec` in, a plain summary out, ``progress`` reported per
series. The features themselves are recomputed on demand; pass ``persist`` to
also write them to the optional :class:`~trader.features.cache.FeatureLake`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.inputs import parse_since

__all__ = ["FeatureBuildSpec", "run_feature_build"]


class FeatureBuildSpec(BaseModel):
    """What features to build, for which instruments.

    ``horizon`` and each ``context_horizons`` entry accept a label (``"5m"``) or
    minutes; ``since`` accepts ``"90d"``, ``"2y"``, a date, an ISO timestamp, or
    ``"all"``.
    """

    symbols: list[str] = Field(min_length=1)
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    exchange: str | None = None
    horizon: int = 5
    context_horizons: list[int] = Field(default_factory=list)
    since: datetime | None = None
    feature_set: str = "price_v1"
    persist: bool = False

    @field_validator("horizon", "context_horizons", mode="before")
    @classmethod
    def _horizons(cls, value: object) -> object:
        from trader.saxo.charts import ChartError, parse_horizon

        try:
            if isinstance(value, (list, tuple)):
                return [parse_horizon(v) for v in value]
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


async def run_feature_build(
    settings: Settings,
    spec: FeatureBuildSpec,
    *,
    progress: Callable[[float], None] | None = None,
) -> list[dict[str, Any]]:
    """Build features for every symbol in ``spec`` and return one summary each.

    Raises:
        InvalidRequest: unknown feature set, or a context horizon that is not a
            whole multiple of the base horizon.
        SeriesNotStored: the lake holds nothing for one of the symbols.
    """
    from trader.features.pipeline import compute_feature_frame
    from trader.features.registry import UnknownFeatureSet, get_feature_set
    from trader.service._panel import load_panel

    try:
        get_feature_set(spec.feature_set)
    except UnknownFeatureSet as exc:
        raise InvalidRequest(str(exc)) from exc

    series = await load_panel(
        settings,
        symbols=spec.symbols,
        uics=spec.uics,
        asset_type=spec.asset_type,
        exchange=spec.exchange,
        horizon=spec.horizon,
        since=spec.since,
    )

    summaries: list[dict[str, Any]] = []
    for index, item in enumerate(series):
        try:
            features, fspec = compute_feature_frame(
                item.frame,
                feature_set=spec.feature_set,
                base_horizon=spec.horizon,
                context_horizons=tuple(spec.context_horizons),
                session=item.session,
            )
        except ValueError as exc:
            raise InvalidRequest(str(exc)) from exc

        touched = 0
        if spec.persist:
            from trader.features.cache import FeatureLake

            touched = FeatureLake(settings.data_dir).write(item.key, fspec, features)

        nan_rows = int(features.isna().any(axis=1).sum())
        summaries.append(
            {
                "key": str(item.key),
                "label": item.label,
                "feature_set": fspec.name,
                "digest": fspec.digest(),
                "rows": len(features),
                "columns": list(features.columns),
                "warmup": fspec.warmup,
                "nan_rows": nan_rows,
                "first": features.index[0].isoformat() if len(features) else None,
                "last": features.index[-1].isoformat() if len(features) else None,
                "persisted_files": touched,
            }
        )
        if progress is not None:
            progress((index + 1) / len(series))

    return summaries
