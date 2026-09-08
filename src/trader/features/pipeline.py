"""The one entry point that turns bars into a feature frame.

Both training and the live ``lstm`` strategy call this, so the features a model
learns on and the features it trades on come from exactly the same code. Pass a
frozen :class:`~trader.features.base.FeatureSpec` (``spec=``) to reproduce a
trained model's features; pass ``feature_set=`` / ``base_horizon=`` to resolve a
fresh one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from trader.features.base import FeatureSpec
from trader.features.registry import get_feature_set

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trader.data.calendars import RegularHours

__all__ = ["compute_feature_frame"]


def compute_feature_frame(
    bars: pd.DataFrame,
    *,
    feature_set: str | None = None,
    base_horizon: int | None = None,
    context_horizons: Sequence[int] = (),
    session: RegularHours | None = None,
    overrides: Mapping[str, Any] | None = None,
    spec: FeatureSpec | None = None,
) -> tuple[pd.DataFrame, FeatureSpec]:
    """Compute features for one canonical bar frame.

    Args:
        bars: canonical base-horizon frame, oldest first (as the lake returns it).
        feature_set: registered set name; required unless ``spec`` is given.
        base_horizon: base bar size in minutes; required unless ``spec`` is given.
        context_horizons: coarser horizons to fold in (must be whole multiples
            of the base). Resampled from ``bars`` with
            :func:`trader.data.bars.resample`.
        session: inferred trading hours, used by the session-relative columns.
        overrides: field overrides passed to ``FeatureSet.resolve`` (ignored when
            ``spec`` is given).
        spec: a frozen spec to reproduce exactly, instead of resolving one.

    Returns:
        ``(features, spec)`` -- ``features`` is indexed by bar ``time`` with
        columns exactly ``spec.columns`` and one row per input bar; warmup rows
        are NaN.

    Raises:
        ValueError: neither ``spec`` nor (``feature_set`` and ``base_horizon``)
            given, or a context horizon is not a multiple of the base.
        trader.features.registry.UnknownFeatureSet: no such feature set.
    """
    from trader.data import bars as bars_mod

    if spec is None:
        if feature_set is None or base_horizon is None:
            raise ValueError("pass spec=, or both feature_set= and base_horizon=")
        fs = get_feature_set(feature_set)
        spec = fs.resolve(
            base_horizon=int(base_horizon),
            context_horizons=tuple(int(h) for h in context_horizons),
            **dict(overrides or {}),
        )
    else:
        fs = get_feature_set(spec.name)

    context: dict[int, pd.DataFrame] = {}
    for horizon in spec.context_horizons:
        if horizon % spec.base_horizon:
            raise ValueError(
                f"context horizon {horizon} is not a multiple of base {spec.base_horizon}"
            )
        context[horizon] = bars_mod.resample(bars, source=spec.base_horizon, target=horizon)

    features = fs.compute(bars, spec=spec, context=context, session=session)
    features = features.reindex(columns=list(spec.columns))
    features.index = pd.DatetimeIndex(bars["time"], name="time")
    return features, spec
