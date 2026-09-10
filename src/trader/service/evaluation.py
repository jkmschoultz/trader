"""Walk-forward evaluation for *any* registered strategy, not just a model.

:func:`trader.service.training.run_cv` walk-forward evaluates a model config by
training a fold and backtesting its held-out window. A classical strategy
(``ma_cross``, ``orb``) has no training step -- it just needs the same fold
geometry, cost model, allocator, and scoring so it lands on the same
leaderboard. This module carves out the shared pieces:

* :func:`score_fold` -- clip the panel to one fold's window and backtest it,
  returning ``result.metrics.as_dict()``.
* :func:`aggregate_folds` -- median / mean / pstdev across folds, plus the
  ``folds_sharpe_gt_0_5`` / ``worst_fold_sharpe`` headline numbers.
* :class:`StrategyCVSpec` + :func:`run_strategy_cv` -- the classical counterpart
  of ``run_cv``: build the strategy per fold and score it, no model in sight.

The selection metric is unchanged: out-of-sample Sharpe.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.inputs import parse_since

__all__ = [
    "StrategyCVSpec",
    "aggregate_folds",
    "run_strategy_cv",
    "score_fold",
]

# strategies that go through the model training path rather than this one
MODEL_STRATEGIES = frozenset({"lstm", "gbm"})


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def score_fold(
    series: Sequence[Any],
    strategy: Any,
    *,
    horizon: int,
    val_end: datetime,
    test_end: datetime | None,
    lookback,
    allocator: Any,
    cost_model: Any,
    leverage: float,
    starting_cash: float = 100_000.0,
) -> dict[str, Any]:
    """Backtest ``strategy`` over one fold's ``[val_end - lookback, test_end]`` window.

    Returns the run's ``metrics.as_dict()``. Raises :class:`InvalidRequest` if
    every series is shorter than the strategy warmup over that window.
    """
    from trader import backtest as bt
    from trader.data import bars as bars_mod

    panel: dict[str, Any] = {}
    instruments: dict[str, Any] = {}
    for s in series:
        clipped = bars_mod.clip(s.frame, start=val_end - lookback, end=test_end)
        if len(clipped) <= strategy.warmup:
            continue
        panel[s.label] = clipped
        instruments[s.label] = bt.Instrument(key=s.key, session=s.session)
    if not panel:
        raise InvalidRequest("fold window is shorter than the strategy warmup")

    result = bt.run(
        panel,
        strategy,
        horizon=horizon,
        allocator=allocator,
        cost_model=cost_model,
        instruments=instruments,
        starting_cash=starting_cash,
        leverage_cap=leverage,
    )
    return result.metrics.as_dict()


def aggregate_folds(fold_metrics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-fold metric dicts into the sweep's aggregate block."""

    def _agg(key: str) -> dict[str, float]:
        vals = [m[key] for m in fold_metrics if m.get(key) is not None and finite(m[key])]
        if not vals:
            return {"median": float("nan"), "mean": float("nan"), "std": float("nan")}
        return {
            "median": statistics.median(vals),
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }

    sharpes = [m["sharpe"] for m in fold_metrics if finite(m.get("sharpe"))]
    return {
        "n_folds": len(fold_metrics),
        "folds_sharpe_gt_0_5": sum(1 for s in sharpes if s > 0.5),
        "worst_fold_sharpe": min(sharpes) if sharpes else float("nan"),
        "sharpe": _agg("sharpe"),
        "total_return": _agg("total_return"),
        "turnover": _agg("turnover"),
        "hit_rate": _agg("hit_rate"),
        "profit_factor": _agg("profit_factor"),
    }


def fold_lookback(horizon: int, warmup: int):
    """Wall-clock span of bars a fold backtest needs before its first decision.

    ``horizon * (warmup + 64)`` bars, floored at a day so a session-relative
    strategy (``orb``) always sees at least one full session before ``val_end``.
    """
    import pandas as pd

    return max(
        pd.Timedelta(minutes=horizon * (warmup + 64)),
        pd.Timedelta(days=1),
    )


class StrategyCVSpec(BaseModel):
    """Walk-forward evaluation of one classical strategy config.

    The model-free counterpart of ``TrainingSpec`` + ``CVConfig``: no windowing,
    no labels, no hyperparameters -- just a registered ``strategy`` name, its
    ``params``, the fold geometry, and the scoring costs.
    """

    symbols: list[str] = Field(min_length=1)
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    exchange: str | None = None
    strategy: str
    params: dict[str, Any] = Field(default_factory=dict)
    horizon: int = 5
    since: datetime | None = None

    folds: int = Field(default=5, ge=1)
    mode: Literal["rolling", "anchored"] = "rolling"
    train_days: float = Field(gt=0)
    val_days: float = Field(gt=0)
    test_days: float = Field(gt=0)
    step_days: float | None = Field(default=None, gt=0)

    fee_bps: float = Field(default=0.0, ge=0)
    spread_bps: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=0.0, ge=0)
    allocator: str = "equal-weight"
    leverage: float = Field(default=1.0, gt=0)
    starting_cash: float = 100_000.0

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


async def run_strategy_cv(
    settings: Settings,
    spec: StrategyCVSpec,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Walk-forward evaluate one classical-strategy config; return folds + aggregate.

    Same return shape as :func:`trader.service.training.run_cv` (minus the
    ``*_macro_f1`` columns), so the tuning report code is shared.

    Raises:
        InvalidRequest: unknown strategy / allocator, bad params, the data span
            cannot fit ``folds`` folds, or a fold is shorter than the warmup.
        SeriesNotStored: the lake holds nothing for one of the symbols.
    """
    import inspect

    from trader import backtest as bt
    from trader import strategies
    from trader.models.dataset import walk_forward_splits
    from trader.service._panel import load_panel

    def report_progress(value: float) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, value)))

    try:
        strategy_cls = strategies.get_strategy(spec.strategy)
    except strategies.UnknownStrategy as exc:
        raise InvalidRequest(str(exc)) from exc
    if spec.strategy in MODEL_STRATEGIES:
        raise InvalidRequest(
            f"{spec.strategy!r} is a model strategy; use run_cv / the model tuning path"
        )

    params = dict(spec.params)
    accepts = inspect.signature(strategy_cls.__init__).parameters
    if "models_dir" in accepts and "models_dir" not in params:
        params["models_dir"] = str(settings.models_dir)

    def _build_strategy() -> Any:
        try:
            return strategy_cls(**params)
        except (TypeError, ValueError) as exc:
            raise InvalidRequest(f"bad params for strategy {spec.strategy!r}: {exc}") from exc

    try:
        allocator = bt.get_allocator(spec.allocator)
    except KeyError as exc:
        raise InvalidRequest(exc.args[0]) from exc
    cost_model = bt.CostModel(
        commission_bps=spec.fee_bps,
        half_spread_bps=spec.spread_bps,
        slippage_bps=spec.slippage_bps,
    )

    series = await load_panel(
        settings,
        symbols=spec.symbols,
        uics=spec.uics,
        asset_type=spec.asset_type,
        exchange=spec.exchange,
        horizon=spec.horizon,
        since=spec.since,
    )
    report_progress(0.05)

    times = _decision_times(series)
    try:
        splits = walk_forward_splits(
            times,
            n_folds=spec.folds,
            train_days=spec.train_days,
            val_days=spec.val_days,
            test_days=spec.test_days,
            step_days=spec.step_days,
            mode=spec.mode,
        )
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc

    folds: list[dict[str, Any]] = []
    for i, split in enumerate(splits):
        strategy = _build_strategy()
        metrics = score_fold(
            series,
            strategy,
            horizon=spec.horizon,
            val_end=split.val_end,
            test_end=split.test_end,
            lookback=fold_lookback(spec.horizon, strategy.warmup),
            allocator=allocator,
            cost_model=cost_model,
            leverage=spec.leverage,
            starting_cash=spec.starting_cash,
        )
        folds.append(
            {
                "fold": i + 1,
                "train_end": split.train_end.isoformat(),
                "val_end": split.val_end.isoformat(),
                "test_end": split.test_end.isoformat() if split.test_end else None,
                "metrics": metrics,
            }
        )
        report_progress(0.05 + 0.93 * (i + 1) / len(splits))

    aggregate = aggregate_folds([f["metrics"] for f in folds])
    report_progress(1.0)
    return {"folds": folds, "aggregate": aggregate}


def _decision_times(series: Sequence[Any]):
    """Sorted int64-ns bar timestamps across every series, for ``walk_forward_splits``."""
    import numpy as np
    import pandas as pd

    chunks = []
    for s in series:
        idx = pd.DatetimeIndex(s.frame["time"]).tz_convert("UTC").tz_localize(None)
        chunks.append(idx.to_numpy("datetime64[ns]").astype("int64"))
    return np.sort(np.concatenate(chunks))
