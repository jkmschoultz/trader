"""Train an LSTM from the lake and register it, without a CLI or an HTTP layer.

Mirrors :func:`trader.service.backtest.run_backtest`: a pydantic
:class:`TrainingSpec` in, a summary dict out, ``progress`` in ``[0, 1]``. The
trained model lands in the registry (``Settings.models_dir``) and from then on
the ``lstm`` strategy trades it through the unchanged backtest path.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.inputs import parse_since

__all__ = ["TrainingSpec", "run_training"]


class TrainingSpec(BaseModel):
    """Everything needed to train and register one model."""

    symbols: list[str] = Field(min_length=1)
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    exchange: str | None = None
    name: str = "lstm"

    horizon: int = 5
    context_horizons: list[int] = Field(default_factory=list)
    feature_set: str = "price_v1"

    stop: float | None = Field(default=0.005, gt=0)
    take: float | None = Field(default=0.01, gt=0)
    max_bars: int = Field(default=24, ge=1)
    min_return: float = Field(default=0.0, ge=0)

    window: int = Field(default=32, ge=2)
    train_end: datetime
    val_end: datetime
    embargo_bars: int | None = Field(default=None, ge=0)
    since: datetime | None = None

    hidden: int = Field(default=64, ge=1)
    layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    bidirectional: bool = False
    epochs: int = Field(default=40, ge=1)
    batch_size: int = Field(default=128, ge=1)
    lr: float = Field(default=1e-3, gt=0)
    class_weight: str | None = "balanced"
    use_sample_weights: bool = False
    seed: int = 0

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

    @field_validator("since", "train_end", "val_end", mode="before")
    @classmethod
    def _dates(cls, value: object) -> datetime | None:
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

    @model_validator(mode="after")
    def _val_after_train(self) -> TrainingSpec:
        if self.val_end <= self.train_end:
            raise ValueError("val_end must be after train_end")
        return self


async def run_training(
    settings: Settings,
    spec: TrainingSpec,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Train, evaluate, and register a model; return its id and metrics.

    Raises:
        InvalidRequest: the ``[model]`` extra is missing, the feature set is
            unknown, or too few samples survive the split.
        SeriesNotStored: the lake holds nothing for one of the symbols.
    """
    import numpy as np

    from trader.features.registry import UnknownFeatureSet, get_feature_set
    from trader.labels.triple_barrier import LabelConfig
    from trader.models._optional import require_torch
    from trader.models.dataset import (
        SequenceBundle,
        SplitSpec,
        WindowSpec,
        build_bundle,
        time_split,
    )
    from trader.models.lstm import LSTMConfig
    from trader.models.registry import ModelRegistry
    from trader.models.scaler import StandardScaler
    from trader.models.training import train_model
    from trader.service._panel import load_panel

    def report_progress(value: float) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, value)))

    try:
        require_torch()
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc

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
    report_progress(0.05)

    label = LabelConfig(
        stop=spec.stop, take=spec.take, max_bars=spec.max_bars, min_return=spec.min_return
    )
    window_spec = WindowSpec(
        window=spec.window,
        feature_set=spec.feature_set,
        base_horizon=spec.horizon,
        context_horizons=tuple(spec.context_horizons),
        label=label,
    )

    bundles = []
    for item in series:
        weights = _sample_weights(item.frame, label) if spec.use_sample_weights else None
        try:
            bundles.append(
                build_bundle(item.frame, spec=window_spec, session=item.session, weights=weights)
            )
        except ValueError as exc:
            raise InvalidRequest(f"{item.label}: {exc}") from exc

    X = np.concatenate([b.X for b in bundles])
    y = np.concatenate([b.y for b in bundles])
    w = np.concatenate([b.w for b in bundles])
    t = np.concatenate([b.t for b in bundles])
    feature_spec = bundles[0].feature_spec
    merged = SequenceBundle(X, y, w, t, bundles[0].feature_names, feature_spec)
    report_progress(0.1)

    embargo = spec.embargo_bars if spec.embargo_bars is not None else spec.window + spec.max_bars
    split = SplitSpec(train_end=spec.train_end, val_end=spec.val_end, embargo_bars=embargo)
    tr, va, te = time_split(merged, split, base_horizon=spec.horizon, max_bars=spec.max_bars)
    if tr.sum() < spec.batch_size or va.sum() == 0:
        raise InvalidRequest(
            f"too few samples after the split (train={int(tr.sum())}, val={int(va.sum())}, "
            f"test={int(te.sum())}); widen --since or move the split dates"
        )

    scaler = StandardScaler().fit(X[tr])
    config = LSTMConfig(
        n_features=X.shape[-1],
        hidden=spec.hidden,
        layers=spec.layers,
        dropout=spec.dropout,
        bidirectional=spec.bidirectional,
    )
    model, train_report = train_model(
        scaler.transform(X[tr]),
        y[tr],
        w[tr],
        scaler.transform(X[va]),
        y[va],
        config=config,
        epochs=spec.epochs,
        Xte=scaler.transform(X[te]) if te.any() else None,
        yte=y[te] if te.any() else None,
        batch_size=spec.batch_size,
        lr=spec.lr,
        class_weight=spec.class_weight,
        use_sample_weights=spec.use_sample_weights,
        seed=spec.seed,
        progress=lambda frac: report_progress(0.1 + 0.85 * frac),
    )
    report_progress(0.95)

    info = ModelRegistry(settings.models_dir).save(
        name=spec.name,
        model=model,
        scaler=scaler,
        feature_spec=feature_spec,
        label=label,
        window=spec.window,
        split=split,
        config=config,
        report=train_report,
        data_spec={
            "symbols": spec.symbols,
            "uics": spec.uics,
            "asset_type": spec.asset_type,
            "since": spec.since.isoformat() if spec.since else None,
            "horizon": spec.horizon,
        },
        optimiser={
            "lr": spec.lr,
            "batch_size": spec.batch_size,
            "epochs": spec.epochs,
            "class_weight": spec.class_weight,
            "use_sample_weights": spec.use_sample_weights,
            "seed": spec.seed,
        },
    )
    report_progress(1.0)

    return {
        "model_id": info.id,
        "metrics": train_report.metrics,
        "class_distribution": train_report.class_distribution,
        "report": train_report.to_dict(),
    }


def _sample_weights(frame, label):
    from trader.labels.triple_barrier import triple_barrier
    from trader.labels.weights import return_attribution_weights

    events = triple_barrier(
        frame,
        stop=label.stop,
        take=label.take,
        max_bars=label.max_bars,
        min_return=label.min_return,
        entry=label.entry,
    )
    return return_attribution_weights(events, frame["time"])
