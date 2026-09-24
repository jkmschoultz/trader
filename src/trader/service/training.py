"""Train an LSTM from the lake and register it, without a CLI or an HTTP layer.

Mirrors :func:`trader.service.backtest.run_backtest`: a pydantic
:class:`TrainingSpec` in, a summary dict out, ``progress`` in ``[0, 1]``. The
trained model lands in the registry (``Settings.models_dir``) and from then on
the ``lstm`` strategy trades it through the unchanged backtest path.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest
from trader.service.inputs import parse_since

__all__ = ["CVConfig", "TrainingSpec", "run_cv", "run_training"]


class TrainingSpec(BaseModel):
    """Everything needed to train and register one model."""

    symbols: list[str] = Field(min_length=1)
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    exchange: str | None = None
    name: str = "lstm"
    #: which model family to train: ``"lstm"`` (torch sequence), ``"gbm"``
    #: (LightGBM on the flattened lag stack) or ``"xgb"`` (XGBoost on the same
    #: lag stack, on CUDA when available). Picks the trainer, the feature
    #: layout, and the registry weights format.
    model_type: Literal["lstm", "gbm", "xgb"] = "lstm"

    horizon: int = 5
    context_horizons: list[int] = Field(default_factory=list)
    feature_set: str = "price_v1"

    stop: float | None = Field(default=0.005, gt=0)
    take: float | None = Field(default=0.01, gt=0)
    max_bars: int = Field(default=24, ge=1)
    min_return: float = Field(default=0.0, ge=0)

    window: int = Field(default=32, ge=2)
    # Required for a single train (:func:`run_training`); ignored by
    # :func:`run_cv`, which derives a split per fold.
    train_end: datetime | None = None
    val_end: datetime | None = None
    embargo_bars: int | None = Field(default=None, ge=0)
    since: datetime | None = None

    # --- LSTM hyperparameters ---
    hidden: int = Field(default=64, ge=1)
    layers: int = Field(default=2, ge=1)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    bidirectional: bool = False
    epochs: int = Field(default=40, ge=1)
    batch_size: int = Field(default=128, ge=1)
    # --- GBM (LightGBM) hyperparameters ---
    num_leaves: int = Field(default=31, ge=2)
    n_estimators: int = Field(default=400, ge=1)
    max_depth: int = Field(default=-1)
    min_child_samples: int = Field(default=20, ge=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, gt=0, le=1)
    # --- shared ---
    lr: float = Field(default=1e-3, gt=0)  # LSTM Adam lr / GBM learning_rate
    class_weight: str | None = "balanced"
    use_sample_weights: bool = False
    seed: int = 0
    #: device for the LSTM (torch) and the XGBoost trees: ``"auto"`` picks CUDA
    #: when available, else CPU. The LightGBM ``gbm`` ignores it.
    device: str = "auto"

    @property
    def layout(self) -> str:
        return "tabular" if self.model_type in ("gbm", "xgb") else "sequence"

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
        if self.train_end is not None and self.val_end is not None:
            if self.val_end <= self.train_end:
                raise ValueError("val_end must be after train_end")
        elif (self.train_end is None) != (self.val_end is None):
            raise ValueError("train_end and val_end must be given together")
        return self


class CVConfig(BaseModel):
    """Walk-forward evaluation knobs: fold geometry, costs, decision rule.

    Spans are in days over the data's wall-clock range. Each fold trains on
    ``train_days``, validates on ``val_days``, then the held-out ``test_days``
    are run through a real backtest; the origin advances by ``step_days``
    (default ``test_days`` -- non-overlapping test windows).
    """

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
    threshold: float = Field(default=0.15, ge=0, lt=1)
    on_no_signal: Literal["hold", "flat"] = "hold"


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
    from trader.models.dataset import SplitSpec, time_split
    from trader.models.registry import ModelRegistry

    def report_progress(value: float) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, value)))

    if spec.train_end is None or spec.val_end is None:
        raise InvalidRequest(
            "run_training needs train_end and val_end; use run_cv for walk-forward"
        )

    merged, series, label, _ = await _build_dataset(settings, spec)
    report_progress(0.1)

    embargo = spec.embargo_bars if spec.embargo_bars is not None else spec.window + spec.max_bars
    split = SplitSpec(train_end=spec.train_end, val_end=spec.val_end, embargo_bars=embargo)
    tr, va, te = time_split(merged, split, base_horizon=spec.horizon, max_bars=spec.max_bars)
    if tr.sum() < spec.batch_size or va.sum() == 0:
        raise InvalidRequest(
            f"too few samples after the split (train={int(tr.sum())}, val={int(va.sum())}, "
            f"test={int(te.sum())}); widen --since or move the split dates"
        )

    model, scaler, config, train_report = _fit_fold(
        merged, tr, va, te, spec, progress=lambda frac: report_progress(0.1 + 0.85 * frac)
    )
    report_progress(0.95)

    info = ModelRegistry(settings.models_dir).save(
        name=spec.name,
        model=model,
        scaler=scaler,
        feature_spec=merged.feature_spec,
        label=label,
        window=spec.window,
        split=split,
        config=config,
        report=train_report,
        model_type=spec.model_type,
        layout=spec.layout,
        data_spec={
            "symbols": spec.symbols,
            "uics": spec.uics,
            "asset_type": spec.asset_type,
            "since": spec.since.isoformat() if spec.since else None,
            "horizon": spec.horizon,
        },
        optimiser={
            "lr": spec.lr,
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


async def run_cv(
    settings: Settings,
    spec: TrainingSpec,
    cv: CVConfig,
    *,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Walk-forward evaluate one config: train each fold, backtest its test window.

    Selection metric is out-of-sample Sharpe -- ``aggregate.folds_sharpe_gt_0_5``
    counts how many folds clear ~0.5. Returns per-fold rows and the aggregate.

    Raises:
        InvalidRequest: ``[model]`` missing, unknown feature set / allocator, the
            data span cannot fit ``cv.folds`` folds, or a fold has too few samples.
        SeriesNotStored: the lake holds nothing for one of the symbols.
    """
    import shutil
    import tempfile
    from pathlib import Path

    from trader import backtest as bt
    from trader.models.dataset import time_split, walk_forward_splits
    from trader.models.registry import ModelRegistry
    from trader.service.evaluation import aggregate_folds, score_fold
    from trader.strategies import get_strategy

    def report_progress(value: float) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, value)))

    merged, series, label, _ = await _build_dataset(settings, spec)
    report_progress(0.05)

    embargo = spec.embargo_bars if spec.embargo_bars is not None else spec.window + spec.max_bars
    try:
        splits = walk_forward_splits(
            merged.t,
            n_folds=cv.folds,
            train_days=cv.train_days,
            val_days=cv.val_days,
            test_days=cv.test_days,
            step_days=cv.step_days,
            embargo_bars=embargo,
            mode=cv.mode,
        )
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc

    try:
        allocator = bt.get_allocator(cv.allocator)
    except KeyError as exc:
        raise InvalidRequest(exc.args[0]) from exc
    cost_model = bt.CostModel(
        commission_bps=cv.fee_bps, half_spread_bps=cv.spread_bps, slippage_bps=cv.slippage_bps
    )

    # every fold backtests the same series with the same frozen spec: compute each
    # series' features once and let the strategy look rows up instead of rerunning
    # the pipeline per bar (which dominated a fold's wall time)
    from trader.features.pipeline import compute_feature_frame

    precomputed = {
        s.label: compute_feature_frame(s.frame, spec=merged.feature_spec, session=s.session)[0]
        for s in series
    }

    folds: list[dict[str, Any]] = []
    for i, split in enumerate(splits):
        tr, va, te = time_split(merged, split, base_horizon=spec.horizon, max_bars=spec.max_bars)
        if tr.sum() < spec.batch_size or va.sum() == 0 or te.sum() == 0:
            raise InvalidRequest(
                f"fold {i + 1}/{len(splits)}: too few samples "
                f"(train={int(tr.sum())}, val={int(va.sum())}, test={int(te.sum())}); "
                "widen --since or shrink the fold spans"
            )

        model, scaler, config, train_report = _fit_fold(
            merged,
            tr,
            va,
            te,
            spec,
            progress=lambda frac, i=i: report_progress(0.05 + 0.9 * (i + frac) / len(splits)),
        )

        tmp = Path(tempfile.mkdtemp(prefix="cv-fold-"))
        try:
            info = ModelRegistry(tmp).save(
                name=spec.name,
                model=model,
                scaler=scaler,
                feature_spec=merged.feature_spec,
                label=label,
                window=spec.window,
                split=split,
                config=config,
                report=train_report,
                model_type=spec.model_type,
                layout=spec.layout,
                data_spec={"symbols": spec.symbols, "horizon": spec.horizon},
            )
            strategy = get_strategy(spec.model_type)(
                model=info.id,
                threshold=cv.threshold,
                on_no_signal=cv.on_no_signal,
                models_dir=str(tmp),
            )
            strategy.use_precomputed(precomputed)
            metrics = score_fold(
                series,
                strategy,
                horizon=spec.horizon,
                val_end=split.val_end,
                test_end=split.test_end,
                lookback=_fold_lookback(spec, strategy.warmup),
                allocator=allocator,
                cost_model=cost_model,
                leverage=cv.leverage,
            )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        folds.append(
            {
                "fold": i + 1,
                "train_end": split.train_end.isoformat(),
                "val_end": split.val_end.isoformat(),
                "test_end": split.test_end.isoformat() if split.test_end else None,
                "val_macro_f1": train_report.metrics.get("val_macro_f1"),
                "test_macro_f1": train_report.metrics.get("test_macro_f1"),
                "metrics": metrics,
            }
        )
    report_progress(0.98)

    aggregate = aggregate_folds([f["metrics"] for f in folds])
    report_progress(1.0)
    return {"folds": folds, "aggregate": aggregate}


def _fold_lookback(spec: TrainingSpec, warmup: int):
    """Wall-clock span of bars a fold backtest needs before its first decision."""
    import pandas as pd

    return pd.Timedelta(minutes=spec.horizon * (warmup + spec.max_bars + 64))


async def _build_dataset(settings: Settings, spec: TrainingSpec):
    """Load the panel and build one merged :class:`SequenceBundle` for ``spec``.

    Shared by :func:`run_training` and :func:`run_cv` -- features and labels do
    not depend on where the train/val/test cuts fall.
    """
    import numpy as np

    from trader.features.registry import UnknownFeatureSet, get_feature_set
    from trader.labels.triple_barrier import LabelConfig
    from trader.models._optional import require_lightgbm, require_torch, require_xgboost
    from trader.models.dataset import SequenceBundle, WindowSpec, build_bundle
    from trader.service._panel import load_panel

    try:
        {"gbm": require_lightgbm, "xgb": require_xgboost}.get(spec.model_type, require_torch)()
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

    label = LabelConfig(
        stop=spec.stop, take=spec.take, max_bars=spec.max_bars, min_return=spec.min_return
    )
    window_spec = WindowSpec(
        window=spec.window,
        feature_set=spec.feature_set,
        base_horizon=spec.horizon,
        context_horizons=tuple(spec.context_horizons),
        label=label,
        layout=spec.layout,
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

    merged = SequenceBundle(
        X=np.concatenate([b.X for b in bundles]),
        y=np.concatenate([b.y for b in bundles]),
        w=np.concatenate([b.w for b in bundles]),
        t=np.concatenate([b.t for b in bundles]),
        feature_names=bundles[0].feature_names,
        feature_spec=bundles[0].feature_spec,
    )
    return merged, series, label, window_spec


def _fit_fold(merged, tr, va, te, spec: TrainingSpec, *, progress=None):
    """Fit the scaler on the train mask and train one model. Returns
    ``(model, scaler, config, train_report)``."""
    from trader.models.scaler import StandardScaler

    X, y, w = merged.X, merged.y, merged.w
    scaler = StandardScaler().fit(X[tr])

    if spec.model_type in ("gbm", "xgb"):
        from trader.models.gbm import GBMConfig, train_gbm
        from trader.models.xgb import train_xgb

        config = GBMConfig(
            num_leaves=spec.num_leaves,
            max_depth=spec.max_depth,
            learning_rate=spec.lr,
            n_estimators=spec.n_estimators,
            min_child_samples=spec.min_child_samples,
            subsample=spec.subsample,
            colsample_bytree=spec.colsample_bytree,
        )
        # same config, same data: xgb is the CUDA-capable trainer for the same trees
        extra = {"device": spec.device} if spec.model_type == "xgb" else {}
        trainer = train_xgb if spec.model_type == "xgb" else train_gbm
        model, train_report = trainer(
            scaler.transform(X[tr]),
            y[tr],
            w[tr],
            scaler.transform(X[va]),
            y[va],
            config=config,
            class_weight=spec.class_weight,
            use_sample_weights=spec.use_sample_weights,
            seed=spec.seed,
            progress=progress,
            **extra,
        )
        return model, scaler, config, train_report

    from trader.models.lstm import LSTMConfig
    from trader.models.training import train_model

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
        device=spec.device,
        progress=progress,
    )
    return model, scaler, config, train_report


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
