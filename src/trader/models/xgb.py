"""Train an XGBoost gradient-boosted classifier, on CUDA when there is one.

The GPU sibling of :mod:`trader.models.gbm`: same 3-class triple-barrier target,
same ``layout="tabular"`` bundle, same :class:`GBMConfig` hyperparameters and
:class:`TrainReport`. It exists for speed -- on a fold-sized problem (50k rows x
1,360 lag columns) a CUDA ``hist`` fit is ~10x faster than single-threaded
LightGBM, which is what a sweep's parallel workers each get.

The mapping from :class:`GBMConfig` is kept as close to LightGBM's semantics as
XGBoost allows: leaf-wise growth (``grow_policy="lossguide"``) capped at
``num_leaves``; ``min_child_samples`` becomes a hessian floor
(``min_child_weight``), scaled by ~0.2 because that is the per-sample hessian
``p(1 - p)`` of a 3-class softmax near uniform; ``class_weight="balanced"``
becomes per-sample weights, as XGBoost has no class-weight option.

Early-stopped on validation multi-logloss. Prediction is moved to the CPU on
load (``set_param(device="cpu")``) so a GPU-trained model backtests and trades
on a machine without one.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from functools import cache

import numpy as np

from trader.models.gbm import GBMConfig, _class_distribution
from trader.models.report import TrainReport

__all__ = ["cuda_available", "resolve_device", "train_xgb"]

#: per-sample hessian of a 3-class softmax near uniform probabilities
_HESSIAN_PER_SAMPLE = 0.2


@cache
def cuda_available() -> bool:
    """Whether XGBoost can actually train on CUDA here (probed once per process)."""
    from trader.models._optional import require_xgboost

    xgb = require_xgboost()
    if not xgb.build_info().get("USE_CUDA"):
        return False
    try:
        probe = xgb.DMatrix(np.zeros((4, 1), dtype=np.float32), label=np.array([0, 1, 0, 1]))
        xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0}, probe, 1)
    except xgb.core.XGBoostError:
        return False
    return True


def resolve_device(device: str) -> str:
    """``"auto"`` -> ``"cuda"`` when usable, else ``"cpu"``; anything else as given."""
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    return device


def _balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """sklearn's ``class_weight="balanced"``: ``n / (k * count[class])`` per sample."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    per_class = np.divide(len(y), n_classes * counts, out=np.zeros_like(counts), where=counts > 0)
    return per_class[y]


def train_xgb(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    wtr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    config: GBMConfig,
    early_stopping_rounds: int = 50,
    class_weight: str | Sequence[float] | None = "balanced",
    use_sample_weights: bool = False,
    seed: int = 0,
    device: str = "auto",
    progress: Callable[[float], None] | None = None,
) -> tuple[object, TrainReport]:
    """Fit an XGBoost classifier and return ``(booster, report)``.

    The booster is trimmed to the early-stopping best iteration and set to
    predict on the CPU; ``booster.inplace_predict(X)`` gives ``(n, 3)``
    class probabilities.

    Raises:
        ValueError: no training samples, or xgboost is not installed.
    """
    if len(Xtr) == 0:
        raise ValueError("no training samples (after purge/embargo?)")

    from trader.models._optional import require_sklearn_metrics, require_xgboost

    xgb = require_xgboost()
    confusion_matrix, f1_score = require_sklearn_metrics()

    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
    Xva = np.ascontiguousarray(Xva, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yva = np.asarray(yva, dtype=np.int64)

    weight = np.ones(len(ytr), dtype=np.float64)
    if class_weight == "balanced":
        weight *= _balanced_weights(ytr, config.n_classes)
    elif class_weight is not None and not isinstance(class_weight, str):
        weight *= np.asarray(class_weight, dtype=np.float64)[ytr]
    if use_sample_weights:
        weight *= np.asarray(wtr, dtype=np.float64)

    dev = resolve_device(device)
    # the native API, not XGBClassifier: the sklearn wrapper insists labels be
    # consecutive, and a fold can lack the (rare) flat class entirely
    params = {
        "objective": "multi:softprob",
        "num_class": config.n_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": dev,
        "grow_policy": "lossguide",
        "max_leaves": config.num_leaves,
        "max_depth": max(config.max_depth, 0),  # LightGBM's -1 (no limit) is XGBoost's 0
        "learning_rate": config.learning_rate,
        "min_child_weight": config.min_child_samples * _HESSIAN_PER_SAMPLE,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "reg_lambda": config.reg_lambda,
        "seed": seed,
        "nthread": 1,
        "verbosity": 0,
    }
    dtrain = xgb.DMatrix(Xtr, label=ytr, weight=weight)
    dval = xgb.DMatrix(Xva, label=yva)
    history: dict = {}

    started = time.perf_counter()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=config.n_estimators,
        evals=[(dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        evals_result=history,
        verbose_eval=False,
    )
    elapsed = time.perf_counter() - started
    if progress is not None:
        progress(1.0)

    best_round = int(getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1))
    # keep only the trees up to the early-stopping best: anything after would
    # leak into inference, which predicts with every tree it is given
    booster = booster[: best_round + 1]
    booster.set_param({"device": "cpu"})  # small predictions; identical on any device

    curve = list(history.get("val", {}).get("mlogloss", []))
    epochs = [
        {"iteration": i + 1, "val_multi_logloss": round(float(v), 5)} for i, v in enumerate(curve)
    ]
    best_iteration = best_round + 1

    yva_pred = (
        booster.inplace_predict(Xva).argmax(axis=1) if len(yva) else np.array([], dtype=np.int64)
    )
    val_acc = float((yva_pred == yva).mean()) if len(yva) else 0.0
    val_f1 = (
        float(f1_score(yva, yva_pred, average="macro", labels=[0, 1, 2], zero_division=0))
        if len(yva)
        else 0.0
    )
    val_confusion = (
        confusion_matrix(yva, yva_pred, labels=[0, 1, 2]).tolist() if len(yva) else [[0, 0, 0]] * 3
    )

    report = TrainReport(
        epochs=epochs,
        best_epoch=best_iteration,
        val_confusion=val_confusion,
        class_distribution={
            "train": _class_distribution(ytr),
            "val": _class_distribution(yva),
        },
        metrics={"val_acc": round(val_acc, 5), "val_macro_f1": round(val_f1, 5)},
        n_features=int(Xtr.shape[1]),
        n_train=int(len(Xtr)),
        n_val=int(len(Xva)),
        elapsed_seconds=elapsed,
        framework_version=f"xgboost {xgb.__version__} ({dev})",
    )
    return booster, report
