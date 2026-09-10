"""Train a LightGBM gradient-boosted classifier. Imports lightgbm -- lazy.

The tabular counterpart to :mod:`trader.models.lstm`: same 3-class
triple-barrier target (down / flat / up), same :class:`TrainReport`, same
registry directory. It consumes the ``layout="tabular"`` bundle -- one flat
feature vector per decision bar -- so there is no sequence dimension here.

Deterministic given ``seed``, early-stopped on validation multi-logloss,
class-weighted by default. Returns the fitted estimator and a JSON-safe
:class:`TrainReport`.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from trader.models.report import TrainReport

__all__ = ["GBMConfig", "train_gbm"]


@dataclass(frozen=True)
class GBMConfig:
    """LightGBM hyperparameters the registry stores under ``hyperparameters``."""

    num_leaves: int = 31
    max_depth: int = -1
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 20
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 0.0
    n_classes: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GBMConfig:
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**fields)


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y, minlength=3)
    return {"down": int(counts[0]), "flat": int(counts[1]), "up": int(counts[2])}


def train_gbm(
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
    progress: Callable[[float], None] | None = None,
) -> tuple[object, TrainReport]:
    """Fit a LightGBM classifier and return ``(estimator, report)``.

    Raises:
        ValueError: no training samples, or lightgbm is not installed.
    """
    if len(Xtr) == 0:
        raise ValueError("no training samples (after purge/embargo?)")

    from trader.models._optional import require_lightgbm

    lgb = require_lightgbm()
    from sklearn.metrics import confusion_matrix, f1_score

    Xtr = np.ascontiguousarray(Xtr, dtype=np.float32)
    Xva = np.ascontiguousarray(Xva, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.int64)
    yva = np.asarray(yva, dtype=np.int64)

    cw = class_weight if isinstance(class_weight, str) else None
    if class_weight is not None and not isinstance(class_weight, str):
        cw = {i: float(v) for i, v in enumerate(class_weight)}

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=config.n_classes,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        subsample_freq=config.subsample_freq,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        class_weight=cw,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
        n_jobs=1,
    )

    started = time.perf_counter()
    dep_warning = getattr(
        __import__("lightgbm.basic", fromlist=["LGBMDeprecationWarning"]),
        "LGBMDeprecationWarning",
        DeprecationWarning,
    )
    with warnings.catch_warnings():
        # lightgbm 4.7 renamed eval_set -> eval_X/eval_y but still accepts the
        # old kwarg; keep eval_set for >= 4.3 compatibility and hush the notice.
        warnings.filterwarnings("ignore", category=dep_warning)
        clf.fit(
            Xtr,
            ytr,
            sample_weight=np.asarray(wtr, dtype=np.float64) if use_sample_weights else None,
            eval_set=[(Xva, yva)],
            eval_metric="multi_logloss",
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
    elapsed = time.perf_counter() - started
    if progress is not None:
        progress(1.0)

    curve = list(clf.evals_result_.get("valid_0", {}).get("multi_logloss", []))
    epochs = [
        {"iteration": i + 1, "val_multi_logloss": round(float(v), 5)} for i, v in enumerate(curve)
    ]
    best_iteration = int(getattr(clf, "best_iteration_", 0) or len(curve) or config.n_estimators)

    yva_pred = clf.predict(Xva)
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
        framework_version=lgb.__version__,
    )
    return clf, report
