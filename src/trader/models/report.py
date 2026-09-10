"""The training report -- model-agnostic, so it carries no torch / lightgbm import.

Both :func:`trader.models.training.train_model` (LSTM) and
:func:`trader.models.gbm.train_gbm` (gradient-boosted trees) fill one of these
and the registry serialises it. The ``epochs`` list is a per-round curve for the
LSTM and the boosting eval history for a GBM; everything else (confusion
matrices, class balance, headline metrics) has the same meaning for both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TrainReport"]


@dataclass
class TrainReport:
    epochs: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    val_confusion: list[list[int]] = field(default_factory=list)
    test_confusion: list[list[int]] | None = None
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    n_features: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    elapsed_seconds: float = 0.0
    torch_version: str = ""
    framework_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "val_confusion": self.val_confusion,
            "test_confusion": self.test_confusion,
            "class_distribution": self.class_distribution,
            "metrics": self.metrics,
            "n_features": self.n_features,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "torch_version": self.torch_version,
            "framework_version": self.framework_version,
        }
