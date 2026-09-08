"""Train an :class:`~trader.models.lstm.LSTMClassifier`. Imports torch -- lazy.

Deterministic given ``seed``, early-stopped on validation macro-F1, class-
weighted by default. Returns the best model (by val F1) and a JSON-safe
:class:`TrainReport` carrying the per-epoch curve and val/test confusion
matrices.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trader.models.lstm import LSTMClassifier, LSTMConfig

__all__ = ["TrainReport", "train_model"]


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
        }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:  # best effort -- some ops have no deterministic kernel
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - never fatal
        pass


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _class_weights(
    y: np.ndarray,
    mode: str | Sequence[float] | None,
    n_classes: int,
) -> torch.Tensor | None:
    if mode is None:
        return None
    if mode == "balanced":
        counts = np.bincount(y, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (n_classes * counts)
    else:
        weights = np.asarray(list(mode), dtype=np.float64)
    return torch.tensor(weights, dtype=torch.float32)


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y, minlength=3)
    return {"down": int(counts[0]), "flat": int(counts[1]), "up": int(counts[2])}


@torch.no_grad()
def _evaluate(model, X, y, criterion, device, batch_size):
    from sklearn.metrics import confusion_matrix, f1_score

    model.eval()
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    losses: list[float] = []
    preds: list[np.ndarray] = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        losses.append(criterion(logits, yb).mean().item())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
    y_true = y.numpy()
    acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    macro_f1 = (
        float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0))
        if len(y_true)
        else 0.0
    )
    confusion = (
        confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
        if len(y_true)
        else [[0, 0, 0]] * 3
    )
    return float(np.mean(losses)) if losses else 0.0, acc, macro_f1, confusion


def train_model(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    wtr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    config: LSTMConfig,
    epochs: int,
    Xte: np.ndarray | None = None,
    yte: np.ndarray | None = None,
    batch_size: int = 128,
    lr: float = 1e-3,
    class_weight: str | Sequence[float] | None = "balanced",
    use_sample_weights: bool = False,
    patience: int = 8,
    grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    progress: Callable[[float], None] | None = None,
) -> tuple[LSTMClassifier, TrainReport]:
    """Fit an LSTM and return ``(best_model, report)``.

    Raises:
        ValueError: no training samples.
    """
    if len(Xtr) == 0:
        raise ValueError("no training samples (after purge/embargo?)")

    _seed_everything(seed)
    dev = _resolve_device(device)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    wtr_t = torch.tensor(wtr, dtype=torch.float32)
    Xva_t = torch.tensor(Xva, dtype=torch.float32)
    yva_t = torch.tensor(yva, dtype=torch.long)

    model = LSTMClassifier(config).to(dev)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    weight = _class_weights(ytr, class_weight, config.n_classes)
    criterion = nn.CrossEntropyLoss(
        weight=weight.to(dev) if weight is not None else None,
        reduction="none",
    )

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t, wtr_t),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
    )

    report = TrainReport(
        n_features=config.n_features,
        n_train=len(Xtr),
        n_val=len(Xva),
        n_test=0 if Xte is None else len(Xte),
        torch_version=torch.__version__,
        class_distribution={
            "train": _class_distribution(ytr),
            "val": _class_distribution(yva),
        },
    )

    best_f1 = -1.0
    best_state: dict | None = None
    best_epoch = 0
    since_improved = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(dev), yb.to(dev), wb.to(dev)
            optimiser.zero_grad()
            per_sample = criterion(model(xb), yb)
            loss = (per_sample * wb).mean() if use_sample_weights else per_sample.mean()
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss, val_acc, val_f1, val_confusion = _evaluate(
            model, Xva_t, yva_t, criterion, dev, batch_size
        )
        report.epochs.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_acc": round(val_acc, 5),
                "val_macro_f1": round(val_f1, 5),
            }
        )

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            report.val_confusion = val_confusion
            since_improved = 0
        else:
            since_improved += 1

        if progress is not None:
            progress(epoch / epochs)
        if since_improved >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    report.best_epoch = best_epoch
    report.elapsed_seconds = time.perf_counter() - started

    val_loss, val_acc, val_f1, val_confusion = _evaluate(
        model, Xva_t, yva_t, criterion, dev, batch_size
    )
    report.val_confusion = val_confusion
    report.metrics = {"val_acc": round(val_acc, 5), "val_macro_f1": round(val_f1, 5)}

    if Xte is not None and yte is not None and len(Xte):
        Xte_t = torch.tensor(Xte, dtype=torch.float32)
        yte_t = torch.tensor(yte, dtype=torch.long)
        te_loss, te_acc, te_f1, te_confusion = _evaluate(
            model, Xte_t, yte_t, criterion, dev, batch_size
        )
        report.test_confusion = te_confusion
        report.class_distribution["test"] = _class_distribution(yte)
        report.metrics.update({"test_acc": round(te_acc, 5), "test_macro_f1": round(te_f1, 5)})

    return model, report
