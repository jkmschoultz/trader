"""A tiny numpy standard-scaler, so the inference path needs only torch.

``sklearn.preprocessing.StandardScaler`` would do, but pulling scikit-learn into
:class:`~trader.strategies.lstm.LSTMStrategy` just to subtract a mean is not
worth it. This fits per feature column across the (n, window, F) training tensor,
serialises to plain lists, and reloads exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["StandardScaler"]

_EPS = 1e-8


@dataclass
class StandardScaler:
    """Zero-mean, unit-variance per feature. ``fit`` on train, ``transform`` on all."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> StandardScaler:
        """Fit on ``x`` of shape ``(n, window, features)`` or ``(n, features)``."""
        flat = x.reshape(-1, x.shape[-1]).astype(np.float64)
        self.mean_ = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        self.scale_ = np.where(std < _EPS, 1.0, std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler.transform called before fit")
        return ((x - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler.to_dict called before fit")
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> StandardScaler:
        return cls(
            mean_=np.asarray(data["mean"], dtype=np.float64),
            scale_=np.asarray(data["scale"], dtype=np.float64),
        )
