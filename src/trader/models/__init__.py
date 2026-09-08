"""Model layer: feature windowing, an LSTM classifier, training, and (4d) a
registry.

The windowing and scaler are numpy/pandas only. The LSTM and the training loop
need the ``[model]`` extra (torch, scikit-learn); import
:mod:`trader.models.lstm` / :mod:`trader.models.training` lazily and call
:func:`trader.models._optional.require_torch` first. This package's top level
stays torch-free so a ``[data]``-only install can still read model manifests.
"""

from __future__ import annotations

from trader.models._optional import require_sklearn_metrics, require_torch
from trader.models.dataset import (
    SequenceBundle,
    SplitSpec,
    WindowSpec,
    build_bundle,
    time_split,
)
from trader.models.scaler import StandardScaler

__all__ = [
    "SequenceBundle",
    "SplitSpec",
    "StandardScaler",
    "WindowSpec",
    "build_bundle",
    "require_sklearn_metrics",
    "require_torch",
    "time_split",
]
