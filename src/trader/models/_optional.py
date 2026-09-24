"""Lazy import guard for the optional ``[model]`` dependencies.

``torch`` and ``scikit-learn`` are a heavy extra. The registry and manifest
paths never import them, so ``trader models list`` and the API's ``/models``
route work on a ``[data]``-only install; only actually training or running a
model needs them. Everything that does calls :func:`require_torch` first.

The error is a plain :class:`ValueError` on purpose: ``run_backtest`` and the
training service already wrap ``ValueError`` into an ``InvalidRequest``, so the
pip hint reaches the CLI and the API without extra plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

_HINT = 'install them with:  pip install -e ".[model]"'


def require_torch() -> torch:
    """Import and return :mod:`torch`, or raise with an install hint."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ValueError(f"this needs the optional model dependencies ({exc}); {_HINT}") from exc
    return torch


def require_sklearn_metrics():
    """Return ``(confusion_matrix, f1_score)`` from scikit-learn, or raise."""
    try:
        from sklearn.metrics import confusion_matrix, f1_score
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ValueError(f"this needs the optional model dependencies ({exc}); {_HINT}") from exc
    return confusion_matrix, f1_score


def require_lightgbm():
    """Import and return :mod:`lightgbm`, or raise with an install hint."""
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ValueError(f"this needs the optional model dependencies ({exc}); {_HINT}") from exc
    return lightgbm


def require_xgboost():
    """Import and return :mod:`xgboost`, or raise with an install hint."""
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ValueError(f"this needs the optional model dependencies ({exc}); {_HINT}") from exc
    return xgboost
