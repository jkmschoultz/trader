"""Does the model know when it knows? Direction-skill metrics from class probabilities.

Accuracy and macro-F1 score every bar equally, but a strategy only trades the
bars where the model is confident. These metrics look at the *edge* the strategy
acts on -- ``p(up) - p(down)`` -- and ask whether a bigger edge really means
better odds:

- ``dir_auc``: how well the edge ranks up-labelled bars above down-labelled ones
  (0.5 = no skill, 1.0 = perfect). Rank-based, so unaffected by class
  weighting shifting the probabilities.
- ``long_top10`` / ``short_top10``: of the 10% of bars with the highest
  (lowest) edge, the share labelled up (down) -- to compare with ``up_rate`` /
  ``down_rate``, the share across all bars.
- ``signal_share`` / ``signal_hit``: the share of bars whose edge clears the
  strategy threshold in either direction, and how often those point the right
  way.

:func:`edge_table` bins the edge into quintiles and shows the up / down share
in each: with real skill the up share rises bin by bin.

Labels are ``0 = down, 1 = flat, 2 = up``; ``proba`` is ``(n, 3)`` in that order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["direction_metrics", "edge_table"]

DOWN, FLAT, UP = 0, 1, 2


def _edge(proba: np.ndarray) -> np.ndarray:
    proba = np.asarray(proba, dtype=float)
    return proba[:, UP] - proba[:, DOWN]


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties; NaN if a side is empty."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    sorted_score = score[order]
    # average rank over each run of equal scores
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    ends = np.r_[starts[1:], len(score)]
    for s, e in zip(starts, ends, strict=True):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def direction_metrics(
    proba: np.ndarray, y: np.ndarray, *, top: float = 0.10, threshold: float = 0.15
) -> dict[str, float]:
    """Direction-skill metrics (see the module docstring), unprefixed, rounded."""
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return {}
    edge = _edge(proba)
    directional = y != FLAT
    k = max(1, int(round(n * top)))
    order = np.argsort(edge, kind="mergesort")
    signal = np.abs(edge) > threshold
    right = np.where(edge > 0, y == UP, y == DOWN)

    out = {
        "dir_auc": _auc(edge[directional], y[directional] == UP),
        "long_top10": float((y[order[-k:]] == UP).mean()),
        "short_top10": float((y[order[:k]] == DOWN).mean()),
        "up_rate": float((y == UP).mean()),
        "down_rate": float((y == DOWN).mean()),
        "signal_share": float(signal.mean()),
        "signal_hit": float(right[signal].mean()) if signal.any() else float("nan"),
    }
    return {key: round(value, 5) for key, value in out.items()}


def edge_table(proba: np.ndarray, y: np.ndarray, *, bins: int = 5) -> list[dict[str, float]]:
    """Up / down share per edge quantile bin, lowest edge first."""
    y = np.asarray(y)
    if len(y) < bins:
        return []
    edge = _edge(proba)
    order = np.argsort(edge, kind="mergesort")
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append(
            {
                "edge_lo": round(float(edge[chunk].min()), 4),
                "edge_hi": round(float(edge[chunk].max()), 4),
                "n": int(len(chunk)),
                "up": round(float((y[chunk] == UP).mean()), 4),
                "down": round(float((y[chunk] == DOWN).mean()), 4),
            }
        )
    return rows
