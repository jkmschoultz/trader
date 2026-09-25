"""Direction-skill metrics: AUC against sklearn, the top / bottom decile, the edge table."""

from __future__ import annotations

import numpy as np
import pytest

from trader.models.confidence import _auc, direction_metrics, edge_table


def _proba(edge):
    """Probabilities whose p(up) - p(down) equals ``edge`` (flat takes the rest)."""
    edge = np.asarray(edge, dtype=float)
    up = 0.4 + edge / 2
    down = 0.4 - edge / 2
    return np.column_stack([down, 1 - up - down, up])


def test_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    score = rng.integers(0, 5, 300).astype(float)  # plenty of ties
    positive = rng.random(300) < 0.4
    assert _auc(score, positive) == pytest.approx(roc_auc_score(positive, score))


def test_a_perfect_ranking():
    y = np.array([0] * 50 + [2] * 50)
    edge = np.r_[np.linspace(-0.5, -0.1, 50), np.linspace(0.1, 0.5, 50)]
    m = direction_metrics(_proba(edge), y)
    assert m["dir_auc"] == 1.0
    assert m["long_top10"] == 1.0 and m["short_top10"] == 1.0
    assert m["up_rate"] == 0.5
    assert m["signal_hit"] == 1.0


def test_no_skill_sits_at_the_base_rate():
    rng = np.random.default_rng(1)
    y = rng.choice([0, 2], 20000, p=[0.6, 0.4])
    m = direction_metrics(_proba(rng.uniform(-0.3, 0.3, 20000)), y)
    assert m["dir_auc"] == pytest.approx(0.5, abs=0.02)
    assert m["long_top10"] == pytest.approx(0.4, abs=0.03)


def test_flat_labels_are_left_out_of_the_auc_but_count_as_misses():
    y = np.array([0, 1, 2, 1])
    edge = np.array([-0.3, 0.9, 0.3, -0.9])  # flats get the most extreme edges
    m = direction_metrics(_proba(edge), y, top=0.25)
    assert m["dir_auc"] == 1.0
    assert m["long_top10"] == 0.0  # the top bar was flat


def test_edge_table_bins_lowest_edge_first():
    y = np.array([0] * 5 + [2] * 5)
    rows = edge_table(_proba(np.linspace(-0.4, 0.4, 10)), y, bins=5)
    assert [r["n"] for r in rows] == [2] * 5
    assert rows[0]["down"] == 1.0 and rows[-1]["up"] == 1.0
    assert rows[0]["edge_hi"] < rows[-1]["edge_lo"]
