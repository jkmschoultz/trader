"""Overlap counts and uniqueness weights on hand-laid-out events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from trader.labels.weights import (
    average_uniqueness,
    label_overlap_counts,
    return_attribution_weights,
)

START = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)


def _bar_time(i: int) -> pd.Timestamp:
    return pd.Timestamp(START + timedelta(minutes=5 * i))


def _events(spans, rets=None):
    """``spans`` is a list of ``(entry_bar, touch_bar)`` index pairs."""
    rets = rets if rets is not None else [0.01] * len(spans)
    return pd.DataFrame(
        {
            "label": [1.0] * len(spans),
            "entry_time": [_bar_time(a) for a, _ in spans],
            "touch_time": [_bar_time(b) for _, b in spans],
            "ret": rets,
        }
    )


BASE = pd.Series([_bar_time(i) for i in range(20)])


def test_disjoint_events_are_fully_unique():
    events = _events([(0, 2), (5, 7), (10, 12)])
    uniqueness = average_uniqueness(events, BASE)
    assert uniqueness.tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_two_fully_overlapping_events_are_half_unique():
    events = _events([(0, 4), (0, 4)])
    uniqueness = average_uniqueness(events, BASE)
    assert uniqueness.tolist() == pytest.approx([0.5, 0.5])


def test_overlap_counts_track_concurrency():
    events = _events([(0, 5), (3, 8)])
    counts = label_overlap_counts(events, BASE)
    assert counts.loc[_bar_time(1)] == 1  # only the first event
    assert counts.loc[_bar_time(4)] == 2  # both open
    assert counts.loc[_bar_time(7)] == 1  # only the second
    assert counts.loc[_bar_time(12)] == 0


def test_attribution_weights_average_to_one_and_favour_the_big_lonely_move():
    events = _events([(0, 4), (0, 4), (10, 12)], rets=[0.01, 0.01, 0.05])
    weights = return_attribution_weights(events, BASE)
    assert weights.mean() == pytest.approx(1.0)
    assert weights.iloc[2] > weights.iloc[0]
