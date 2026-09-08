"""Sample weights for overlapping triple-barrier events.

Triple-barrier events overlap in time -- while one is still open, the next few
start -- so treating every labelled row as an independent observation
over-counts the crowded stretches. Two shallow corrections, enough for a first
sequence model where class balance is the bigger lever:

* **average uniqueness** -- the mean, over an event's life, of ``1 /
  (concurrent events)``. A run of tightly overlapping events each score low.
* **return-attribution weights** -- ``|ret| * average_uniqueness``, normalised
  to mean 1, so a big move that barely overlaps anything counts most.

The full sequential-bootstrap treatment (López de Prado, ch. 4) is deliberately
left for later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "average_uniqueness",
    "label_overlap_counts",
    "return_attribution_weights",
]


def _spans(events: pd.DataFrame) -> pd.DataFrame:
    """Resolved events with a usable ``[entry_time, touch_time]`` span."""
    usable = events.dropna(subset=["label", "entry_time", "touch_time"])
    return usable[usable["touch_time"] >= usable["entry_time"]]


def label_overlap_counts(events: pd.DataFrame, base_time: pd.Series) -> pd.Series:
    """How many events are open on each bar of ``base_time``."""
    index = pd.DatetimeIndex(base_time)
    counts = pd.Series(0, index=index, dtype="int64")
    spans = _spans(events)
    for entry, touch in zip(spans["entry_time"], spans["touch_time"], strict=True):
        counts.loc[entry:touch] += 1
    return counts


def average_uniqueness(events: pd.DataFrame, base_time: pd.Series) -> pd.Series:
    """Mean ``1 / concurrency`` over each event's life, indexed like ``events``."""
    spans = _spans(events)
    concurrency = label_overlap_counts(spans, base_time).replace(0, np.nan)
    inverse = 1.0 / concurrency

    out = pd.Series(np.nan, index=events.index, dtype="float64")
    for idx, entry, touch in zip(
        spans.index, spans["entry_time"], spans["touch_time"], strict=True
    ):
        window = inverse.loc[entry:touch]
        if len(window):
            out.loc[idx] = float(window.mean())
    return out


def return_attribution_weights(events: pd.DataFrame, base_time: pd.Series) -> pd.Series:
    """``|ret| * average_uniqueness`` per event, normalised to mean 1.

    Events with no usable span get weight 0.
    """
    uniqueness = average_uniqueness(events, base_time)
    raw = events["ret"].abs() * uniqueness
    raw = raw.fillna(0.0)
    total = raw.sum()
    if total <= 0:
        return raw
    return raw * (len(raw) / total)
