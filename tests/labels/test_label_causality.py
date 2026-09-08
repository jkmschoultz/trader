"""A label may look forward -- but only as far as its own barriers.

Perturbing a bar beyond ``entry_offset + max_bars`` from a decision bar must not
change that bar's label or its event metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

from trader.labels.triple_barrier import triple_barrier


@pytest.mark.parametrize("max_bars", [8, 20])
def test_future_beyond_the_window_does_not_change_a_label(random_walk_frame, max_bars):
    frame = random_walk_frame(500, seed=3)
    reference = triple_barrier(frame, stop=0.01, take=0.01, max_bars=max_bars)

    cut = 300
    horizon_end = cut + 1 + max_bars  # last bar row `cut` may legitimately read
    tampered = frame.copy()
    beyond = tampered.index > horizon_end
    tampered.loc[beyond, ["open", "high", "low", "close"]] *= 4.0
    perturbed = triple_barrier(tampered, stop=0.01, take=0.01, max_bars=max_bars)

    a = reference.iloc[: cut + 1]
    b = perturbed.iloc[: cut + 1]
    for col in ("label", "entry_price", "touch_price", "barrier", "bars_held", "ret"):
        left, right = a[col], b[col]
        both_nan = left.isna() & right.isna()
        assert (both_nan | (left.fillna(0) == right.fillna(0))).all(), col


def test_a_touch_is_unaffected_by_anything_after_it(random_walk_frame):
    frame = random_walk_frame(400, seed=7)
    ref = triple_barrier(frame, stop=0.008, take=0.008, max_bars=30)

    # take the first row that resolved on a barrier touch, mangle everything
    # strictly after its touch bar, and check the row is identical
    touched = ref[ref["barrier"].isin(["stop", "take"])]
    assert len(touched) > 0
    pos = frame.index[frame["time"] == touched.index[0]][0]
    held = int(touched.iloc[0]["bars_held"])
    touch_bar = pos + held

    tampered = frame.copy()
    tampered.loc[tampered.index > touch_bar, ["open", "high", "low", "close"]] *= 5.0
    again = triple_barrier(tampered, stop=0.008, take=0.008, max_bars=30)

    before, after = ref.iloc[pos], again.iloc[pos]
    assert before["label"] == after["label"]
    assert before["barrier"] == after["barrier"]
    assert np.isclose(before["ret"], after["ret"])
