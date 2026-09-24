"""Hand-checked triple-barrier cases: one per outcome, plus the tie rule."""

from __future__ import annotations

import numpy as np
import pytest

from trader.labels.triple_barrier import (
    BarrierParams,
    barrier_params_from_target,
    target_from_barrier_params,
    triple_barrier,
)


def test_clean_take_hit(bar_frame, flat_rows):
    # decision bar 0, entry at bar 1 open = 100, take barrier = 101
    rows = [
        (100, 100, 100, 100),
        (100, 100.2, 99.9, 100.1),
        (100.1, 101.6, 100.0, 101.5),  # bar 2 high clears 101
    ] + flat_rows(101.5, 30)
    events = triple_barrier(bar_frame(rows), stop=0.01, take=0.01, max_bars=5)
    row = events.iloc[0]

    assert row["label"] == 1.0
    assert row["barrier"] == "take"
    assert row["entry_price"] == 100.0
    assert row["touch_price"] == pytest.approx(101.0)
    assert row["bars_held"] == 2.0
    assert row["ret"] == pytest.approx(0.01)


def test_clean_stop_hit(bar_frame, flat_rows):
    rows = [
        (100, 100, 100, 100),
        (100, 100.1, 99.8, 99.9),
        (99.9, 100.0, 98.4, 98.5),  # bar 2 low breaks 99
    ] + flat_rows(98.5, 30)
    row = triple_barrier(bar_frame(rows), stop=0.01, take=0.01, max_bars=5).iloc[0]

    assert row["label"] == -1.0
    assert row["barrier"] == "stop"
    assert row["touch_price"] == pytest.approx(99.0)
    assert row["ret"] == pytest.approx(-0.01)


def test_one_bar_spanning_both_barriers_takes_the_stop(bar_frame, flat_rows):
    rows = [
        (100, 100, 100, 100),
        (100, 100.1, 99.9, 100.0),
        (100.0, 102.0, 98.0, 100.0),  # spans 99 and 101 in one bar
    ] + flat_rows(100.0, 30)
    row = triple_barrier(bar_frame(rows), stop=0.01, take=0.01, max_bars=5).iloc[0]

    assert row["barrier"] == "stop"
    assert row["label"] == -1.0


def test_timeout_above_the_deadband_is_labelled_up(bar_frame):
    # drifts up ~0.4% over the window, never touching the 1% barriers
    rows = [
        (100 + 0.05 * i, 100 + 0.05 * i + 0.05, 100 + 0.05 * i - 0.05, 100 + 0.05 * i)
        for i in range(20)
    ]
    events = triple_barrier(bar_frame(rows), stop=0.01, take=0.01, max_bars=5, min_return=0.001)
    row = events.iloc[0]

    assert row["barrier"] == "time"
    assert row["label"] == 1.0
    assert row["bars_held"] == 5.0
    assert row["ret"] > 0.001


def test_timeout_inside_the_deadband_is_labelled_flat(bar_frame, flat_rows):
    row = triple_barrier(
        bar_frame(flat_rows(100.0, 40)), stop=0.05, take=0.05, max_bars=5, min_return=0.001
    ).iloc[0]

    assert row["barrier"] == "time"
    assert row["label"] == 0.0
    assert row["ret"] == pytest.approx(0.0)


def test_trailing_rows_have_no_label(bar_frame, flat_rows):
    events = triple_barrier(bar_frame(flat_rows(100.0, 20)), stop=0.01, take=0.01, max_bars=5)
    # entry_offset (1) + max_bars (5) rows cannot resolve
    assert events["label"].iloc[-6:].isna().all()
    assert events["label"].iloc[:-6].notna().all()


def test_index_is_the_decision_bar_and_entry_is_the_next_open(bar_frame, flat_rows):
    frame = bar_frame(flat_rows(100.0, 20))
    events = triple_barrier(frame, stop=0.01, take=0.01, max_bars=5)
    assert list(events.index) == list(frame["time"])
    assert events.index.name == "time"
    assert events["entry_time"].iloc[0] == frame["time"].iloc[1]


def test_none_barrier_is_never_touched(bar_frame, flat_rows):
    rows = [(100, 100, 100, 100)] + [(100, 100, 90, 95)] + flat_rows(95.0, 20)
    row = triple_barrier(bar_frame(rows), stop=None, take=0.01, max_bars=5).iloc[0]
    # a 5% drop with no stop barrier -> falls through to the vertical barrier
    assert row["barrier"] == "time"


def test_barrier_params_round_trip_with_a_target():
    from trader.strategies.base import Target

    target = Target(1.0, stop=0.005, take=0.01, max_bars=12)
    params = barrier_params_from_target(target)
    assert params == BarrierParams(stop=0.005, take=0.01, max_bars=12)

    kwargs = target_from_barrier_params(params)
    assert Target(-1.0, **kwargs).max_bars == 12


def test_barrier_params_reject_a_target_without_a_vertical_barrier():
    from trader.strategies.base import Target

    with pytest.raises(ValueError, match="max_bars"):
        barrier_params_from_target(Target(1.0, stop=0.01))


def test_labels_are_only_minus_one_zero_or_one(random_walk_frame):
    events = triple_barrier(random_walk_frame(500), stop=0.01, take=0.01, max_bars=20)
    labels = set(events["label"].dropna().unique())
    assert labels <= {-1.0, 0.0, 1.0}
    assert np.isnan(events["label"].iloc[-1])


def test_a_gap_through_the_stop_is_labelled_at_the_open(bar_frame, flat_rows):
    rows = [
        (100, 100, 100, 100),
        (100, 100.1, 99.9, 100.0),  # entry at 100; stop 99
        (97.0, 97.5, 96.5, 97.0),  # opens at 97: an overnight gap through the stop
    ] + flat_rows(97.0, 30)
    row = triple_barrier(bar_frame(rows), stop=0.01, take=0.01, max_bars=5).iloc[0]

    assert row["label"] == -1.0
    assert row["barrier"] == "stop"
    assert row["touch_price"] == pytest.approx(97.0)
    assert row["ret"] == pytest.approx(-0.03)


def test_a_gap_through_the_take_is_a_take_even_if_the_bar_then_breaks_the_stop(
    bar_frame, flat_rows
):
    rows = [
        (100, 100, 100, 100),
        (100, 100.1, 99.9, 100.0),  # entry at 100; take 101, stop 99
        (102.0, 102.0, 98.0, 98.5),  # opens above the take, later falls through the stop
    ] + flat_rows(98.5, 30)
    row = triple_barrier(bar_frame(rows), stop=0.01, take=0.01, max_bars=5).iloc[0]

    assert row["label"] == 1.0
    assert row["barrier"] == "take"
    assert row["touch_price"] == pytest.approx(102.0)
    assert row["ret"] == pytest.approx(0.02)
