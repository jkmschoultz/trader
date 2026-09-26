"""No-skill baselines: reproducible coin flips, buy-and-hold, and bracketed entries."""

from __future__ import annotations

import numpy as np
import pytest

from trader.strategies.base import Hold, Target
from trader.strategies.baseline import Baseline


def _walk(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return list(100 + np.cumsum(rng.normal(0, 0.5, n)))


def test_random_directions_are_reproducible_and_seeded(make_bars, make_context):
    closes = _walk()

    def sides(seed):
        strat = Baseline(direction="random", seed=seed, hold_bars=1)
        return [strat.on_bar(make_context(make_bars(closes[:n]))).weight for n in range(1, 60)]

    assert sides(0) == sides(0)
    assert sides(0) != sides(1)
    assert set(sides(0)) == {1.0, -1.0}
    assert abs(np.mean(sides(0))) < 0.4  # roughly balanced


def test_scheduled_mode_redraws_every_hold_bars(make_bars, make_context):
    strat = Baseline(direction="long", hold_bars=3)
    closes = _walk()
    acted = [
        not isinstance(strat.on_bar(make_context(make_bars(closes[:n]))), Hold)
        for n in range(1, 10)
    ]
    assert acted == [True, False, False, True, False, False, True, False, False]


def test_bracketed_mode_enters_only_when_flat(make_bars, make_context):
    strat = Baseline(direction="long", stop=0.01, take=0.02, max_bars=10)
    bars = make_bars(_walk())
    flat = strat.on_bar(make_context(bars, units=0.0))
    assert isinstance(flat, Target)
    assert (flat.stop, flat.take, flat.max_bars) == (0.01, 0.02, 10)
    assert isinstance(strat.on_bar(make_context(bars, units=5.0)), Hold)


def test_atr_bracket_is_the_labels_atr_multiple(make_bars, make_context):
    from trader.labels.triple_barrier import barrier_unit

    closes = _walk(80)
    bars = make_bars(closes, highs=[c + 0.4 for c in closes], lows=[c - 0.4 for c in closes])
    strat = Baseline(direction="short", stop=2.0, take=3.0, max_bars=5, barrier_scale="atr")
    assert strat.warmup == 15
    decision = strat.on_bar(make_context(bars))
    unit = barrier_unit(bars, "atr")[-1]
    assert decision.weight == -1.0
    assert decision.stop == pytest.approx(2.0 * unit)
    assert decision.take == pytest.approx(3.0 * unit)


def test_entry_prob_thins_out_entries(make_bars, make_context):
    closes = _walk(400)
    strat = Baseline(direction="long", stop=0.01, max_bars=5, entry_prob=0.25)
    entries = sum(
        isinstance(strat.on_bar(make_context(make_bars(closes[:n]))), Target) for n in range(1, 400)
    )
    assert 60 < entries < 140


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        Baseline(direction="sideways")
    with pytest.raises(ValueError):
        Baseline(entry_prob=0.0)


def test_long_flat_flips_between_long_and_out(make_bars, make_context):
    from trader.strategies.base import Flat

    closes = _walk()
    strat = Baseline(direction="long_flat", hold_bars=1)
    kinds = {type(strat.on_bar(make_context(make_bars(closes[:n])))).__name__ for n in range(1, 60)}
    assert kinds == {"Target", "Flat"}
    decisions = [strat.on_bar(make_context(make_bars(closes[:n]))) for n in range(1, 60)]
    assert all(d.weight == 1.0 for d in decisions if not isinstance(d, Flat))


def test_primary_gate_keeps_the_baseline_flat_while_momentum_is_down():
    import pandas as pd

    from trader.strategies.base import BarContext as Ctx
    from trader.strategies.base import Flat, PortfolioView, Position

    times = pd.date_range("2024-01-01", periods=24 * 12, freq="h", tz="UTC")
    down = np.linspace(200, 100, len(times))  # every day lower than the last
    frame = pd.DataFrame({"time": times, "open": down, "high": down, "low": down, "close": down})

    def ctx(n):
        return Ctx(
            label="X",
            now=times[n - 1],
            horizon=60,
            history=frame.iloc[:n],
            position=Position("X"),
            portfolio=PortfolioView(cash=1.0, equity=1.0, positions={}),
        )

    strat = Baseline(direction="long", hold_bars=1, primary="tsmom", primary_lookbacks="1/2")
    assert all(isinstance(strat.on_bar(ctx(n)), Flat) for n in range(80, len(times), 17))
    ungated = Baseline(direction="long", hold_bars=1)
    assert isinstance(ungated.on_bar(ctx(200)), Target)


def test_a_gated_baseline_asks_folds_for_a_year_of_days():
    from trader.service.evaluation import fold_lookback

    strat = Baseline(direction="long", primary="tsmom")
    assert strat.history_days == 255
    assert fold_lookback(60, strat.warmup, strat.history_days).days >= 255
    assert Baseline(direction="long").history_days == 0
