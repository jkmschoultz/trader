"""The engine: causal fills, bracket exits, tradability, and multi-instrument
accounting that reconciles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from trader.backtest import CostModel, EqualWeight, run
from trader.data.lake import normalise
from trader.strategies.base import (
    BarContext,
    Decision,
    Flat,
    Hold,
    InstrumentStrategy,
    Order,
    Orders,
    Side,
    Target,
)

_START = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)


def make_frame(
    *,
    opens: list[float],
    closes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    states: list[str | None] | None = None,
    horizon: int = 5,
) -> pd.DataFrame:
    n = len(opens)
    closes = closes if closes is not None else list(opens)
    return normalise(
        pd.DataFrame(
            {
                "time": [_START + timedelta(minutes=horizon * i) for i in range(n)],
                "open": opens,
                "high": highs
                if highs is not None
                else [max(o, c) for o, c in zip(opens, closes, strict=True)],
                "low": lows
                if lows is not None
                else [min(o, c) for o, c in zip(opens, closes, strict=True)],
                "close": closes,
                "volume": [1_000.0] * n,
                "interest": [None] * n,
                "close_ask": [None] * n,
                "trading_state": states if states is not None else [None] * n,
            }
        )
    )


class Always(InstrumentStrategy):
    """Return the same decision on every bar."""

    def __init__(self, decision: Decision, *, warmup: int = 0) -> None:
        self._decision = decision
        self.warmup = warmup

    def on_bar(self, ctx: BarContext) -> Decision:
        return self._decision


class Script(InstrumentStrategy):
    """Return decisions from a list, one per call, then :class:`Hold`."""

    def __init__(self, decisions: list[Decision], *, warmup: int = 0) -> None:
        self._decisions = list(decisions)
        self._i = 0
        self.warmup = warmup
        self.seen_closes: list[float] = []

    def on_bar(self, ctx: BarContext) -> Decision:
        self.seen_closes.append(float(ctx.history["close"].iloc[-1]))
        if self._i < len(self._decisions):
            self._i += 1
            return self._decisions[self._i - 1]
        return Hold()


def _reconciles(result) -> bool:
    delta = result.final_equity - result.starting_equity
    return abs(float(result.trades["pnl"].sum()) - delta) < 1e-6


# --------------------------------------------------------------- causal fills


def test_a_decision_fills_at_the_next_bar_open_not_the_signal_bar():
    frame = make_frame(opens=[10, 20, 30, 40], closes=[11, 21, 31, 41])
    result = run({"X": frame}, {"X": Always(Target(1.0))}, horizon=5)

    first = result.fills.iloc[0]
    # Decision made on bar 0 (open 10 / close 11); it fills at bar 1's open, 20.
    assert first["price"] == 20.0
    assert first["time"] == frame["time"].iloc[1]


def test_the_strategy_never_sees_the_bar_it_trades_into():
    frame = make_frame(opens=[10, 11, 12, 13, 14], closes=[10.5, 11.5, 12.5, 13.5, 99.0])
    script = Script([Target(1.0)])
    run({"X": frame}, {"X": script}, horizon=5)

    assert 99.0 not in script.seen_closes
    assert script.seen_closes == [10.5, 11.5, 12.5, 13.5]


# ------------------------------------------------------------- bracket exits


def test_long_stop_is_hit_intrabar_and_exits_at_the_stop():
    # Entry at bar 1 open = 20, stop 5% -> 19. Bar 2 dips to 18.
    frame = make_frame(
        opens=[20, 20, 20, 20],
        closes=[20, 20, 20, 20],
        highs=[20, 20, 20.5, 20],
        lows=[20, 20, 18.0, 20],
    )
    result = run({"X": frame}, {"X": Script([Target(1.0, stop=0.05)])}, horizon=5)

    exits = result.trades[result.trades["exit_reason"] == "stop"]
    assert len(exits) == 1
    assert exits.iloc[0]["exit_price"] == 19.0


def test_take_profit_is_hit_intrabar_and_exits_at_the_target():
    frame = make_frame(
        opens=[20, 20, 20, 20],
        closes=[20, 20, 20, 20],
        highs=[20, 20, 23.0, 20],
        lows=[20, 20, 19.5, 20],
    )
    result = run({"X": frame}, {"X": Script([Target(1.0, stop=0.05, take=0.10)])}, horizon=5)

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take"
    assert trade["exit_price"] == 22.0


def test_when_one_bar_hits_both_barriers_the_stop_wins():
    frame = make_frame(
        opens=[20, 20, 20, 20],
        closes=[20, 20, 20, 20],
        highs=[20, 20, 23.0, 20],
        lows=[20, 20, 18.0, 20],
    )
    result = run({"X": frame}, {"X": Script([Target(1.0, stop=0.05, take=0.10)])}, horizon=5)

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == 19.0


def test_time_barrier_exits_at_the_open_after_the_countdown():
    frame = make_frame(opens=[10, 20, 30, 40, 50], closes=[10, 20, 30, 40, 50])
    result = run({"X": frame}, {"X": Script([Target(1.0, max_bars=2)])}, horizon=5)

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "max_bars"
    assert trade["exit_price"] == 40.0  # bar 3 open
    assert trade["bars_held"] == 2


# --------------------------------------------------------------- tradability


def test_a_non_tradable_bar_defers_the_fill_to_the_next_one():
    frame = make_frame(
        opens=[10, 20, 30, 40],
        closes=[10, 20, 30, 40],
        states=[None, "Halted", "Automated", "Automated"],
    )
    result = run({"X": frame}, {"X": Always(Target(1.0))}, horizon=5)

    assert result.fills.iloc[0]["price"] == 30.0  # bar 2 open, bar 1 was halted
    assert result.fills.iloc[0]["time"] == frame["time"].iloc[2]


# ---------------------------------------------------------- multi-instrument


def test_equal_weight_two_instruments_reconciles_and_attributes():
    a = make_frame(opens=[10, 11, 12, 13, 12, 11])
    b = make_frame(opens=[50, 49, 48, 47, 48, 49])
    result = run(
        {"A": a, "B": b},
        Always(Target(1.0)),
        horizon=5,
        allocator=EqualWeight(),
        cost_model=CostModel(commission_bps=1.0, slippage_bps=1.0),
    )

    assert _reconciles(result)
    attributed = sum(s.pnl for s in result.by_instrument.values())
    assert abs(attributed - (result.final_equity - result.starting_equity)) < 1e-6
    assert set(result.by_instrument) == {"A", "B"}


def test_gross_exposure_stays_within_the_leverage_cap():
    a = make_frame(opens=[10] * 8)
    b = make_frame(opens=[10] * 8)
    result = run(
        {"A": a, "B": b}, Always(Target(1.0)), horizon=5, allocator=EqualWeight(), leverage_cap=1.0
    )
    peak, pos = 0.0, {"A": 0.0, "B": 0.0}
    for _, fill in result.fills.iterrows():
        pos[fill["label"]] += fill["units"]
        peak = max(peak, sum(abs(v) for v in pos.values()) * 10.0)
    assert peak <= 100_000.0 + 1e-6


# ----------------------------------------------------------------- Orders


def test_orders_bypass_the_allocator_and_equity_sizing():
    frame = make_frame(opens=[10, 20, 30, 40])
    strat = Script([Orders([Order(Side.BUY, 100)])])
    result = run(
        {"X": frame},
        {"X": strat},
        horizon=5,
        allocator=EqualWeight(),
        leverage_cap=0.01,  # would size to ~0 units through the allocator
    )
    assert result.fills.iloc[0]["units"] == 100
    assert result.fills.iloc[0]["price"] == 20.0


# ------------------------------------------------------------------ basics


def test_a_flat_strategy_never_trades():
    frame = make_frame(opens=[10, 11, 12, 13])
    result = run({"X": frame}, {"X": Always(Flat())}, horizon=5)
    assert result.trades.empty
    assert result.fills.empty
    assert result.final_equity == result.starting_equity


def test_positions_are_flat_at_the_end():
    frame = make_frame(opens=[10, 11, 12, 13, 14])
    result = run({"X": frame}, {"X": Always(Target(1.0))}, horizon=5)
    assert not result.trades.empty
    assert result.trades.iloc[-1]["exit_reason"] == "end"
    assert _reconciles(result)


def test_warmup_delays_the_first_decision():
    frame = make_frame(opens=list(range(10, 30)))
    result = run({"X": frame}, {"X": Script([Target(1.0)], warmup=5)}, horizon=5)
    assert result.fills.iloc[0]["time"] == frame["time"].iloc[5]


def test_a_gap_through_the_stop_fills_at_the_open_not_the_stop():
    # entry at bar 1 open = 20, stop 5% -> 19; bar 2 opens at 17 (an overnight gap)
    frame = make_frame(
        opens=[20, 20, 17, 17],
        closes=[20, 20, 17, 17],
        highs=[20, 20, 17.5, 17],
        lows=[20, 20, 16.5, 17],
    )
    result = run({"X": frame}, {"X": Script([Target(1.0, stop=0.05)])}, horizon=5)

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == 17.0  # the market never traded at 19


def test_a_gap_through_the_take_fills_at_the_better_open():
    # take 10% -> 22; bar 2 opens at 23, then falls through the stop intrabar
    frame = make_frame(
        opens=[20, 20, 23, 20],
        closes=[20, 20, 18, 20],
        highs=[20, 20, 23, 20],
        lows=[20, 20, 18, 20],
    )
    result = run({"X": frame}, {"X": Script([Target(1.0, stop=0.05, take=0.10)])}, horizon=5)

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "take"  # the open came first
    assert trade["exit_price"] == 23.0


def test_a_short_gapping_up_through_its_stop_fills_at_the_open():
    # short at 20, stop 5% -> 21; bar 2 opens at 22
    frame = make_frame(
        opens=[20, 20, 22, 22],
        closes=[20, 20, 22, 22],
        highs=[20, 20, 22.5, 22],
        lows=[20, 20, 21.5, 22],
    )
    result = run({"X": frame}, {"X": Script([Target(-1.0, stop=0.05)])}, horizon=5)

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "stop"
    assert trade["exit_price"] == 22.0
