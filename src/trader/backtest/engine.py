"""The event-driven backtest engine.

One pass over a set of same-horizon bar series. At each event -- a timestamp
where at least one instrument printed a bar -- the engine:

1. marks every instrument to its last completed close;
2. checks bracket exits (stop, take-profit, time barrier) against the bar that
   just closed, stop before take when a single bar hits both;
3. asks each instrument's strategy for a :data:`~trader.strategies.base.Decision`
   from history that excludes the not-yet-closed bar;
4. runs the allocator over the resulting convictions and sizes the target
   weights to whole units, filling at the just-opened bar's open price.

The one hard rule is causality: a decision made on the bar opening at ``t`` is
filled at the open of the bar at ``t + horizon`` and never sees a price from it.
``docs/backtest.md`` has the reasoning and the known limitations.
"""

from __future__ import annotations

import copy
import logging
import math
from bisect import bisect_left
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from trader.backtest.allocator import Allocator, AllocatorContext, EqualWeight, PassThrough
from trader.backtest.costs import CostModel
from trader.backtest.metrics import Metrics, compute, instrument_stats, periods_per_year
from trader.backtest.result import BacktestResult
from trader.data.calendars import RegularHours
from trader.data.lake import SeriesKey, normalise
from trader.saxo.charts import TRADABLE_STATES
from trader.strategies.base import (
    BarContext,
    Flat,
    Hold,
    InstrumentStrategy,
    Order,
    Orders,
    PanelContext,
    PortfolioStrategy,
    PortfolioView,
    Position,
    Side,
    Strategy,
    Target,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Instrument:
    """Per-instrument facts the engine needs beyond the bars themselves."""

    key: SeriesKey | None = None
    lot_size: float = 1.0
    tick_size: float | None = None
    session: RegularHours | None = None


@dataclass
class _Pos:
    """Live position state for one instrument."""

    label: str
    units: float = 0.0
    avg_price: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_price: float = 0.0
    bars_remaining: int | None = None


@dataclass
class _OpenTrade:
    entry_time: datetime
    entry_price: float
    direction: int
    units: float
    #: Entry-side commission accrued but not yet booked. Drawn down pro-rata as
    #: the position is reduced, so every cent lands in exactly one trade row.
    costs: float = 0.0


@dataclass
class _EngineConfig:
    horizon: int
    starting_cash: float
    leverage_cap: float
    warmup: int | None
    rebalance_band: float = 0.25


def run(
    panel: Mapping[str, pd.DataFrame],
    strategy: Strategy | Mapping[str, Strategy],
    *,
    horizon: int,
    allocator: Allocator | None = None,
    cost_model: CostModel | Mapping[str, CostModel] | None = None,
    starting_cash: float = 100_000.0,
    instruments: Mapping[str, Instrument] | None = None,
    leverage_cap: float = 1.0,
    warmup: int | None = None,
    rebalance_band: float = 0.25,
    progress: Callable[[float], None] | None = None,
) -> BacktestResult:
    """Backtest ``strategy`` over ``panel`` and return a :class:`BacktestResult`.

    Args:
        panel: label -> canonical bar frame. Every frame must be the same
            ``horizon``; a label needs at least two bars.
        strategy: one :class:`~trader.strategies.base.InstrumentStrategy` (run as
            an independent deep copy per instrument), one
            :class:`~trader.strategies.base.PortfolioStrategy`, or a mapping of
            label -> instrument strategy.
        horizon: bar size in minutes, shared by every series.
        allocator: conviction -> weight rule. Defaults to
            :class:`~trader.backtest.allocator.PassThrough` for a single
            instrument, :class:`~trader.backtest.allocator.EqualWeight` otherwise.
        cost_model: one model for every instrument, or one per label. Defaults to
            a frictionless :class:`~trader.backtest.costs.CostModel`.
        instruments: per-label :class:`Instrument` (lot size, session, ...).
        leverage_cap: ceiling on gross exposure as a fraction of equity.
        warmup: override the bars-of-history gate; defaults to the largest
            ``strategy.warmup``.
        rebalance_band: leave a position alone unless the new target is at least
            this fraction away from it (or the sign changed). Stops equity- and
            price-drift from churning a constant conviction every bar.
        progress: called with a fraction in ``(0, 1]`` as the event loop
            advances, throttled to about fifty updates. For a UI or a long run;
            no effect on the result.
    """
    return _Engine(
        panel,
        strategy,
        allocator=allocator,
        cost_model=cost_model,
        instruments=instruments,
        cfg=_EngineConfig(horizon, starting_cash, leverage_cap, warmup, rebalance_band),
        progress=progress,
    ).run()


class _Engine:
    def __init__(
        self,
        panel: Mapping[str, pd.DataFrame],
        strategy: Strategy | Mapping[str, Strategy],
        *,
        allocator: Allocator | None,
        cost_model: CostModel | Mapping[str, CostModel] | None,
        instruments: Mapping[str, Instrument] | None,
        cfg: _EngineConfig,
        progress: Callable[[float], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._progress = progress
        self.labels = list(panel)
        if not self.labels:
            raise ValueError("panel is empty")

        self.frames = {label: normalise(panel[label]) for label in self.labels}
        for label, frame in self.frames.items():
            if len(frame) < 2:
                raise ValueError(f"panel[{label!r}] needs at least two bars")

        self.instruments = {
            label: (instruments or {}).get(label) or Instrument() for label in self.labels
        }

        base_cost = cost_model if cost_model is not None else CostModel()
        self.costs = {
            label: (base_cost[label] if isinstance(base_cost, Mapping) else base_cost)
            for label in self.labels
        }

        self.instrument_strategies, self.portfolio_strategy = self._build_strategies(strategy)
        warmups = [s.warmup for s in self.instrument_strategies.values()]
        if self.portfolio_strategy is not None:
            warmups.append(self.portfolio_strategy.warmup)
        self.warmup = cfg.warmup if cfg.warmup is not None else max(warmups or [0])

        if allocator is not None:
            self.allocator: Allocator = allocator
        else:
            self.allocator = PassThrough() if len(self.labels) == 1 else EqualWeight()

        # Per-label column arrays and an ordered timestamp list for bisecting.
        self.col: dict[str, dict[str, list]] = {}
        self.times: dict[str, list[pd.Timestamp]] = {}
        self.row_at: dict[str, dict[pd.Timestamp, int]] = {}
        for label, frame in self.frames.items():
            self.col[label] = {
                name: frame[name].tolist()
                for name in ("open", "high", "low", "close", "close_ask", "trading_state")
            }
            self.times[label] = list(frame["time"])
            self.row_at[label] = {ts: i for i, ts in enumerate(self.times[label])}

        merged: set[pd.Timestamp] = set()
        for label in self.labels:
            merged.update(self.times[label])
        self.events: list[pd.Timestamp] = sorted(merged)
        self.step = pd.Timedelta(minutes=cfg.horizon)

        # Mutable run state.
        self.cash = cfg.starting_cash
        self.pos = {label: _Pos(label) for label in self.labels}
        self.mark = {label: math.nan for label in self.labels}
        self.conviction = {label: 0.0 for label in self.labels}
        self.pending_bracket: dict[str, tuple | None] = {label: None for label in self.labels}
        self.open_trade: dict[str, _OpenTrade | None] = {label: None for label in self.labels}
        self.started = {label: False for label in self.labels}
        self._portfolio_started = False

        self.fills: list[dict] = []
        self.trades: list[dict] = []
        self.equity_points: list[tuple[pd.Timestamp, float]] = []
        self.exposure_hits = 0
        self.traded_notional = 0.0

    # ---------------------------------------------------------------- setup

    def _build_strategies(
        self, strategy: Strategy | Mapping[str, Strategy]
    ) -> tuple[dict[str, InstrumentStrategy], PortfolioStrategy | None]:
        if isinstance(strategy, Mapping):
            missing = set(self.labels) - set(strategy)
            if missing:
                raise ValueError(f"no strategy for {sorted(missing)}")
            return {label: strategy[label] for label in self.labels}, None
        if isinstance(strategy, PortfolioStrategy):
            return {}, strategy
        if isinstance(strategy, InstrumentStrategy):
            return {label: copy.deepcopy(strategy) for label in self.labels}, None
        raise TypeError(f"{type(strategy).__name__} is not a Strategy")

    # ------------------------------------------------------------- main loop

    def run(self) -> BacktestResult:
        total = len(self.events)
        tick = max(1, total // 50)
        for i, event in enumerate(self.events):
            active = self._active_labels(event)
            for label, row in active:
                self._bracket_exits(label, row, event)

            decided = self._collect_decisions(event, active)
            self._apply_decisions(event, dict(active), decided)

            equity = self._equity()
            self.equity_points.append((event, equity))
            if any(self.pos[label].units for label in self.labels):
                self.exposure_hits += 1

            if self._progress is not None and (i % tick == 0 or i == total - 1):
                self._progress((i + 1) / total)

        self._flatten_all(self.events[-1])
        return self._result()

    def _active_labels(self, event: pd.Timestamp) -> list[tuple[str, int]]:
        active: list[tuple[str, int]] = []
        for label in self.labels:
            row = self.row_at[label].get(event)
            if row is None:
                continue
            if row >= 1:
                self.mark[label] = self.col[label]["close"][row - 1]
            active.append((label, row))
        return active

    def _collect_decisions(
        self, event: pd.Timestamp, active: list[tuple[str, int]]
    ) -> dict[str, object]:
        if self.portfolio_strategy is not None:
            return self._portfolio_decisions(event, active)

        decided: dict[str, object] = {}
        for label, row in active:
            if row < 1 or row < self.warmup:
                continue
            strat = self.instrument_strategies[label]
            ctx = self._bar_context(label, event, row)
            if not self.started[label]:
                strat.on_start(ctx)
                self.started[label] = True
            decided[label] = strat.on_bar(ctx)
        return decided

    def _portfolio_decisions(
        self, event: pd.Timestamp, active: list[tuple[str, int]]
    ) -> dict[str, object]:
        histories = {
            label: self.frames[label].iloc[: self._cutoff(label, event)]
            for label in self.labels
            if self._cutoff(label, event) >= 1
        }
        if not histories or max(len(h) for h in histories.values()) < self.warmup:
            return {}
        ctx = PanelContext(
            now=event,
            horizon=self.cfg.horizon,
            histories=histories,
            active=frozenset(label for label, _ in active),
            portfolio=self._portfolio_view(),
            sessions={label: self.instruments[label].session for label in self.labels},
        )
        if not self._portfolio_started:
            self.portfolio_strategy.on_start(ctx)
            self._portfolio_started = True
        out = self.portfolio_strategy.on_bar(ctx) or {}
        active_labels = {label for label, _ in active}
        return {label: dec for label, dec in out.items() if label in active_labels}

    def _apply_decisions(
        self, event: pd.Timestamp, active: dict[str, int], decided: Mapping[str, object]
    ) -> None:
        to_size: list[str] = []
        for label, decision in decided.items():
            row = active[label]
            if not _tradable(self.col[label]["trading_state"][row]):
                continue  # not a bar you could have traded; the decision waits
            if isinstance(decision, Hold):
                continue
            if isinstance(decision, Orders):
                self._execute_orders(label, row, event, decision.orders)
                continue
            if isinstance(decision, Flat):
                self.conviction[label] = 0.0
                self.pending_bracket[label] = None
            elif isinstance(decision, Target):
                self.conviction[label] = decision.weight
                self.pending_bracket[label] = (
                    (decision.stop, decision.take, decision.max_bars) if decision.weight else None
                )
            else:  # pragma: no cover - defensive
                raise TypeError(f"unexpected decision {decision!r}")
            to_size.append(label)

        if not to_size:
            return
        weights = self.allocator.weights(self.conviction, self._allocator_context(event))
        equity = self._equity()
        for label in to_size:
            row = active[label]
            target = _size(
                weights.get(label, 0.0),
                equity,
                self.col[label]["open"][row],
                self.instruments[label].lot_size,
            )
            self._rebalance_to(label, row, event, target)

    # -------------------------------------------------------------- fills

    def _rebalance_to(self, label: str, row: int, event: pd.Timestamp, target: float) -> None:
        current = self.pos[label].units
        delta = target - current
        lot = self.instruments[label].lot_size
        if abs(delta) < lot * 0.5:
            return
        # A sign change (entering, exiting, flipping) always trades; otherwise the
        # move must clear the rebalance band, so a drifting target does not churn.
        sign_change = (target > 0) != (current > 0) or target == 0 or current == 0
        if not sign_change and abs(delta) < self.cfg.rebalance_band * max(
            abs(target), abs(current)
        ):
            return
        side = Side.BUY if delta > 0 else Side.SELL
        price = self.costs[label].fill_price(
            side, self.col[label]["open"][row], self._ref_bar(label, row)
        )
        self._fill(label, event, side, delta, price, "signal")

    def _execute_orders(
        self, label: str, row: int, event: pd.Timestamp, orders: tuple[Order, ...]
    ) -> None:
        for order in orders:
            delta = order.signed_units
            side = order.side
            price = self.costs[label].fill_price(
                side, self.col[label]["open"][row], self._ref_bar(label, row)
            )
            if order.stop or order.take or order.max_bars:
                self.pending_bracket[label] = (order.stop, order.take, order.max_bars)
            self._fill(label, event, side, delta, price, "orders")
        self.conviction[label] = (
            math.copysign(1.0, self.pos[label].units) if self.pos[label].units else 0.0
        )

    def _bracket_exits(self, label: str, row: int, event: pd.Timestamp) -> None:
        pos = self.pos[label]
        if pos.units == 0 or row < 1:
            return
        bar_open = self.col[label]["open"][row - 1]
        bar_high = self.col[label]["high"][row - 1]
        bar_low = self.col[label]["low"][row - 1]
        long = pos.units > 0

        level: float | None = None
        reason = ""
        # A bar that *opens* beyond a barrier (an overnight or weekend gap) fills
        # at that open, not at the barrier: the market never traded at the
        # barrier price. The open comes first in time, so it also decides which
        # barrier was hit first -- a gap through the take is a take, even if the
        # rest of the bar later falls through the stop.
        gapped_stop = pos.stop_price and (
            bar_open <= pos.stop_price if long else bar_open >= pos.stop_price
        )
        gapped_take = pos.take_price and (
            bar_open >= pos.take_price if long else bar_open <= pos.take_price
        )
        if gapped_stop:
            level, reason = bar_open, "stop"
        elif gapped_take:
            level, reason = bar_open, "take"
        elif long and pos.stop_price and bar_low <= pos.stop_price:
            level, reason = pos.stop_price, "stop"
        elif long and pos.take_price and bar_high >= pos.take_price:
            level, reason = pos.take_price, "take"
        elif not long and pos.stop_price and bar_high >= pos.stop_price:
            level, reason = pos.stop_price, "stop"
        elif not long and pos.take_price and bar_low <= pos.take_price:
            level, reason = pos.take_price, "take"

        if level is not None:
            self._exit_at(label, event, row, level, reason)
            return

        if pos.bars_remaining is not None:
            pos.bars_remaining -= 1
            if pos.bars_remaining <= 0:
                self._exit_at(label, event, row, self.col[label]["open"][row], "max_bars")

    def _exit_at(
        self, label: str, event: pd.Timestamp, row: int, level: float, reason: str
    ) -> None:
        pos = self.pos[label]
        side = Side.SELL if pos.units > 0 else Side.BUY
        price = self.costs[label].fill_price(side, level, self._ref_bar(label, row))
        self._fill(label, event, side, -pos.units, price, reason)
        self.conviction[label] = 0.0
        self.pending_bracket[label] = None

    def _fill(
        self,
        label: str,
        event: pd.Timestamp,
        side: Side,
        delta: float,
        price: float,
        reason: str,
    ) -> None:
        if delta == 0:
            return
        commission = self.costs[label].commission(delta, price)
        self.cash -= delta * price + commission
        self.traded_notional += abs(delta) * price

        pos = self.pos[label]
        old, new = pos.units, pos.units + delta
        self._account_fill(label, event, price, old, new, delta, commission, reason)

        pos.units = new
        if new == 0:
            pos.avg_price = pos.entry_price = pos.stop_price = pos.take_price = 0.0
            pos.bars_remaining = None
        elif old == 0 or (old > 0) != (new > 0):
            pos.avg_price = pos.entry_price = price
            self._arm_bracket(label, price)
        elif abs(new) > abs(old):
            pos.avg_price = (pos.avg_price * abs(old) + price * abs(delta)) / abs(new)

        self.fills.append(
            {
                "time": event,
                "label": label,
                "side": side.value,
                "units": delta,
                "price": price,
                "commission": commission,
                "cash_after": self.cash,
                "reason": reason,
            }
        )

    def _arm_bracket(self, label: str, entry_price: float) -> None:
        pos = self.pos[label]
        bracket = self.pending_bracket[label]
        if bracket is None:
            pos.stop_price = pos.take_price = 0.0
            pos.bars_remaining = None
            return
        stop, take, max_bars = bracket
        direction = 1.0 if pos.units > 0 else -1.0
        pos.stop_price = entry_price * (1.0 - direction * stop) if stop else 0.0
        pos.take_price = entry_price * (1.0 + direction * take) if take else 0.0
        pos.bars_remaining = max_bars

    def _account_fill(
        self,
        label: str,
        event: pd.Timestamp,
        price: float,
        old: float,
        new: float,
        delta: float,
        commission: float,
        reason: str,
    ) -> None:
        """Fold one fill into the round-trip blotter.

        Every fill either opens, adds to, reduces, or flips a position. A reduce
        (partial or full) books a closed-trade row for the quantity it removed,
        drawing down its share of the entry commission so the books reconcile:
        ``sum(trade.pnl) == final_equity - starting_equity``.
        """
        if old == 0:
            self.open_trade[label] = _OpenTrade(event, price, _sign(new), abs(new), commission)
            return

        trade = self.open_trade[label]
        if trade is None:  # pragma: no cover - defensive
            self.open_trade[label] = _OpenTrade(event, price, _sign(new), abs(new), commission)
            return

        same_sign = (old > 0) == (new > 0)
        if same_sign and abs(new) >= abs(old):
            added = abs(delta)
            trade.entry_price = (trade.entry_price * abs(old) + price * added) / abs(new)
            trade.units = abs(new)
            trade.costs += commission
            return

        reduce_qty = min(abs(old), abs(delta))
        cost_share = trade.costs * (reduce_qty / trade.units) if trade.units else 0.0
        trade.costs -= cost_share
        self._book_close(label, event, price, reduce_qty, cost_share + commission, reason)

        if reduce_qty < abs(old):
            trade.units = abs(old) - reduce_qty  # entry price unchanged by a reduce
        else:
            self.open_trade[label] = (
                _OpenTrade(event, price, _sign(new), abs(new), 0.0) if new != 0 else None
            )

    def _book_close(
        self,
        label: str,
        event: pd.Timestamp,
        exit_price: float,
        qty: float,
        costs: float,
        reason: str,
    ) -> None:
        trade = self.open_trade[label]
        gross = trade.direction * (exit_price - trade.entry_price) * qty
        notional = trade.entry_price * qty
        self.trades.append(
            {
                "label": label,
                "direction": trade.direction,
                "units": qty,
                "entry_time": trade.entry_time,
                "exit_time": event,
                "entry_price": trade.entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross,
                "costs": costs,
                "pnl": gross - costs,
                "return": (gross - costs) / notional if notional else 0.0,
                "bars_held": int((event - trade.entry_time) / self.step),
                "exit_reason": reason,
            }
        )

    def _flatten_all(self, event: pd.Timestamp) -> None:
        for label in self.labels:
            pos = self.pos[label]
            if pos.units == 0:
                continue
            mark = self.mark[label]
            if math.isnan(mark):
                mark = pos.avg_price
            side = Side.SELL if pos.units > 0 else Side.BUY
            price = self.costs[label].fill_price(side, mark, self._ref_bar_mark(label))
            self._fill(label, event, side, -pos.units, price, "end")
        if self.equity_points:
            self.equity_points[-1] = (self.equity_points[-1][0], self._equity())

    # ------------------------------------------------------------- helpers

    def _cutoff(self, label: str, event: pd.Timestamp) -> int:
        """Number of bars for ``label`` that have fully closed as of ``event``."""
        return bisect_left(self.times[label], event)

    def _bar_context(self, label: str, event: pd.Timestamp, row: int) -> BarContext:
        return BarContext(
            label=label,
            now=event,
            horizon=self.cfg.horizon,
            history=self.frames[label].iloc[:row],
            position=Position(label, self.pos[label].units, self.pos[label].avg_price),
            portfolio=self._portfolio_view(),
            session=self.instruments[label].session,
        )

    def _portfolio_view(self) -> PortfolioView:
        return PortfolioView(
            cash=self.cash,
            equity=self._equity(),
            positions={
                label: Position(label, self.pos[label].units, self.pos[label].avg_price)
                for label in self.labels
            },
        )

    def _allocator_context(self, event: pd.Timestamp) -> AllocatorContext:
        return AllocatorContext(
            histories={
                label: self.frames[label].iloc[: self._cutoff(label, event)]
                for label in self.labels
            },
            portfolio=self._portfolio_view(),
            leverage_cap=self.cfg.leverage_cap,
        )

    def _equity(self) -> float:
        total = self.cash
        for label in self.labels:
            units = self.pos[label].units
            if units:
                mark = self.mark[label]
                total += units * (mark if not math.isnan(mark) else self.pos[label].avg_price)
        return total

    def _ref_bar(self, label: str, row: int) -> dict[str, float]:
        """The last completed bar's close/ask -- the causal spread reference."""
        idx = max(row - 1, 0)
        return {
            "close": self.col[label]["close"][idx],
            "close_ask": self.col[label]["close_ask"][idx],
        }

    def _ref_bar_mark(self, label: str) -> dict[str, float]:
        return {"close": self.mark[label], "close_ask": math.nan}

    def _result(self) -> BacktestResult:
        equity = pd.Series(
            [value for _, value in self.equity_points],
            index=pd.DatetimeIndex([ts for ts, _ in self.equity_points], name="time"),
            name="equity",
            dtype="float64",
        )
        if equity.empty:
            equity = pd.Series(
                [self.cfg.starting_cash, self.cfg.starting_cash],
                index=pd.DatetimeIndex([self.events[0], self.events[-1]], name="time"),
                name="equity",
            )
        fills = pd.DataFrame(self.fills, columns=_FILL_COLUMNS)
        trades = pd.DataFrame(self.trades, columns=_TRADE_COLUMNS)

        sessions = [
            self.instruments[label].session
            for label in self.labels
            if self.instruments[label].session is not None
        ]
        session_minutes = _session_minutes(sessions[0]) if sessions else None
        ppy = periods_per_year(self.cfg.horizon, session_minutes)

        n_events = max(len(self.events), 1)
        metrics = compute(
            equity,
            trades,
            periods_per_year=ppy,
            exposure=self.exposure_hits / n_events,
            turnover=self.traded_notional / self.cfg.starting_cash
            if self.cfg.starting_cash
            else float("nan"),
        )
        by_instrument = {
            label: instrument_stats(trades, label, self.cfg.starting_cash) for label in self.labels
        }
        config = {
            "labels": list(self.labels),
            "horizon": self.cfg.horizon,
            "starting_cash": self.cfg.starting_cash,
            "leverage_cap": self.cfg.leverage_cap,
            "warmup": self.warmup,
            "allocator": type(self.allocator).__name__,
            "periods_per_year": ppy,
            "events": len(self.events),
        }
        return BacktestResult(
            equity=equity,
            fills=fills,
            trades=trades,
            metrics=metrics,
            by_instrument=by_instrument,
            config=config,
        )


_FILL_COLUMNS = ("time", "label", "side", "units", "price", "commission", "cash_after", "reason")
_TRADE_COLUMNS = (
    "label",
    "direction",
    "units",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "gross_pnl",
    "costs",
    "pnl",
    "return",
    "bars_held",
    "exit_reason",
)


def _tradable(state: object) -> bool:
    """True unless the bar carries a venue state that is not continuous trading."""
    if state is None or state is pd.NA:
        return True
    if isinstance(state, float) and math.isnan(state):
        return True
    return str(state) in TRADABLE_STATES


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _size(weight: float, equity: float, price: float, lot: float) -> float:
    """Whole lot-rounded units for ``weight`` of ``equity`` at ``price``."""
    if weight == 0.0 or price <= 0 or equity <= 0:
        return 0.0
    lots = math.floor(abs(weight) * equity / price / lot)
    return math.copysign(lots * lot, weight)


def _session_minutes(hours: RegularHours) -> float | None:
    open_m = hours.open_time.hour * 60 + hours.open_time.minute
    close_m = hours.close_time.hour * 60 + hours.close_time.minute
    span = close_m - open_m
    if span <= 0:
        span += 1440
    return float(span)


__all__ = ["Instrument", "run", "Metrics", "BacktestResult"]
