# Backtest engine: the timing model and what it assumes

Phase 2 is the `Strategy` interface (`trader.strategies`) and the event-driven
engine that runs it (`trader.backtest`). This is the record of the decisions
baked into the engine, so they are not re-litigated by accident later — and so
the numbers it prints are read with the right caveats.

## The one hard rule: no lookahead

Saxo bars are timestamped at their **opening** instant. A bar `B` opening at
`t` covers `[t, t + horizon)` and is only fully known at `t + horizon`.

The engine walks the union of every instrument's bar timestamps. At event `e`:

1. the bar that opened at `e - horizon` has just **closed** — it is the newest
   bar the strategy may see;
2. the strategy's decision is filled at the **open of the bar at `e`**, a price
   that is knowable at exactly the instant the decision is made.

So "signal on the close, fill at the next open", stated for open-stamped bars.
`ctx.history` in a `BarContext` ends at the just-closed bar; the bar being
traded into is not in it and cannot be read. The last bar of a series is never
shown to a strategy (it never closes within the run) — the engine flattens into
its close instead, as an `exit_reason == "end"` trade.

## Bracket exits

A `Target(weight, stop, take, max_bars)` attaches a bracket, with `stop` and
`take` as **fractions of the entry price** and `max_bars` a bar count — the
three barriers of Phase 4's triple-barrier labelling (`docs/labels.md`), so the
parameterisation is shared on purpose.

Brackets are checked intrabar against each completed bar's `high`/`low`:

- **stop before take.** If a single bar's range covers both levels, the engine
  takes the stop. Real fills are path-dependent and unknowable from OHLC; the
  conservative assumption is that the adverse level came first. This biases
  bracket strategies' results *down*, which is the safe direction.
- **`max_bars`** counts completed bars since entry and exits at the next bar's
  open when it reaches zero.
- Every bracket exit fills through the cost model, like any other fill — the
  barrier price is the reference, not the guaranteed fill.

## Fills, sizing, and allocation

- `Target.weight` is a **conviction in `[-1, 1]`**, not a position. An
  `Allocator` turns the set of per-instrument convictions into portfolio-equity
  weights under a gross-exposure (`leverage_cap`) ceiling. `PassThrough` is the
  single-instrument default; `EqualWeight` the multi-instrument one.
- A weight is sized to **whole lot-rounded units** at the next bar's open,
  against *current* portfolio equity (`cash + Σ units · last close`).
- `rebalance_band` (default 0.25) suppresses a fill unless the new target is at
  least that far from the current position, or the sign changed. Without it,
  equity and price drift churn a constant conviction every bar.
- `Orders([...])` bypasses the allocator and the equity sizing entirely — the
  units are taken as given.
- A bar whose `trading_state` is set and is not one of
  `trader.saxo.charts.TRADABLE_STATES` cannot be filled; the decision carries to
  the next tradable bar.

## Costs

`CostModel` is commission + spread + slippage, all additive, all pushing the
fill against the trade (a buy fills higher, a sell lower):

- **spread** is half the bid/ask, taken from the bar's `close_ask` when it has
  one (quote-driven instruments, FX) and from `half_spread_bps` otherwise. The
  spread reference is always the **last completed bar**, never the one being
  entered on — using the entry bar's own ask would be lookahead.
- **commission** is `commission_bps` of notional plus `commission_per_unit`,
  floored at `commission_min` per fill.

## Metrics and annualisation

`periods_per_year` is derived from the horizon and, when an inferred session
calendar is available, the session length: a 6.5-hour equity session at
5-minute bars is 78 bars/day, not `1440 / 5`. The trading year is assumed to be
**252 days**. Ratios use a zero risk-free rate and sample standard deviation.

CAGR is only reported once the run spans at least a week of bars — annualising a
return measured over a few days produces a number with no meaning.

Per-instrument attribution (`result.by_instrument`) is trade-based and
reconciles: `Σ by_instrument.pnl == final_equity − starting_equity`, because
every position is flat at the end and every commission lands in exactly one
trade row.

## Known limitations (revisit before Phase 3+ leans on this)

- **One horizon per run.** Every series in a `panel` must share a bar size.
  Multi-timeframe inputs (what the LSTM wants) are resampled/aligned upstream,
  not by the engine.
- **`history` is sliced per bar.** `frame.iloc[:i]` each event is fine for
  5-minute runs over months; a multi-year 1-minute run makes this the
  bottleneck — including the classical walk-forward sweeps in
  `docs/benchmarks.md`, which run one `bt.run` per fold per config. A rolling
  view is the fix when that run matters.
- **No borrow cost, no financing, no dividends.** Shorts and overnight holds are
  free. Fine for intraday research; a cost-model extension before anything
  holds overnight for real.
- **Fills are next-open only.** No limit orders, no partial fills, no queue
  position. `Orders` is the escape hatch for anything more specific.
