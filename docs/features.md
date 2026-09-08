# Feature pipeline: the causality rule and the multi-timeframe join

Phase 4a is `trader.features`: a registry of named feature sets and one entry
point, `compute_feature_frame`, that both training and the live `lstm` strategy
call. This is the record of the decisions baked into it.

## The one hard rule: row `i` sees bars `0..i` and nothing after

Every column is computed with backward-only pandas operations — `.rolling`,
`.ewm`, `.expanding`, `.diff`, `.shift(k)` with `k >= 0`. No `center=True`, no
negative shift. Feature row `i` is therefore a pure function of input bars
`0..i`, and is "known at the close of base bar `i`" — the same instant the
backtest engine hands `history.iloc[:i+1]` to a strategy.

The first `spec.warmup` rows come out as NaN and are **kept** that way. They are
dropped only at the dataset boundary (`trader.models.dataset.build_bundle`) or
ignored by the strategy until `ctx.bars_seen >= warmup`. Dropping them earlier
would just make the causality test harder to state.

`warmup` is a deliberately loose upper bound (EWM columns never fully shed their
seed). For a single horizon it is roughly `3 x max(rsi, macd_slow, atr)`; adding
a context horizon multiplies that by the horizon ratio, so `mtf_v1` with a
1-hour context on 5-minute bars reports a warmup near 950 bars. That is honest —
a 1-hour MACD genuinely needs that much 5-minute history — and irrelevant in
practice because training runs span months.

## The feature sets

`price_v1` — single timeframe, 17 columns:

- returns: `ret_1`, `ret_5`, `ret_15` (log returns over 1 / 5 / 15 bars)
- volatility: `vol_20`, `vol_60` (rolling stdev of 1-bar log returns), `atr_pct`
  (ATR / close), `range_pct` ((high − low) / close)
- momentum: `rsi`, `macd`, `macd_signal`, `macd_hist`
- flow: `volz` (volume z-score)
- session: `vwap_dist` (close / session-cumulative VWAP − 1, reset each local
  trading day), `ovn_gap` (first bar of a local day vs the previous close, else
  0), `tod_sin` / `tod_cos` (time-of-day on a 24-hour circle), `session_phase`
  (fraction elapsed through the regular session, or of the day when no session
  calendar is available)

`mtf_v1` — `price_v1` on the base horizon plus, for each context horizon `h`,
the columns `h{h}_ret_1`, `h{h}_rsi`, `h{h}_vol`, `h{h}_macd_hist`,
`h{h}_atr_pct`.

The session-relative columns take their timezone and day boundaries from the
`RegularHours` calendar inferred in `trader.data.calendars`; with no calendar
they fall back to UTC calendar days.

## Multi-timeframe: the as-of join

Context bars are built by `trader.data.bars.resample` (whole-multiple horizons
only) — never fetched separately, so the base and context series are guaranteed
to align on the same grid.

A context bar opening at `T` covers `[T, T + h)` and is only known once it
closes at `T + h`. A base row opening at `s` is itself only known at
`s + base_horizon`. So `attach_context` gives each base row the newest context
row whose close is **strictly before** the base row's close
(`np.searchsorted(..., side="left") - 1` on the context close instants). Strict,
not `<=`: a context bar that closes at the exact instant a base bar closes costs
one extra base bar of latency rather than risking a simultaneous-close leak.
This is the same conservative bias as the engine's "stop wins ties".

The causality tests (`tests/features/test_causality.py`) enforce all of the
above: perturbing any bar after a cut — including a future context bar — must
leave every feature row at or before the cut byte-identical.

## Persistence

Default: nothing is persisted. `compute_feature_frame` recomputes every time —
one code path, no staleness, trivially testable. `FeatureBuildSpec.persist=True`
writes to `trader.features.cache.FeatureLake`
(`data/features/{asset_type}/{uic}/{base_horizon}/{set}@{digest}/`), keyed by the
spec digest so a changed spec can never read a stale file. Turn it on only if a
training loop profiles feature computation as its bottleneck.

## Adding a feature set

Subclass `FeatureSet`, implement `resolve` (return a frozen `FeatureSpec` with
`columns` filled in) and `compute` (return a frame indexed by `time`, columns
exactly `spec.columns`, warmup rows NaN), and decorate with
`@register_feature_set("name")`. Import it from `trader/features/__init__.py` for
the registration side effect. A new set does not invalidate existing models —
each model stores its own frozen `FeatureSpec` and reloads it for inference.
