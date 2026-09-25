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
they fall back to UTC calendar days. Note that a run driven by `--uic` skips the
exchange lookup, so it gets no calendar and these columns use UTC.

`structure_v1` (`trader.features.structure`) — `mtf_v1` plus ~58 session and
price-structure columns, all from OHLC (no volume needed, so FX works):

- sessions: `sess_asia` / `sess_london` / `sess_ny` / `sess_overlap` flags,
  `london_elapsed`, `ny_elapsed` (from the 09:30 cash open), `fix_hours` (to the
  London 16:00 fix), `dow_sin` / `dow_cos`
- Asian range (roll → London open) and New York opening range (09:30–10:00),
  running while they form and held after: `*_hi_dist`, `*_lo_dist`,
  `*_range_atr`, and an `asia_ok` / `or_ok` flag
- levels: prior day high / low / close and range, prior week high / low, the
  day's running high / low, floor pivots (P, R1, S1), and two round-number grids
- per swing order `k` (default 3 and 12): distance to and age of the last
  confirmed swing high / low, `hh` / `hl` (higher high / higher low), and the
  last leg's direction, size, Fibonacci retracement (`fib_retr`), signed
  distance to the 0.382 / 0.5 / 0.618 / 0.786 levels, and the nearest of them

Its clocks are fixed market clocks, not the inferred calendar: `America/New_York`
and `Europe/London`, DST-aware, with the **trading day rolling at 17:00 New
York** (Sunday evening is Monday). Distances are in **ATR units**, clipped to
±25, so one model reads EURUSD and BTC on the same scale. A column that cannot
exist for an instrument (the Asian range of an ETF that only trades US hours) is
0 with its `*_ok` flag at 0 — never NaN, which would drop every training row.
Levels that only need more history (prior week, a first swing) are NaN until
it exists, like warmup.

**Swing points are the lookahead trap.** A high at bar `j` is a swing high only
once `k` later bars have failed to beat it, so it is recorded at bar `j + k`.
The causality test covers the whole set, and a deliberately forward-looking
swing window makes it fail.

The spec carries `swing_orders` and `level_days` (15). `level_days` is a
wall-clock span, not a bar count, because a day is 96 bars of 15m FX but 26 of
an ETF. So it is not folded into `warmup`: `ModelStrategy` extends its recompute
tail to cover it, which keeps live prior-week levels identical to training. Both
fields are omitted from `to_dict` at their defaults, so older specs keep their
digests.

`regime_v1` (`trader.features.regime`) — `structure_v1` plus 27 volatility,
trend and gap columns, again OHLC only:

- volatility regime: `pk_vol_20` / `gk_vol_20` / `rs_vol_20` (Parkinson,
  Garman-Klass, Rogers-Satchell: whole-candle estimators, more accurate than
  close-to-close per bar), `gk_ratio` (20-bar vs 100-bar Garman-Klass, above 1 =
  expanding), and from completed trading days `dvol_20` (20-day close-to-close
  vol, gaps included), `dvol_ratio` (5-day / 20-day), `dvol_pct` (where
  `dvol_20` sits in its last 60 days, 0..1), `dvolvol` (how unstable daily
  ranges are)
- trend quality: `mom_1` / `mom_5` / `mom_20` (log return vs the close 1 / 5 /
  20 trading days back, divided by `dvol_20 * sqrt(k)`), `dma20_dist` (vs the
  20-day average close, in daily-vol units), `er_20` / `er_60` (efficiency
  ratio: net move / path length, 1 = straight line), `adx` / `di_diff` (Wilder
  ADX strength and direction, scaled to 0..1), `r2_60` / `slope_60` (fit and
  size in ATRs of a 60-bar straight line), `ac1_60` (lag-1 return
  autocorrelation: positive = trending, negative = snapping back),
  `ema50_dist` / `ema200_dist` / `ema50_slope` (ATR units), `upfrac_20`
- gaps: `gap_z` (the trading day's open vs the prior close, in daily-vol units),
  `gap_fill` (0..1 share of the gap retraced so far, 0 when there is no real
  gap), `gap_trend` (+1 gap with the prior 5-day trend, −1 against, 0 small
  gap), `day_ret_z` (close vs the day's open)

Daily columns use only **completed** trading days (shifted one day), joined back
onto every bar of the current day. Built from whole trading days, they read the
same on 24-hour FX and a 6.5-hour ETF; the bar-count columns (`*_20`, `*_60`)
do not. The daily columns need ~80 trading days, so `level_days` is 120: a
series loses its first ~4 months to NaN, and the live recompute tail spans 120
days. On FX the gap columns stay near 0 (only the Sunday open gaps).

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
