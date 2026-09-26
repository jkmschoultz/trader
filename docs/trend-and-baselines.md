# Daily trend-following, no-skill baselines, and Coinbase crypto data

After intraday models on EURUSD and IBIT found no direction signal that survived
walk-forward testing, the next step was a documented, low-turnover effect
(time-series momentum) measured against baselines that have no skill at all.
This records the pieces that were added and what the first backtests showed.

## Crypto data from Coinbase (`trader data crypto`)

Saxo SIM has no spot BTC or ETH, and the ETF proxies only start in 2024.
`trader.data.coinbase` pulls public Coinbase candles (no API key) into the same
lake:

```bash
trader data crypto BTC-USD ETH-USD --horizon 1d,1h   # full history, then resumes
```

- Stored under asset type `Crypto`, keyed by a **pseudo-Uic**
  (`pseudo_uic("BTC-USD")`, a stable hash), with the product id in
  `instruments.json`. `load_panel` resolves `--symbol BTC-USD --asset-type
  Crypto` from that registry, so no Saxo login is needed. `trader data symbols`
  skips these series.
- Coinbase serves 1m / 5m / 15m / 1h / 6h / 1d candles. Daily candles are UTC
  days, 7 per week.
- **No bid/ask** (`close_ask` is NaN), so a backtest needs an explicit cost
  assumption: `--spread-bps` (half-spread) and `--fee-bps`.
- **Fractional sizing.** The engine sizes positions in whole lots. With the
  default lot of 1, a $50k allocation to $60k bitcoin rounds to zero.
  `PanelSeries.lot_size` is therefore `coinbase.LOT_SIZE` (1e-8, one satoshi)
  for Crypto, and 1 elsewhere.

## `tsmom` — time-series momentum

Long an instrument whose price rose over the lookback, short one that fell
(Moskowitz, Ooi & Pedersen 2012). Default lookbacks `21,63,126,252` bars (about
1, 3, 6 and 12 months of trading days); the conviction is the mean of their
signs. It rebalances every `rebalance` bars (default 5, weekly on daily bars)
and holds in between. Options: `skip`, `long_only`. With `equal-weight` only
the sign is used.

## `baseline` — what a result has to beat

- `direction=random`: a coin flip per decision. The flip is a hash of
  `(seed, instrument, bar time)`, so runs are reproducible. Use several seeds to
  see the spread luck produces.
- `direction=long` / `short`: always one side. Use `hold_bars` larger than the
  data for buy-and-hold.
- **Scheduled mode** (no bracket): re-draw every `hold_bars` bars. This is the
  `tsmom` baseline.
- **Bracketed mode** (`stop` / `take` / `max_bars`, optional
  `barrier_scale=atr`): enter whenever flat, with probability `entry_prob`,
  using the same bracket a model is trained on. This is the model baseline.

## First results (tuning lake, data before 2026-06-01, daily bars)

> The crypto numbers in this section were computed before the metrics switched to
> calendar-time annualisation. That undercounted crypto's 365-day year, so CAGR
> and Sharpe read ~20% low. Rankings are unaffected; the corrected figures are in
> the next section.

**FX**, 7 USD majors, 2000-01 to 2026-05, equal-weight, spread from the stored
ask plus 0.2 bp slippage:

| | CAGR | Sharpe | Max DD |
|---|---|---|---|
| tsmom | −0.3% | −0.02 | −29% |
| buy-and-hold all 7 | +0.1% | 0.04 | −21% |
| random, 10 seeds | −0.6% … −2.7% | −0.15 … −0.74 | −29% … −54% |

Daily FX half-spreads averaged ~4.75 bp over 2000–2026: 6–11 bp in the 2000s,
1–4 bp in the 2020s. Adding costs back (turnover × average cost) gives tsmom
about +0.7%/yr before costs and random about −0.2%/yr. So there is a small
trend edge in FX, but it is smaller than the spread. Carry (overnight
financing) is not modelled.

**Crypto**, BTC + ETH, 2017-01 to 2026-05, equal-weight, 5 bp half-spread +
10 bp fee per side:

| | CAGR | Sharpe | Max DD |
|---|---|---|---|
| tsmom | +24.6% | 0.67 | −71% |
| tsmom long-only | +30.2% | 0.82 | −63% |
| buy-and-hold | +45.1% | 0.88 | −93% |
| random, 10 seeds | −7.8% … −19.7% | −0.27 … 0.05 | −84% … −99% |
| BTC alone: tsmom long-only | +29.4% | 0.86 | −62% |
| BTC alone: buy-and-hold | +37.1% | 0.84 | −84% |

Momentum beats every random seed by a wide margin, so the trend information in
crypto is real. Against buy-and-hold over this bull-market sample, long-only
momentum gives about the same Sharpe with a much smaller worst drawdown. The
short side loses money in a market that rose ~40x.

**Reading turnover:** the engine reports traded notional over *starting* equity,
so it inflates for fast-compounding assets. BTC buy-and-hold shows 75x from one
round trip. Compare crypto turnover only between runs over the same period.

## Volatility targeting (`--allocator vol-target`)

`VolTarget` sizes each position at `conviction × target_ann_vol / σ / n_active`.
σ is the last `lookback` bars' annualised volatility, with bars per year counted
from the timestamps: 365 for crypto, ~260 for weekday FX. Each name is capped
at `max_weight` (default 1), then gross exposure at the leverage cap. Set it from
the CLI with `--allocator-param target_ann_vol=0.3 --allocator-param lookback=30`.
The backtest, the classical walk-forward and the model walk-forward all accept
`allocator_params`.

Full-period backtests, BTC + ETH, 2017-01 to 2026-05, calendar annualisation:

| | CAGR | Sharpe | Max DD |
|---|---|---|---|
| buy-and-hold, equal-weight | +71% | 1.06 | −93% |
| buy-and-hold, vol-target 30% | +35% | 1.04 | −57% |
| tsmom long-only, equal-weight | +47% | 0.99 | −63% |
| **tsmom long-only, vol-target 30%** | **+22%** | **1.16** | **−23%** |
| tsmom long-only, vol-target 50% | +33% | 1.12 | −36% |
| tsmom long/short, vol-target 30% | +24% | 1.07 | −34% |

FX tsmom with a 10% target: Sharpe 0.08. Still no edge.

## Robustness: walk-forward over settings (crypto)

`state/research/crypto_tsmom_wf.json`: long-only tsmom, vol-target 30%, eight
rolling one-year test windows (2018 to 2025). The grid covered lookbacks
`21/63/126/252, 21, 63, 126, 252, 21/63, 126/252` and rebalance `1, 5, 21` bars.

- All 21 configs had a positive median yearly Sharpe (0.56 to 1.29).
- Under the same windows, vol-targeted buy-and-hold had 0.43
  (`crypto_bh_vt_wf.json`). Ten random long/flat baselines had 0.05 to 0.37
  (`crypto_random_lf_wf.json`). The *worst* tsmom config beat the *best* random
  seed.
- The default (`21/63/126/252`, weekly) made money in all 8 windows, with a worst
  in-window drawdown of −22%. Most of its edge comes from stepping aside in crash
  years: 2018 +36% vs −9%, and 2022 +4% vs −30% for vol-targeted holding. It lags
  in the strongest bull years (2021 +165% vs +237%).
- Picking the top config from this sweep (21-day, weekly) would be fitting to
  the sample. The default blend was fixed before any results were seen, so it is
  the one to carry forward.

## Holdout sanity check (Jun–Sep 2026)

The default long-only tsmom with vol-target 30% returned −1.5% on the full lake
from 2026-06-01 to 2026-09-25. Buy-and-hold returned +24%, and vol-targeted
holding +16%. This is the strategy working as designed. BTC fell from $119k
(Oct 2025) to $65k (mid-Jul 2026), so the signal stayed negative and the
strategy stayed out through the June–July low. It re-entered at half conviction
in late August, after a ~30% bounce. That late re-entry is the known cost of
trend-following. Four months cannot judge the strategy either way; the check
was for bugs, and none showed up.

## Meta-labelling: a model filter on top of tsmom (`--primary tsmom`)

`trader.labels.primary` computes the tsmom gate (conviction > 0) from completed
UTC days built out of the bars themselves, so it works on any horizon.
`primary_gate` (vectorised) and `gate_now` / `LiveGate` (at decision time)
agree bar for bar (`tests/labels/test_primary.py`).

- `trader train|tune --primary tsmom [--primary-lookbacks 21/63/126/252]`
  keeps only the samples where the gate is long. The manifest records
  `"primary"`, which also enters the model id.
- A primary model trades only while the gate is long. A "down" call means flat,
  never short.
- `baseline --param primary=tsmom` applies the same gate with no model.
- `Strategy.history_days` lets a strategy ask walk-forward folds for calendar
  days of history beyond its bar warmup. The gate needs about a year of daily
  closes. Without this, a gated strategy read "off" for most of each fold.
- Walk-forward folds for a primary model are laid over every bar, not over the
  gated samples, so they line up with classical folds over the same dates.
- `trader tune --no-bracket` holds a model's positions until the signal
  changes, instead of attaching the trained exits.

**Result (tuning lake): the filter does not help.** BTC + ETH hourly bars,
`regime_v1` + 4h context, XGB window 4. Six anchored folds with one-year test
windows (mid-2019 to mid-2025). Vol-target 30%; 5 bp + 10 bp costs per side.

| | median fold Sharpe |
|---|---|
| daily tsmom long-only, weekly rebalance | **+1.02** |
| gate alone, hourly bars, rebalanced daily | +0.69 |
| model as a hold/skip filter, 24h labels, no exits, threshold 0.3 | +0.35 |
| model as a hold/skip filter, threshold 0.1 | −0.57 |
| gate + 6-ATR / 168h exits, no model | +0.02 |
| gate + exits + model filter, threshold 0.2 | −0.33 |
| gate + exits + model filter, threshold 0 | −1.61 |

Reports are in `state/research/crypto_{meta_xgb,meta_xgb_hold,gate_bracket,gate_hold,tsmom_anchored}_wf.json`.
Stop/take exits cut the trends momentum profits from. The model picks entries
no better than taking all of them, and as a hold/skip filter it trades twice as
much as the gate alone, which costs more than any skill it adds. Plain daily
tsmom remains the best configuration found.

## A wider crypto basket (22 coins)

`trader data crypto` now also holds LTC, BCH, ETC, XLM, LINK, SOL, ADA, DOGE,
AVAX, DOT, XRP, ATOM, ZEC, XTZ, ALGO, UNI, AAVE and FIL. It also holds the
delisted EOS (to 2025-12) and MATIC (to 2025-10, renamed POL). Faded and delisted
coins are included on purpose: a basket of today's winners would carry
survivorship bias. An unknown product is skipped with a message.

**Untradable gaps.** Coinbase suspended XRP from 2021-01 to 2023-07. Momentum
was long when trading stopped, so a backtest would have held through the hole
and booked a +170% jump on relisting. `load_panel` now uses only the bars
after a series' last gap of more than 30 days (`bars.after_last_long_gap`),
with a warning, so the strategy warms up afresh. A delisted series simply ends,
and the engine holds any open position at the last price until the run ends,
which amounts to selling at the last close.

Walk-forward over the same 8 yearly windows (long-only tsmom, weekly,
vol-target 30%):

| | median Sharpe | years > 0.5 | worst in-year DD |
|---|---|---|---|
| tsmom, BTC + ETH | 0.99 | 5/8 | −22% |
| tsmom, 22 coins | 0.94 | 6/8 | −26% |
| buy-and-hold vol-target, 22 coins | 0.35 | 3/8 | −49% |
| random long/flat, 22 coins, 5 seeds | 0.14 … 0.23 | 2–3/8 | |

More coins barely change the result. Crypto is close to one asset: the average
pairwise daily-return correlation since 2022 is 0.61, with most coins at
0.65–0.84 to BTC. Real diversification for momentum has to come from other
asset classes (indices, commodities, bonds).

## Other asset classes (Saxo daily bars)

Backfilled daily bars: index CFDs US500.I (from 1984), USNAS100.I, GER40.I,
FRA40.I; spot gold and silver (XAUUSD, XAGUSD, from 2002); and ETFs SPY, QQQ,
EFA, EEM, EWJ, TLT, IEF, GLD, SLV, USO, VNQ. Saxo lists bonds and commodities
only as single futures contracts; stitching those into continuous series is
left for later, so ETFs stand in.

**Data caveats.** ETF prices are *not* dividend-adjusted (SPY's first close is
the as-traded $43.94). Long-only results therefore miss income: ~3%/yr for TLT,
~4% for VNQ. Some old splits are unadjusted: EFA and EEM in 2005, SLV in 2008.
Multi-asset runs start in 2009 to stay clear of them.

**Mixed baskets.** `backtest` / `tune` now take a repeatable `--asset-type`.
Once means every symbol; repeated, it pairs with `--symbol` in order, like
`--uic`. `load_panel(asset_type=[...])` does the same.

Walk-forward over 16 yearly windows (2010–2025), vol-target 30%, costs
5 bp + 10 bp per side on everything (conservative for ETFs and FX). Basket
"traditional" = US500.I, GER40.I, EWJ, EEM, TLT, XAUUSD, XAGUSD, USO, VNQ.

| | median Sharpe |
|---|---|
| traditional buy-and-hold, vol-targeted | **0.49** |
| traditional tsmom long-only, no added costs | 0.38 |
| traditional tsmom long-only | 0.15 |
| traditional tsmom long/short | −0.04 |
| traditional random long/flat, 3 seeds | −0.12 … 0.02 |
| traditional + 7 FX, tsmom long-only | 0.02 |
| traditional + BTC/ETH, tsmom long-only | 0.45 |

Trend-following on traditional assets beat random but not holding. This is
the known 2010s "trend drought" (steady rises reward holding), and costs take
about half of its edge. Putting crypto *into* the same vol-target basket
dilutes it: equal risk shares give crypto ~4% positions, and the book becomes
mostly traditional momentum.

**Sleeves work better** (full-period daily curves, 2018 to 2026-05; computed
outside the engine by mixing the two equity curves, rebalanced daily):

| | CAGR | vol | Sharpe | max DD |
|---|---|---|---|---|
| crypto tsmom (BTC+ETH, long-only, vt 30%) | +19.4% | 19.0% | 1.03 | −23% |
| traditional buy-and-hold, vt | +5.9% | 10.2% | 0.62 | −23% |
| 50% crypto tsmom / 50% traditional B&H | +13.1% | 11.3% | 1.15 | −18% |

The sleeves' daily returns correlate at 0.11, so a split portfolio is smoother
than either sleeve. The engine has no per-sleeve allocation yet.
