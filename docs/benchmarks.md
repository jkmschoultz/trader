# Strategy benchmarks: one walk-forward leaderboard

Every strategy type — classical (`ma_cross`, `orb`) and model-backed (`lstm`,
`gbm`) — is scored the same way, so their numbers are directly comparable:

- **Same folds.** `walk_forward_splits` over the panel's bar timestamps:
  `train_days` → `val_days` → `test_days`, origin advancing by `step_days`
  (default `test_days`, non-overlapping test windows).
- **Same held-out backtest.** Each fold's `[val_end − lookback, test_end]` window
  runs through `bt.run` with one `CostModel(commission_bps=fee_bps,
  half_spread_bps=spread_bps, slippage_bps=slippage_bps)`, one allocator, one
  `leverage_cap`. For a classical strategy the train/val span is only warm-up.
- **Same score.** Median out-of-sample Sharpe across folds; tie-break lower
  median turnover. The report also carries `folds_sharpe_gt_0_5`,
  `worst_fold_sharpe`, and median/mean/pstdev of return, turnover, hit rate and
  profit factor.

Model sweeps go through `trader.service.training.run_cv` (train per fold);
classical sweeps through `trader.service.evaluation.run_strategy_cv` (no
training). Both are driven by `trader tune --strategy <name> …` /
`POST /api/tuning`, and `score_fold` / `aggregate_folds` are shared, so the
`{folds, aggregate}` shape and the ranking are identical.

## Running a benchmark

Pick the config on the deep AAPL 1-minute history first, then re-confirm on the
full book (AAPL + NVDA + US500) once those series are backfilled at `1m`.

```bash
# classical: opening-range breakout
trader tune --strategy orb --symbol AAPL --asset-type CfdOnStock --horizon 1m \
            --folds 4 --train-days 30 --val-days 7 --test-days 7 \
            --fee-bps 0.5 --spread-bps 1.0 --slippage-bps 0.5 --leverage 1.0 \
            --grid open_minutes=5,15,30 --grid stop=0.002,0.004,0.008 \
            --grid take=0.004,0.008,0.016 --grid long_only=true,false \
            --workers 4 --out state/bench-orb-1m.json

# classical: MA cross, intraday-only
trader tune --strategy ma_cross --symbol AAPL --asset-type CfdOnStock --horizon 1m \
            --folds 4 --train-days 30 --val-days 7 --test-days 7 \
            --fee-bps 0.5 --spread-bps 1.0 --slippage-bps 0.5 \
            --param flat_eod=true --param long_only=true \
            --grid fast=10,20,50 --grid slow=100,200,400 \
            --workers 4 --out state/bench-macross-1m.json

# model: LightGBM on the same folds (trains in seconds)
trader tune --strategy gbm --model-type gbm --symbol AAPL --asset-type CfdOnStock \
            --horizon 1m --folds 4 --train-days 30 --val-days 7 --test-days 7 \
            --fee-bps 0.5 --spread-bps 1.0 --slippage-bps 0.5 \
            --grid num_leaves=31,63 --grid lr=0.03,0.1 --grid n_estimators=300,600 \
            --grid stop=0.004,0.008 --grid threshold=0.1,0.2 \
            --workers 4 --out state/bench-gbm-1m.json
```

Always let the symbols resolve over the network (no `--uic`) so a `RegularHours`
calendar is inferred — otherwise Sharpe/vol annualise against a 24 h day (~5.9×
inflation for a 6.5 h US equity session).

## Chosen benchmark configs

_Fill in from the winning row of each sweep. These are the fixed yardsticks a
model has to beat, not tuned alpha._

| strategy | config | horizon | folds | median OOS Sharpe | worst fold | median turnover |
|---|---|---|---|---|---|---|
| `ma_cross` (flat_eod, long_only) | `fast=…, slow=…` | 1m | | | | |
| `orb` | `open_minutes=…, stop=…, take=…, long_only=…` | 1m | | | | |
| `gbm` | `num_leaves=…, lr=…, n_estimators=…, stop=…, threshold=…` | 1m | | | | |
| `lstm` | best from `state/sweep-*.json` | 1m | | | | |

## Notes / caveats

- **1-minute depth on Saxo SIM is per-symbol.** AAPL `CfdOnStock/211` has years;
  NVDA and the US500 index CFD need `trader data backfill … --horizon 1m` first
  and may only be weeks-to-months deep. Size `train_days` / `val_days` /
  `test_days` × `folds` to fit the intersection of the three coverages, or run
  AAPL-only for the model comparison and use the 3-name book only for the final
  confirmation.
- The classical path bleeds at most `fold_lookback` (≈ 1 day, floored) of the val
  tail into each fold's score — the same quirk `run_cv` already has.
- `StandardScaler` is fit and stored for the GBM too; it is a no-op for trees,
  kept only so the pipeline and the registry directory stay uniform.
