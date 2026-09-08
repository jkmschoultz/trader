# The model layer: windowing, the split, the registry, and the strategy

Phases 4c-4e are `trader.models` plus `trader.service.training` and the API/CLI
around them. `torch` and `scikit-learn` are the optional `[model]` extra; the
registry and manifest paths never import them, so `trader models list` and
`GET /api/models` work on a `[data]`-only install.

## From bars to `(X, y, w, t)`

`build_bundle(bars, *, spec, session, scaler, weights)`:

1. `compute_feature_frame` → features indexed by `time` (`docs/features.md`).
2. `triple_barrier` → labels indexed by the decision bar (`docs/labels.md`).
3. Sliding windows of length `spec.window`. `X[k]` is the `window` feature rows
   **ending at** decision bar `i`; `y[k]` is that bar's label. Because the label
   describes a trade entered at bar `i + 1`'s open, everything in `X[k]` is
   strictly older than the outcome `y[k]` measures — no leak.
4. Drop any window containing a warmup NaN, and any row whose label is NaN.
5. Map `{-1, 0, +1}` → `{0, 1, 2}`.
6. `t[k]` is the decision bar's timestamp in epoch nanoseconds, kept for the
   split.

`w` is per-sample weights: ones, unless `return_attribution_weights` are passed
in (`use_sample_weights`).

Multi-instrument training concatenates each instrument's bundle; the feature spec
must match across them.

## The split: purge and embargo

`time_split(bundle, SplitSpec(train_end, val_end, embargo_bars), *, base_horizon,
max_bars)` returns boolean `(train, val, test)` masks over `t`:

- **chronological** — train is `t < train_end`, val is `[train_end, val_end)`,
  test is `>= val_end`.
- **purge** — drop a train (or val) sample whose label window
  `[t, t + (1 + max_bars) bars]` reaches past its boundary, so no training label
  overlaps a validation bar.
- **embargo** — drop samples in the `embargo_bars` band immediately before each
  boundary. `run_training` defaults this to `window + max_bars`.

One split, not walk-forward CV — the `SplitSpec` shape (two dates) extends to a
list of folds later without touching callers.

## The model

`LSTMClassifier` is one `nn.LSTM` over the `(batch, window, features)` sequence,
the last (or mean-pooled) hidden state, dropout, and a linear head to three
classes. **One** LSTM, not a branch per timeframe — multi-timeframe context
already enters as extra feature columns via the as-of join, so a merged-feature
model is simpler and loses nothing.

`train_model` is deterministic given `seed`, uses inverse-frequency class weights
by default, clips gradients, early-stops on validation macro-F1, and keeps the
best `state_dict`. It returns the best model and a JSON-safe `TrainReport` (per-
epoch curve, val + test confusion matrices, class distribution, timings).

## The registry

`ModelRegistry` (`Settings.models_dir`, default `data_dir/models/`) stores one
model per directory, `{name}-{YYYYMMDD-HHMMSS}-{digest8}`:

| file | contents |
|---|---|
| `manifest.json` | id, created_at, git commit, data spec, horizons, feature-set name + digest, window, barriers, split, hyperparameters, val/test metrics, class distribution, framework versions |
| `weights.pt` | `torch` state_dict |
| `scaler.json` | the numpy `StandardScaler` (mean, scale per feature) |
| `feature_spec.json` | the frozen `FeatureSpec` — inference recomputes features from exactly this |
| `report.json` | the full `TrainReport` |

`save` builds the directory under a sibling `.tmp` name and `os.replace`s it into
place, so an interrupted or failed save leaves nothing partial behind. `list` and
`get` parse manifests only (no torch); `load_torch` rebuilds the net and scaler.
The digest in the id fingerprints the feature spec, barriers, window, and
hyperparameters, so two genuinely different models never collide and a re-run of
the same recipe is obvious.

## `LSTMStrategy` — the model as a `Strategy`

`@register("lstm")`. It loads a registered model in `__init__` (the only place,
with `on_bar`, that touches torch), sets `warmup = window + feature_warmup`, and
each bar:

1. recompute features from `ctx.history` using the model's **frozen**
   `FeatureSpec` — closed bars only, so it stays causal;
2. take the last `window` rows, apply the saved scaler, run the net, softmax;
3. `edge = p(up) - p(down)`. `edge > threshold` → `Target(+weight, **barriers)`;
   `edge < -threshold` → the short; otherwise `Hold()` (or `Flat()` if
   `on_no_signal="flat"`).

`barriers` are the model's trained stop/take/max_bars (unless `bracket=False`).
So the bracket the strategy trades is the barrier the model was labelled on.

Nothing in the engine, the backtest service, or the Results view changed — the
LSTM evaluates through the existing `POST /api/backtests` /
`trader backtest --strategy lstm --param model=<id>` path. `run_backtest` injects
`settings.models_dir` into any strategy whose `__init__` accepts it, so the
caller never has to spell out where the registry lives.

Known cost: `on_bar` recomputes features over the whole `ctx.history` slice every
bar — the same O(n) per-bar cost `docs/backtest.md` flags for the engine. An
incremental feature buffer is the follow-up if a long 1-minute run needs it.

## End to end

```
trader data backfill AAPL:xnas --asset-type Stock --horizon 5m,15m,60m --since 2y
trader features build --symbol AAPL:xnas --asset-type Stock --horizon 5m --context 15m,1h --feature-set mtf_v1
trader labels --symbol AAPL:xnas --asset-type Stock --horizon 5m --stop 0.005 --take 0.01 --max-bars 24
trader train  --symbol AAPL:xnas --asset-type Stock --horizon 5m --context 15m,1h --feature-set mtf_v1 \
              --window 32 --stop 0.005 --take 0.01 --max-bars 24 --train-end 2025-04-01 --val-end 2025-06-01
trader models list
trader backtest --symbol AAPL:xnas --asset-type Stock --strategy lstm --param model=<id> --horizon 5m --since 60d
```

The API mirrors this: `POST /api/training` (a job on the same store as
backtests), `GET /api/models[/{id}]`, `GET /api/catalog/feature-sets`, then the
unchanged `POST /api/backtests` with `{"strategy": "lstm", "params": {"model":
"<id>"}}`. The Train tab in the UI drives the whole loop.
