# The model layer: windowing, the split, the registry, and the strategy

Phases 4c-4e are `trader.models` plus `trader.service.training` and the API/CLI
around them. `torch`, `scikit-learn` and `lightgbm` are the optional `[model]`
extra; the registry and manifest paths never import them, so `trader models list`
and `GET /api/models` work on a `[data]`-only install.

There are two model families, chosen by `--model-type` (`TrainingSpec.model_type`):
`lstm` (a torch sequence model) and `gbm` (a LightGBM gradient-boosted classifier
on the flattened lag stack). They share the labels, the features, the split, the
registry directory, and the decision rule — only the trainer, the feature
`layout`, and the weights format differ.

## From bars to `(X, y, w, t)`

`build_bundle(bars, *, spec, session, scaler, weights)`:

1. `compute_feature_frame` → features indexed by `time` (`docs/features.md`).
2. `triple_barrier` → labels indexed by the decision bar (`docs/labels.md`).
3. Sliding windows of length `spec.window`. `X[k]` is the `window` feature rows
   **ending at** decision bar `i`; `y[k]` is that bar's label. Because the label
   describes a trade entered at bar `i + 1`'s open, everything in `X[k]` is
   strictly older than the outcome `y[k]` measures — no leak.
4. Drop any window containing a warmup NaN, and any row whose label is NaN.
5. `spec.layout` picks the shape: `"sequence"` keeps `X` as `(n, window, F)` for
   the LSTM; `"tabular"` flattens it to `(n, window * F)` (oldest lag first,
   columns renamed `col__t-k`) for the GBM. `run_cv` sets it from `model_type`.
6. Map `{-1, 0, +1}` → `{0, 1, 2}`.
7. `t[k]` is the decision bar's timestamp in epoch nanoseconds, kept for the
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

`SplitSpec` also carries an optional `test_end` (right-bound the test set, with a
matching right-side purge) and `train_start` (a rolling, fixed-width train
window instead of an anchored one). `run_training` uses neither — they exist for
walk-forward.

## Walk-forward CV and the sweep

`walk_forward_splits(t, *, n_folds, train_days, val_days, test_days, embargo_bars,
mode, step_days)` returns `n_folds` chronological `SplitSpec`s over the span of
`t`: each fold is `train_days` → `val_days` → `test_days`, the origin advancing by
`step_days` (default `test_days`, so test windows tile without overlap).
`mode="rolling"` fixes the train width; `"anchored"` lets it expand. It raises if
the data span cannot fit the folds.

`run_cv(settings, spec, cv)` (`trader.service.training`) builds the
`SequenceBundle` **once** (features and labels do not depend on the cuts), then
per fold: `time_split` → fit the scaler on that fold's train → `train_model` →
**run the held-out window through a real `bt.run`** with a realistic `CostModel`.
It aggregates median / mean / std of Sharpe, return, turnover, hit rate and
profit factor across folds, and counts `folds_sharpe_gt_0_5`. Classification
accuracy is not the selection metric — out-of-sample Sharpe is.

`run_strategy_cv(settings, StrategyCVSpec)` (`trader.service.evaluation`) is the
model-free counterpart: no windowing, no training — build a **classical**
strategy (`ma_cross`, `orb`, …) from its params per fold and score the same
held-out window through the same `bt.run` + `CostModel`. `score_fold` and
`aggregate_folds` are shared with `run_cv`, so both paths produce one comparable
`{folds, aggregate}` shape.

`run_tuning(settings, TuningSpec)` (`trader.service.tuning`) sweeps a `grid`:
the cartesian product, each combo through the right CV path, ranked by median OOS
Sharpe (tie-break lower turnover). `TuningSpec.strategy` selects the path —
`"lstm"` / `"gbm"` sweep `TrainingSpec` / `CVConfig` fields and train per fold;
any other registered name sweeps that strategy's constructor args (plus the
scoring knobs `allocator`, `leverage`, `fee_bps`, `spread_bps`, `slippage_bps`)
with `params` holding the fixed args. A per-config failure is an error row, not
fatal. Configs run in parallel over a
`ProcessPoolExecutor` (`max_workers`, default `min(4, configs, cpu//2)`; `1` =
in-process) using a `spawn` context since torch is already loaded in the parent;
each finished config emits an `on_message` line (`config i/n done — best median
Sharpe …`) onto the job. `build_report` / `render` / `save_report` mirror
`trader.data.depth` — a JSON-safe dict, a terminal table, a file under `state/`.
Driven by `trader tune --symbol … --grid window=16,32 --grid stop=0.004,0.008
--folds 5 --workers 4` and `POST /api/tuning` (a `kind="tuning"` job).

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
`TrainingSpec.device` (`--device`, default `auto`) picks the torch device: CUDA
when `torch.cuda.is_available()`, else CPU. The best `state_dict` is copied to
CPU and the registry loads with `map_location="cpu"`, so a GPU-trained model
backtests and trades on a CPU-only box. A `tune` sweep's workers each open their
own CUDA context (~0.5 GB), so `--workers` is bounded by GPU memory, not cores.

`GBMClassifier` (`--model-type gbm`) is a `lightgbm.LGBMClassifier`
(`objective="multiclass"`, `num_class=3`) on the `layout="tabular"` lag stack.
`train_gbm` (`trader.models.gbm`) early-stops on validation multi-logloss,
supports the same `class_weight="balanced"` and per-sample weights, and fills the
same model-agnostic `TrainReport` (`epochs` is the boosting eval curve;
`framework_version` records the lightgbm version). It trains in seconds, so a GBM
sweep is the cheap yardstick a heavier model has to beat. `TrainReport` now lives
in `trader.models.report`, importable without torch or lightgbm.

`--model-type xgb` (`trader.models.xgb.train_xgb`) trains the same trees with
XGBoost, on CUDA when there is one (`device="auto"`). It exists for speed: on a
fold-sized problem (50k rows × 1,360 lag columns) a CUDA `hist` fit took 7 s
against 77 s for single-threaded LightGBM, and a whole EURUSD 15m fold (build,
train, 180-day backtest) runs in ~30 s. It takes the same `GBMConfig`, mapped as
closely as XGBoost allows: leaf-wise growth capped at `num_leaves`;
`min_child_samples` becomes `min_child_weight` at ×0.2 (the per-sample hessian of
a 3-class softmax near uniform); `class_weight="balanced"` becomes per-sample
weights. It uses the native `xgb.train` API with `num_class=3` because the
scikit-learn wrapper rejects a fold that lacks the (rare) flat class. The booster
is trimmed to its early-stopping best iteration before it is saved, since
inference predicts with every tree it is given. The registry writes XGBoost's
JSON model (`model.json`), and `load_xgb` pins prediction to the CPU, so a
GPU-trained model trades on a box without a GPU. LightGBM's own GPU build was
only ~1.45× faster and does not share one card well across sweep workers, so
`gbm` stays CPU-only.

## The registry

`ModelRegistry` (`Settings.models_dir`, default `data_dir/models/`) stores one
model per directory, `{name}-{YYYYMMDD-HHMMSS}-{digest8}`:

| file | contents |
|---|---|
| `manifest.json` | id, created_at, git commit, **`model_type`** (`lstm` \| `gbm`), **`layout`** (`sequence` \| `tabular`), data spec, horizons, feature-set name + digest, window, barriers, split, hyperparameters, val/test metrics, class distribution, framework versions |
| `weights.pt` *or* `model.txt` | `torch` state_dict (`lstm`) or a LightGBM text model (`gbm`); `manifest["files"]["weights"]` names it |
| `scaler.json` | the numpy `StandardScaler` (mean, scale per feature) — kept for the GBM too, a harmless no-op for trees |
| `feature_spec.json` | the frozen `FeatureSpec` — inference recomputes features from exactly this |
| `report.json` | the full `TrainReport` |

`save` builds the directory under a sibling `.tmp` name and `os.replace`s it into
place, so an interrupted or failed save leaves nothing partial behind. `list` and
`get` parse manifests only (no torch/lightgbm); `load_model(id)` dispatches on
`model_type` to `load_torch` / `load_gbm`, each returning `(predictor, scaler,
info)`. A manifest with no `model_type` / `layout` reads as `lstm` / `sequence`,
so models trained before this change still load. The digest in the id
fingerprints `model_type`, the feature spec, barriers, window, and
hyperparameters, so two genuinely different models never collide and a re-run of
the same recipe is obvious.

## `ModelStrategy` — the model as a `Strategy`

`trader.strategies.model_base.ModelStrategy` is the shared base;
`LSTMStrategy` (`@register("lstm")`) and `GBMStrategy` (`@register("gbm")`) each
add only a `_require_deps` and a `_predict_proba`. The base loads a registered
model in `__init__` via `load_model` (the only place, with `on_bar`, that touches
the model deps), sets `warmup = window + feature_warmup`, and each bar:

1. recompute features from `ctx.history` using the model's **frozen**
   `FeatureSpec` — closed bars only, so it stays causal;
2. take the last `window` rows, apply the saved scaler, and reshape per the
   manifest `layout` (`(1, window, F)` for the LSTM, `(1, window * F)` for the
   GBM);
3. `_predict_proba` → class probabilities (torch softmax / `Booster.predict`);
   `edge = p(up) - p(down)`. `edge > threshold` → `Target(+weight, **barriers)`;
   `edge < -threshold` → the short; otherwise `Hold()` (or `Flat()` if
   `on_no_signal="flat"`).

`barriers` are the model's trained stop/take/max_bars (unless `bracket=False`).
So the bracket the strategy trades is the barrier the model was labelled on.

Nothing in the engine, the backtest service, or the Results view changed — the
LSTM evaluates through the existing `POST /api/backtests` /
`trader backtest --strategy lstm --param model=<id>` path. `run_backtest` injects
`settings.models_dir` into any strategy whose `__init__` accepts it, so the
caller never has to spell out where the registry lives.

`on_bar` recomputes features from a **bounded tail** of `ctx.history` —
`warmup + _TAIL_PAD * context_ratio` bars — not the whole slice, so a backtest is
O(N) rather than O(N²). Features are causal, so the retained `window` rows match a
full-history compute once the EMAs have burned in (`< 1e-5`).

Even bounded, that is one pipeline run per bar (~60 ms for `mtf_v1`), and it was
~90% of a walk-forward fold's wall time. `use_precomputed({label: features})`
lets a caller hand the strategy each series' feature frame computed **once** over
the whole series with the model's frozen spec; `on_bar` then slices the last
`window` rows up to the current bar instead of recomputing. Causality makes the
two equivalent — row `i` of a full-history compute depends only on bars `0..i` —
and the precomputed rows are exactly the ones training saw. A bar whose time is
not in the frame falls back to the recompute. `run_cv` precomputes per config, so
a fold's backtest is a lookup (~10x faster end to end); live trading and
`run_backtest` still recompute.

## End to end

```
trader data backfill AAPL:xnas --asset-type Stock --horizon 5m,15m,60m --since 2y
trader features build --symbol AAPL:xnas --asset-type Stock --horizon 5m --context 15m,1h --feature-set mtf_v1
trader labels --symbol AAPL:xnas --asset-type Stock --horizon 5m --stop 0.005 --take 0.01 --max-bars 24
trader train  --symbol AAPL:xnas --asset-type Stock --horizon 5m --context 15m,1h --feature-set mtf_v1 \
              --window 32 --stop 0.005 --take 0.01 --max-bars 24 --train-end 2025-04-01 --val-end 2025-06-01
trader models list
trader backtest --symbol AAPL:xnas --asset-type Stock --strategy lstm --param model=<id> --horizon 5m --since 60d

# or sweep configs by walk-forward CV instead of a single train:
trader tune --symbol AAPL:xnas --asset-type Stock --horizon 15m --context 1h,4h --feature-set mtf_v1 \
            --grid window=16,32 --grid stop=0.004,0.008 --folds 5 --fee-bps 0.2 --spread-bps 1.0 --out state/sweep.json

# a LightGBM sweep on the same folds (trains in seconds):
trader tune --symbol AAPL:xnas --asset-type Stock --strategy gbm --model-type gbm --horizon 15m \
            --grid num_leaves=31,63 --grid lr=0.03,0.1 --grid threshold=0.1,0.2 --folds 5 --fee-bps 0.2

# and a classical baseline through the exact same harness:
trader tune --symbol AAPL:xnas --asset-type Stock --strategy orb --horizon 1m \
            --grid open_minutes=5,15,30 --grid stop=0.002,0.004 --grid long_only=true,false --folds 4
```

The API mirrors this: `POST /api/training` (a job on the same store as
backtests), `GET /api/models[/{id}]`, `GET /api/catalog/feature-sets`, then the
unchanged `POST /api/backtests` with `{"strategy": "lstm", "params": {"model":
"<id>"}}`. The Train tab in the UI drives the whole loop.
