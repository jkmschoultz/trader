# Triple-barrier labelling: matching the engine's bracket

Phase 4b is `trader.labels`. `triple_barrier(frame, *, stop, take, max_bars,
min_return, entry)` turns a bar frame into `{-1, 0, +1}` training targets by
asking, for every bar: *if the model fires here, you enter at the next bar's
open — which barrier does the trade hit first?*

## The barriers are the engine's `Target` bracket

`stop` and `take` are positive fractions of the entry price; `max_bars` is a
bar count. These are exactly the three barriers a
`trader.strategies.base.Target(weight, stop, take, max_bars)` attaches in the
backtest engine (`docs/backtest.md`). The parameterisation is shared on purpose:
a model trained on these labels, traded through `LSTMStrategy` with `bracket=True`,
places the bracket it was taught.

Semantics, matched to the engine bar for bar:

- **Entry** is the next bar's open (`entry="next_open"`, the default and the only
  mode the engine supports). `entry="close"` exists for research.
- For each later bar, `high` is checked against `entry * (1 + take)` and `low`
  against `entry * (1 - stop)`.
- **Gaps fill at the open.** If a bar *opens* beyond a barrier (typically an
  overnight gap on an instrument that closes while its underlying keeps
  trading, like a bitcoin ETF), the exit fills at that open, not at the barrier
  price the market never traded at. The open comes first in time, so it also
  decides which barrier was hit: opening above the take is a take even if the
  bar later falls through the stop.
- **Stop wins ties.** If one bar's range spans both levels without a gap, the
  label is the stop (`-1`). Real fills are path-dependent and unknowable from OHLC; assuming
  the adverse touch came first biases labels the safe way, the same downward
  bias the engine applies.
- If neither barrier is touched within `max_bars` bars, the **vertical barrier**
  fires: exit at that bar's open, label `sign(return)`, with `|return| <=
  min_return` collapsing to `0` (a dead-band so a flat drift is not called a
  direction).

`touch_price` is the fill price: the barrier level, or the open when the bar
gapped through it -- the same rule the engine's bracket exits use. Before this
rule, a stop was booked at its level even after a 3% overnight gap, which made
both the labels and the backtests quietly too kind to losing trades.

## Indexing and the trailing NaN band

The result frame is indexed by the **decision bar** (the bar the model sees),
one row per input bar, so it aligns 1:1 with a feature frame on `time`. Columns:
`label`, `entry_time`, `entry_price`, `touch_time`, `touch_price`, `barrier`
(`"stop"` / `"take"` / `"time"` / None), `bars_held`, `ret`.

The last `1 + max_bars` rows have no room for a full forward window and get
`label = NaN`. Those are dropped at the dataset boundary, never earlier. The
causality tests (`tests/labels/test_causality.py`) check the other direction: a
label may look forward, but perturbing a bar beyond `entry_offset + max_bars`
from the decision bar must not change it.

## Sample weights — shallow, on purpose

Triple-barrier events overlap: while one is open the next few start, so treating
every labelled row as independent over-counts crowded stretches.
`trader.labels.weights` provides two corrections:

- `average_uniqueness` — the mean of `1 / concurrency` over an event's life.
- `return_attribution_weights` — `|ret| * average_uniqueness`, normalised to
  mean 1; a big move that barely overlaps anything counts most.

Training uses class weighting by default and multiplies in these per-sample
weights only when `use_sample_weights=True` — for both the LSTM
(`train_model`) and the GBM (`train_gbm`, which passes them straight to
`LGBMClassifier.fit(sample_weight=...)`). The full sequential-bootstrap treatment
(López de Prado, ch. 4) is deliberately left for a later phase — overlap
weighting matters most for bagged trees on non-overlapping events, and class
balance is the bigger lever for a first sequence model.

The GBM consumes the same labels through a flattened feature layout: `build_bundle`
with `WindowSpec(layout="tabular")` turns the `(n, window, F)` lag stack into one
`(n, window * F)` row per decision bar, columns renamed `col__t-k`. The label,
the barriers, and the split are untouched.

## Inspecting a parameterisation

`trader labels --symbol AAPL --stop 0.005 --take 0.01 --max-bars 24` (or
`POST /api/…` via the service) prints the class balance, the stop/take/time
barrier breakdown, mean `|ret|`, and median bars held — enough to tell whether a
barrier choice produces a usable label distribution before committing a training
run to it.
