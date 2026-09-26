"""Shared base for strategies that trade a registered classifier.

A model lands as just another :class:`InstrumentStrategy`: recompute the frozen
feature spec over a bounded tail, scale, ask the model for class probabilities,
and turn ``p(up) - p(down)`` into a :class:`Target`. The bracket attached is the
one the model was *trained* on (the triple-barrier params from the manifest), so
what it trades matches what it learned.

Concrete subclasses (:class:`~trader.strategies.lstm.LSTMStrategy`,
:class:`~trader.strategies.gbm.GBMStrategy`) only supply ``_require_deps`` and
``_predict_proba``; everything else -- loading, warmup, tail sizing, the decision
rule, causality -- lives here. ``_layout`` (from the manifest) decides whether
the model is fed an ``(1, window, F)`` sequence or a flat ``(1, window*F)`` row.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target

if TYPE_CHECKING:
    import pandas as pd

# ``on_bar`` only needs the last ``window`` feature rows, and every indicator's
# lookback is bounded by ``FeatureSpec.warmup``. Recomputing over the whole
# ``ctx.history`` each bar is what makes a model backtest O(N^2); slicing to a
# bounded tail makes it O(N) for ``price_v1`` and shallow context sets. The pad
# past ``warmup`` is EWM burn-in headroom -- an EMA still carries a little of its
# seed for many spans -- scaled per call by the coarsest context ratio.
_TAIL_PAD = 300
# bars of history for an ATR-scaled bracket: the ATR's EWM seed decays as
# (13/14)^n, so 700 bars reproduce the full-history value the labels used
_ATR_TAIL = 700


class ModelStrategy(InstrumentStrategy):
    """Go long when ``p(up) - p(down)`` clears ``threshold``, short when it clears ``-threshold``.

    Args:
        model: id of a model in the registry (``Settings.models_dir``).
        threshold: dead-band on the up-minus-down probability edge.
        weight: conviction magnitude handed to the allocator.
        bracket: attach the model's trained stop / take / max_bars to each entry.
        on_no_signal: inside the dead-band, ``"hold"`` the position (let the
            bracket manage the exit) or go ``"flat"``.
        models_dir: override the registry root (mainly for tests).
    """

    def __init__(
        self,
        *,
        model: str,
        threshold: float = 0.15,
        weight: float = 1.0,
        bracket: bool = True,
        on_no_signal: Literal["hold", "flat"] = "hold",
        models_dir: str | None = None,
    ) -> None:
        import pandas as pd

        from trader.config import get_settings
        from trader.features.base import FeatureSpec
        from trader.labels.triple_barrier import target_from_barrier_params
        from trader.models.registry import ModelRegistry

        self._require_deps()

        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {threshold}")
        if not -1.0 <= weight <= 1.0 or weight == 0.0:
            raise ValueError(f"weight must be a non-zero conviction in [-1, 1], got {weight}")

        root = Path(models_dir) if models_dir else get_settings().models_dir
        registry = ModelRegistry(root)
        self._predictor, self._scaler, self._info = registry.load_model(model)

        self._spec = FeatureSpec.from_file(registry.path(model) / "feature_spec.json")
        self._window = int(self._info.window)
        self._layout = self._info.layout
        self.warmup = self._window + self._spec.warmup

        base = self._spec.base_horizon
        ratio = max((-(-h // base) for h in self._spec.context_horizons), default=1)
        self._tail = self.warmup + _TAIL_PAD * ratio
        # level features (prior day / week) reach back a wall-clock span, which
        # is a different bar count for 24h FX and a 6.5h equity session
        self._tail_span = (
            pd.Timedelta(days=self._spec.level_days) if self._spec.level_days else None
        )

        self._threshold = float(threshold)
        self._weight = float(weight)
        self._flat_on_no_signal = on_no_signal == "flat"
        self._barrier = target_from_barrier_params(self._info.barriers) if bracket else {}
        # "atr": the stored stop / take are ATR multiples, converted per decision
        self._barrier_scale = self._info.barriers.get("scale", "fraction")
        # meta-labelling: trade only while the primary rule the model was trained
        # under is long; the gate is re-read once per completed day
        from trader.labels.primary import LiveGate, PrimarySpec

        self._primary = (
            LiveGate(PrimarySpec.from_dict(self._info.primary)) if self._info.primary else None
        )
        if self._primary is not None:
            self.history_days = self._primary.spec.span_days
        # label -> (bar times as int64 ns, feature rows); see use_precomputed
        self._precomputed: dict[str, tuple] = {}

    def use_precomputed(self, features: Mapping[str, pd.DataFrame]) -> None:
        """Serve ``on_bar`` from feature frames computed once over each full series.

        ``features`` maps an instrument label to ``compute_feature_frame`` output
        for that label's whole bar frame, built with this model's frozen spec.
        Features are causal, so row ``i`` of a full-history compute is exactly
        what a recompute at bar ``i`` would give (minus the tail's EWM burn-in
        drift) -- this turns a fold backtest from one pipeline run per bar into
        a lookup. A bar whose time is not in the frame falls back to recomputing.
        """
        import pandas as pd

        # the lake stores microsecond times: normalise to ns so lookups compare
        # against Timestamp.value, which is always ns
        self._precomputed = {
            label: (
                pd.DatetimeIndex(frame.index).as_unit("ns").asi8,
                frame.to_numpy(dtype="float32"),
            )
            for label, frame in features.items()
        }

    def _window_rows(self, ctx: BarContext):
        """The last ``window`` feature rows as of ``ctx``'s most recent closed bar."""
        import numpy as np
        import pandas as pd

        cached = self._precomputed.get(ctx.label)
        if cached is not None:
            times, values = cached
            last = pd.Timestamp(ctx.history["time"].iloc[-1]).value
            end = int(np.searchsorted(times, last, side="right"))
            if end and times[end - 1] == last:
                return values[max(0, end - self._window) : end]

        from trader.features.pipeline import compute_feature_frame

        start = max(0, len(ctx.history) - self._tail)
        if self._tail_span is not None and start:
            times = ctx.history["time"]
            cutoff = times.iloc[-1] - self._tail_span
            start = min(start, int(times.searchsorted(cutoff, side="left")))
        tail = ctx.history.iloc[start:]
        features, _ = compute_feature_frame(tail, spec=self._spec, session=ctx.session)
        return features.to_numpy(dtype="float32")[-self._window :]

    def _bracket(self, ctx: BarContext) -> dict | None:
        """This decision's bracket kwargs; None if an ATR bracket has no ATR yet.

        Uses :func:`~trader.labels.triple_barrier.barrier_unit`, the same
        function the labels were built with, on the bars up to this one.
        """
        if not self._barrier or self._barrier_scale == "fraction":
            return self._barrier

        import math

        from trader.labels.triple_barrier import barrier_unit

        unit = float(barrier_unit(ctx.history.iloc[-_ATR_TAIL:], self._barrier_scale)[-1])
        if not math.isfinite(unit) or unit <= 0:
            return None
        stop, take = self._barrier["stop"], self._barrier["take"]
        return {
            **self._barrier,
            "stop": stop * unit if stop is not None else None,
            "take": take * unit if take is not None else None,
        }

    # --- subclass hooks -----------------------------------------------------

    @abstractmethod
    def _require_deps(self) -> None:
        """Raise ``ValueError`` (with a ``[model]`` hint) if a dep is missing."""

    @abstractmethod
    def _predict_proba(self, x):
        """Class probabilities for one framed example, as a length-3 array."""

    # --- the strategy ------------------------------------------------------

    def on_bar(self, ctx: BarContext) -> Decision:
        import numpy as np

        if ctx.bars_seen < self.warmup:
            return Hold()
        if self._primary is not None and not self._primary(ctx.history, ctx.horizon, ctx.label):
            return Flat()

        window = self._window_rows(ctx)
        if window.shape[0] < self._window or np.isnan(window).any():
            return Hold()

        if self._layout == "tabular":
            x = self._scaler.transform(window.reshape(1, -1))
        else:
            x = self._scaler.transform(window[None, :, :])

        probs = np.asarray(self._predict_proba(np.ascontiguousarray(x)), dtype=float).reshape(-1)
        edge = float(probs[2] - probs[0])  # p(up) - p(down)
        if abs(edge) > self._threshold:
            if edge < 0 and self._primary is not None:
                return Flat()  # the primary is long-only: a "down" call means stay out
            barrier = self._bracket(ctx)
            if barrier is None:
                return Hold()
            return Target(self._weight if edge > 0 else -self._weight, **barrier)
        return Flat() if self._flat_on_no_signal else Hold()
