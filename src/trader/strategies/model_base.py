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
from pathlib import Path
from typing import Literal

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target

# ``on_bar`` only needs the last ``window`` feature rows, and every indicator's
# lookback is bounded by ``FeatureSpec.warmup``. Recomputing over the whole
# ``ctx.history`` each bar is what makes a model backtest O(N^2); slicing to a
# bounded tail makes it O(N) for ``price_v1`` and shallow context sets. The pad
# past ``warmup`` is EWM burn-in headroom -- an EMA still carries a little of its
# seed for many spans -- scaled per call by the coarsest context ratio.
_TAIL_PAD = 300


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

        self._threshold = float(threshold)
        self._weight = float(weight)
        self._flat_on_no_signal = on_no_signal == "flat"
        self._barrier = target_from_barrier_params(self._info.barriers) if bracket else {}

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

        from trader.features.pipeline import compute_feature_frame

        tail = ctx.history.iloc[-self._tail :]
        features, _ = compute_feature_frame(tail, spec=self._spec, session=ctx.session)
        window = features.to_numpy(dtype="float32")[-self._window :]
        if window.shape[0] < self._window or np.isnan(window).any():
            return Hold()

        if self._layout == "tabular":
            x = self._scaler.transform(window.reshape(1, -1))
        else:
            x = self._scaler.transform(window[None, :, :])

        probs = np.asarray(self._predict_proba(np.ascontiguousarray(x)), dtype=float).reshape(-1)
        edge = float(probs[2] - probs[0])  # p(up) - p(down)
        if edge > self._threshold:
            return Target(self._weight, **self._barrier)
        if edge < -self._threshold:
            return Target(-self._weight, **self._barrier)
        return Flat() if self._flat_on_no_signal else Hold()
