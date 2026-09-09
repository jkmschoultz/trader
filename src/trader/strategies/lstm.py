"""Trade a registered LSTM classifier -- the model lands as just another Strategy.

The engine, the backtest service, and the Results view need no changes: this is
an :class:`InstrumentStrategy` that turns the model's up/down probability into a
:class:`Target`. The bracket it attaches is the one the model was *trained* on
(the triple-barrier params from the manifest), so what it trades matches what it
learned.

Only ``__init__`` and ``on_bar`` touch torch, via
:func:`trader.models._optional.require_torch`; the module imports cleanly on a
``[data]``-only install, so ``available()`` always lists ``lstm`` and a missing
extra only bites when someone actually runs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from trader.strategies.base import BarContext, Decision, Flat, Hold, InstrumentStrategy, Target
from trader.strategies.registry import register

# ``on_bar`` only needs the last ``window`` feature rows, and every indicator's
# lookback is bounded by ``FeatureSpec.warmup``. Recomputing over the whole
# ``ctx.history`` each bar is what makes an LSTM backtest O(N^2); slicing to a
# bounded tail makes it O(N) for ``price_v1`` and shallow context sets. The pad
# past ``warmup`` is EWM burn-in headroom -- an EMA still carries a little of its
# seed for many spans -- scaled per call by the coarsest context ratio, since a
# context EMA burns in over that many base bars. Deep context horizons make
# ``warmup`` itself large; an engine-level precompute is the follow-up there.
_TAIL_PAD = 300


@register("lstm")
class LSTMStrategy(InstrumentStrategy):
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
        from trader.models._optional import require_torch
        from trader.models.registry import ModelRegistry

        require_torch()

        if not 0.0 <= threshold < 1.0:
            raise ValueError(f"threshold must be in [0, 1), got {threshold}")
        if not -1.0 <= weight <= 1.0 or weight == 0.0:
            raise ValueError(f"weight must be a non-zero conviction in [-1, 1], got {weight}")

        root = Path(models_dir) if models_dir else get_settings().models_dir
        registry = ModelRegistry(root)
        self._net, self._scaler, self._info = registry.load_torch(model)

        self._spec = FeatureSpec.from_file(registry.path(model) / "feature_spec.json")
        self._window = int(self._info.window)
        self.warmup = self._window + self._spec.warmup

        base = self._spec.base_horizon
        ratio = max((-(-h // base) for h in self._spec.context_horizons), default=1)
        self._tail = self.warmup + _TAIL_PAD * ratio

        self._threshold = float(threshold)
        self._weight = float(weight)
        self._flat_on_no_signal = on_no_signal == "flat"
        self._barrier = target_from_barrier_params(self._info.barriers) if bracket else {}

    def on_bar(self, ctx: BarContext) -> Decision:
        import numpy as np
        import torch

        if ctx.bars_seen < self.warmup:
            return Hold()

        from trader.features.pipeline import compute_feature_frame

        tail = ctx.history.iloc[-self._tail :]
        features, _ = compute_feature_frame(tail, spec=self._spec, session=ctx.session)
        window = features.to_numpy(dtype="float32")[-self._window :]
        if window.shape[0] < self._window or np.isnan(window).any():
            return Hold()

        scaled = self._scaler.transform(window[None, :, :])
        with torch.no_grad():
            logits = self._net(torch.from_numpy(np.ascontiguousarray(scaled)))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        edge = float(probs[2] - probs[0])  # p(up) - p(down)
        if edge > self._threshold:
            return Target(self._weight, **self._barrier)
        if edge < -self._threshold:
            return Target(-self._weight, **self._barrier)
        return Flat() if self._flat_on_no_signal else Hold()
