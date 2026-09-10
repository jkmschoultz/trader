"""Trade a registered LSTM classifier -- the model lands as just another Strategy.

The engine, the backtest service, and the Results view need no changes: the
shared :class:`~trader.strategies.model_base.ModelStrategy` does the loading,
warmup, tail sizing, and the ``p(up) - p(down)`` decision rule. This subclass
only adds the torch forward pass.

Only ``__init__`` and ``on_bar`` touch torch, via
:func:`trader.models._optional.require_torch`; the module imports cleanly on a
``[data]``-only install, so ``available()`` always lists ``lstm`` and a missing
extra only bites when someone actually runs it.
"""

from __future__ import annotations

from trader.strategies.model_base import ModelStrategy
from trader.strategies.registry import register


@register("lstm")
class LSTMStrategy(ModelStrategy):
    """``ModelStrategy`` backed by a torch :class:`~trader.models.lstm.LSTMClassifier`."""

    def _require_deps(self) -> None:
        from trader.models._optional import require_torch

        require_torch()

    def _predict_proba(self, x):
        import numpy as np
        import torch

        with torch.no_grad():
            logits = self._predictor(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
            return torch.softmax(logits, dim=1).cpu().numpy()[0]
