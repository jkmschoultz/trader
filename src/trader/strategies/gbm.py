"""Trade a registered LightGBM classifier -- the tabular sibling of ``lstm``.

Same :class:`~trader.strategies.model_base.ModelStrategy` machinery; the only
difference is a LightGBM ``Booster.predict`` in place of a torch forward pass,
and the features arrive flattened (``layout="tabular"``) rather than as a
sequence. The module imports cleanly without lightgbm installed, so
``available()`` always lists ``gbm``.
"""

from __future__ import annotations

from trader.strategies.model_base import ModelStrategy
from trader.strategies.registry import register


@register("gbm")
class GBMStrategy(ModelStrategy):
    """``ModelStrategy`` backed by a LightGBM gradient-boosted classifier."""

    def _require_deps(self) -> None:
        from trader.models._optional import require_lightgbm

        require_lightgbm()

    def _predict_proba(self, x):
        import numpy as np

        proba = self._predictor.predict(np.ascontiguousarray(x, dtype=np.float32))
        return np.asarray(proba, dtype=float).reshape(1, -1)[0]
