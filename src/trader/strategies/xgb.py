"""Trade a registered XGBoost classifier -- the GPU-trained sibling of ``gbm``.

Same :class:`~trader.strategies.model_base.ModelStrategy` machinery and the same
flattened (``layout="tabular"``) features; the only difference is XGBoost's
``Booster.inplace_predict`` in place of LightGBM's. The registry loads the
booster pinned to the CPU, so a model trained on CUDA trades anywhere. The
module imports cleanly without xgboost installed, so ``available()`` always
lists ``xgb``.
"""

from __future__ import annotations

from trader.strategies.model_base import ModelStrategy
from trader.strategies.registry import register


@register("xgb")
class XGBStrategy(ModelStrategy):
    """``ModelStrategy`` backed by an XGBoost gradient-boosted classifier."""

    def _require_deps(self) -> None:
        from trader.models._optional import require_xgboost

        require_xgboost()

    def _predict_proba(self, x):
        import numpy as np

        proba = self._predictor.inplace_predict(np.ascontiguousarray(x, dtype=np.float32))
        return np.asarray(proba, dtype=float).reshape(1, -1)[0]
