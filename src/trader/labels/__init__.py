"""Triple-barrier labelling: turn bars into ``{-1, 0, +1}`` training targets.

The barriers -- ``stop``, ``take``, ``max_bars`` -- are the same three a
:class:`trader.strategies.base.Target` attaches in the backtest engine, so a
model trained on these labels trades the bracket it was taught. Needs the
``[data]`` extra (pandas/numpy).
"""

from __future__ import annotations

from trader.labels.triple_barrier import (
    BarrierParams,
    LabelConfig,
    barrier_params_from_target,
    target_from_barrier_params,
    triple_barrier,
)
from trader.labels.weights import (
    average_uniqueness,
    label_overlap_counts,
    return_attribution_weights,
)

__all__ = [
    "BarrierParams",
    "LabelConfig",
    "average_uniqueness",
    "barrier_params_from_target",
    "label_overlap_counts",
    "return_attribution_weights",
    "target_from_barrier_params",
    "triple_barrier",
]
