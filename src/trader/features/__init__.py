"""Causal feature pipeline: the :class:`~trader.features.base.FeatureSet`
interface, a name registry, and the built-in feature sets.

Importing this package registers the built-ins (``price_v1``, ``mtf_v1``,
``structure_v1``), so ``registry.get_feature_set("price_v1")`` works without
importing the module directly. Everything here reads the canonical bar frame
(pandas/numpy), which comes with the ``[data]`` extra.
"""

from __future__ import annotations

from trader.features import sets, structure  # noqa: F401 - registration side effect
from trader.features.base import FeatureSet, FeatureSpec
from trader.features.pipeline import compute_feature_frame
from trader.features.registry import (
    UnknownFeatureSet,
    available_feature_sets,
    get_feature_set,
    register_feature_set,
)

__all__ = [
    "FeatureSet",
    "FeatureSpec",
    "UnknownFeatureSet",
    "available_feature_sets",
    "compute_feature_frame",
    "get_feature_set",
    "register_feature_set",
]
