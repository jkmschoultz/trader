"""Framework-agnostic orchestration shared by the CLI and the API.

The CLI command bodies and the FastAPI routes both need the same few things --
turn a strategy name and some params into a run, read a slice of the lake, list
what strategies and allocators exist, drive a backfill -- with no argparse
namespace and no HTTP request in sight. That logic lives here so neither
front-end owns it.

Like :mod:`trader.backtest`, the run helpers need the ``[data]`` extra; the
catalogue helpers do not.
"""

from __future__ import annotations

from trader.service.catalog import (
    AllocatorInfo,
    FeatureSetInfo,
    ParamInfo,
    StrategyInfo,
    list_allocators,
    list_feature_sets,
    list_strategies,
)
from trader.service.inputs import coerce_scalar, parse_params, parse_since

__all__ = [
    "AllocatorInfo",
    "FeatureSetInfo",
    "ParamInfo",
    "StrategyInfo",
    "coerce_scalar",
    "list_allocators",
    "list_feature_sets",
    "list_strategies",
    "parse_params",
    "parse_since",
]
