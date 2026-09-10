"""What strategies and allocators exist, and how to parameterise them.

The CLI shows these in ``--help`` text; the UI builds a form from them. Both want
the same thing: a name, a one-line description, and the constructor parameters
with their types and defaults.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trader.config import Settings

__all__ = [
    "AllocatorInfo",
    "FeatureSetInfo",
    "ModelSummary",
    "ParamInfo",
    "StrategyInfo",
    "list_allocators",
    "list_feature_sets",
    "list_models",
    "list_strategies",
]


@dataclass(frozen=True)
class ParamInfo:
    """One constructor parameter of a strategy or allocator."""

    name: str
    type: str
    default: object | None
    required: bool


@dataclass(frozen=True)
class StrategyInfo:
    name: str
    summary: str
    params: list[ParamInfo] = field(default_factory=list)


@dataclass(frozen=True)
class AllocatorInfo:
    name: str
    summary: str
    params: list[ParamInfo] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureSetInfo:
    """One registered feature set: its name, blurb, and resolved columns."""

    name: str
    summary: str
    columns: list[str]
    context_capable: bool


@dataclass(frozen=True)
class ModelSummary:
    """A registered model, trimmed to what a list view needs."""

    id: str
    name: str
    created_at: str
    model_type: str
    horizon: int
    context_horizons: list[int]
    feature_set: str
    window: int
    barriers: dict[str, Any]
    metrics: dict[str, Any]
    symbols: list[str]


def _summary(obj: object) -> str:
    """First non-empty line of a docstring."""
    doc = inspect.getdoc(obj) or ""
    return next((line.strip() for line in doc.splitlines() if line.strip()), "")


def _annotation(param: inspect.Parameter) -> str:
    ann = param.annotation
    if ann is inspect.Parameter.empty:
        return "str"
    if isinstance(ann, str):  # PEP 563 -- modules use `from __future__ import annotations`
        return ann
    return getattr(ann, "__name__", str(ann))


def _params(func: object) -> list[ParamInfo]:
    try:
        sig = inspect.signature(func)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return []
    out: list[ParamInfo] = []
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        required = param.default is inspect.Parameter.empty
        out.append(
            ParamInfo(
                name=name,
                type=_annotation(param),
                default=None if required else param.default,
                required=required,
            )
        )
    return out


def list_strategies() -> list[StrategyInfo]:
    """Every registered strategy, sorted by name."""
    from trader import strategies

    infos: list[StrategyInfo] = []
    for name in strategies.available():
        cls = strategies.get_strategy(name)
        infos.append(StrategyInfo(name=name, summary=_summary(cls), params=_params(cls.__init__)))
    return infos


def list_allocators() -> list[AllocatorInfo]:
    """Every registered allocator, sorted by name."""
    from trader.backtest import available_allocators

    return [
        AllocatorInfo(name=name, summary=_summary(cls), params=_params(cls.__init__))
        for name, cls in available_allocators().items()
    ]


def list_feature_sets() -> list[FeatureSetInfo]:
    """Every registered feature set, sorted by name, with its base columns."""
    from trader.features import available_feature_sets, get_feature_set

    infos: list[FeatureSetInfo] = []
    for name in available_feature_sets():
        feature_set = get_feature_set(name)
        spec = feature_set.resolve(base_horizon=5, context_horizons=())
        infos.append(
            FeatureSetInfo(
                name=name,
                summary=_summary(type(feature_set)),
                columns=list(spec.columns),
                context_capable=feature_set.context_capable,
            )
        )
    return infos


def list_models(settings: Settings) -> list[ModelSummary]:
    """Every registered model under ``settings.models_dir``, newest first. No torch."""
    from trader.models.registry import ModelRegistry

    return [
        ModelSummary(
            id=info.id,
            name=info.name,
            created_at=info.created_at,
            model_type=info.model_type,
            horizon=info.base_horizon,
            context_horizons=info.context_horizons,
            feature_set=info.feature_set,
            window=info.window,
            barriers=info.barriers,
            metrics=info.metrics,
            symbols=info.symbols,
        )
        for info in ModelRegistry(settings.models_dir).list()
    ]
