"""What strategies and allocators exist, and how to parameterise them.

The CLI shows these in ``--help`` text; the UI builds a form from them. Both want
the same thing: a name, a one-line description, and the constructor parameters
with their types and defaults.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

__all__ = [
    "AllocatorInfo",
    "ParamInfo",
    "StrategyInfo",
    "list_allocators",
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
