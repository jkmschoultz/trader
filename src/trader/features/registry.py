"""A name -> feature-set registry, mirroring :mod:`trader.strategies.registry`.

Feature sets register themselves at import time. :mod:`trader.features` imports
the built-ins on package import, so ``get_feature_set("price_v1")`` works without
importing the module by hand.
"""

from __future__ import annotations

from trader.features.base import FeatureSet

_FEATURE_SETS: dict[str, FeatureSet] = {}


class UnknownFeatureSet(KeyError):
    """Asked for a feature-set name that was never registered."""


def register_feature_set(name: str):
    """Class decorator: register an instance of ``cls`` under ``name``.

    Raises:
        ValueError: ``name`` is already taken -- re-registering is almost always
            a copy-paste bug, so it fails loudly.
    """

    def decorate(cls: type[FeatureSet]) -> type[FeatureSet]:
        if name in _FEATURE_SETS:
            raise ValueError(
                f"feature set {name!r} is already registered to {_FEATURE_SETS[name]!r}"
            )
        cls.name = name
        _FEATURE_SETS[name] = cls()
        return cls

    return decorate


def get_feature_set(name: str) -> FeatureSet:
    """Look up a registered feature set.

    Raises:
        UnknownFeatureSet: nothing is registered under ``name``; the message
            lists what is.
    """
    try:
        return _FEATURE_SETS[name]
    except KeyError:
        listed = ", ".join(available_feature_sets()) or "none"
        raise UnknownFeatureSet(f"unknown feature set {name!r}; available: {listed}") from None


def available_feature_sets() -> list[str]:
    """Every registered feature-set name, sorted."""
    return sorted(_FEATURE_SETS)
