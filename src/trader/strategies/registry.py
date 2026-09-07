"""A name -> strategy-class registry.

Strategies register themselves at import time with :func:`register`, so the CLI
and, later, the UI can offer them by name without importing each module by hand.
:mod:`trader.strategies` imports the built-ins on package import; third-party
strategies just need to be imported once before :func:`get_strategy` is called.
"""

from __future__ import annotations

from trader.strategies.base import Strategy

_STRATEGIES: dict[str, type[Strategy]] = {}


class UnknownStrategy(KeyError):
    """Asked for a strategy name that was never registered."""


def register(name: str):
    """Class decorator: register ``cls`` under ``name``.

    Raises:
        ValueError: ``name`` is already taken. Re-registering is almost always a
            copy-paste bug, so it fails loudly rather than silently replacing.
    """

    def decorate(cls: type[Strategy]) -> type[Strategy]:
        if name in _STRATEGIES:
            raise ValueError(f"strategy {name!r} is already registered to {_STRATEGIES[name]!r}")
        cls.name = name
        _STRATEGIES[name] = cls
        return cls

    return decorate


def get_strategy(name: str) -> type[Strategy]:
    """Look up a registered strategy class.

    Raises:
        UnknownStrategy: no strategy is registered under ``name``; the message
            lists what is.
    """
    try:
        return _STRATEGIES[name]
    except KeyError:
        raise UnknownStrategy(
            f"unknown strategy {name!r}; available: {', '.join(available()) or 'none'}"
        ) from None


def available() -> list[str]:
    """Every registered strategy name, sorted."""
    return sorted(_STRATEGIES)
