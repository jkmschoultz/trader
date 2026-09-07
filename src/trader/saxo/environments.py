"""Saxo OpenAPI environment host definitions.

Two environments exist: ``sim`` (simulation, funded with fake money) and ``live``
(real money). Every host differs between them, so they are grouped here rather
than assembled ad hoc at call sites -- mixing a live gateway with a sim token is
a failure mode worth making structurally hard.

``sim`` is the default everywhere in this codebase. Nothing selects ``live``
implicitly; see ``docs/live-trading.md``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import NamedTuple


class Environment(StrEnum):
    SIM = "sim"
    LIVE = "live"


class Hosts(NamedTuple):
    """Base URLs for one Saxo environment.

    Attributes:
        gateway: REST base, e.g. ``.../openapi``. Service-group paths such as
            ``/port/v1/balances`` are appended to this.
        authorize: OAuth2 authorization endpoint (browser redirect target).
        token: OAuth2 token endpoint (code exchange and refresh).
        streaming: WebSocket base for streaming subscriptions.
    """

    gateway: str
    authorize: str
    token: str
    streaming: str


_SIM = Hosts(
    gateway="https://gateway.saxobank.com/sim/openapi",
    authorize="https://sim.logonvalidation.net/authorize",
    token="https://sim.logonvalidation.net/token",
    streaming="wss://streaming.saxotrader.com/sim/openapi/streamingws/connect",
)

_LIVE = Hosts(
    gateway="https://gateway.saxobank.com/openapi",
    authorize="https://live.logonvalidation.net/authorize",
    token="https://live.logonvalidation.net/token",
    streaming="wss://streaming.saxotrader.com/openapi/streamingws/connect",
)

_HOSTS: dict[Environment, Hosts] = {
    Environment.SIM: _SIM,
    Environment.LIVE: _LIVE,
}


def hosts_for(environment: Environment) -> Hosts:
    """Return the host set for ``environment``."""
    return _HOSTS[Environment(environment)]
