"""Exchange reference data and session parsing."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from trader.config import Settings
from trader.saxo.client import SaxoClient
from trader.saxo.exchanges import (
    REGULAR_STATES,
    Exchange,
    ExchangeSession,
    get_exchange,
    list_exchanges,
)

# Shaped like the real SIM response (verified against /ref/v1/exchanges/NASDAQ):
# no TimeZoneId anywhere -- a small integer TimeZone code plus an abbreviation
# and the zone's *current* offset instead.
NASDAQ = {
    "ExchangeId": "NASDAQ",
    "Name": "NASDAQ",
    "CountryCode": "US",
    "Currency": "USD",
    "TimeZone": 3,
    "TimeZoneAbbreviation": "EDT",
    "TimeZoneOffset": "-04:00:00",
    "AllDay": False,
    "ExchangeSessions": [
        {
            "State": "PreMarket",
            "StartTime": "2024-03-01T09:00:00Z",
            "EndTime": "2024-03-01T14:30:00Z",
        },
        {
            "State": "AutomatedTrading",
            "StartTime": "2024-03-01T14:30:00Z",
            "EndTime": "2024-03-01T21:00:00Z",
        },
    ],
}


class _StubAuth:
    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        return None


def _client(handler) -> SaxoClient:
    return SaxoClient(Settings(), auth=_StubAuth(), transport=httpx.MockTransport(handler))


def test_sessions_parse_into_utc_instants():
    exchange = Exchange.model_validate(NASDAQ)
    assert len(exchange.sessions) == 2
    assert exchange.sessions[1].start == datetime(2024, 3, 1, 14, 30, tzinfo=UTC)
    assert exchange.sessions[1].end == datetime(2024, 3, 1, 21, 0, tzinfo=UTC)


def test_only_continuous_trading_counts_as_regular():
    """Pre-trading and post-trading break a backtest's fill assumptions."""
    exchange = Exchange.model_validate(NASDAQ)
    assert not exchange.sessions[0].is_regular
    assert exchange.sessions[1].is_regular
    assert REGULAR_STATES == {"AutomatedTrading", "CallAuctionTrading"}


def test_the_real_saxo_timezone_fields_parse():
    """Saxo has no TimeZoneId field -- only a code, an abbreviation, and an offset."""
    exchange = Exchange.model_validate(NASDAQ)
    assert exchange.time_zone_code == 3
    assert exchange.time_zone_abbreviation == "EDT"
    assert exchange.time_zone_offset == "-04:00:00"


def test_a_session_is_half_open_at_its_end():
    """Adjacent sessions share a boundary; it must belong to exactly one."""
    session = ExchangeSession.model_validate(
        {
            "State": "AutomatedTrading",
            "StartTime": "2024-03-01T14:30:00Z",
            "EndTime": "2024-03-01T21:00:00Z",
        }
    )
    assert session.contains(datetime(2024, 3, 1, 14, 30, tzinfo=UTC))
    assert not session.contains(datetime(2024, 3, 1, 21, 0, tzinfo=UTC))


def test_an_all_day_venue_reports_no_sessions():
    exchange = Exchange.model_validate(
        {"ExchangeId": "SBFX", "AllDay": True, "ExchangeSessions": []}
    )
    assert exchange.all_day
    assert exchange.sessions == []


async def test_list_exchanges_unwraps_the_data_envelope():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Data": [NASDAQ, {"ExchangeId": "LSE_SETS"}]})

    async with _client(handler) as client:
        exchanges = await list_exchanges(client)

    assert [e.exchange_id for e in exchanges] == ["NASDAQ", "LSE_SETS"]
    assert captured["$top"] == "1000", "the default page size hides most exchanges"


async def test_get_exchange_addresses_one_exchange_by_id():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/ref/v1/exchanges/NASDAQ")
        return httpx.Response(200, json=NASDAQ)

    async with _client(handler) as client:
        exchange = await get_exchange(client, "NASDAQ")

    assert exchange.name == "NASDAQ"
    assert exchange.country_code == "US"


async def test_unknown_exchange_fields_are_ignored():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**NASDAQ, "SomeNewFieldSaxoAdded": 1})

    async with _client(handler) as client:
        assert (await get_exchange(client, "NASDAQ")).exchange_id == "NASDAQ"
