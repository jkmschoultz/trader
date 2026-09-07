"""Symbol resolution and instrument reference data."""

from __future__ import annotations

import httpx
import pytest

from trader.config import Settings
from trader.saxo.client import SaxoClient
from trader.saxo.instruments import (
    AmbiguousInstrument,
    Instrument,
    InstrumentNotFound,
    get_details,
    resolve,
    search,
)

APPLE_STOCK = {
    "AssetType": "Stock",
    "CurrencyCode": "USD",
    "Description": "Apple Inc.",
    "ExchangeId": "NASDAQ",
    "Identifier": 211,
    "Symbol": "AAPL:xnas",
    "TradableAs": ["Stock", "CfdOnStock"],
}
APPLE_CFD = {**APPLE_STOCK, "AssetType": "CfdOnStock", "Identifier": 15334}


class _StubAuth:
    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        return None


def _client(handler) -> SaxoClient:
    return SaxoClient(Settings(), auth=_StubAuth(), transport=httpx.MockTransport(handler))


def _returning(*items):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Data": list(items)})

    return handler


# ---------------------------------------------------------------- the Uic key


def test_search_reads_the_uic_from_identifier():
    """Search calls it Identifier; details calls the same number Uic."""
    assert Instrument.model_validate(APPLE_STOCK).uic == 211


def test_the_same_model_also_reads_a_uic_key():
    assert Instrument.model_validate({"Uic": 211, "Symbol": "AAPL:xnas"}).uic == 211


# -------------------------------------------------------------------- search


async def test_search_passes_keywords_asset_types_and_top():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Data": [APPLE_STOCK]})

    async with _client(handler) as client:
        await search(client, "AAPL", asset_types=("Stock", "Etf"), limit=5)

    assert captured["Keywords"] == "AAPL"
    assert captured["AssetTypes"] == "Stock,Etf"
    assert captured["$top"] == "5"


async def test_search_without_asset_types_omits_the_filter():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Data": []})

    async with _client(handler) as client:
        await search(client, "AAPL", asset_types=None)

    assert "AssetTypes" not in captured


async def test_search_returns_an_empty_list_when_nothing_matches():
    async with _client(_returning()) as client:
        assert await search(client, "zzzz") == []


# ------------------------------------------------------------------- resolve


async def test_resolve_returns_the_single_exact_match():
    async with _client(_returning(APPLE_STOCK)) as client:
        instrument = await resolve(client, "AAPL:xnas")

    assert instrument.uic == 211
    assert instrument.exchange_id == "NASDAQ"


async def test_resolve_prefers_an_exact_symbol_over_fuzzy_neighbours():
    """Saxo's keyword search is fuzzy; a typo must not resolve to another company."""
    neighbour = {
        **APPLE_STOCK,
        "Symbol": "AAPLX:xnas",
        "Identifier": 999,
        "Description": "Something Else",
    }

    async with _client(_returning(neighbour, APPLE_STOCK)) as client:
        instrument = await resolve(client, "AAPL:xnas")

    assert instrument.uic == 211


async def test_resolve_refuses_to_choose_between_asset_types():
    """A stock and a CFD on it are different Uics with different histories."""
    async with _client(_returning(APPLE_STOCK, APPLE_CFD)) as client:
        with pytest.raises(AmbiguousInstrument) as excinfo:
            await resolve(client, "AAPL:xnas")

    assert {c.uic for c in excinfo.value.candidates} == {211, 15334}
    assert "CfdOnStock" in str(excinfo.value)


async def test_naming_the_asset_type_resolves_the_ambiguity():
    async with _client(_returning(APPLE_STOCK, APPLE_CFD)) as client:
        instrument = await resolve(client, "AAPL:xnas", asset_type="CfdOnStock")

    assert instrument.uic == 15334


async def test_resolve_falls_back_to_a_fuzzy_hit_for_a_bare_ticker():
    """ "AAPL" is not the Saxo symbol, but it should still find AAPL:xnas."""
    async with _client(_returning(APPLE_STOCK)) as client:
        assert (await resolve(client, "AAPL")).uic == 211


async def test_resolve_raises_when_nothing_matches():
    async with _client(_returning()) as client:
        with pytest.raises(InstrumentNotFound):
            await resolve(client, "NOSUCHTICKER")


async def test_resolve_raises_when_the_asset_type_excludes_every_match():
    async with _client(_returning(APPLE_STOCK)) as client:
        with pytest.raises(InstrumentNotFound, match="as FxSpot"):
            await resolve(client, "AAPL:xnas", asset_type="FxSpot")


# ------------------------------------------------------------------- details


async def test_get_details_maps_the_trading_increments():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/ref/v1/instruments/details/211/Stock")
        return httpx.Response(
            200,
            json={
                "Uic": 211,
                "Symbol": "AAPL:xnas",
                "AssetType": "Stock",
                "Description": "Apple Inc.",
                "CurrencyCode": "USD",
                "IsTradable": True,
                "TradingStatus": "Tradable",
                "TickSize": 0.01,
                "LotSize": 1,
                "MinimumTradeSize": 1,
                "Exchange": {"ExchangeId": "NASDAQ", "Name": "NASDAQ", "CountryCode": "US"},
            },
        )

    async with _client(handler) as client:
        details = await get_details(client, 211, "Stock")

    assert details.uic == 211
    assert details.tick_size == 0.01
    assert details.exchange_id == "NASDAQ"
    assert details.is_tradable


async def test_details_keeps_the_raw_payload_for_tick_size_schemes():
    """Some instruments price in bands rather than a single tick size."""
    scheme = {"DefaultTickSize": 0.05, "Elements": [{"HighPrice": 10, "TickSize": 0.01}]}

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"Uic": 5, "AssetType": "Stock", "TickSizeScheme": scheme},
        )

    async with _client(handler) as client:
        details = await get_details(client, 5, "Stock")

    assert details.tick_size is None
    assert details.raw["TickSizeScheme"] == scheme


async def test_unknown_detail_fields_are_ignored():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Uic": 5, "AssetType": "Stock", "BrandNewField": 1})

    async with _client(handler) as client:
        assert (await get_details(client, 5, "Stock")).uic == 5
