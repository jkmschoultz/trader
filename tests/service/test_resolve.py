"""resolve_symbol: the bare-ticker exchange tiebreak for the UI."""

from __future__ import annotations

import httpx
import pytest
import respx

from trader.config import Settings
from trader.saxo.client import SaxoClient
from trader.saxo.instruments import InstrumentNotFound
from trader.service.errors import InvalidRequest
from trader.service.instruments import resolve_symbol

GATEWAY = "https://gateway.saxobank.com/sim/openapi"


def _hit(symbol: str, uic: int, exchange: str = "") -> dict:
    return {
        "Symbol": symbol,
        "Identifier": uic,
        "AssetType": "Stock",
        "ExchangeId": exchange,
        "Description": "Apple Inc.",
    }


AAPL_LISTINGS = [
    _hit("AAPL:xnas", 211, "NASDAQ"),
    _hit("AAPL:xmil", 46521, "MILAN"),
    _hit("AAPL:xams", 46520, "AMSTERDAM"),
]


def _client(tmp_path) -> SaxoClient:
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state", saxo={"token_24h": "tok"})
    return SaxoClient(settings)


@respx.mock
async def test_bare_ticker_picks_the_first_preferred_exchange(tmp_path):
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": AAPL_LISTINGS})
    )
    async with _client(tmp_path) as client:
        hit = await resolve_symbol(client, "AAPL", asset_type="Stock", prefer=["xnas", "xnys"])
    assert hit.symbol == "AAPL:xnas"
    assert hit.uic == 211


@respx.mock
async def test_bare_ticker_falls_through_to_the_next_preference(tmp_path):
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": AAPL_LISTINGS})
    )
    async with _client(tmp_path) as client:
        hit = await resolve_symbol(client, "AAPL", asset_type="Stock", prefer=["xtks", "xmil"])
    assert hit.symbol == "AAPL:xmil"


@respx.mock
async def test_no_preference_matches_reports_the_candidates(tmp_path):
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": AAPL_LISTINGS})
    )
    async with _client(tmp_path) as client:
        with pytest.raises(InvalidRequest) as exc:
            await resolve_symbol(client, "AAPL", asset_type="Stock", prefer=["xtks"])
    assert "AAPL:xnas" in str(exc.value) and "AAPL:xmil" in str(exc.value)


@respx.mock
async def test_an_explicit_venue_is_not_second_guessed(tmp_path):
    # Saxo returns two rows even for the exact symbol; resolve() would raise
    # AmbiguousInstrument, and resolve_symbol must not silently pick one.
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(
            200, json={"Data": [_hit("AAPL:xmil", 46521), _hit("AAPL:xmil", 99999)]}
        )
    )
    async with _client(tmp_path) as client:
        with pytest.raises(InvalidRequest):
            await resolve_symbol(client, "AAPL:xmil", asset_type="Stock", prefer=["xnas"])


@respx.mock
async def test_nothing_matches_is_instrument_not_found(tmp_path):
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": []})
    )
    async with _client(tmp_path) as client:
        with pytest.raises(InstrumentNotFound):
            await resolve_symbol(client, "ZZZZ", asset_type="Stock", prefer=["xnas"])
