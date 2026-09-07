"""GET /api/instruments/search: hit mapping, query forwarding, error paths."""

from __future__ import annotations

import httpx
import pytest
import respx
from starlette.testclient import TestClient

from trader.api import create_app
from trader.config import Settings

GATEWAY = "https://gateway.saxobank.com/sim/openapi"

APPLE = {
    "AssetType": "Stock",
    "CurrencyCode": "USD",
    "Description": "Apple Inc.",
    "ExchangeId": "NASDAQ",
    "Identifier": 211,
    "Symbol": "AAPL:xnas",
}
APPLE_CFD = {**APPLE, "AssetType": "CfdOnStock", "Identifier": 15334}


@pytest.fixture
def authed_client(tmp_path) -> TestClient:
    """A client whose settings carry a 24h token, so auth needs no network."""
    settings = Settings(
        data_dir=tmp_path,
        state_dir=tmp_path / "state",
        saxo={"token_24h": "tok-for-tests"},
    )
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@respx.mock
def test_search_maps_saxo_hits(authed_client):
    route = respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": [APPLE, APPLE_CFD]})
    )
    body = authed_client.get("/api/instruments/search", params={"q": "AAPL"}).json()

    assert route.called
    assert [h["uic"] for h in body] == [211, 15334]
    assert body[0] == {
        "symbol": "AAPL:xnas",
        "uic": 211,
        "asset_type": "Stock",
        "exchange_id": "NASDAQ",
        "description": "Apple Inc.",
    }


@respx.mock
def test_search_forwards_the_query_asset_type_and_limit(authed_client):
    route = respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": []})
    )
    authed_client.get(
        "/api/instruments/search",
        params={"q": "nvda", "asset_type": "Stock", "limit": 5},
    )

    sent = route.calls.last.request
    assert sent.url.params["Keywords"] == "nvda"
    assert sent.url.params["AssetTypes"] == "Stock"
    assert sent.url.params["$top"] == "5"


@respx.mock
def test_search_upstream_error_is_502(authed_client):
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(403, json={"Message": "forbidden"})
    )
    resp = authed_client.get("/api/instruments/search", params={"q": "AAPL"})
    assert resp.status_code == 502


def test_search_without_a_session_is_409(client):
    # The shared `client` fixture has no token and an empty state dir, so the
    # Saxo client cannot mint an access token -> ReauthRequired -> 409.
    resp = client.get("/api/instruments/search", params={"q": "AAPL"})
    assert resp.status_code == 409
    assert "trader auth login" in resp.json()["detail"]
