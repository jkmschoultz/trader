"""Portfolio queries and their parameter contracts."""

from __future__ import annotations

import httpx

from trader.config import Settings
from trader.saxo.accounts import get_balance, get_my_balance, get_user, list_accounts
from trader.saxo.client import SaxoClient


class _StubAuth:
    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        return None


def _client(handler) -> SaxoClient:
    return SaxoClient(Settings(), auth=_StubAuth(), transport=httpx.MockTransport(handler))


async def test_get_user_maps_pascal_case_fields():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"UserId": "22693120", "Name": "Test User", "ClientKey": "ck==", "Culture": "en"},
        )

    async with _client(handler) as client:
        user = await get_user(client)

    assert user.user_id == "22693120"
    assert user.client_key == "ck=="


async def test_list_accounts_unwraps_the_data_envelope():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "Data": [
                    {
                        "AccountKey": "ak==",
                        "AccountId": "123",
                        "Currency": "EUR",
                        "AccountType": "Normal",
                        "Active": True,
                    }
                ]
            },
        )

    async with _client(handler) as client:
        accounts = await list_accounts(client)

    assert len(accounts) == 1
    assert accounts[0].account_id == "123"
    assert accounts[0].currency == "EUR"


async def test_balance_always_sends_client_key():
    """Saxo 400s on AccountKey alone, despite the docs calling the keys alternatives."""
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Currency": "EUR", "CashBalance": 1000.0})

    async with _client(handler) as client:
        await get_balance(client, client_key="ck==", account_key="ak==")

    assert captured["ClientKey"] == "ck=="
    assert captured["AccountKey"] == "ak=="


async def test_client_level_balance_omits_account_key():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Currency": "EUR", "CashBalance": 1000.0})

    async with _client(handler) as client:
        await get_balance(client, client_key="ck==")

    assert captured["ClientKey"] == "ck=="
    assert "AccountKey" not in captured


async def test_my_balance_needs_no_keys():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/balances/me")
        assert not request.url.params
        return httpx.Response(
            200, json={"Currency": "EUR", "CashBalance": 1_000_000.0, "TotalValue": 999_989.41}
        )

    async with _client(handler) as client:
        balance = await get_my_balance(client)

    assert balance.cash_balance == 1_000_000.0
    assert balance.total_value == 999_989.41


async def test_unknown_fields_are_ignored():
    """Saxo adds response fields over time; that must not break parsing."""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"Currency": "EUR", "CashBalance": 5.0, "SomeNewFieldSaxoAdded": 1}
        )

    async with _client(handler) as client:
        balance = await get_my_balance(client)

    assert balance.cash_balance == 5.0
