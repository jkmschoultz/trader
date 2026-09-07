"""Account, user, and balance queries -- the ``port`` service group.

Phase 0 needs only enough of this to prove end-to-end connectivity. The
``AccountKey`` fetched here is a required parameter on most trading and
portfolio calls later, so it is resolved and cached once.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from trader.saxo.client import SaxoClient


class UserInfo(BaseModel):
    user_id: str = Field(alias="UserId")
    name: str = Field(default="", alias="Name")
    client_key: str = Field(alias="ClientKey")
    culture: str = Field(default="", alias="Culture")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class AccountInfo(BaseModel):
    account_key: str = Field(alias="AccountKey")
    account_id: str = Field(alias="AccountId")
    currency: str = Field(default="", alias="Currency")
    account_type: str = Field(default="", alias="AccountType")
    active: bool = Field(default=True, alias="Active")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class Balance(BaseModel):
    currency: str = Field(default="", alias="Currency")
    cash_balance: float = Field(default=0.0, alias="CashBalance")
    total_value: float = Field(default=0.0, alias="TotalValue")
    margin_available: float = Field(default=0.0, alias="MarginAvailableForTrading")
    unrealized_pl: float = Field(default=0.0, alias="UnrealizedPositionsValue")

    model_config = {"populate_by_name": True, "extra": "ignore"}


async def get_user(client: SaxoClient) -> UserInfo:
    """Return the authenticated user."""
    return UserInfo.model_validate(await client.get("/port/v1/users/me"))


async def list_accounts(client: SaxoClient) -> list[AccountInfo]:
    """Return every account visible to the authenticated user."""
    payload = await client.get("/port/v1/accounts/me")
    return [AccountInfo.model_validate(item) for item in payload.get("Data", [])]


async def get_balance(
    client: SaxoClient,
    *,
    client_key: str,
    account_key: str | None = None,
) -> Balance:
    """Return the balance for one account, or the client-level total.

    ``client_key`` is required even when narrowing to a single account: Saxo
    rejects ``AccountKey`` on its own with a 400 ``InvalidModelState``, despite
    the reference docs presenting the two keys as alternatives.

    Args:
        client_key: from :func:`get_user`.
        account_key: narrows to one account; omit for the client-level total.
    """
    payload = await client.get("/port/v1/balances", ClientKey=client_key, AccountKey=account_key)
    return Balance.model_validate(payload)


async def get_my_balance(client: SaxoClient) -> Balance:
    """Return the client-level balance without needing any keys."""
    return Balance.model_validate(await client.get("/port/v1/balances/me"))
