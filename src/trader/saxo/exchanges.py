"""Exchange reference data and trading sessions -- the ``ref`` service group.

Saxo publishes each exchange's sessions as a list of ``(State, StartTime,
EndTime)`` windows in UTC. This module is the raw API surface; the calendar
built on top of it lives in ``trader.data.calendars``.

One limitation shapes everything downstream: **the session list Saxo returns is
a short rolling window** around now, not a historical calendar. It answers "is
NASDAQ open right now, and when does it next open" -- which is what a live
session needs -- and cannot answer "was 4 July a trading day in 2021". Historical
session structure is recovered from the bar data instead; see
``trader.data.calendars.infer_sessions``.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from trader.saxo.client import SaxoClient

# Saxo's session states. Only these two carry continuous two-sided trading;
# the rest are auctions, breaks, and pre/post windows where a backtest's fill
# assumptions do not hold.
REGULAR_STATES = frozenset({"AutomatedTrading", "CallAuctionTrading"})

# Every state observed across a full /ref/v1/exchanges listing (SIM, 266
# exchanges), for reference when reading a raw session list. Not exhaustive --
# treat an unrecognised state as informational, not an error; only membership in
# REGULAR_STATES is load-bearing.
KNOWN_STATES = REGULAR_STATES | {
    "Closed",
    "PreTrading",
    "PostTrading",
    "PreMarket",  # NASDAQ/AMEX/NYSE label their pre/post windows this way
    "PostMarket",  # rather than PreTrading/PostAutomatedTrading.
    "PostAutomatedTrading",
    "TradingAtLast",
    "OpeningAuction",
    "Auction",
    "TradingHalt",
    "Break",
    "Suspended",
}


class ExchangeSession(BaseModel):
    """One ``State`` window on an exchange, in UTC."""

    state: str = Field(default="", alias="State")
    start: datetime = Field(alias="StartTime")
    end: datetime = Field(alias="EndTime")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def is_regular(self) -> bool:
        """True for continuous trading, false for auctions, breaks, and closes."""
        return self.state in REGULAR_STATES

    def contains(self, moment: datetime) -> bool:
        """True when ``moment`` falls in ``[start, end)``."""
        return self.start <= moment < self.end


class Exchange(BaseModel):
    """An exchange and its currently published sessions."""

    exchange_id: str = Field(alias="ExchangeId")
    name: str = Field(default="", alias="Name")
    country_code: str = Field(default="", alias="CountryCode")
    currency: str = Field(default="", alias="Currency")
    # Saxo has no IANA or Windows zone name anywhere in this response -- checked
    # against a full /ref/v1/exchanges listing. What it gives instead is a small
    # internal integer code, a 3-4 letter abbreviation ("EDT", "CEST"), and the
    # zone's *current* UTC offset (already reflecting today's DST state, so it
    # silently goes stale across a DST transition). trader.data.calendars maps
    # the code to a real IANA zone for the handful this project trades, and
    # falls back to the offset -- wrong across DST, right today -- for the rest.
    time_zone_code: int | None = Field(default=None, alias="TimeZone")
    time_zone_abbreviation: str = Field(default="", alias="TimeZoneAbbreviation")
    time_zone_offset: str = Field(default="", alias="TimeZoneOffset")
    all_day: bool = Field(default=False, alias="AllDay")
    sessions: list[ExchangeSession] = Field(default_factory=list, alias="ExchangeSessions")

    model_config = {"populate_by_name": True, "extra": "ignore"}


async def list_exchanges(client: SaxoClient) -> list[Exchange]:
    """Return every exchange Saxo exposes, with sessions."""
    payload = await client.get("/ref/v1/exchanges", **{"$top": 1000})
    return [Exchange.model_validate(item) for item in payload.get("Data", [])]


async def get_exchange(client: SaxoClient, exchange_id: str) -> Exchange:
    """Return one exchange by its Saxo ``ExchangeId`` (e.g. ``NASDAQ``)."""
    return Exchange.model_validate(await client.get(f"/ref/v1/exchanges/{exchange_id}"))
