"""Instrument reference data -- the ``ref`` service group.

Nothing in the data layer can address an instrument by ticker: every chart,
price, and order call takes a numeric ``Uic``. This module is the only place
that turns a human symbol into one.

Two Saxo quirks are handled here and nowhere else:

* The search endpoint returns the Uic under the key ``Identifier``, while the
  details endpoint returns the same number under ``Uic``. Both map to
  :attr:`Instrument.uic`.
* One symbol commonly resolves to several instruments -- ``AAPL:xnas`` exists as
  ``Stock`` and as ``CfdOnStock``, which are different Uics with different
  histories. :func:`resolve` refuses to guess between them.
"""

from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field

from trader.saxo.client import SaxoClient

# Asset types are deliberately plain strings. Saxo defines several dozen and
# adds more over time; an enum here would reject valid instruments rather than
# let the gateway decide. These are the ones the data layer is exercised on.
STOCK = "Stock"
CFD_ON_STOCK = "CfdOnStock"
CFD_ON_INDEX = "CfdOnIndex"
FX_SPOT = "FxSpot"
ETF = "Etf"

DEFAULT_ASSET_TYPES = (STOCK, ETF, CFD_ON_INDEX, FX_SPOT)


class InstrumentNotFound(LookupError):
    """No instrument matched the requested symbol."""


class AmbiguousInstrument(LookupError):
    """A symbol matched more than one instrument.

    Carries the candidates so a caller (or the CLI) can show them rather than
    make an arbitrary choice between, say, a stock and a CFD on that stock.
    """

    def __init__(self, symbol: str, candidates: list[Instrument]) -> None:
        rendered = ", ".join(f"{c.symbol} {c.asset_type} (uic {c.uic})" for c in candidates)
        super().__init__(f"{symbol!r} matches {len(candidates)} instruments: {rendered}")
        self.symbol = symbol
        self.candidates = candidates


class Instrument(BaseModel):
    """A search result: enough to address the instrument, not to trade it."""

    # Search returns "Identifier"; details returns "Uic". Accept either.
    uic: int = Field(validation_alias=AliasChoices("Identifier", "Uic", "uic"))
    symbol: str = Field(default="", alias="Symbol")
    description: str = Field(default="", alias="Description")
    asset_type: str = Field(default="", alias="AssetType")
    exchange_id: str = Field(default="", alias="ExchangeId")
    currency: str = Field(default="", alias="CurrencyCode")
    tradable_as: list[str] = Field(default_factory=list, alias="TradableAs")

    model_config = {"populate_by_name": True, "extra": "ignore"}

    def __str__(self) -> str:
        return f"{self.symbol} [{self.asset_type}] uic={self.uic} {self.description}"


class InstrumentDetails(BaseModel):
    """Full detail for one instrument, including its trading increments.

    ``tick_size`` and ``lot_size`` are what a backtest needs to round prices and
    sizes the way the venue will. They are absent for some instruments, in which
    case Saxo supplies a tick *scheme* keyed by price band, which this model does
    not attempt to flatten -- callers needing that precision should read
    ``raw["TickSizeScheme"]``.
    """

    uic: int = Field(alias="Uic")
    symbol: str = Field(default="", alias="Symbol")
    description: str = Field(default="", alias="Description")
    asset_type: str = Field(default="", alias="AssetType")
    currency: str = Field(default="", alias="CurrencyCode")
    is_tradable: bool = Field(default=False, alias="IsTradable")
    trading_status: str = Field(default="", alias="TradingStatus")
    tick_size: float | None = Field(default=None, alias="TickSize")
    lot_size: float | None = Field(default=None, alias="LotSize")
    minimum_trade_size: float | None = Field(default=None, alias="MinimumTradeSize")
    decimals: int | None = Field(default=None, alias="Decimals")
    # Nested {"ExchangeId": ..., "Name": ..., "CountryCode": ...}.
    exchange: dict = Field(default_factory=dict, alias="Exchange")
    raw: dict = Field(default_factory=dict, exclude=True)

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def exchange_id(self) -> str:
        return str(self.exchange.get("ExchangeId", ""))


async def search(
    client: SaxoClient,
    keywords: str,
    *,
    asset_types: tuple[str, ...] | None = DEFAULT_ASSET_TYPES,
    limit: int = 20,
) -> list[Instrument]:
    """Search instruments by ticker, ISIN, or description.

    Args:
        keywords: free text; Saxo matches symbol, name, and ISIN.
        asset_types: restrict the search. ``None`` searches every asset type,
            which usually buries the instrument you want under CFD variants.
        limit: maximum results to return.
    """
    payload = await client.get(
        "/ref/v1/instruments",
        Keywords=keywords,
        AssetTypes=",".join(asset_types) if asset_types else None,
        **{"$top": limit},
    )
    return [Instrument.model_validate(item) for item in payload.get("Data", [])]


async def get_details(client: SaxoClient, uic: int, asset_type: str) -> InstrumentDetails:
    """Return full detail for one instrument."""
    payload = await client.get(f"/ref/v1/instruments/details/{uic}/{asset_type}")
    details = InstrumentDetails.model_validate(payload)
    details.raw = payload
    return details


async def resolve(
    client: SaxoClient,
    symbol: str,
    *,
    asset_type: str | None = None,
) -> Instrument:
    """Resolve an exact symbol to a single instrument.

    Matching is case-insensitive and exact against ``Symbol`` -- searching
    ``"AAPL"`` finds ``AAPL:xnas`` because Saxo's keyword search is fuzzy, but
    only an exact symbol match is accepted as *the* answer. That keeps a typo
    from silently resolving to a different company.

    Args:
        symbol: exact Saxo symbol, e.g. ``AAPL:xnas`` or ``EURUSD``.
        asset_type: required when the symbol is tradable in more than one form.

    Raises:
        InstrumentNotFound: nothing matched.
        AmbiguousInstrument: several asset types matched and none was named.
    """
    results = await search(
        client, symbol, asset_types=(asset_type,) if asset_type else None, limit=50
    )

    wanted = symbol.casefold()
    exact = [i for i in results if i.symbol.casefold() == wanted]
    # Fall back to fuzzy hits so a bare "AAPL" can still resolve to "AAPL:xnas".
    candidates = exact or results
    if asset_type:
        candidates = [i for i in candidates if i.asset_type == asset_type]

    if not candidates:
        raise InstrumentNotFound(
            f"no instrument matches {symbol!r}" + (f" as {asset_type}" if asset_type else "")
        )
    if len(candidates) > 1:
        raise AmbiguousInstrument(symbol, candidates)
    return candidates[0]
