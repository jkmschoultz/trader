"""Symbol resolution with a bare-ticker convenience for the UI.

``trader.saxo.instruments.resolve`` refuses to guess between ``AAPL:xnas`` and
``AAPL:xmil``. The CLI keeps that strictness; the UI is more forgiving -- typing
``AAPL`` picks the listing on the first matching *preferred exchange*
(``settings.saxo.preferred_exchanges``). An explicit ``AAPL:xmil`` is always
taken as given.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from trader.service.errors import InvalidRequest

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trader.saxo.client import SaxoClient
    from trader.saxo.instruments import Instrument

__all__ = ["resolve_symbol"]


async def resolve_symbol(
    client: SaxoClient,
    symbol: str,
    *,
    asset_type: str | None = None,
    prefer: Sequence[str] = (),
) -> Instrument:
    """Resolve ``symbol`` to one instrument, breaking bare-ticker ties by exchange.

    Raises:
        InvalidRequest: still ambiguous after the preferred-exchange tiebreak.
        trader.saxo.instruments.InstrumentNotFound: nothing matched.
    """
    from trader.saxo.instruments import AmbiguousInstrument, resolve

    try:
        return await resolve(client, symbol, asset_type=asset_type)
    except AmbiguousInstrument as exc:
        if ":" in symbol:  # an explicit venue was given -- do not second-guess it
            raise _as_invalid(symbol, exc.candidates) from exc
        for suffix in prefer:
            tail = f":{suffix.casefold()}"
            hits = [c for c in exc.candidates if c.symbol.casefold().endswith(tail)]
            if len(hits) == 1:
                return hits[0]
        raise _as_invalid(symbol, exc.candidates) from exc


def _as_invalid(symbol: str, candidates: list[Instrument]) -> InvalidRequest:
    lines = "\n".join(
        f"  {c.symbol}  {c.asset_type}  uic={c.uic}  {c.description}" for c in candidates
    )
    return InvalidRequest(
        f"{symbol!r} matches several instruments; give one of these symbols, "
        f"or set asset_type:\n{lines}"
    )
