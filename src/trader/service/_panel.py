"""Load a set of instrument bar series from the lake, ready to run something over.

Extracted from :mod:`trader.service.backtest` so features, labels, and training
share exactly the symbol resolution, lake read, ``since`` clip, and session
inference a backtest already does. The one step that needs the network is
resolving a bare symbol to a Uic; pass the Uic to skip it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from trader.config import Settings
from trader.service.errors import SeriesNotStored

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from trader.data.calendars import RegularHours
    from trader.data.lake import SeriesKey
    from trader.saxo.client import SaxoClient

log = logging.getLogger(__name__)

__all__ = ["PanelSeries", "load_panel"]


@dataclass(frozen=True)
class PanelSeries:
    """One resolved instrument: its lake key, bars, and inferred session hours."""

    label: str
    key: SeriesKey
    frame: pd.DataFrame
    session: RegularHours | None
    exchange_id: str | None
    #: the engine's sizing unit; fractional for crypto, 1 unit elsewhere
    lot_size: float = 1.0


async def load_panel(
    settings: Settings,
    *,
    symbols: Sequence[str],
    uics: Sequence[int] = (),
    asset_type: str | Sequence[str] | None = None,
    exchange: str | None = None,
    horizon: int,
    since: datetime | None = None,
) -> list[PanelSeries]:
    """Resolve every symbol, read its bars in ``[since, ...]``, infer a calendar.

    ``symbols`` and ``uics`` pair up positionally; a missing Uic is resolved over
    the network, which is also the only reason a :class:`~trader.saxo.client.SaxoClient`
    is opened.

    Raises:
        SeriesNotStored: the lake holds nothing for one of the symbols in range.
        trader.saxo.instruments.InstrumentNotFound: a symbol matched nothing.
        trader.service.errors.InvalidRequest: a bare ticker stayed ambiguous
            after the preferred-exchange tiebreak.
    """
    from trader.data import bars as bars_mod
    from trader.data import calendars
    from trader.data.lake import BarLake, SeriesKey
    from trader.saxo.charts import horizon_label
    from trader.saxo.client import SaxoAPIError, SaxoClient
    from trader.saxo.exchanges import get_exchange
    from trader.service.instruments import resolve_symbol

    store = BarLake(settings.data_dir)
    padded_uics: list[int | None] = list(uics) + [None] * (len(symbols) - len(uics))
    types = _per_symbol_types(asset_type, len(symbols))
    # instruments from outside Saxo (Coinbase crypto) resolve from the local registry
    padded_uics = [
        uic if uic is not None else _local_uic(settings, symbol, atype)
        for symbol, uic, atype in zip(symbols, padded_uics, types, strict=True)
    ]
    need_network = any(uic is None for uic in padded_uics)

    out: list[PanelSeries] = []

    def _read(label: str, key: SeriesKey) -> pd.DataFrame:
        frame = store.read(key)
        if since is not None:
            frame = bars_mod.clip(frame, start=since)
        frame, cut = bars_mod.after_last_long_gap(frame)
        if cut is not None:
            log.warning("%s: a gap of over 30 days; using bars from %s only", label, cut)
        if frame.empty:
            raise SeriesNotStored(
                label,
                horizon_label(horizon),
                asset_type=key.asset_type,
                uic=key.uic,
                since=since.isoformat() if since else None,
            )
        return frame

    async def _add(
        symbol: str, uic: int | None, asset_type: str | None, client: SaxoClient | None
    ) -> None:
        exchange_id: str | None = None
        if uic is None:
            assert client is not None  # need_network implies a client is open
            instrument = await resolve_symbol(
                client,
                symbol,
                asset_type=asset_type,
                prefer=[e for e in (exchange, *settings.saxo.preferred_exchanges) if e],
            )
            label, uic, resolved_type = instrument.symbol, instrument.uic, instrument.asset_type
            exchange_id = instrument.exchange_id
        else:
            label, resolved_type = symbol, (asset_type or "Stock")

        key = SeriesKey(resolved_type, int(uic), horizon)
        frame = _read(label, key)

        session = None
        if client is not None and exchange_id:
            try:
                exch = await get_exchange(client, exchange_id)
                session = calendars.infer_regular_hours(frame, calendars.resolve_timezone(exch))
            except SaxoAPIError as exc:
                log.warning("no session calendar for %s: %s", label, exc)

        from trader.data import coinbase

        lot = coinbase.LOT_SIZE if key.asset_type == coinbase.ASSET_TYPE else 1.0
        out.append(PanelSeries(label, key, frame, session, exchange_id, lot))

    if need_network:
        async with SaxoClient(settings) as client:
            for symbol, uic, atype in zip(symbols, padded_uics, types, strict=True):
                await _add(symbol, uic, atype, client)
    else:
        for symbol, uic, atype in zip(symbols, padded_uics, types, strict=True):
            await _add(symbol, uic, atype, None)

    return out


def _per_symbol_types(asset_type: str | Sequence[str] | None, n: int) -> list[str | None]:
    """One asset type per symbol: a single value (or a one-item list) applies to all,
    a list pairs positionally -- a mixed basket of ETFs, index CFDs, FX and crypto.
    """
    from trader.service.errors import InvalidRequest

    if asset_type is None or isinstance(asset_type, str):
        return [asset_type] * n
    types = list(asset_type)
    if len(types) == 1:
        return types * n
    if len(types) != n:
        raise InvalidRequest(
            f"{len(types)} asset types for {n} symbols; give one for all, or one per symbol"
        )
    return types


def _local_uic(settings: Settings, symbol: str, asset_type: str | None) -> int | None:
    """The pseudo-Uic of a non-Saxo instrument (asset type ``Crypto``), or None.

    Saxo instruments still resolve over the network; this only answers for
    asset types Saxo does not serve, recorded by their importer.
    """
    from trader.data.coinbase import ASSET_TYPE
    from trader.data.instruments import InstrumentRegistry

    if asset_type != ASSET_TYPE:
        return None
    for record in InstrumentRegistry(settings.data_dir):
        if record.asset_type == ASSET_TYPE and record.symbol.upper() == symbol.upper():
            return record.uic
    from trader.service.errors import InvalidRequest

    raise InvalidRequest(
        f"{symbol} is not in the lake; import it with: trader data crypto {symbol}"
    )
