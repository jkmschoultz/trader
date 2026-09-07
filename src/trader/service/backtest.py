"""Turn a :class:`BacktestSpec` into a :class:`~trader.backtest.result.BacktestResult`.

This is the body of the old ``trader backtest`` command with the argparse and
the ``print`` calls taken out, so the API can run exactly the same thing. It
reads bars from the lake, resolves any symbols that were not given as a Uic
(the only step that needs the network), infers a session calendar per
instrument, and calls :func:`trader.backtest.run`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trader.config import Settings
from trader.service.errors import InvalidRequest, SeriesNotStored
from trader.service.inputs import parse_since

log = logging.getLogger(__name__)

__all__ = ["BacktestSpec", "run_backtest"]


class BacktestSpec(BaseModel):
    """Everything needed to run one backtest.

    ``symbols`` and ``uics`` pair up positionally: give a Uic to skip the
    symbol-resolution network call for that instrument. ``horizon`` accepts a
    label (``"5m"``) or minutes; ``since`` accepts ``"90d"``, ``"2y"``, a date,
    an ISO timestamp, or ``"all"``.
    """

    symbols: list[str] = Field(min_length=1)
    strategy: str
    uics: list[int] = Field(default_factory=list)
    asset_type: str | None = None
    # Preferred exchange suffix (e.g. "xnas") for resolving a bare ticker;
    # ignored when the symbol already names a venue or a uic is given.
    exchange: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    horizon: int = 5
    since: datetime | None = None
    allocator: str = "equal-weight"
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    starting_cash: float = 100_000.0
    leverage: float = 1.0

    @field_validator("horizon", mode="before")
    @classmethod
    def _horizon(cls, value: object) -> int:
        from trader.saxo.charts import ChartError, parse_horizon

        try:
            return parse_horizon(value)  # type: ignore[arg-type]
        except ChartError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("since", mode="before")
    @classmethod
    def _since(cls, value: object) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value  # type: ignore[return-value]
        return parse_since(str(value))

    @field_validator("uics")
    @classmethod
    def _uics_fit(cls, value: list[int], info: Any) -> list[int]:
        symbols = info.data.get("symbols") or []
        if len(value) > len(symbols):
            raise ValueError("more uics than symbols")
        return value


async def run_backtest(
    settings: Settings,
    spec: BacktestSpec,
    *,
    progress: Callable[[float], None] | None = None,
):
    """Run ``spec`` and return a :class:`~trader.backtest.result.BacktestResult`.

    Raises:
        InvalidRequest: unknown strategy or allocator, or bad strategy params.
        SeriesNotStored: the lake has no bars for one of the symbols.
        trader.saxo.instruments.InstrumentNotFound: a symbol matched nothing.
    """
    from trader import backtest as bt
    from trader import strategies
    from trader.data import bars as bars_mod
    from trader.data import calendars
    from trader.data.lake import BarLake, SeriesKey
    from trader.saxo.charts import horizon_label
    from trader.saxo.client import SaxoAPIError, SaxoClient
    from trader.saxo.exchanges import get_exchange
    from trader.service.instruments import resolve_symbol

    try:
        strategy = strategies.get_strategy(spec.strategy)(**spec.params)
    except strategies.UnknownStrategy as exc:
        raise InvalidRequest(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise InvalidRequest(f"bad params for strategy {spec.strategy!r}: {exc}") from exc

    try:
        allocator = bt.get_allocator(spec.allocator)
    except KeyError as exc:
        raise InvalidRequest(exc.args[0]) from exc

    store = BarLake(settings.data_dir)
    padded_uics = list(spec.uics) + [None] * (len(spec.symbols) - len(spec.uics))
    need_network = any(uic is None for uic in padded_uics)

    panel: dict[str, Any] = {}
    instruments: dict[str, Any] = {}

    def _read(label: str, key: SeriesKey):
        frame = store.read(key)
        if spec.since is not None:
            frame = bars_mod.clip(frame, start=spec.since)
        if frame.empty:
            raise SeriesNotStored(
                label,
                horizon_label(spec.horizon),
                asset_type=key.asset_type,
                uic=key.uic,
                since=spec.since.isoformat() if spec.since else None,
            )
        return frame

    async def _add(symbol: str, uic: int | None, client: SaxoClient | None) -> None:
        exchange_id = None
        if uic is None:
            instrument = await resolve_symbol(
                client,
                symbol,
                asset_type=spec.asset_type,
                prefer=[e for e in (spec.exchange, *settings.saxo.preferred_exchanges) if e],
            )
            label, uic, asset_type = instrument.symbol, instrument.uic, instrument.asset_type
            exchange_id = instrument.exchange_id
        else:
            label, asset_type = symbol, (spec.asset_type or "Stock")

        key = SeriesKey(asset_type, int(uic), spec.horizon)
        frame = _read(label, key)

        session = None
        if client is not None and exchange_id:
            try:
                exchange = await get_exchange(client, exchange_id)
                session = calendars.infer_regular_hours(frame, calendars.resolve_timezone(exchange))
            except SaxoAPIError as exc:
                log.warning("no session calendar for %s: %s", label, exc)

        panel[label] = frame
        instruments[label] = bt.Instrument(key=key, session=session)

    if need_network:
        async with SaxoClient(settings) as client:
            for symbol, uic in zip(spec.symbols, padded_uics, strict=True):
                await _add(symbol, uic, client)
    else:
        for symbol, uic in zip(spec.symbols, padded_uics, strict=True):
            await _add(symbol, uic, None)

    return bt.run(
        panel,
        strategy,
        horizon=spec.horizon,
        allocator=allocator,
        cost_model=bt.CostModel(
            commission_bps=spec.fee_bps,
            half_spread_bps=spec.spread_bps,
            slippage_bps=spec.slippage_bps,
        ),
        instruments=instruments,
        starting_cash=spec.starting_cash,
        leverage_cap=spec.leverage,
        progress=progress,
    )
