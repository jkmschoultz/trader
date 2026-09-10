"""Read the bar lake and drive backfills, without a CLI or an HTTP layer.

- :func:`lake_series` -- what the lake holds, cheaply (coverage only).
- :func:`read_bars` -- one series as chart-ready rows, decimated to a point cap,
  with the gaps it contains.
- :func:`search_instruments` -- ticker -> Uic, over the network.
- :func:`run_backfill` -- fetch bars into the lake, reporting progress per page.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from trader.config import Settings
from trader.service.errors import SeriesNotStored
from trader.service.inputs import parse_since

log = logging.getLogger(__name__)

__all__ = [
    "BackfillSpec",
    "InstrumentHit",
    "SeriesInfo",
    "lake_series",
    "read_bars",
    "refresh_symbols",
    "run_backfill",
    "search_instruments",
]

DEFAULT_MAX_POINTS = 5000


@dataclass(frozen=True)
class SeriesInfo:
    asset_type: str
    uic: int
    horizon: int
    horizon_label: str
    rows: int
    first: datetime
    last: datetime
    files: int
    # Saxo symbol for this uic, from the instrument registry. "" until a
    # backfill or `trader data symbols` has recorded it.
    symbol: str = ""


@dataclass(frozen=True)
class InstrumentHit:
    symbol: str
    uic: int
    asset_type: str
    exchange_id: str
    description: str


def lake_series(settings: Settings) -> list[SeriesInfo]:
    """Every stored series with its coverage, sorted by key."""
    from trader.data.instruments import InstrumentRegistry
    from trader.data.lake import BarLake
    from trader.saxo.charts import horizon_label

    store = BarLake(settings.data_dir)
    registry = InstrumentRegistry(settings.data_dir)
    out: list[SeriesInfo] = []
    for key in store.series():
        coverage = store.coverage(key)
        if coverage is None:
            continue
        out.append(
            SeriesInfo(
                asset_type=key.asset_type,
                uic=key.uic,
                horizon=key.horizon,
                horizon_label=horizon_label(key.horizon),
                rows=coverage.rows,
                first=coverage.first,
                last=coverage.last,
                files=coverage.files,
                symbol=registry.symbol_for(key.asset_type, key.uic),
            )
        )
    return out


def read_bars(
    settings: Settings,
    *,
    asset_type: str,
    uic: int,
    horizon: int,
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: int = DEFAULT_MAX_POINTS,
) -> dict[str, Any]:
    """One series as ``{bars, gaps, rows, returned, decimated}``.

    ``bars`` rows are ``{time, open, high, low, close, volume}`` with ``time`` a
    Unix timestamp in seconds (the format lightweight-charts wants). The series
    is strided down to at most ``max_points`` points, keeping the last bar;
    ``gaps`` is measured on the full slice before decimation.
    """
    import pandas as pd

    from trader.data import bars as bars_mod
    from trader.data.lake import BarLake, SeriesKey
    from trader.saxo.charts import horizon_label

    store = BarLake(settings.data_dir)
    key = SeriesKey(asset_type, int(uic), int(horizon))
    frame = store.read(key, start=start, end=end)
    if frame.empty:
        raise SeriesNotStored(f"{asset_type}:{uic}", horizon_label(horizon))

    discontinuities = bars_mod.gaps(frame, horizon)
    rows = len(frame)

    step = 1 if rows <= max_points else math.ceil(rows / max_points)
    if step > 1:
        keep = frame.iloc[::step]
        if frame.index[-1] not in keep.index:
            keep = pd.concat([keep, frame.iloc[[-1]]])
        frame = keep

    bars = [
        {
            "time": int(row["time"].timestamp()),
            "open": _f(row["open"]),
            "high": _f(row["high"]),
            "low": _f(row["low"]),
            "close": _f(row["close"]),
            "volume": _f(row["volume"]),
        }
        for _, row in frame.iterrows()
    ]
    return {
        "bars": bars,
        "gaps": [
            {
                "after": g.after.isoformat(),
                "before": g.before.isoformat(),
                "missing": g.missing,
            }
            for g in discontinuities
        ],
        "rows": rows,
        "returned": len(bars),
        "decimated": step > 1,
    }


def _f(value: object) -> float | None:
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


async def search_instruments(
    settings: Settings,
    keywords: str,
    *,
    asset_type: str | None = None,
    limit: int = 20,
) -> list[InstrumentHit]:
    """Search Saxo for instruments matching ``keywords``. Needs a live session."""
    from trader.saxo.client import SaxoClient
    from trader.saxo.instruments import search

    async with SaxoClient(settings) as client:
        results = await search(
            client,
            keywords,
            asset_types=(asset_type,) if asset_type else None,
            limit=limit,
        )
    return [
        InstrumentHit(
            symbol=i.symbol,
            uic=i.uic,
            asset_type=i.asset_type,
            exchange_id=i.exchange_id,
            description=i.description,
        )
        for i in results
    ]


async def _record_instrument(client: Any, registry: Any, uic: int, asset_type: str) -> None:
    """Fetch reference data for one uic and store it. Never fatal to a backfill."""
    from trader.saxo.client import SaxoAPIError
    from trader.saxo.instruments import get_details

    try:
        details = await get_details(client, int(uic), asset_type)
    except (SaxoAPIError, LookupError) as exc:
        log.warning("could not fetch reference data for %s:%s: %s", asset_type, uic, exc)
        return
    registry.put(
        asset_type,
        uic,
        symbol=details.symbol,
        description=details.description,
        currency=details.currency,
        exchange_id=details.exchange_id,
    )


async def refresh_symbols(
    settings: Settings,
    *,
    refresh: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[Any]:
    """Fill the instrument registry from Saxo for every uic in the lake.

    By default only uics with no recorded symbol are fetched; ``refresh=True``
    re-fetches every one. Returns the resulting records, sorted by key.
    """
    from trader.data.instruments import InstrumentRegistry
    from trader.data.lake import BarLake
    from trader.saxo.client import SaxoClient

    store = BarLake(settings.data_dir)
    registry = InstrumentRegistry(settings.data_dir)

    pairs = sorted({(key.asset_type, key.uic) for key in store.series()})
    todo = [
        (asset_type, uic)
        for asset_type, uic in pairs
        if refresh or not registry.symbol_for(asset_type, uic)
    ]

    if todo:
        async with SaxoClient(settings) as client:
            for asset_type, uic in todo:
                await _record_instrument(client, registry, uic, asset_type)
                if progress is not None:
                    progress(f"{registry.label(asset_type, uic)}")

    return registry.records()


class BackfillSpec(BaseModel):
    """What to fetch into the lake. ``horizons`` accepts labels or minutes."""

    symbol: str
    asset_type: str | None = None
    # Preferred exchange suffix (e.g. "xnas") for resolving a bare ticker.
    exchange: str | None = None
    uic: int | None = None
    horizons: list[int] = Field(default_factory=lambda: [1])
    since: datetime | None = None
    max_bars: int | None = None
    max_pages: int = 1000
    resume: bool = True

    @field_validator("horizons", mode="before")
    @classmethod
    def _horizons(cls, value: object) -> list[int]:
        from trader.saxo.charts import ChartError, parse_horizon

        items = value if isinstance(value, (list, tuple)) else [value]
        try:
            return [parse_horizon(item) for item in items]
        except ChartError as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("since", mode="before")
    @classmethod
    def _since(cls, value: object) -> datetime | None:
        if value is None or isinstance(value, datetime):
            return value  # type: ignore[return-value]
        return parse_since(str(value))


async def run_backfill(
    settings: Settings,
    spec: BackfillSpec,
    *,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Backfill every horizon in ``spec``; return one summary dict per horizon.

    ``progress`` is called with a short status line after each committed page.
    """
    from trader.data import ingest
    from trader.data.instruments import InstrumentRegistry
    from trader.data.lake import BarLake, SeriesKey
    from trader.saxo.client import SaxoClient
    from trader.service.instruments import resolve_symbol

    store = BarLake(settings.data_dir)
    registry = InstrumentRegistry(settings.data_dir)
    summaries: list[dict[str, Any]] = []

    async with SaxoClient(settings) as client:
        if spec.uic is None:
            instrument = await resolve_symbol(
                client,
                spec.symbol,
                asset_type=spec.asset_type,
                prefer=[e for e in (spec.exchange, *settings.saxo.preferred_exchanges) if e],
            )
            uic, asset_type = instrument.uic, instrument.asset_type
            registry.put(
                asset_type,
                uic,
                symbol=instrument.symbol,
                description=instrument.description,
                currency=instrument.currency,
                exchange_id=instrument.exchange_id,
            )
        else:
            uic, asset_type = spec.uic, (spec.asset_type or "Stock")
            await _record_instrument(client, registry, uic, asset_type)

        for horizon in spec.horizons:
            key = SeriesKey(asset_type, int(uic), horizon)

            def _on_page(result: ingest.BackfillResult, _key: SeriesKey = key) -> None:
                if progress is not None:
                    progress(
                        f"{_key}: {result.pages} pages, "
                        f"{result.bars_fetched:,} bars, {result.bars_added:,} new"
                    )

            result = await ingest.backfill(
                client,
                store,
                key,
                since=spec.since,
                max_bars=spec.max_bars,
                max_pages=spec.max_pages,
                resume=spec.resume,
                on_page=_on_page,
            )
            summaries.append(
                {
                    "key": str(key),
                    "pages": result.pages,
                    "bars_fetched": result.bars_fetched,
                    "bars_added": result.bars_added,
                    "bars_revised": result.bars_revised,
                    "first": result.first.isoformat() if result.first else None,
                    "last": result.last.isoformat() if result.last else None,
                    "elapsed_seconds": round(result.elapsed_seconds, 1),
                    "stopped_because": result.stopped_because,
                }
            )

    return summaries
