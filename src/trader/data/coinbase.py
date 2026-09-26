"""Crypto bars from Coinbase's public market-data API, into the same lake.

Saxo's SIM environment has no spot BTC or ETH, and the ETF proxies (IBIT, ETHA)
only exist since 2024 -- too short for anything that needs a year of lookback.
Coinbase publishes BTC-USD candles back to 2015 and ETH-USD back to 2016, with
no API key.

Series are stored under asset type ``"Crypto"`` with a *pseudo-Uic*: a stable
hash of the product id (:func:`pseudo_uic`), because the lake is keyed by a
number. The product id goes into the instrument registry, so
``--symbol BTC-USD --asset-type Crypto`` resolves locally with no Saxo call (see
:func:`trader.service._panel.load_panel`).

Coinbase candles carry no bid/ask, so ``close_ask`` is NaN and a backtest must
be given a cost assumption (``--spread-bps`` / ``--fee-bps``). Daily candles are
UTC days; crypto trades all week, so there are 7 bars per week.
"""

from __future__ import annotations

import logging
import time
import zlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

    from trader.data.lake import BarLake, WriteResult

log = logging.getLogger(__name__)

__all__ = ["ASSET_TYPE", "GRANULARITY", "fetch_candles", "import_product", "pseudo_uic"]

ASSET_TYPE = "Crypto"
EXCHANGE_ID = "COINBASE"
BASE_URL = "https://api.exchange.coinbase.com"
#: lake horizon (minutes) -> Coinbase granularity (seconds); the only sizes it serves
GRANULARITY: dict[int, int] = {1: 60, 5: 300, 15: 900, 60: 3600, 360: 21600, 1440: 86400}
_PAGE = 300  # candles per request, Coinbase's maximum
_PAUSE = 0.2  # seconds between requests; the public limit is ~10/s
#: before any Coinbase product listed; a fetch from here finds the real start
EARLIEST = datetime(2015, 1, 1, tzinfo=UTC)
#: smallest tradable amount (1 satoshi). The engine sizes in whole lots, so the
#: default lot of 1 would round a $50k allocation to zero bitcoin.
LOT_SIZE = 1e-8


def pseudo_uic(product: str) -> int:
    """A stable positive int for a product id, standing in for a Saxo Uic."""
    return zlib.crc32(f"coinbase:{product.upper()}".encode()) & 0x7FFFFFFF


def _get(client: httpx.Client, product: str, params: dict) -> list:
    for attempt in range(5):
        response = client.get(f"{BASE_URL}/products/{product}/candles", params=params)
        if response.status_code == 429:  # rate limited: back off and retry
            time.sleep(1.0 + attempt)
            continue
        response.raise_for_status()
        return response.json()
    response.raise_for_status()
    return []


def fetch_candles(
    product: str,
    horizon: int,
    *,
    start: datetime,
    end: datetime | None = None,
    client: httpx.Client | None = None,
) -> pd.DataFrame:
    """Every candle for ``product`` at ``horizon`` in ``[start, end)``, as a lake frame.

    Pages forward in 300-candle windows. Windows before the product listed come
    back empty and are skipped.

    Raises:
        ValueError: Coinbase does not serve ``horizon``.
        httpx.HTTPStatusError: an unknown product or a persistent API error.
    """
    import httpx

    from trader.data.lake import normalise

    if horizon not in GRANULARITY:
        raise ValueError(f"Coinbase serves {sorted(GRANULARITY)} minute candles, not {horizon}")
    step = timedelta(seconds=GRANULARITY[horizon])
    end = end or datetime.now(UTC)
    own = client is None
    client = client or httpx.Client(timeout=30.0, headers={"User-Agent": "trader"})
    rows: list[list[float]] = []
    try:
        cursor = start
        while cursor < end:
            window_end = min(cursor + step * _PAGE, end)
            params = {
                "granularity": GRANULARITY[horizon],
                "start": cursor.isoformat(),
                "end": window_end.isoformat(),
            }
            rows.extend(_get(client, product, params))
            cursor = window_end
            time.sleep(_PAUSE)
    finally:
        if own:
            client.close()

    if not rows:
        return normalise(pd.DataFrame({"time": []}))
    # Coinbase rows: [unix time, low, high, open, close, volume]
    frame = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    frame = frame[frame["time"] < pd.Timestamp(end)]
    return normalise(frame)


def import_product(
    lake: BarLake,
    product: str,
    horizon: int,
    *,
    since: datetime | None = None,
    registry=None,
) -> WriteResult:
    """Fetch ``product`` candles into ``lake`` and record the symbol in ``registry``.

    ``since=None`` resumes from the last stored bar, or starts from the
    beginning of Coinbase history when nothing is stored yet.
    """
    from trader.data.lake import SeriesKey

    product = product.upper()
    key = SeriesKey(ASSET_TYPE, pseudo_uic(product), horizon)
    if since is None:
        coverage = lake.coverage(key)
        since = coverage.last if coverage is not None else EARLIEST
    frame = fetch_candles(product, horizon, start=since)
    result = lake.write(key, frame)
    if registry is not None:
        registry.put(
            ASSET_TYPE,
            key.uic,
            symbol=product,
            description=f"{product} (Coinbase spot)",
            currency=product.split("-")[-1],
            exchange_id=EXCHANGE_ID,
        )
    log.info("%s@%dm: %d rows in, %d new", product, horizon, result.rows_in, result.rows_added)
    return result
