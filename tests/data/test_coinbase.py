"""Coinbase import: paging, the row layout, the lake write, and local symbol resolution."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from trader.data import coinbase
from trader.data.instruments import InstrumentRegistry
from trader.data.lake import BarLake, SeriesKey


def _fake_client(calls):
    """Serve one daily candle per day in each requested window, newest first like Coinbase."""

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        start = datetime.fromisoformat(request.url.params["start"])
        end = datetime.fromisoformat(request.url.params["end"])
        day = 86400
        rows = []
        t = int(start.timestamp())
        while t < int(end.timestamp()):
            price = 100.0 + (t - 1_600_000_000) / day
            rows.append([t, price - 1, price + 1, price - 0.5, price, 10.0])  # l, h, o, c, v
            t += day
        return httpx.Response(200, json=list(reversed(rows)))

    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _no_pause(monkeypatch):
    monkeypatch.setattr(coinbase, "_PAUSE", 0.0)


def test_pseudo_uic_is_stable_and_case_blind():
    assert coinbase.pseudo_uic("btc-usd") == coinbase.pseudo_uic("BTC-USD")
    assert coinbase.pseudo_uic("BTC-USD") != coinbase.pseudo_uic("ETH-USD")
    assert coinbase.pseudo_uic("BTC-USD") > 0


def test_fetch_pages_in_300_candle_windows_and_maps_columns():
    calls: list[dict] = []
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2021, 12, 1, tzinfo=UTC)  # ~700 days -> 3 pages
    frame = coinbase.fetch_candles(
        "BTC-USD", 1440, start=start, end=end, client=_fake_client(calls)
    )

    assert len(calls) == 3
    assert all(c["granularity"] == "86400" for c in calls)
    assert frame["time"].is_monotonic_increasing and frame["time"].is_unique
    assert len(frame) == (end - start).days
    row = frame.iloc[0]
    assert row["low"] == row["close"] - 1 and row["high"] == row["close"] + 1
    assert row["open"] == row["close"] - 0.5
    assert frame["close_ask"].isna().all()  # no quotes: backtests need a cost assumption


def test_unsupported_horizon_is_rejected():
    with pytest.raises(ValueError, match="Coinbase serves"):
        coinbase.fetch_candles("BTC-USD", 240, start=datetime(2020, 1, 1, tzinfo=UTC))


def test_import_writes_the_lake_and_registers_the_symbol(tmp_path, monkeypatch):
    calls: list[dict] = []
    client = _fake_client(calls)
    real = coinbase.fetch_candles
    monkeypatch.setattr(
        coinbase,
        "fetch_candles",
        lambda product, horizon, *, start: real(
            product, horizon, start=start, end=datetime(2020, 3, 1, tzinfo=UTC), client=client
        ),
    )
    lake, registry = BarLake(tmp_path), InstrumentRegistry(tmp_path)
    coinbase.import_product(
        lake, "btc-usd", 1440, since=datetime(2020, 1, 1, tzinfo=UTC), registry=registry
    )

    key = SeriesKey("Crypto", coinbase.pseudo_uic("BTC-USD"), 1440)
    assert lake.coverage(key).rows == 60
    assert registry.symbol_for("Crypto", key.uic) == "BTC-USD"


async def test_crypto_symbols_resolve_locally_without_saxo(tmp_path):
    from trader.config import Settings
    from trader.data.lake import normalise
    from trader.service._panel import load_panel
    from trader.service.errors import InvalidRequest

    uic = coinbase.pseudo_uic("ETH-USD")
    lake = BarLake(tmp_path)
    frame = normalise(
        __import__("pandas").DataFrame(
            {
                "time": __import__("pandas").date_range("2024-01-01", periods=5, tz="UTC"),
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
            }
        )
    )
    lake.write(SeriesKey("Crypto", uic, 1440), frame)
    InstrumentRegistry(tmp_path).put("Crypto", uic, symbol="ETH-USD")
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")

    (series,) = await load_panel(settings, symbols=["eth-usd"], asset_type="Crypto", horizon=1440)
    assert series.key == SeriesKey("Crypto", uic, 1440)
    assert len(series.frame) == 5
    assert series.lot_size == coinbase.LOT_SIZE  # whole-unit sizing would buy zero BTC

    with pytest.raises(InvalidRequest, match="trader data crypto DOGE-USD"):
        await load_panel(settings, symbols=["DOGE-USD"], asset_type="Crypto", horizon=1440)


def test_asset_types_pair_positionally_for_a_mixed_basket():
    from trader.service._panel import _per_symbol_types
    from trader.service.errors import InvalidRequest

    assert _per_symbol_types(None, 2) == [None, None]
    assert _per_symbol_types("Etf", 3) == ["Etf"] * 3
    assert _per_symbol_types(["Etf"], 2) == ["Etf", "Etf"]
    assert _per_symbol_types(["Etf", "Crypto"], 2) == ["Etf", "Crypto"]
    with pytest.raises(InvalidRequest, match="3 asset types for 2 symbols"):
        _per_symbol_types(["Etf", "Crypto", "FxSpot"], 2)
