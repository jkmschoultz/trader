"""run_backfill: it threads horizons, the page callback, and summaries through
``trader.data.ingest`` (whose walk logic is tested in tests/data/test_ingest.py)."""

from __future__ import annotations

import httpx
import pytest
import respx

from trader.config import Settings
from trader.data.lake import BarLake, SeriesKey
from trader.service.data import BackfillSpec, run_backfill

GATEWAY = "https://gateway.saxobank.com/sim/openapi"


@pytest.fixture
def net_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        state_dir=tmp_path / "state",
        saxo={"token_24h": "tok-for-tests"},
    )


def _one_bar_page() -> httpx.Response:
    sample = {
        "Time": "2024-03-01T14:30:00Z",
        "Open": 100.0,
        "High": 101.0,
        "Low": 99.0,
        "Close": 100.5,
        "Volume": 1000.0,
    }
    return httpx.Response(200, json={"Data": [sample], "ChartInfo": {}})


@respx.mock
async def test_fetches_each_horizon_and_reports_progress(net_settings):
    chart = respx.get(f"{GATEWAY}/chart/v3/charts").mock(return_value=_one_bar_page())
    messages: list[str] = []

    spec = BackfillSpec(
        symbol="AAPL:xnas", asset_type="Stock", uic=211, horizons=["1m", "5m"], since="1d"
    )
    summaries = await run_backfill(net_settings, spec, progress=messages.append)

    assert chart.called
    assert [s["key"] for s in summaries] == ["Stock:211@1m", "Stock:211@5m"]
    assert all(s["pages"] >= 1 and s["bars_fetched"] >= 1 for s in summaries)

    # on_page -> progress, tagged with the series key
    assert any("Stock:211@1m" in m for m in messages)
    assert any("Stock:211@5m" in m for m in messages)

    # the bar actually landed in the lake for both horizons
    lake = BarLake(net_settings.data_dir)
    assert not lake.read(SeriesKey("Stock", 211, 1)).empty
    assert not lake.read(SeriesKey("Stock", 211, 5)).empty


@respx.mock
async def test_resolves_the_symbol_when_no_uic_is_given(net_settings):
    ref = respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(
            200,
            json={
                "Data": [
                    {
                        "Identifier": 211,
                        "Symbol": "AAPL:xnas",
                        "AssetType": "Stock",
                        "ExchangeId": "NASDAQ",
                    }
                ]
            },
        )
    )
    respx.get(f"{GATEWAY}/chart/v3/charts").mock(return_value=_one_bar_page())

    spec = BackfillSpec(symbol="AAPL:xnas", horizons=["1m"], since="1d")
    summaries = await run_backfill(net_settings, spec, progress=lambda _m: None)

    assert ref.called
    assert summaries[0]["key"] == "Stock:211@1m"


@respx.mock
async def test_a_bare_ticker_honours_the_chosen_exchange(net_settings):
    listings = [
        {"Identifier": 211, "Symbol": "AAPL:xnas", "AssetType": "Stock"},
        {"Identifier": 46521, "Symbol": "AAPL:xmil", "AssetType": "Stock"},
        {"Identifier": 46520, "Symbol": "AAPL:xams", "AssetType": "Stock"},
    ]
    respx.get(f"{GATEWAY}/ref/v1/instruments").mock(
        return_value=httpx.Response(200, json={"Data": listings})
    )
    respx.get(f"{GATEWAY}/chart/v3/charts").mock(return_value=_one_bar_page())

    # config default prefers xnas; the spec's exchange overrides it
    spec = BackfillSpec(
        symbol="AAPL", asset_type="Stock", exchange="xmil", horizons=["1m"], since="1d"
    )
    summaries = await run_backfill(net_settings, spec, progress=lambda _m: None)

    assert summaries[0]["key"] == "Stock:46521@1m"
