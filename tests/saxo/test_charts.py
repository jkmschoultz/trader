"""Chart fetching, response-shape normalisation, and the backwards walk."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from trader.config import Settings
from trader.saxo.charts import (
    CHARTS_PATH,
    HORIZONS,
    MAX_COUNT,
    Bar,
    ChartError,
    get_chart,
    horizon_label,
    iter_history,
    parse_horizon,
)
from trader.saxo.client import SaxoClient


class _StubAuth:
    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        return None


def _client(handler) -> SaxoClient:
    return SaxoClient(Settings(), auth=_StubAuth(), transport=httpx.MockTransport(handler))


def _samples(start: datetime, count: int, horizon: int) -> list[dict]:
    return [
        {
            "Time": (start + timedelta(minutes=horizon * i)).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
            "Open": 100.0 + i,
            "High": 101.0 + i,
            "Low": 99.0 + i,
            "Close": 100.5 + i,
            "Volume": 1000 + i,
        }
        for i in range(count)
    ]


# ------------------------------------------------------------------- horizons


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("1m", 1),
        ("5m", 5),
        ("1h", 60),
        ("4h", 240),
        ("1d", 1440),
        ("1w", 10080),
        (60, 60),
        ("60", 60),
    ],
)
def test_parse_horizon_accepts_labels_and_minutes(given, expected):
    assert parse_horizon(given) == expected


@pytest.mark.parametrize("given", ["7m", 7, "1y", "banana", 0])
def test_parse_horizon_rejects_horizons_saxo_does_not_serve(given):
    with pytest.raises(ChartError):
        parse_horizon(given)


def test_horizon_label_round_trips_every_supported_horizon():
    for horizon in HORIZONS:
        assert parse_horizon(horizon_label(horizon)) == horizon


# --------------------------------------------------------------- bar parsing


def test_bar_parses_the_exchange_traded_shape():
    bar = Bar.from_sample(
        {
            "Time": "2024-03-01T14:30:00.000000Z",
            "Open": 1.0,
            "High": 2.0,
            "Low": 0.5,
            "Close": 1.5,
            "Volume": 4200,
        }
    )
    assert (bar.open, bar.high, bar.low, bar.close) == (1.0, 2.0, 0.5, 1.5)
    assert bar.volume == 4200
    assert bar.close_ask is None
    assert bar.trading_state is None
    assert bar.tradable is None


def test_bar_parses_the_quote_driven_shape_as_bid_plus_ask_close():
    """FX bars carry no last-traded price, only bid and ask."""
    bar = Bar.from_sample(
        {
            "Time": "2024-03-01T14:30:00.000000Z",
            "OpenBid": 1.0850,
            "HighBid": 1.0860,
            "LowBid": 1.0840,
            "CloseBid": 1.0855,
            "OpenAsk": 1.0851,
            "CloseAsk": 1.0856,
        }
    )
    assert bar.close == 1.0855
    assert bar.close_ask == 1.0856
    # The spread survives into the lake rather than being averaged into a mid.
    assert round(bar.close_ask - bar.close, 6) == 0.0001
    assert bar.volume is None


@pytest.mark.parametrize(
    ("state", "tradable"),
    [
        ("Automated", True),
        ("AutomatedTrading", True),
        ("CallAuctionTrading", True),
        ("Closed", False),
        ("PreTrading", False),
        ("SomeFutureStateNotYetSeen", False),
    ],
)
def test_tradable_reflects_the_markets_own_state_label(state, tradable):
    """Saxo's vocabulary is open-ended; an unrecognised state reads as not tradable."""
    bar = Bar.from_sample(
        {
            "Time": "2024-03-01T14:30:00.000000Z",
            "Open": 1.0,
            "High": 1.0,
            "Low": 1.0,
            "Close": 1.0,
            "MarketTradingState": state,
        }
    )
    assert bar.trading_state == state
    assert bar.tradable is tradable


def test_bar_rejects_a_sample_with_neither_shape():
    with pytest.raises(ChartError):
        Bar.from_sample({"Time": "2024-03-01T14:30:00.000000Z", "Something": 1})


# ------------------------------------------------------------------ get_chart


async def test_get_chart_uses_the_v3_endpoint():
    """v1 returns a 404 from an IIS error page, not a Saxo error body -- verified against SIM."""
    captured_path = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_path
        captured_path = request.url.path
        return httpx.Response(200, json={"Data": []})

    async with _client(handler) as client:
        await get_chart(client, uic=1, asset_type="Stock", horizon=1)

    assert captured_path.endswith(CHARTS_PATH)
    assert captured_path.endswith("/chart/v3/charts")


async def test_get_chart_sends_saxos_parameter_names():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Data": _samples(datetime(2024, 3, 1, tzinfo=UTC), 3, 1)})

    async with _client(handler) as client:
        await get_chart(
            client,
            uic=211,
            asset_type="Stock",
            horizon=1,
            count=500,
            mode="UpTo",
            time=datetime(2024, 3, 1, 15, 0, tzinfo=UTC),
        )

    assert captured["Uic"] == "211"
    assert captured["AssetType"] == "Stock"
    assert captured["Horizon"] == "1"
    assert captured["Count"] == "500"
    assert captured["Mode"] == "UpTo"
    assert captured["Time"] == "2024-03-01T15:00:00.000000Z"


async def test_get_chart_clamps_count_to_saxos_maximum():
    """Over-asking is silently truncated, which would break page arithmetic."""
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Data": []})

    async with _client(handler) as client:
        await get_chart(client, uic=1, asset_type="Stock", horizon=1, count=99_999)

    assert captured["Count"] == str(MAX_COUNT)


async def test_get_chart_omits_time_and_mode_for_the_latest_page():
    captured: dict[str, str] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(dict(request.url.params))
        return httpx.Response(200, json={"Data": []})

    async with _client(handler) as client:
        await get_chart(client, uic=1, asset_type="Stock", horizon=1)

    assert "Time" not in captured
    assert "Mode" not in captured


async def test_get_chart_sorts_a_reversed_page():
    """Ordering is not contractual, and a reversed page would corrupt the walk."""
    ordered = _samples(datetime(2024, 3, 1, tzinfo=UTC), 5, 1)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"Data": list(reversed(ordered))})

    async with _client(handler) as client:
        page = await get_chart(client, uic=1, asset_type="Stock", horizon=1)

    times = [bar.time for bar in page.bars]
    assert times == sorted(times)


async def test_get_chart_reads_chart_info():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "Data": [],
                "ChartInfo": {
                    "ExchangeId": "NASDAQ",
                    "Horizon": 1,
                    "FirstSampleTime": "2020-01-02T14:30:00.000000Z",
                    "DelayedByMinutes": 15,
                },
            },
        )

    async with _client(handler) as client:
        page = await get_chart(client, uic=1, asset_type="Stock", horizon=1)

    assert page.info.exchange_id == "NASDAQ"
    assert page.info.delayed_by_minutes == 15
    assert page.info.first_sample_time == datetime(2020, 1, 2, 14, 30, tzinfo=UTC)


async def test_get_chart_rejects_an_unsupported_horizon_without_sending():
    """A 400 from Saxo costs a request against the rate limit; the guard is free."""
    sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return httpx.Response(200, json={"Data": []})

    async with _client(handler) as client:
        with pytest.raises(ChartError):
            await get_chart(client, uic=1, asset_type="Stock", horizon=7)

    assert sent == 0


# --------------------------------------------------------------- iter_history


async def test_iter_history_walks_backwards_anchoring_one_bar_before_each_page():
    anchors: list[str | None] = []
    newest = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    async def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        anchors.append(params.get("Time"))
        page_index = len(anchors) - 1
        if page_index >= 3:
            return httpx.Response(200, json={"Data": []})
        start = newest - timedelta(minutes=10 * (page_index + 1))
        return httpx.Response(200, json={"Data": _samples(start, 10, 1)})

    async with _client(handler) as client:
        walk = iter_history(client, uic=1, asset_type="Stock", horizon=1, count=10)
        pages = [page async for page in walk]

    assert len(pages) == 3
    assert anchors[0] is None  # first page asks for the latest bars
    # Each later anchor is one bar before the previous page's oldest sample.
    assert anchors[1] == "2024-03-01T11:49:00.000000Z"
    assert anchors[2] == "2024-03-01T11:39:00.000000Z"
    assert walk.stopped_because == "empty-page"
    assert walk.bars == 30


async def test_iter_history_stops_when_a_page_fails_to_advance():
    """Saxo clamps at the start of history instead of returning nothing."""
    calls = 0
    start = datetime(2024, 3, 1, tzinfo=UTC)

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        # Always the same page, however far back we anchor.
        return httpx.Response(200, json={"Data": _samples(start, 5, 1)})

    async with _client(handler) as client:
        walk = iter_history(client, uic=1, asset_type="Stock", horizon=1, count=5, max_pages=50)
        pages = [page async for page in walk]

    assert len(pages) == 1, "the repeated page must not be yielded twice"
    assert calls == 2
    assert walk.stopped_because == "no-progress"
    assert walk.exhausted_history


async def test_iter_history_stops_at_since():
    newest = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    async def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        anchor = (
            datetime.strptime(params["Time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            if "Time" in params
            else newest
        )
        return httpx.Response(200, json={"Data": _samples(anchor - timedelta(minutes=9), 10, 1)})

    async with _client(handler) as client:
        walk = iter_history(
            client,
            uic=1,
            asset_type="Stock",
            horizon=1,
            count=10,
            since=newest - timedelta(minutes=25),
        )
        pages = [page async for page in walk]

    assert walk.stopped_because == "reached-since"
    assert walk.earliest <= newest - timedelta(minutes=25)
    assert len(pages) == 3
    assert not walk.exhausted_history


async def test_iter_history_stops_at_the_reported_first_sample():
    first_sample = datetime(2024, 3, 1, 11, 40, tzinfo=UTC)
    newest = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    async def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        anchor = (
            datetime.strptime(params["Time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            if "Time" in params
            else newest
        )
        return httpx.Response(
            200,
            json={
                "Data": _samples(anchor - timedelta(minutes=9), 10, 1),
                "ChartInfo": {"FirstSampleTime": "2024-03-01T11:40:00.000000Z"},
            },
        )

    async with _client(handler) as client:
        walk = iter_history(client, uic=1, asset_type="Stock", horizon=1, count=10)
        [page async for page in walk]

    assert walk.stopped_because == "first-sample"
    assert walk.exhausted_history
    assert walk.reported_first_sample == first_sample


async def test_iter_history_respects_the_page_cap():
    newest = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)

    async def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        anchor = (
            datetime.strptime(params["Time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            if "Time" in params
            else newest
        )
        return httpx.Response(200, json={"Data": _samples(anchor - timedelta(minutes=9), 10, 1)})

    async with _client(handler) as client:
        walk = iter_history(client, uic=1, asset_type="Stock", horizon=1, count=10, max_pages=4)
        pages = [page async for page in walk]

    assert len(pages) == 4
    assert walk.stopped_because == "page-cap"
    assert not walk.exhausted_history, "a capped walk says nothing about Saxo's limit"
