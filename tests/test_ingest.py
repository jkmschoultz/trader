"""Backfill orchestration: resume in both directions, and interrupt safety."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest

from trader.config import Settings
from trader.data.ingest import backfill, backfill_all, days_ago
from trader.data.lake import BarLake, SeriesKey
from trader.saxo.client import SaxoAPIError, SaxoClient

KEY = SeriesKey("Stock", 211, 1)
NEWEST = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


class _StubAuth:
    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        return None


def _client(handler) -> SaxoClient:
    return SaxoClient(Settings(), auth=_StubAuth(), transport=httpx.MockTransport(handler))


class FakeChartFeed:
    """A synthetic 1-minute series with a fixed first sample and newest bar.

    Serves ``Mode=UpTo`` pages the way Saxo does, including clamping at the start
    of history rather than returning an empty page.
    """

    def __init__(self, *, first: datetime, newest: datetime = NEWEST, page: int = 10) -> None:
        self.first = first
        self.newest = newest
        self.page = page
        self.requests: list[datetime | None] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        anchor = (
            datetime.strptime(params["Time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            if "Time" in params
            else self.newest
        )
        self.requests.append(anchor)

        end = min(anchor, self.newest)
        if end < self.first:
            return httpx.Response(200, json={"Data": [], "ChartInfo": self._info()})

        start = max(self.first, end - timedelta(minutes=self.page - 1))
        count = int((end - start).total_seconds() // 60) + 1
        samples = [
            {
                "Time": (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Volume": 1000,
            }
            for i in range(count)
        ]
        return httpx.Response(200, json={"Data": samples, "ChartInfo": self._info()})

    def _info(self) -> dict:
        return {"FirstSampleTime": self.first.strftime("%Y-%m-%dT%H:%M:%S.000000Z")}


# -------------------------------------------------------------- first backfill


async def test_backfill_walks_to_the_start_of_history(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(minutes=45))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        result = await backfill(client, lake, KEY, count=10)

    assert result.bars_added == 46
    assert result.first == NEWEST - timedelta(minutes=45)
    assert result.last == NEWEST
    assert result.reached_start_of_history
    assert lake.coverage(KEY).rows == 46


async def test_backfill_stops_at_since(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(days=30))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        result = await backfill(client, lake, KEY, since=NEWEST - timedelta(minutes=25), count=10)

    assert result.stopped_because["backward"] == "reached-since"
    assert not result.reached_start_of_history
    assert lake.coverage(KEY).first <= NEWEST - timedelta(minutes=25)


async def test_max_bars_caps_the_walk(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(days=30))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        result = await backfill(client, lake, KEY, max_bars=25, count=10)

    assert result.stopped_because["backward"] == "max-bars"
    # Applied after a page completes, so it overshoots by at most one page.
    assert 25 <= result.bars_fetched < 25 + 10


# ---------------------------------------------------------------------- resume


async def test_a_second_backfill_of_a_current_series_is_two_requests(tmp_path):
    """Topping up must not re-download: one probe forward, one backward.

    The forward walk stops on its first page, having reached stored data. The
    backward walk costs one more page to confirm there is still nothing older --
    the lake records which bars it holds, not that a previous run exhausted
    Saxo's history, so that has to be re-established each run.
    """
    feed = FakeChartFeed(first=NEWEST - timedelta(minutes=45))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        await backfill(client, lake, KEY, count=10)
        before = len(feed.requests)
        result = await backfill(client, lake, KEY, count=10)

    assert len(feed.requests) - before == 2
    assert result.bars_added == 0
    assert result.stopped_because == {"forward": "reached-since", "backward": "empty-page"}


async def test_resume_extends_backwards_towards_an_earlier_since(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(days=30))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        await backfill(client, lake, KEY, since=NEWEST - timedelta(minutes=20), count=10)
        shallow = lake.coverage(KEY)

        await backfill(client, lake, KEY, since=NEWEST - timedelta(minutes=60), count=10)
        deep = lake.coverage(KEY)

    assert deep.first < shallow.first
    assert deep.last == shallow.last
    assert deep.rows > shallow.rows


async def test_resume_extends_forwards_when_new_bars_have_printed(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(minutes=45))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        await backfill(client, lake, KEY, count=10)
        stored = lake.coverage(KEY)

        # Twenty more minutes print, then we top up.
        feed.newest = NEWEST + timedelta(minutes=20)
        result = await backfill(client, lake, KEY, count=10)

    assert result.stopped_because["forward"] == "reached-since"
    assert lake.coverage(KEY).last == feed.newest
    assert lake.coverage(KEY).first == stored.first


async def test_no_resume_refetches_and_deduplicates(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(minutes=45))
    lake = BarLake(tmp_path)

    async with _client(feed.handler) as client:
        await backfill(client, lake, KEY, count=10)
        result = await backfill(client, lake, KEY, count=10, resume=False)

    assert result.bars_fetched == 46
    assert result.bars_added == 0, "a full refetch repairs in place, it does not duplicate"
    assert result.bars_revised == 46
    assert lake.coverage(KEY).rows == 46


# ------------------------------------------------------------ interrupt safety


async def test_each_page_is_committed_before_the_next_is_fetched(tmp_path):
    """An interrupted backfill must keep everything already fetched."""
    feed = FakeChartFeed(first=NEWEST - timedelta(days=30))
    lake = BarLake(tmp_path)
    rows_seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        coverage = lake.coverage(KEY)
        rows_seen.append(coverage.rows if coverage else 0)
        if len(rows_seen) > 4:
            # A 400 rather than a dropped connection: the client does not retry
            # it, so the walk dies on the fifth page without backoff delays.
            return httpx.Response(400, json={"Message": "boom", "ErrorCode": "Nope"})
        return feed.handler(request)

    async with _client(handler) as client:
        with pytest.raises(SaxoAPIError):
            await backfill(client, lake, KEY, count=10, max_pages=100)

    # The lake grew before each successive request, not only at the end.
    assert rows_seen == [0, 10, 20, 30, 40]
    assert lake.coverage(KEY).rows == 40, "four fetched pages survived the failure"


# ------------------------------------------------------------------- utilities


async def test_backfill_all_runs_each_key(tmp_path):
    feed = FakeChartFeed(first=NEWEST - timedelta(minutes=20))
    lake = BarLake(tmp_path)
    keys = [SeriesKey("Stock", 211, 1), SeriesKey("Stock", 211, 5)]

    async with _client(feed.handler) as client:
        results = await backfill_all(client, lake, keys, count=10)

    assert len(results) == 2
    assert {r.key for r in results} == set(keys)
    assert sorted(lake.series()) == sorted(keys)


def test_days_ago_is_utc_aware():
    moment = days_ago(30)
    assert moment.tzinfo is UTC
    assert 29 < (datetime.now(UTC) - moment).days + 1 < 32
