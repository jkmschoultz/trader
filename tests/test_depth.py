"""The history-depth spike.

The property that matters is that the spike never reports its own page cap as
Saxo's limit -- a probe that was cut short must say so.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from trader.config import Settings
from trader.data.depth import probe, probe_horizon, render, save_report
from trader.saxo.client import SaxoClient

NEWEST = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


class _StubAuth:
    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        return None


def _client(handler) -> SaxoClient:
    return SaxoClient(Settings(), auth=_StubAuth(), transport=httpx.MockTransport(handler))


def _feed(*, first: datetime | None, page: int = 10, report_first_sample: bool = True):
    """A chart endpoint whose history begins at ``first`` (None = unlimited)."""

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        horizon = int(params["Horizon"])
        anchor = (
            datetime.strptime(params["Time"], "%Y-%m-%dT%H:%M:%S.%f%z")
            if "Time" in params
            else NEWEST
        )
        step = timedelta(minutes=horizon)
        info = {}
        if first is not None and report_first_sample:
            info["FirstSampleTime"] = first.strftime("%Y-%m-%dT%H:%M:%S.000000Z")

        end = min(anchor, NEWEST)
        start = end - step * (page - 1)
        if first is not None:
            if end < first:
                return httpx.Response(200, json={"Data": [], "ChartInfo": info})
            start = max(start, first)

        count = int((end - start) / step) + 1
        samples = [
            {
                "Time": (start + step * i).strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                "Open": 1.0,
                "High": 2.0,
                "Low": 0.5,
                "Close": 1.5,
                "Volume": 10,
            }
            for i in range(count)
        ]
        return httpx.Response(200, json={"Data": samples, "ChartInfo": info})

    return handler


async def test_a_probe_that_reaches_the_start_of_history_is_conclusive():
    first = NEWEST - timedelta(minutes=45)

    async with _client(_feed(first=first)) as client:
        result = await probe_horizon(client, uic=1, asset_type="Stock", horizon=1, count=10)

    assert result.conclusive
    assert result.stopped_because == "first-sample"
    assert result.first == first
    assert result.bars == 46
    assert result.reported_first_sample == first


async def test_clamping_without_a_first_sample_time_is_also_conclusive():
    """Saxo omits FirstSampleTime for some instruments; the walk must still know."""
    first = NEWEST - timedelta(minutes=45)

    async with _client(_feed(first=first, report_first_sample=False)) as client:
        result = await probe_horizon(client, uic=1, asset_type="Stock", horizon=1, count=10)

    assert result.conclusive
    assert result.stopped_because in {"no-progress", "empty-page"}
    assert result.first == first
    assert result.reported_first_sample is None


async def test_a_probe_cut_short_by_the_page_cap_is_not_conclusive():
    """This is the whole point: our own limit must never be read as Saxo's."""
    async with _client(_feed(first=None)) as client:
        result = await probe_horizon(
            client, uic=1, asset_type="Stock", horizon=1, count=10, max_pages=3
        )

    assert not result.conclusive
    assert result.stopped_because == "page-cap"
    assert result.pages == 3


async def test_span_days_measures_the_range_actually_retrieved():
    first = NEWEST - timedelta(days=10)

    async with _client(_feed(first=first, page=1440)) as client:
        result = await probe_horizon(client, uic=1, asset_type="Stock", horizon=1440, count=20)

    assert result.span_days == pytest.approx(10.0, abs=0.01)
    assert result.horizon_label == "1d"


async def test_probe_covers_every_requested_horizon():
    async with _client(_feed(first=NEWEST - timedelta(days=400))) as client:
        report = await probe(
            client,
            uic=211,
            asset_type="Stock",
            symbol="AAPL:xnas",
            horizons=(1, 60, 1440),
            max_pages=3,
            count=10,
        )

    assert [p["horizon"] for p in report["probes"]] == [1, 60, 1440]
    assert report["symbol"] == "AAPL:xnas"
    assert report["uic"] == 211
    assert report["requests"] == sum(p["pages"] for p in report["probes"])


async def test_probe_rejects_a_horizon_saxo_does_not_serve():
    async with _client(_feed(first=None)) as client:
        with pytest.raises(ValueError, match="not one Saxo serves"):
            await probe(client, uic=1, asset_type="Stock", horizons=(7,))


async def test_render_flags_the_inconclusive_horizons():
    async with _client(_feed(first=None)) as client:
        report = await probe(
            client, uic=1, asset_type="Stock", horizons=(1,), max_pages=2, count=10
        )

    text = render(report)
    assert "cut short" in text
    assert "--max-pages" in text


async def test_render_stays_quiet_when_every_probe_is_conclusive():
    async with _client(_feed(first=NEWEST - timedelta(minutes=30))) as client:
        report = await probe(
            client, uic=1, asset_type="Stock", horizons=(1,), max_pages=20, count=10
        )

    assert "cut short" not in render(report)


async def test_a_saved_report_round_trips_through_json(tmp_path):
    async with _client(_feed(first=NEWEST - timedelta(minutes=30))) as client:
        report = await probe(
            client, uic=1, asset_type="Stock", horizons=(1,), max_pages=20, count=10
        )

    path = save_report(report, tmp_path / "nested" / "depth.json")
    loaded = json.loads(path.read_text())

    assert loaded["uic"] == 1
    assert loaded["probes"][0]["conclusive"] is True
    # Datetimes serialise as strings rather than failing the dump.
    assert isinstance(loaded["probes"][0]["first"], str)
