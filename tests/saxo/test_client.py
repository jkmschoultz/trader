"""Client retry, rate-limit, and error-shaping behaviour.

These use a stub transport rather than the network, so they assert on the
policy -- what gets retried and what does not -- which is the part that matters
for correctness when real money is involved.
"""

from __future__ import annotations

import httpx
import pytest

from trader.config import Settings
from trader.saxo.client import SaxoAPIError, SaxoClient


class _StubAuth:
    """Stands in for SaxoAuth; counts forced refreshes."""

    def __init__(self) -> None:
        self.refreshes = 0

    async def get_access_token(self) -> str:
        return "token"

    async def refresh_now(self):
        self.refreshes += 1
        return None


def _client(handler, **overrides) -> tuple[SaxoClient, _StubAuth]:
    settings = Settings(**overrides)
    auth = _StubAuth()
    client = SaxoClient(settings, auth=auth, transport=httpx.MockTransport(handler))
    return client, auth


async def test_get_returns_decoded_json():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(200, json={"Data": [1, 2, 3]})

    client, _ = _client(handler)
    async with client:
        assert await client.get("/port/v1/accounts/me") == {"Data": [1, 2, 3]}


async def test_every_request_carries_a_unique_request_id():
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["x-request-id"])
        return httpx.Response(200, json={})

    client, _ = _client(handler)
    async with client:
        await client.get("/port/v1/balances/me")
        await client.get("/port/v1/balances/me")

    # Saxo 409s on identical operations within 15s that share a request id.
    assert len(set(seen)) == 2


async def test_get_retries_server_errors_then_succeeds():
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"ok": True})

    client, _ = _client(handler)
    async with client:
        assert await client.get("/chart/v1/charts") == {"ok": True}
    assert attempts == 3


async def test_post_is_not_retried_on_server_error():
    """A write that may have reached the exchange must never be replayed."""
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="unavailable")

    client, _ = _client(handler)
    async with client:
        with pytest.raises(SaxoAPIError) as exc:
            await client.post("/trade/v2/orders", json={"Uic": 21}, is_order=True)

    assert attempts == 1, "an order must be attempted exactly once"
    assert exc.value.status_code == 503


async def test_post_is_not_retried_on_timeout():
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("timed out", request=request)

    client, _ = _client(handler)
    async with client:
        with pytest.raises(SaxoAPIError):
            await client.post("/trade/v2/orders", json={}, is_order=True)

    assert attempts == 1


async def test_rate_limit_is_retried_using_the_reset_header():
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={"X-RateLimit-ChartMinute-Reset": "1"},
                text="too many requests",
            )
        return httpx.Response(200, json={"ok": True})

    client, _ = _client(handler)
    async with client:
        assert await client.get("/chart/v1/charts") == {"ok": True}
    assert attempts == 2


async def test_retry_after_picks_the_bucket_that_actually_ran_out():
    """Verified against SIM: a 429 from bursting the per-minute chart quota still
    carries an unrelated AppDay quota header alongside it, with its own -Reset
    that is nowhere near exhausted but numerically enormous by comparison
    (~23h vs ~1min). Picking the first -Reset header seen, rather than the one
    whose own -Remaining is actually 0, previously stalled the client for most
    of a day on a routine per-minute throttle."""
    response = httpx.Response(
        429,
        headers={
            "X-RateLimit-AppDay-Limit": "10000000",
            "X-RateLimit-AppDay-Remaining": "9999521",
            "X-RateLimit-AppDay-Reset": "83016",
            "X-RateLimit-ChartMinute-Limit": "120",
            "X-RateLimit-ChartMinute-Remaining": "0",
            "X-RateLimit-ChartMinute-Reset": "50",
        },
    )
    assert SaxoClient._retry_after(response) == 50.0


async def test_retry_after_falls_back_to_the_soonest_reset_without_remaining_counts():
    response = httpx.Response(
        429,
        headers={"X-RateLimit-AppDay-Reset": "83016", "X-RateLimit-ChartMinute-Reset": "50"},
    )
    assert SaxoClient._retry_after(response) == 50.0


async def test_a_burst_that_only_exhausts_the_minute_quota_does_not_stall(monkeypatch):
    """End-to-end: the same header shape as the real SIM 429 must not turn into
    a long sleep. TokenBucket.pause_for is monkeypatched to record what it was
    told rather than actually sleeping, since asyncio.sleep(50) has no place in
    a unit test."""
    from trader.saxo.ratelimit import TokenBucket

    paused_for: list[float] = []
    original = TokenBucket.pause_for

    def spy(self, seconds):
        paused_for.append(seconds)
        original(self, seconds)

    monkeypatch.setattr(TokenBucket, "pause_for", spy)

    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(
                429,
                headers={
                    "X-RateLimit-AppDay-Remaining": "9999521",
                    "X-RateLimit-AppDay-Reset": "83016",
                    "X-RateLimit-ChartMinute-Remaining": "0",
                    "X-RateLimit-ChartMinute-Reset": "1",
                },
            )
        return httpx.Response(200, json={"ok": True})

    client, _ = _client(handler)
    async with client:
        assert await client.get("/chart/v1/charts") == {"ok": True}

    assert paused_for == [1.0], "must pause for the exhausted minute bucket, not the day bucket"


async def test_unauthorized_triggers_one_forced_refresh():
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(401, text="unauthorized")
        return httpx.Response(200, json={"ok": True})

    client, auth = _client(handler)
    async with client:
        assert await client.get("/port/v1/users/me") == {"ok": True}
    assert auth.refreshes == 1


async def test_client_error_surfaces_saxo_message():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"ErrorCode": "InvalidRequest", "Message": "Uic is required"}
        )

    client, _ = _client(handler)
    async with client:
        with pytest.raises(SaxoAPIError) as exc:
            await client.get("/chart/v1/charts")

    assert "InvalidRequest" in str(exc.value)
    assert "Uic is required" in str(exc.value)
    assert exc.value.status_code == 400


async def test_no_content_returns_none():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(204)

    client, _ = _client(handler)
    async with client:
        assert await client.delete("/trade/v2/orders/123") is None


def test_service_group_is_the_first_path_segment():
    assert SaxoClient._service_group("/port/v1/balances") == "port"
    assert SaxoClient._service_group("chart/v1/charts") == "chart"
    assert SaxoClient._service_group("/trade/v2/orders") == "trade"


async def test_none_query_parameters_are_dropped():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert "Time" not in request.url.params
        assert request.url.params["Uic"] == "21"
        return httpx.Response(200, json={})

    client, _ = _client(handler)
    async with client:
        await client.get("/chart/v1/charts", Uic=21, Time=None)
