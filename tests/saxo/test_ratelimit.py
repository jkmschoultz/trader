"""Token bucket and per-service-group limiting."""

from __future__ import annotations

import asyncio

from trader.saxo.ratelimit import ServiceGroupLimiter, TokenBucket


async def test_bucket_allows_burst_up_to_capacity():
    bucket = TokenBucket(capacity=5, rate_per_second=1.0)
    start = asyncio.get_running_loop().time()
    for _ in range(5):
        await bucket.acquire()
    # The initial burst is free; only the sixth call should wait.
    assert asyncio.get_running_loop().time() - start < 0.05


async def test_bucket_throttles_once_drained():
    bucket = TokenBucket(capacity=1, rate_per_second=20.0)
    await bucket.acquire()
    start = asyncio.get_running_loop().time()
    await bucket.acquire()
    # Refilling one token at 20/s takes 50ms.
    assert asyncio.get_running_loop().time() - start >= 0.04


async def test_pause_for_blocks_further_issue():
    bucket = TokenBucket(capacity=10, rate_per_second=100.0)
    bucket.pause_for(0.1)
    start = asyncio.get_running_loop().time()
    await bucket.acquire()
    assert asyncio.get_running_loop().time() - start >= 0.09


async def test_service_groups_are_metered_independently():
    limiter = ServiceGroupLimiter(requests_per_minute=60)
    chart = await limiter.bucket("chart")
    port = await limiter.bucket("port")
    assert chart is not port
    assert (await limiter.bucket("chart")) is chart


async def test_order_cap_is_one_per_second():
    limiter = ServiceGroupLimiter(requests_per_minute=6000)
    await limiter.acquire("trade", is_order=True)
    start = asyncio.get_running_loop().time()
    await limiter.acquire("trade", is_order=True)
    # Saxo rejects more than one order per second per session.
    assert asyncio.get_running_loop().time() - start >= 0.9


async def test_non_orders_are_not_subject_to_the_order_cap():
    limiter = ServiceGroupLimiter(requests_per_minute=6000)
    start = asyncio.get_running_loop().time()
    for _ in range(5):
        await limiter.acquire("trade", is_order=False)
    assert asyncio.get_running_loop().time() - start < 0.5


async def test_request_counts_are_tracked():
    limiter = ServiceGroupLimiter(requests_per_minute=6000)
    await limiter.acquire("chart")
    await limiter.acquire("chart")
    await limiter.acquire("port")
    assert limiter.request_counts == {"chart": 2, "port": 1}
