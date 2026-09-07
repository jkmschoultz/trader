"""Client-side rate limiting for the Saxo OpenAPI.

Saxo enforces 120 requests/minute *per session per service group*, plus a
separate 1 order/second cap. Limiting on our side matters more than it might
seem: the backfill in Phase 1 pages through history as fast as it can, and a
429 mid-backfill wastes the request and stalls the loop. Staying just under the
cap is faster in practice than being throttled.

The bucket is per service group -- the first path segment of an OpenAPI URL
(``port``, ``chart``, ``trade``, ``ref``) -- because that is the dimension Saxo
actually meters.

This limiter's memory is per process. It correctly keeps one long-running
process (a live trading session, one backfill run) under the configured rate,
but has no way to know about quota another process already spent against the
same Saxo session in the last minute -- confirmed while building Phase 1, where
running several short-lived scripts against SIM back to back still produced a
429 despite each one individually staying under its own limit. When that
happens the client must still recover correctly rather than stall; see
``SaxoClient._retry_after``.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict


class TokenBucket:
    """An asyncio token bucket refilling at ``rate`` tokens per second."""

    def __init__(self, capacity: float, rate_per_second: float) -> None:
        self._capacity = capacity
        self._rate = rate_per_second
        self._tokens = capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Wait until ``tokens`` are available, then consume them."""
        while True:
            async with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._updated) * self._rate
                )
                self._updated = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                wait = deficit / self._rate
            # Sleep outside the lock so other callers can still be served.
            await asyncio.sleep(wait)

    def pause_for(self, seconds: float) -> None:
        """Drain the bucket so nothing is issued for roughly ``seconds``.

        Used when Saxo returns a 429 with a reset hint: the server's view of our
        quota is authoritative and overrides the local estimate.
        """
        self._tokens = min(self._tokens, -seconds * self._rate)
        self._updated = time.monotonic()


class ServiceGroupLimiter:
    """Holds one :class:`TokenBucket` per service group, created on demand."""

    def __init__(self, requests_per_minute: int) -> None:
        self._rpm = requests_per_minute
        self._buckets: dict[str, TokenBucket] = {}
        self._guard = asyncio.Lock()
        # Saxo's own per-session order cap. Shared across all callers.
        self._order_bucket = TokenBucket(capacity=1.0, rate_per_second=1.0)
        self._counts: defaultdict[str, int] = defaultdict(int)

    async def bucket(self, service_group: str) -> TokenBucket:
        async with self._guard:
            if service_group not in self._buckets:
                self._buckets[service_group] = TokenBucket(
                    capacity=self._rpm, rate_per_second=self._rpm / 60.0
                )
            return self._buckets[service_group]

    async def acquire(self, service_group: str, *, is_order: bool = False) -> None:
        bucket = await self.bucket(service_group)
        await bucket.acquire()
        if is_order:
            await self._order_bucket.acquire()
        self._counts[service_group] += 1

    async def penalise(self, service_group: str, seconds: float) -> None:
        (await self.bucket(service_group)).pause_for(seconds)

    @property
    def request_counts(self) -> dict[str, int]:
        """Requests issued per service group, for diagnostics."""
        return dict(self._counts)
