"""Backfill orchestration: Saxo chart pages into the Parquet lake.

:func:`backfill` is the one entry point. It is written to be interrupted --
every page is committed to the lake as it arrives, so killing a six-hour
1-minute backfill loses at most one page, and re-running it resumes from what is
already stored rather than starting over.

Resume works in two directions, and a routine top-up needs both:

* **Forward** -- fetch what has printed since the last stored bar. Cheap: the
  walk stops as soon as it reaches stored data.
* **Backward** -- extend the stored history further into the past, towards
  ``since``.

Running a backfill on a series that is already current therefore costs one
request, not a full re-download.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from trader.data.lake import BarLake, SeriesKey, bars_to_frame
from trader.saxo.charts import MAX_COUNT, iter_history
from trader.saxo.client import SaxoClient

log = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    """What one :func:`backfill` call fetched and stored."""

    key: SeriesKey
    pages: int = 0
    bars_fetched: int = 0
    bars_added: int = 0
    bars_revised: int = 0
    first: datetime | None = None
    last: datetime | None = None
    # One entry per direction actually walked, e.g. {"forward": "reached-since"}.
    stopped_because: dict[str, str] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def reached_start_of_history(self) -> bool:
        """True when the backward walk ran out of history rather than stopping early."""
        return self.stopped_because.get("backward") in {"first-sample", "no-progress", "empty-page"}

    def __str__(self) -> str:
        if not self.bars_fetched:
            return f"{self.key}: already current, nothing fetched"
        span = ""
        if self.first and self.last:
            span = f" covering {self.first:%Y-%m-%d %H:%M} -> {self.last:%Y-%m-%d %H:%M} UTC"
        reasons = ", ".join(f"{k}={v}" for k, v in self.stopped_because.items())
        return (
            f"{self.key}: {self.bars_added:,} new bars "
            f"({self.bars_fetched:,} fetched over {self.pages} pages{span}) "
            f"in {self.elapsed_seconds:.1f}s [{reasons}]"
        )


async def backfill(
    client: SaxoClient,
    lake: BarLake,
    key: SeriesKey,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    max_bars: int | None = None,
    max_pages: int = 1000,
    count: int = MAX_COUNT,
    resume: bool = True,
) -> BackfillResult:
    """Fetch bars for ``key`` and merge them into ``lake``.

    Args:
        since: how far back to go. ``None`` means as far as Saxo will serve,
            which for 1-minute bars is worth measuring first -- see
            :mod:`trader.data.depth`.
        until: fetch nothing newer than this. Defaults to the latest bar.
        max_bars: stop after roughly this many bars. Applied per direction after
            a page completes, so the true total may overshoot by up to one page.
        resume: extend existing coverage instead of refetching it. Turn this off
            to re-download a series whose stored bars are suspect -- writes
            deduplicate, so a full refetch repairs in place.

    Returns:
        A :class:`BackfillResult`. Its ``stopped_because`` is worth reading: a
        backward walk that stopped at ``page-cap`` has more history available.
    """
    started = time.monotonic()
    result = BackfillResult(key=key)
    coverage = lake.coverage(key) if resume else None
    step = timedelta(minutes=key.horizon)

    if coverage is None:
        await _walk(
            client,
            lake,
            key,
            result,
            "backward",
            since=since,
            until=until,
            max_bars=max_bars,
            max_pages=max_pages,
            count=count,
        )
    else:
        log.info("%s: resuming from stored %s -> %s", key, coverage.first, coverage.last)

        # Forward: from now (or `until`) back to the newest stored bar.
        if until is None or until > coverage.last:
            await _walk(
                client,
                lake,
                key,
                result,
                "forward",
                since=coverage.last,
                until=until,
                max_bars=max_bars,
                max_pages=max_pages,
                count=count,
            )

        # Backward: older than the oldest stored bar, towards `since`.
        if since is None or since < coverage.first:
            await _walk(
                client,
                lake,
                key,
                result,
                "backward",
                since=since,
                until=coverage.first - step,
                max_bars=max_bars,
                max_pages=max_pages,
                count=count,
            )

    result.elapsed_seconds = time.monotonic() - started
    log.info("%s", result)
    return result


async def _walk(
    client: SaxoClient,
    lake: BarLake,
    key: SeriesKey,
    result: BackfillResult,
    direction: str,
    *,
    since: datetime | None,
    until: datetime | None,
    max_bars: int | None,
    max_pages: int,
    count: int,
) -> None:
    """Run one directional walk, committing each page to the lake as it lands."""
    walk = iter_history(
        client,
        uic=key.uic,
        asset_type=key.asset_type,
        horizon=key.horizon,
        since=since,
        until=until,
        count=count,
        max_pages=max_pages,
    )

    fetched_here = 0
    async for page in walk:
        written = lake.write(key, bars_to_frame(page.bars))
        result.pages += 1
        result.bars_fetched += len(page.bars)
        result.bars_added += written.rows_added
        result.bars_revised += written.rows_updated
        fetched_here += len(page.bars)

        earliest, latest = page.earliest, page.latest
        result.first = earliest if result.first is None else min(result.first, earliest)
        result.last = latest if result.last is None else max(result.last, latest)

        log.debug(
            "%s %s page %d: %d bars from %s, %d new",
            key,
            direction,
            result.pages,
            len(page.bars),
            earliest,
            written.rows_added,
        )

        if max_bars is not None and fetched_here >= max_bars:
            result.stopped_because[direction] = "max-bars"
            return

    result.stopped_because[direction] = walk.stopped_because or "complete"


async def backfill_all(
    client: SaxoClient,
    lake: BarLake,
    keys: list[SeriesKey],
    *,
    since: datetime | None = None,
    **options: object,
) -> list[BackfillResult]:
    """Backfill several series in sequence.

    Deliberately sequential. The client's rate limiter already keeps a single
    walk near Saxo's per-service-group ceiling, so running walks concurrently
    would not fetch faster -- it would just make each one slower and the logs
    harder to follow.
    """
    results = []
    for key in keys:
        results.append(await backfill(client, lake, key, since=since, **options))  # type: ignore[arg-type]
    return results


def days_ago(days: float) -> datetime:
    """UTC instant ``days`` before now -- convenience for CLI arguments."""
    return datetime.now(UTC) - timedelta(days=days)
