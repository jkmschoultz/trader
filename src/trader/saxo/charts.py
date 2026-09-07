"""Historical OHLC bars -- the ``chart`` service group.

Saxo serves bars one page at a time and only lets you walk *backwards* through
history in useful sizes, so :class:`HistoryWalk` is the shape everything else
uses: an async iterator of pages, newest first, that stops on its own when the
instrument's history runs out.

The endpoint is ``/chart/v3/charts``. ``v1`` returns a 404 from an IIS error
page rather than a Saxo error body, so a wrong version here looks like a broken
gateway rather than a wrong URL -- worth stating, since the reference docs are
organised by service group and not by version.

Everything below was measured against SIM rather than taken from the docs:
``Count`` clamps silently at 1200, ``Mode=UpTo`` is inclusive of its anchor
instant, and equity bars cover regular trading hours only.

Two response shapes
-------------------
Exchange-traded instruments return ``Open/High/Low/Close/Volume`` plus a
per-bar ``MarketTradingState``. FX and other quote-driven instruments return no
last-traded price at all, only ``OpenBid/HighBid/.../CloseAsk``, and no trading
state. :class:`Bar` normalises both:

* ``open/high/low/close`` is the **bid** series for quote-driven instruments and
  the last-traded series for exchange instruments.
* ``close_ask`` is populated only for quote-driven instruments, so the spread --
  the dominant cost in intraday FX -- survives into the lake as
  ``close_ask - close`` rather than being averaged away into a mid price.
* ``trading_state`` is the venue's own session label for that bar, and is the
  most reliable session information available: it beats inferring hours from
  timestamps, for the instruments that report it.

Paging termination
------------------
Walking backwards has an obvious failure mode: if a request returns a page whose
oldest bar is not older than the previous page's, the loop never advances. That
happens at the true start of history, where Saxo clamps rather than returning
empty. :class:`HistoryWalk` treats lack of progress as a stop condition, which
is why it cannot spin forever even against an endpoint that misbehaves.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field

from trader.saxo.client import SaxoClient

log = logging.getLogger(__name__)

# The chart endpoint. v1 and v2 are gone; see the module docstring.
CHARTS_PATH = "/chart/v3/charts"

# Saxo accepts only these horizons, in minutes. Anything else is a 400.
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 360, 480, 1440, 10080, 43200)

# Per-request maximum, confirmed against SIM: asking for 2000 returns 1200 with
# no error. Silent truncation would make a backfill's progress arithmetic wrong,
# so requests are clamped rather than trusted.
MAX_COUNT = 1200

# The MarketTradingState values that mean continuous two-sided trading. Bars in
# any other state -- auctions, halts -- are real prints but not ones a backtest
# can assume it could have traded at.
TRADABLE_STATES = frozenset({"Automated", "AutomatedTrading", "CallAuctionTrading"})

Mode = Literal["From", "UpTo"]

_HORIZON_ALIASES: dict[str, int] = {
    "1m": 1, "2m": 2, "3m": 3, "5m": 5, "10m": 10, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480,
    "1d": 1440, "1w": 10080, "1mo": 43200,
}  # fmt: skip


class ChartError(ValueError):
    """A chart request that Saxo would reject, caught before it is sent."""


def parse_horizon(value: str | int) -> int:
    """Normalise ``"5m"``, ``"1h"``, ``"1d"``, or ``60`` to a horizon in minutes.

    Raises:
        ChartError: the horizon is not one Saxo serves.
    """
    if isinstance(value, str):
        text = value.strip().casefold()
        minutes = _HORIZON_ALIASES.get(text)
        if minutes is None:
            # Accept a bare number, and also a well-formed but unsupported
            # duration like "7m" or "3h", so those get the "not served by Saxo"
            # message below rather than being dismissed as unreadable.
            try:
                minutes = _minutes_from_duration(text)
            except ValueError as exc:
                raise ChartError(
                    f"unrecognised horizon {value!r}; use a plain number of minutes "
                    "or a label like 5m, 1h, or 1d"
                ) from exc
    else:
        minutes = int(value)

    if minutes not in HORIZONS:
        served = ", ".join(map(str, HORIZONS))
        raise ChartError(f"horizon {minutes} is not served by Saxo; choose one of {served}")
    return minutes


def _minutes_from_duration(text: str) -> int:
    """Convert ``"7"``, ``"7m"``, ``"3h"``, or ``"2d"`` to minutes.

    Raises:
        ValueError: the text is not a duration at all.
    """
    scale = {"m": 1, "h": 60, "d": 1440, "w": 10080}.get(text[-1:], None)
    if scale is None:
        return int(text)
    return int(text[:-1]) * scale


def horizon_label(minutes: int) -> str:
    """Inverse of :func:`parse_horizon`, for filenames and display."""
    for label, value in _HORIZON_ALIASES.items():
        if value == minutes:
            return label
    return f"{minutes}m"


class Bar(BaseModel):
    """One OHLC sample, timestamped at the bar's **opening** instant in UTC."""

    time: datetime = Field(alias="Time")
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    interest: float | None = None
    # Only for quote-driven instruments. spread = close_ask - close.
    close_ask: float | None = None
    # The venue's own session label, e.g. "Automated". Absent for FX.
    trading_state: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    @property
    def tradable(self) -> bool | None:
        """Whether the venue was in continuous trading, or None if it did not say."""
        if self.trading_state is None:
            return None
        return self.trading_state in TRADABLE_STATES

    @classmethod
    def from_sample(cls, sample: dict[str, Any]) -> Bar:
        """Build a bar from either Saxo response shape."""
        if "Close" in sample:
            return cls(
                Time=sample["Time"],
                open=sample["Open"],
                high=sample["High"],
                low=sample["Low"],
                close=sample["Close"],
                volume=sample.get("Volume"),
                interest=sample.get("Interest"),
                trading_state=sample.get("MarketTradingState"),
            )
        if "CloseBid" in sample:
            return cls(
                Time=sample["Time"],
                open=sample["OpenBid"],
                high=sample["HighBid"],
                low=sample["LowBid"],
                close=sample["CloseBid"],
                close_ask=sample.get("CloseAsk"),
                trading_state=sample.get("MarketTradingState"),
            )
        raise ChartError(f"chart sample has neither Close nor CloseBid: {sorted(sample)}")


class ChartInfo(BaseModel):
    """Metadata Saxo attaches to a chart response.

    ``first_sample_time`` is the authoritative answer to "how far back does this
    instrument go" -- when a page reaches it, there is nothing older to fetch.
    Saxo omits it for some instruments, which is precisely why the history-depth
    spike measures depth empirically instead of trusting this field.
    """

    exchange_id: str = Field(default="", alias="ExchangeId")
    horizon: int | None = Field(default=None, alias="Horizon")
    first_sample_time: datetime | None = Field(default=None, alias="FirstSampleTime")
    delayed_by_minutes: int = Field(default=0, alias="DelayedByMinutes")

    model_config = {"populate_by_name": True, "extra": "ignore"}


class ChartPage(NamedTuple):
    """One page of bars, oldest first, with the response's metadata."""

    bars: list[Bar]
    info: ChartInfo

    @property
    def earliest(self) -> datetime | None:
        return self.bars[0].time if self.bars else None

    @property
    def latest(self) -> datetime | None:
        return self.bars[-1].time if self.bars else None


async def get_chart(
    client: SaxoClient,
    *,
    uic: int,
    asset_type: str,
    horizon: int,
    count: int = MAX_COUNT,
    mode: Mode | None = None,
    time: datetime | None = None,
) -> ChartPage:
    """Fetch one page of bars.

    Args:
        horizon: bar size in minutes; must be one of :data:`HORIZONS`.
        count: bars to request, clamped to :data:`MAX_COUNT`.
        mode: ``"UpTo"`` ends the page at ``time``, ``"From"`` starts it there.
            Ignored when ``time`` is None, which returns the most recent bars.
            ``UpTo`` includes a bar opening exactly at ``time``.
        time: anchor instant. Naive values are read as UTC.

    Returns:
        A :class:`ChartPage` whose bars are sorted oldest first.
    """
    if horizon not in HORIZONS:
        raise ChartError(f"horizon {horizon} is not served by Saxo")

    params: dict[str, Any] = {
        "Uic": uic,
        "AssetType": asset_type,
        "Horizon": horizon,
        "Count": min(count, MAX_COUNT),
        "FieldGroups": "ChartInfo,Data",
    }
    if time is not None:
        params["Mode"] = mode or "UpTo"
        params["Time"] = _format_time(time)

    payload = await client.get(CHARTS_PATH, **params) or {}
    bars = [Bar.from_sample(s) for s in payload.get("Data", [])]
    # Saxo returns oldest-first today, but the ordering is not contractual and a
    # reversed page would corrupt the backwards walk. Sort rather than assume.
    bars.sort(key=lambda b: b.time)
    return ChartPage(bars, ChartInfo.model_validate(payload.get("ChartInfo", {})))


class HistoryWalk:
    """A backwards walk over one instrument's history, page by page.

    Async-iterate it to get pages newest-first; each page is sorted oldest-first
    internally. When iteration ends, :attr:`stopped_because` says why, which is
    the whole point of the class -- "I reached the date I asked for" and "Saxo
    ran out of history" look identical from the outside otherwise, and the
    history-depth spike exists to tell them apart.

    Stop reasons:
        ``reached-since``: walked back past the requested ``since``.
        ``first-sample``: reached the ``FirstSampleTime`` Saxo reported.
        ``no-progress``: a page failed to reach further back than the last one,
            which is how Saxo signals the true start of history for instruments
            that report no ``FirstSampleTime``.
        ``empty-page``: Saxo returned no bars at all.
        ``page-cap``: hit ``max_pages`` with history still available.
    """

    def __init__(
        self,
        client: SaxoClient,
        *,
        uic: int,
        asset_type: str,
        horizon: int,
        since: datetime | None = None,
        until: datetime | None = None,
        count: int = MAX_COUNT,
        max_pages: int = 1000,
    ) -> None:
        self._client = client
        self._uic = uic
        self._asset_type = asset_type
        self._horizon = horizon
        self._since = _as_utc(since)
        self._until = _as_utc(until)
        self._count = count
        self._max_pages = max_pages

        self.stopped_because: str | None = None
        self.pages = 0
        self.bars = 0
        self.earliest: datetime | None = None
        self.latest: datetime | None = None
        self.reported_first_sample: datetime | None = None

    async def __aiter__(self) -> AsyncIterator[ChartPage]:
        step = timedelta(minutes=self._horizon)
        anchor = self._until
        previous_earliest: datetime | None = None

        for page_number in range(1, self._max_pages + 1):
            page = await get_chart(
                self._client,
                uic=self._uic,
                asset_type=self._asset_type,
                horizon=self._horizon,
                count=self._count,
                mode="UpTo",
                time=anchor,
            )
            if page.info.first_sample_time is not None:
                self.reported_first_sample = _as_utc(page.info.first_sample_time)

            if not page.bars:
                self.stopped_because = "empty-page"
                return

            earliest = page.earliest
            if previous_earliest is not None and earliest >= previous_earliest:
                # Saxo clamps to the start of history rather than returning an
                # empty page, so yielding this one would repeat bars forever.
                log.debug(
                    "uic=%s horizon=%s: page %d did not advance past %s",
                    self._uic,
                    self._horizon,
                    page_number,
                    previous_earliest,
                )
                self.stopped_because = "no-progress"
                return

            yield page

            self.pages += 1
            self.bars += len(page.bars)
            self.earliest = earliest
            if self.latest is None:
                self.latest = page.latest
            previous_earliest = earliest

            if self._since is not None and earliest <= self._since:
                self.stopped_because = "reached-since"
                return
            if self.reported_first_sample is not None and earliest <= self.reported_first_sample:
                self.stopped_because = "first-sample"
                return

            # Anchor the next page one bar before this page's oldest sample.
            # UpTo is inclusive of its anchor, so reusing `earliest` refetches it.
            anchor = earliest - step

        self.stopped_because = "page-cap"

    @property
    def exhausted_history(self) -> bool:
        """True when the walk ended because Saxo had nothing older to give."""
        return self.stopped_because in {"first-sample", "no-progress", "empty-page"}


def iter_history(
    client: SaxoClient,
    *,
    uic: int,
    asset_type: str,
    horizon: int,
    since: datetime | None = None,
    until: datetime | None = None,
    count: int = MAX_COUNT,
    max_pages: int = 1000,
) -> HistoryWalk:
    """Walk an instrument's history backwards.

    The returned :class:`HistoryWalk` is async-iterable and, once drained,
    reports how far it got and why it stopped::

        walk = iter_history(client, uic=211, asset_type="Stock", horizon=1)
        async for page in walk:
            ...
        print(walk.stopped_because, walk.earliest)

    Args:
        since: stop once bars older than this have been returned. ``None`` walks
            to the start of available history.
        until: start the walk here instead of at the most recent bar.
        max_pages: hard cap, so a misbehaving endpoint cannot loop forever.
    """
    return HistoryWalk(
        client,
        uic=uic,
        asset_type=asset_type,
        horizon=horizon,
        since=since,
        until=until,
        count=count,
        max_pages=max_pages,
    )


def _as_utc(moment: datetime | None) -> datetime | None:
    """Attach UTC to a naive datetime; convert an aware one."""
    if moment is None:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _format_time(moment: datetime) -> str:
    """Render an instant the way Saxo's chart endpoint expects it."""
    return _as_utc(moment).strftime("%Y-%m-%dT%H:%M:%S.000000Z")
