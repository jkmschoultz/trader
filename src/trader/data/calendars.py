"""Session calendars: when an instrument actually trades.

There are two separate questions here, and they need different answers.

**"Is it open right now, and when does it next open?"** is a live-trading
question, answered by :class:`SessionCalendar`, built from the session windows
Saxo publishes in ``/ref/v1/exchanges``. Those windows cover a short rolling
period around now -- authoritative for the present, useless for the past.

**"Which minutes of 12 March 2023 were tradable?"** is a research question, and
Saxo will not answer it. :func:`infer_regular_hours` recovers it from the bars
themselves: the modal first and last bar of each local trading day. Using the
*mode* rather than the extremes is what makes it robust -- an early close on
Christmas Eve or a single stray pre-market print shifts the extremes but not the
typical day.

Local time exists in this module and nowhere else in the data layer. Everything
crossing the boundary in or out is UTC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta, tzinfo
from datetime import timezone as fixed_offset_zone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from trader.data.bars import Gap
from trader.saxo.exchanges import Exchange, ExchangeSession

log = logging.getLogger(__name__)

# Saxo's /ref/v1/exchanges gives no IANA or Windows zone name anywhere in the
# response -- confirmed by surveying a full SIM listing (266 exchanges). What it
# gives instead is a small internal integer ``TimeZone`` code, an abbreviation
# ("EDT", "CEST"), and the zone's *current* offset. Only ~20 codes cover the
# whole listing, so mapping the codes we actually trade against is tractable;
# this table was built from that survey and should be extended the same way
# (fetch /ref/v1/exchanges, group by TimeZone) as new venues are added.
#
# A code intentionally maps to one representative IANA zone even where several
# real cities share it (code 4 "CEST" covers Frankfurt, Paris, and Milan, all of
# which observe CET/CEST identically) -- what matters for bar alignment is the
# UTC offset and DST calendar, and those agree across the group.
_SAXO_TIMEZONE_CODE_TO_IANA: dict[int, str] = {
    0: "UTC",
    1: "Europe/London",  # BST -- LSE, LIFFE, LME, IPE, ...
    2: "Asia/Singapore",  # SGT
    3: "America/New_York",  # EDT -- AMEX, NASDAQ, NYSE, COMEX, ...
    4: "Europe/Berlin",  # CEST -- most of continental Europe
    5: "America/Chicago",  # CDT -- CME, CBOT, CBOE, GLOBEX, ...
    6: "America/Los_Angeles",  # PDT
    7: "Asia/Hong_Kong",  # HKT
    8: "Australia/Sydney",  # AEST
    9: "Pacific/Auckland",  # NZST
    10: "Asia/Tokyo",  # JST
    12: "Europe/Moscow",  # MSK
    16: "Africa/Johannesburg",  # SAST
    20: "America/Sao_Paulo",  # BRT
    284: "Asia/Shanghai",  # CST (mainland China venues; no DST)
    290: "Asia/Tokyo",  # JST -- APAC dual-listed group
}


def resolve_timezone(exchange: Exchange) -> tzinfo:
    """Best available zone for ``exchange``, degrading rather than failing.

    Tries Saxo's internal ``TimeZone`` code against :data:`_SAXO_TIMEZONE_CODE_TO_IANA`,
    then falls back to a fixed offset built from ``TimeZoneOffset``. The
    fallback is right *today* and wrong across a DST transition, since the
    offset Saxo reports already reflects today's DST state rather than being a
    zone definition -- so a code mapping is always preferable, which is why an
    unmapped code is logged.
    """
    if exchange.time_zone_code is not None:
        iana = _SAXO_TIMEZONE_CODE_TO_IANA.get(exchange.time_zone_code)
        if iana is not None:
            try:
                return ZoneInfo(iana)
            except ZoneInfoNotFoundError:
                pass

    offset = _parse_offset(exchange.time_zone_offset)
    if offset is not None:
        log.warning(
            "exchange %s has unmapped TimeZone code %r (%s); using its current offset %s, "
            "which will be wrong across the next DST transition",
            exchange.exchange_id,
            exchange.time_zone_code,
            exchange.time_zone_abbreviation or "no abbreviation",
            exchange.time_zone_offset,
        )
        return fixed_offset_zone(offset, exchange.time_zone_abbreviation or "fixed")

    log.warning("exchange %s has no usable time zone; falling back to UTC", exchange.exchange_id)
    return UTC


@dataclass(frozen=True)
class SessionCalendar:
    """The sessions Saxo currently publishes for one exchange.

    Only valid around the moment it was fetched. :meth:`is_open` returns None
    rather than a guess for instants outside the published window, so a caller
    can tell "closed" apart from "unknown".
    """

    exchange_id: str
    timezone: tzinfo
    all_day: bool
    sessions: tuple[ExchangeSession, ...]

    @classmethod
    def from_exchange(cls, exchange: Exchange) -> SessionCalendar:
        return cls(
            exchange_id=exchange.exchange_id,
            timezone=resolve_timezone(exchange),
            all_day=exchange.all_day,
            sessions=tuple(sorted(exchange.sessions, key=lambda s: s.start)),
        )

    @property
    def covers(self) -> tuple[datetime, datetime] | None:
        """The span the published sessions actually describe."""
        if not self.sessions:
            return None
        return self.sessions[0].start, self.sessions[-1].end

    def state_at(self, moment: datetime | None = None) -> str | None:
        """Saxo's session state at ``moment``, or None if it is not covered."""
        moment = _as_utc(moment or datetime.now(UTC))
        for session in self.sessions:
            if session.contains(moment):
                return session.state
        return None

    def is_open(self, moment: datetime | None = None) -> bool | None:
        """True during continuous trading, False when closed, None if unknown.

        ``all_day`` venues -- FX, most CFDs -- report no sessions at all, and are
        reported open. They still close at weekends, which the published sessions
        do not describe; a weekend check belongs to the strategy, not here.
        """
        if self.all_day and not self.sessions:
            return True
        state = self.state_at(moment)
        if state is None:
            return None
        return state in {"AutomatedTrading", "CallAuctionTrading"}

    def next_transition(self, moment: datetime | None = None) -> tuple[datetime, str] | None:
        """The next published session boundary and the state it begins."""
        moment = _as_utc(moment or datetime.now(UTC))
        for session in self.sessions:
            if session.start > moment:
                return session.start, session.state
        return None

    def next_open(self, moment: datetime | None = None) -> datetime | None:
        """Start of the next continuous-trading session, if one is published."""
        moment = _as_utc(moment or datetime.now(UTC))
        for session in self.sessions:
            if session.is_regular and session.start > moment:
                return session.start
        return None


@dataclass(frozen=True)
class RegularHours:
    """Typical trading hours for an instrument, inferred from its own bars.

    ``open_time`` and ``close_time`` are local wall-clock times in ``timezone``,
    and ``close_time`` is the opening time of the last bar of a normal day -- not
    the instant the bell rings. Comparing a bar's timestamp against them is
    therefore inclusive at both ends.
    """

    timezone: tzinfo
    open_time: time
    close_time: time
    weekdays: frozenset[int]
    days_observed: int

    def contains(self, moment: datetime) -> bool:
        """True when ``moment`` falls inside a typical session."""
        local = _as_utc(moment).astimezone(self.timezone)
        if local.weekday() not in self.weekdays:
            return False
        if self.open_time <= self.close_time:
            return self.open_time <= local.time() <= self.close_time
        # An overnight session, e.g. a venue whose day starts before midnight.
        return local.time() >= self.open_time or local.time() <= self.close_time

    def __str__(self) -> str:
        days = ",".join(
            ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d] for d in sorted(self.weekdays)
        )
        return (
            f"{self.open_time:%H:%M}-{self.close_time:%H:%M} {self.timezone} "
            f"on {days} (from {self.days_observed} days)"
        )


def session_table(frame: pd.DataFrame, timezone: tzinfo) -> pd.DataFrame:
    """Summarise a bar frame one local trading day per row.

    Columns: ``date`` (local), ``bars``, ``first``, ``last`` (local wall-clock
    times), ``open``, ``close``, ``high``, ``low``, ``volume``. This is the raw
    material for :func:`infer_regular_hours` and a useful sanity check by itself
    -- a day with a tenth of the usual bar count is a data problem, not a
    half-day, unless the exchange says otherwise.
    """
    if frame.empty:
        return pd.DataFrame(
            columns=["date", "bars", "first", "last", "open", "close", "high", "low", "volume"]
        )

    local = frame["time"].dt.tz_convert(timezone)
    grouped = frame.assign(_date=local.dt.date, _clock=local.dt.time).groupby("_date", sort=True)
    table = pd.DataFrame(
        {
            "bars": grouped.size(),
            "first": grouped["_clock"].first(),
            "last": grouped["_clock"].last(),
            "open": grouped["open"].first(),
            "close": grouped["close"].last(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "volume": grouped["volume"].sum(min_count=1),
        }
    )
    return table.reset_index(names="date")


def infer_regular_hours(
    frame: pd.DataFrame,
    timezone: tzinfo,
    *,
    min_days: int = 5,
) -> RegularHours | None:
    """Infer typical session hours from the bars themselves.

    Takes the modal first-bar and last-bar time across local trading days, and
    the set of weekdays that trade at all. The mode ignores half-days, holidays,
    and the odd out-of-hours print, which is exactly what a *typical* session
    means.

    Args:
        min_days: refuse to infer from fewer local days than this. A calendar
            guessed from two days of data is worse than no calendar.

    Returns:
        The inferred hours, or None when there is too little data.
    """
    table = session_table(frame, timezone)
    if len(table) < min_days:
        return None

    open_time = table["first"].mode()
    close_time = table["last"].mode()
    if open_time.empty or close_time.empty:
        return None

    weekdays = {pd.Timestamp(d).weekday() for d in table["date"]}
    return RegularHours(
        timezone=timezone,
        open_time=open_time.iloc[0],
        close_time=close_time.iloc[0],
        weekdays=frozenset(weekdays),
        days_observed=len(table),
    )


def filter_regular_hours(frame: pd.DataFrame, hours: RegularHours) -> pd.DataFrame:
    """Keep only the bars falling inside typical trading hours.

    Restricting an intraday model to regular hours is usually right: extended
    hours trade at a fraction of the volume with several times the spread, so
    they contribute mostly noise and unfillable signals.
    """
    if frame.empty:
        return frame
    keep = frame["time"].map(hours.contains)
    return frame[keep].reset_index(drop=True)


def drop_session_breaks(found: list[Gap], hours: RegularHours) -> list[Gap]:
    """Keep only the gaps that represent real missing data.

    A gap is kept when both of its edges sit inside typical trading hours on the
    same local day. Overnight and weekend breaks fail that test, which is what
    separates "the exchange was shut" from "we lost an hour of bars".
    """
    kept: list[Gap] = []
    for gap in found:
        if not (hours.contains(gap.after) and hours.contains(gap.before)):
            continue
        left = _as_utc(gap.after).astimezone(hours.timezone).date()
        right = _as_utc(gap.before).astimezone(hours.timezone).date()
        if left == right:
            kept.append(gap)
    return kept


def trading_days(frame: pd.DataFrame, timezone: tzinfo) -> list[datetime]:
    """Local dates on which at least one bar was printed."""
    if frame.empty:
        return []
    local = frame["time"].dt.tz_convert(timezone)
    return sorted({d for d in local.dt.normalize().dt.to_pydatetime()})


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _parse_offset(text: str) -> timedelta | None:
    """Parse Saxo's ``"-05:00:00"`` offset format."""
    if not text:
        return None
    sign = -1 if text.startswith("-") else 1
    try:
        hours, minutes, *rest = (int(part) for part in text.lstrip("+-").split(":"))
    except ValueError:
        return None
    return sign * timedelta(hours=hours, minutes=minutes, seconds=rest[0] if rest else 0)
