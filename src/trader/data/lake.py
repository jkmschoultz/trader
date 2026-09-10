"""The Parquet bar lake.

One series -- an ``(asset_type, uic, horizon)`` triple -- is stored as a set of
Parquet files partitioned by time, under::

    data/bars/{asset_type}/{uic}/{horizon}/{period}.parquet

Design decisions worth keeping
------------------------------
**Writes are idempotent.** :meth:`BarLake.write` merges into whatever is already
stored, deduplicating on bar time and letting the incoming row win. Re-running a
backfill therefore costs time and nothing else, which is what makes an
interrupted backfill safe to simply repeat.

**Partition width scales with the horizon.** A month of 1-minute bars is a
sensible file; a month of monthly bars is one row. The period key widens as the
horizon does, so files stay in a useful size range across every timeframe.

**Files are replaced atomically.** A merge writes a sibling temp file and renames
it over the target, so an interrupted write leaves the previous file intact
rather than a truncated one. Losing a page of a backfill is cheap; a corrupt
Parquet file in the middle of a series is not.

**Timestamps are UTC, always.** Local time enters the system only in
``trader.data.calendars``, at the point a human or an exchange session needs it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from trader.saxo.charts import Bar, horizon_label

log = logging.getLogger(__name__)

# The canonical bar schema. Every frame entering or leaving the lake has exactly
# these columns, in this order, so downstream code never branches on asset type.
BAR_COLUMNS: tuple[str, ...] = (
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "interest",
    "close_ask",
    "trading_state",
)
_PRICE_COLUMNS = ("open", "high", "low", "close", "volume", "interest", "close_ask")

# The venue's own session label for each bar. Kept as raw text rather than
# reduced to a boolean: Saxo's vocabulary is open-ended, and a state this code
# does not recognise today should still reach a later reader intact. Parquet
# dictionary-encodes it, so the storage cost of a low-cardinality string is
# close to nothing.
_STATE_COLUMN = "trading_state"
STATE_DTYPE = "str"

# Pinned explicitly rather than left to pandas. `to_datetime` infers a unit from
# its input -- microseconds from Python datetimes, nanoseconds from some Parquet
# files -- and concatenating two frames that disagree silently upcasts. Saxo
# timestamps carry microsecond precision, so that is the unit to standardise on.
TIME_DTYPE = "datetime64[us, UTC]"


class SeriesKey(NamedTuple):
    """Identifies one stored bar series."""

    asset_type: str
    uic: int
    horizon: int

    def __str__(self) -> str:
        return f"{self.asset_type}:{self.uic}@{horizon_label(self.horizon)}"


@dataclass(frozen=True)
class Coverage:
    """What the lake holds for one series."""

    key: SeriesKey
    rows: int
    first: datetime
    last: datetime
    files: int

    def __str__(self) -> str:
        return coverage_line(self)


def coverage_line(coverage: Coverage, symbol: str = "") -> str:
    """One-line coverage summary, optionally symbol-prefixed.

    ``symbol`` comes from :mod:`trader.data.instruments`, which the lake itself
    does not depend on -- callers that have a registry pass the label through,
    everyone else gets the bare ``asset_type:uic@horizon`` form.
    """
    key = coverage.key
    head = f"{symbol}:" if symbol else ""
    return (
        f"{head}{key.asset_type}:{key.uic}@{horizon_label(key.horizon)}: "
        f"{coverage.rows:,} bars "
        f"{coverage.first:%Y-%m-%d %H:%M} -> {coverage.last:%Y-%m-%d %H:%M} UTC "
        f"({coverage.files} file{'s' if coverage.files != 1 else ''})"
    )


@dataclass(frozen=True)
class WriteResult:
    """Outcome of one :meth:`BarLake.write`."""

    rows_in: int
    rows_added: int
    rows_updated: int
    files_touched: int


def empty_frame() -> pd.DataFrame:
    """An empty frame with the canonical schema and dtypes."""
    frame = pd.DataFrame(
        {
            "time": pd.Series([], dtype=TIME_DTYPE),
            **{name: pd.Series([], dtype="float64") for name in _PRICE_COLUMNS},
            _STATE_COLUMN: pd.Series([], dtype=STATE_DTYPE),
        }
    )
    return frame[list(BAR_COLUMNS)]


def as_utc_timestamp(moment: pd.Timestamp | datetime) -> pd.Timestamp:
    """Coerce an instant to a UTC :class:`pandas.Timestamp`, aware or not.

    ``pd.Timestamp(value, tz="UTC")`` raises when ``value`` already carries a
    zone, so comparisons against bar times need this rather than the constructor.
    """
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def bars_to_frame(bars: Iterable[Bar]) -> pd.DataFrame:
    """Convert :class:`~trader.saxo.charts.Bar` objects to a canonical frame."""
    records = [
        {
            "time": bar.time,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
            "interest": bar.interest,
            "close_ask": bar.close_ask,
            "trading_state": bar.trading_state,
        }
        for bar in bars
    ]
    if not records:
        return empty_frame()
    return normalise(pd.DataFrame.from_records(records))


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce any bar-shaped frame to the canonical schema, sorted and unique.

    Missing optional columns are filled with NaN rather than rejected, so a
    frame built from exchange-traded bars (no ``close_ask``) and one built from
    FX bars (no ``volume``) concatenate cleanly.
    """
    if frame.empty:
        return empty_frame()

    out = frame.copy()
    if "time" not in out.columns:
        raise ValueError("bar frame has no 'time' column")

    out["time"] = pd.to_datetime(out["time"], utc=True).astype(TIME_DTYPE)
    for name in _PRICE_COLUMNS:
        out[name] = pd.to_numeric(out[name], errors="coerce") if name in out else float("nan")
    out[_STATE_COLUMN] = (
        out[_STATE_COLUMN]
        if _STATE_COLUMN in out
        else pd.Series([pd.NA] * len(out), index=out.index)
    ).astype(STATE_DTYPE)

    out = out[list(BAR_COLUMNS)]
    # Keep the last row for a repeated timestamp: a re-fetch is a revision.
    out = out.drop_duplicates(subset="time", keep="last").sort_values("time")
    return out.reset_index(drop=True)


def period_key(moment: pd.Timestamp | datetime, horizon: int) -> str:
    """Partition label for ``moment`` at ``horizon``.

    Intraday bars partition monthly, hourly-to-daily yearly, and anything from
    daily up lands in a single file -- so no partition holds either millions of
    rows or a handful.
    """
    stamp = pd.Timestamp(moment)
    if horizon < 60:
        return f"{stamp.year:04d}-{stamp.month:02d}"
    if horizon < 1440:
        return f"{stamp.year:04d}"
    return "all"


class BarLake:
    """Reads and writes bar series under a root directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def directory(self, key: SeriesKey) -> Path:
        return self._root / "bars" / key.asset_type / str(key.uic) / str(key.horizon)

    def path_for(self, key: SeriesKey, period: str) -> Path:
        return self.directory(key) / f"{period}.parquet"

    def files(self, key: SeriesKey) -> list[Path]:
        """Every partition file for ``key``, in chronological order.

        ``sorted`` is chronological because the period labels are zero-padded
        (``2024-01`` < ``2024-10``); the single ``all`` partition is alone in its
        directory, so it never has to sort against dated ones.
        """
        directory = self.directory(key)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.parquet"))

    def series(self) -> list[SeriesKey]:
        """Every series present in the lake."""
        base = self._root / "bars"
        if not base.is_dir():
            return []
        found: list[SeriesKey] = []
        for horizon_dir in base.glob("*/*/*"):
            if not horizon_dir.is_dir():
                continue
            asset_type, uic, horizon = horizon_dir.parts[-3:]
            try:
                found.append(SeriesKey(asset_type, int(uic), int(horizon)))
            except ValueError:
                log.debug("ignoring unrecognised lake directory %s", horizon_dir)
        return sorted(found)

    def read(
        self,
        key: SeriesKey,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return stored bars for ``key`` in ``[start, end]``, oldest first."""
        paths = self.files(key)
        if not paths:
            return empty_frame()

        frames = [pd.read_parquet(path) for path in paths]
        out = normalise(pd.concat(frames, ignore_index=True))
        if start is not None:
            out = out[out["time"] >= as_utc_timestamp(start)]
        if end is not None:
            out = out[out["time"] <= as_utc_timestamp(end)]
        return out.reset_index(drop=True)

    def write(self, key: SeriesKey, frame: pd.DataFrame | Sequence[Bar]) -> WriteResult:
        """Merge bars into the lake, returning what changed.

        Rows are routed to partitions by their own timestamps, so a page
        straddling a month boundary updates both files.
        """
        incoming = normalise(frame) if isinstance(frame, pd.DataFrame) else bars_to_frame(frame)
        if incoming.empty:
            return WriteResult(0, 0, 0, 0)

        directory = self.directory(key)
        directory.mkdir(parents=True, exist_ok=True)

        added = updated = touched = 0
        periods = incoming["time"].map(lambda ts: period_key(ts, key.horizon))
        for period, chunk in incoming.groupby(periods, sort=True):
            path = self.path_for(key, str(period))
            existing = pd.read_parquet(path) if path.exists() else empty_frame()
            existing = normalise(existing)

            overlap = int(chunk["time"].isin(existing["time"]).sum())
            added += len(chunk) - overlap
            updated += overlap

            merged = normalise(pd.concat([existing, chunk], ignore_index=True))
            _write_atomic(merged, path)
            touched += 1

        log.debug("%s: +%d new, %d revised across %d file(s)", key, added, updated, touched)
        return WriteResult(len(incoming), added, updated, touched)

    def coverage(self, key: SeriesKey) -> Coverage | None:
        """Summarise what is stored for ``key``, or None if nothing is."""
        paths = self.files(key)
        if not paths:
            return None
        # Read only the time column: coverage on a large series should not pay
        # for the price data it does not look at.
        times = pd.concat([pd.read_parquet(p, columns=["time"]) for p in paths], ignore_index=True)
        if times.empty:
            return None
        stamps = pd.to_datetime(times["time"], utc=True).drop_duplicates().sort_values()
        return Coverage(
            key=key,
            rows=len(stamps),
            first=stamps.iloc[0].to_pydatetime(),
            last=stamps.iloc[-1].to_pydatetime(),
            files=len(paths),
        )

    def drop(self, key: SeriesKey) -> int:
        """Delete every partition for ``key``. Returns the file count removed."""
        paths = self.files(key)
        for path in paths:
            path.unlink()
        directory = self.directory(key)
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
        return len(paths)


def _write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to ``path`` via a temp file and a rename.

    The temp file is a sibling so the rename stays within one filesystem, where
    ``os.replace`` is atomic.
    """
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        frame.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
