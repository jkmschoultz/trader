"""Bar-frame operations: grid alignment, resampling, and gap detection.

These are the checks that decide whether a stored series is fit to train on.
A model fed bars that silently skip an hour, or whose timestamps drift off the
horizon grid, learns the drift.

Everything here takes and returns canonical frames (see
:data:`trader.data.lake.BAR_COLUMNS`) and never mutates its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from trader.data.lake import BAR_COLUMNS, as_utc_timestamp, empty_frame, normalise

# How each column combines when several bars collapse into one. trading_state
# takes the first source bar's label, same as open -- a bin usually shares one
# state throughout, and the opening state is the one that decided whether the
# bin's own open print was tradable.
_AGGREGATION = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "interest": "last",
    "close_ask": "last",
    "trading_state": "first",
}

# The grid anchor for alignment checks: midnight UTC, Thursday 1 January 1970.
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


@dataclass(frozen=True)
class Gap:
    """A run of missing bars between two present ones."""

    after: datetime
    before: datetime
    missing: int

    @property
    def duration(self) -> timedelta:
        return self.before - self.after

    def __str__(self) -> str:
        return (
            f"{self.missing:,} bars missing between "
            f"{self.after:%Y-%m-%d %H:%M} and {self.before:%Y-%m-%d %H:%M} UTC"
        )


def is_aligned(frame: pd.DataFrame, horizon: int) -> bool:
    """True when every bar time sits exactly on the ``horizon``-minute grid."""
    return len(off_grid(frame, horizon)) == 0


def off_grid(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Return the rows whose timestamps are not on the horizon grid.

    The grid is anchored at the Unix epoch, which is midnight UTC on a Thursday.
    That is exact for every horizon Saxo serves up to daily. Weekly and monthly
    bars are anchored to calendar boundaries instead, so they are exempt rather
    than reported as universally misaligned.
    """
    if frame.empty or horizon >= 10080:
        return frame.iloc[0:0]
    # Taken as a Timedelta remainder rather than an integer one: the underlying
    # resolution of a timestamp column is not fixed, so raw int64 values are
    # microseconds or nanoseconds depending on where the frame came from.
    remainder = (frame["time"] - _EPOCH) % pd.Timedelta(minutes=horizon)
    return frame[remainder != pd.Timedelta(0)]


def align_to_grid(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Snap bar times down to the horizon grid, merging any collisions.

    Use this only after inspecting :func:`off_grid`. Snapping is the right repair
    for sub-second jitter in a feed; it is the wrong repair for a series fetched
    at the wrong horizon, which it would quietly compress instead of flagging.
    """
    if frame.empty or horizon >= 10080:
        return normalise(frame)
    snapped = frame.copy()
    snapped["time"] = snapped["time"].dt.floor(f"{horizon}min")
    return normalise(snapped)


def resample(frame: pd.DataFrame, *, source: int, target: int) -> pd.DataFrame:
    """Aggregate bars from a ``source`` horizon up to a ``target`` horizon.

    Bins with no source bars are dropped rather than forward-filled: an empty
    bin means the market was closed, and inventing a flat bar there would teach
    a model that price stands still at exactly the times it cannot be traded.

    Args:
        source: horizon of ``frame``, in minutes.
        target: horizon to produce; must be a whole multiple of ``source``.

    Raises:
        ValueError: the target is not a multiple of the source, or either
            horizon is weekly or longer, where fixed-minute bins do not line up
            with calendar boundaries. Fetch those horizons from Saxo directly.
    """
    if target < source or target % source:
        raise ValueError(f"target horizon {target} is not a whole multiple of source {source}")
    if max(source, target) >= 10080:
        raise ValueError(
            "weekly and monthly bars do not tile into fixed-minute bins; "
            "request that horizon from the chart endpoint instead"
        )
    if frame.empty:
        return empty_frame()
    if target == source:
        return normalise(frame)

    rule = f"{target}min"
    indexed = normalise(frame).set_index("time")
    binned = indexed.resample(rule, label="left", closed="left")
    out = binned.agg(_AGGREGATION)

    # `sum` turns an all-NaN volume bin into 0.0; restore NaN so "the venue
    # reports no volume" stays distinguishable from "zero shares traded".
    reported = binned["volume"].count().reindex(out.index, fill_value=0)
    out.loc[reported == 0, "volume"] = float("nan")

    out = out.dropna(subset=["open", "high", "low", "close"], how="all")
    return normalise(out.reset_index())


def gaps(frame: pd.DataFrame, horizon: int, *, min_missing: int = 1) -> list[Gap]:
    """Find every discontinuity larger than one bar.

    This is calendar-blind on purpose: it reports the overnight and weekend
    breaks of an exchange-traded instrument alongside genuine data loss. Pass
    the result through :func:`trader.data.calendars.drop_session_breaks` to keep
    only the gaps that fall inside trading hours.

    Args:
        min_missing: ignore gaps shorter than this many bars.
    """
    if len(frame) < 2:
        return []
    step = pd.Timedelta(minutes=horizon)
    times = frame["time"].reset_index(drop=True)
    deltas = times.diff()

    found: list[Gap] = []
    for position in deltas.index[deltas > step]:
        missing = int(deltas.iloc[position] / step) - 1
        if missing >= min_missing:
            found.append(
                Gap(
                    after=times.iloc[position - 1].to_pydatetime(),
                    before=times.iloc[position].to_pydatetime(),
                    missing=missing,
                )
            )
    return found


def clip(
    frame: pd.DataFrame,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Restrict a frame to ``[start, end]``, inclusive at both ends."""
    out = frame
    if start is not None:
        out = out[out["time"] >= as_utc_timestamp(start)]
    if end is not None:
        out = out[out["time"] <= as_utc_timestamp(end)]
    return out.reset_index(drop=True)


def describe(frame: pd.DataFrame, horizon: int) -> dict[str, object]:
    """A one-glance summary of a series, for the CLI and for logs."""
    if frame.empty:
        return {"rows": 0}
    discontinuities = gaps(frame, horizon)
    span = frame["time"].iloc[-1] - frame["time"].iloc[0]
    expected = int(span / pd.Timedelta(minutes=horizon)) + 1
    return {
        "rows": len(frame),
        "first": frame["time"].iloc[0].to_pydatetime(),
        "last": frame["time"].iloc[-1].to_pydatetime(),
        # Against a continuous grid, so anything with a session calendar scores
        # well below 1.0; it is a comparison tool between fetches of the same
        # instrument, not an absolute quality score.
        "density": round(len(frame) / expected, 4) if expected else 0.0,
        "gaps": len(discontinuities),
        "largest_gap_bars": max((g.missing for g in discontinuities), default=0),
        "off_grid": len(off_grid(frame, horizon)),
        "has_volume": bool(frame["volume"].notna().any()),
        "has_spread": bool(frame["close_ask"].notna().any()),
        "has_trading_state": bool(frame["trading_state"].notna().any()),
        "columns": list(BAR_COLUMNS),
    }
