"""The history-depth spike: how much history does Saxo actually serve?

This exists because the answer is not documented, varies by horizon and by
instrument, and determines what can be built. A multi-timeframe LSTM on
1-minute bars needs years of them; if Saxo serves weeks, the model design has to
change, and it is far cheaper to learn that now than after the feature pipeline
is written around an assumption.

So this module measures rather than assumes. It walks each horizon back until
Saxo stops giving, and records how far it got, how many requests that cost, and
*why* the walk ended -- the last of which is the part that matters. A walk that
ends at ``page-cap`` was cut off by this tool and says nothing about Saxo's
limit; only ``first-sample`` and ``no-progress`` are real answers.

Nothing here writes to the lake. The spike is a measurement, and its output is a
report you keep -- ``docs/history-depth.md`` records what it returned.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from trader.saxo.charts import HORIZONS, MAX_COUNT, horizon_label, iter_history
from trader.saxo.client import SaxoClient

log = logging.getLogger(__name__)

# The horizons the strategy layer actually plans to use. Probing all fifteen
# costs requests without informing the design.
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 15, 60, 1440)

# Stop reasons that represent Saxo's own limit rather than ours.
CONCLUSIVE = frozenset({"first-sample", "no-progress", "empty-page"})


@dataclass
class DepthProbe:
    """The measured history depth for one instrument at one horizon."""

    horizon: int
    horizon_label: str
    bars: int
    pages: int
    first: datetime | None
    last: datetime | None
    span_days: float
    stopped_because: str
    conclusive: bool
    reported_first_sample: datetime | None
    elapsed_seconds: float

    def __str__(self) -> str:
        if not self.bars:
            return f"{self.horizon_label:>5}: no data ({self.stopped_because})"
        qualifier = "" if self.conclusive else "  (cut short -- more available)"
        return (
            f"{self.horizon_label:>5}: {self.bars:>8,} bars  "
            f"{self.span_days:>8,.0f} days back to {self.first:%Y-%m-%d}  "
            f"{self.pages:>3} pages{qualifier}"
        )


async def probe_horizon(
    client: SaxoClient,
    *,
    uic: int,
    asset_type: str,
    horizon: int,
    max_pages: int = 40,
    count: int = MAX_COUNT,
) -> DepthProbe:
    """Walk one horizon back as far as Saxo allows.

    Args:
        max_pages: the spike's own cap. At the default page size this is roughly
            48,000 bars -- 33 days of 1-minute bars, or 130 years of daily ones.
            Raise it when a probe comes back inconclusive.

    Returns:
        A :class:`DepthProbe`. Check ``conclusive`` before believing ``span_days``
        is Saxo's limit rather than ``max_pages``.
    """
    started = time.monotonic()
    walk = iter_history(
        client,
        uic=uic,
        asset_type=asset_type,
        horizon=horizon,
        count=count,
        max_pages=max_pages,
    )
    # Pages are counted, not kept: a 1-minute probe pulls tens of thousands of
    # bars, and the spike only needs their extent.
    async for _page in walk:
        pass

    elapsed = time.monotonic() - started
    span = (
        (walk.latest - walk.earliest).total_seconds() / 86400
        if walk.earliest and walk.latest
        else 0.0
    )
    reason = walk.stopped_because or "complete"
    return DepthProbe(
        horizon=horizon,
        horizon_label=horizon_label(horizon),
        bars=walk.bars,
        pages=walk.pages,
        first=walk.earliest,
        last=walk.latest,
        span_days=round(span, 2),
        stopped_because=reason,
        conclusive=reason in CONCLUSIVE,
        reported_first_sample=walk.reported_first_sample,
        elapsed_seconds=round(elapsed, 2),
    )


async def probe(
    client: SaxoClient,
    *,
    uic: int,
    asset_type: str,
    symbol: str = "",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_pages: int = 40,
    count: int = MAX_COUNT,
) -> dict[str, object]:
    """Probe several horizons and return a report.

    The report is JSON-serialisable via :func:`save_report` so a run can be
    committed and compared against a later one -- Saxo's retention has changed
    before and would otherwise change silently underneath a trained model.
    """
    for horizon in horizons:
        if horizon not in HORIZONS:
            raise ValueError(f"horizon {horizon} is not one Saxo serves")

    probes: list[DepthProbe] = []
    for horizon in horizons:
        log.info("probing %s at %s", symbol or uic, horizon_label(horizon))
        result = await probe_horizon(
            client,
            uic=uic,
            asset_type=asset_type,
            horizon=horizon,
            max_pages=max_pages,
            count=count,
        )
        log.info("%s", result)
        probes.append(result)

    return {
        "measured_at": datetime.now(UTC),
        "symbol": symbol,
        "uic": uic,
        "asset_type": asset_type,
        "page_size": count,
        "max_pages": max_pages,
        "requests": sum(p.pages for p in probes),
        "probes": [asdict(p) for p in probes],
    }


def render(report: dict[str, object]) -> str:
    """Format a report as a table for the terminal."""
    probes = [DepthProbe(**p) for p in report["probes"]]  # type: ignore[arg-type]
    header = (
        f"History depth for {report['symbol'] or report['uic']} "
        f"({report['asset_type']}, uic {report['uic']})"
    )
    lines = [header, "-" * len(header)]
    lines.extend(str(p) for p in probes)

    inconclusive = [p for p in probes if not p.conclusive and p.bars]
    if inconclusive:
        lines.append("")
        lines.append(
            "Cut short by the probe's own page cap at: "
            + ", ".join(p.horizon_label for p in inconclusive)
            + ". Re-run with a higher --max-pages for a real limit."
        )
    lines.append("")
    lines.append(f"{report['requests']} chart requests issued.")
    return "\n".join(lines)


def save_report(report: dict[str, object], path: Path) -> Path:
    """Write a report as JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
