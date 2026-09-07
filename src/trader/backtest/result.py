"""The object a backtest returns: equity curve, blotter, and metrics.

Kept separate from the engine so tests and notebooks can build one from parts,
and so :meth:`BacktestResult.to_dict` -- the JSON report, in the same spirit as
``trader.data.depth.save_report`` -- has one obvious home.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from trader.backtest.metrics import InstrumentStats, Metrics


def _json_safe(value):
    """Recursively replace non-finite floats (NaN, inf) with ``None``.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens, which ``JSON.parse``
    and strict decoders reject. A backtest over a handful of bars legitimately
    produces a NaN CAGR and an infinite profit factor, so the report has to be
    cleaned at this boundary rather than assumed finite.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class BacktestResult:
    """Everything one :func:`~trader.backtest.engine.run` produced."""

    equity: pd.Series
    fills: pd.DataFrame
    trades: pd.DataFrame
    metrics: Metrics
    by_instrument: Mapping[str, InstrumentStats]
    config: Mapping[str, object]

    @property
    def starting_equity(self) -> float:
        return float(self.equity.iloc[0])

    @property
    def final_equity(self) -> float:
        return float(self.equity.iloc[-1])

    def to_dict(self) -> dict[str, object]:
        """A JSON-serialisable summary (timestamps as ISO strings, no NaN/inf)."""
        return _json_safe(
            {
                "config": dict(self.config),
                "starting_equity": self.starting_equity,
                "final_equity": self.final_equity,
                "metrics": self.metrics.as_dict(),
                "by_instrument": {
                    label: stats.as_dict() for label, stats in self.by_instrument.items()
                },
                "equity": [
                    {"time": ts.isoformat(), "equity": float(value)}
                    for ts, value in self.equity.items()
                ],
                "trades": [
                    {
                        **row,
                        "entry_time": pd.Timestamp(row["entry_time"]).isoformat(),
                        "exit_time": pd.Timestamp(row["exit_time"]).isoformat(),
                    }
                    for row in self.trades.to_dict("records")
                ],
                "fills": [
                    {**row, "time": pd.Timestamp(row["time"]).isoformat()}
                    for row in self.fills.to_dict("records")
                ],
                "n_fills": int(len(self.fills)),
            }
        )

    def __str__(self) -> str:
        labels = ", ".join(str(label) for label in self.config.get("labels", []))
        lines = [
            f"Backtest: {labels}",
            f"  {self.config.get('allocator', '')} allocator, "
            f"horizon {self.config.get('horizon', '?')}m, "
            f"{self.config.get('events', '?')} events",
            f"  Equity {self.starting_equity:,.0f} -> {self.final_equity:,.0f}",
            "",
            str(self.metrics),
        ]
        if len(self.by_instrument) > 1:
            lines.append("")
            lines.append("  Per instrument:")
            width = max(len(label) for label in self.by_instrument)
            for label, stats in self.by_instrument.items():
                lines.append(
                    f"    {label:<{width}}  {stats.n_trades:>3} trades  "
                    f"pnl {stats.pnl:>+12,.0f}  ({stats.return_on_start:+.2%})  "
                    f"hit {stats.hit_rate:.0%}"
                )
        return "\n".join(lines)
