"""Performance metrics for an equity curve and a trade blotter.

Nothing here is annualisation-agnostic, so the caller must supply
``periods_per_year`` -- the number of bars in a trading year at the backtest's
horizon. :func:`periods_per_year` derives it from the horizon and, when known,
the instrument's session length, because a 5-minute bar over a 6.5-hour equity
session is 78 bars/day, not ``1440 / 5``. The trading-year length is assumed to
be 252 days.

Ratios use a zero risk-free rate and sample standard deviation (``ddof=1``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class Metrics:
    """Summary statistics for one backtest run."""

    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    exposure: float
    turnover: float
    n_trades: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        rows = [
            ("Total return", f"{self.total_return:+.2%}"),
            ("CAGR", f"{self.cagr:+.2%}"),
            ("Ann. volatility", f"{self.ann_vol:.2%}"),
            ("Sharpe", f"{self.sharpe:.2f}"),
            ("Sortino", f"{self.sortino:.2f}"),
            ("Max drawdown", f"{self.max_drawdown:.2%}"),
            ("Calmar", f"{self.calmar:.2f}"),
            ("Hit rate", f"{self.hit_rate:.1%}"),
            ("Avg win / loss", f"{self.avg_win:+.2%} / {self.avg_loss:+.2%}"),
            ("Profit factor", f"{self.profit_factor:.2f}"),
            ("Exposure", f"{self.exposure:.1%}"),
            ("Turnover", f"{self.turnover:.1f}x"),
            ("Trades", f"{self.n_trades}"),
        ]
        width = max(len(name) for name, _ in rows)
        return "\n".join(f"  {name:<{width}}  {value}" for name, value in rows)


@dataclass(frozen=True)
class InstrumentStats:
    """Per-instrument trade attribution. Sums back to the portfolio result."""

    n_trades: int
    pnl: float
    return_on_start: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def periods_per_year(horizon: int, session_minutes: float | None = None) -> float:
    """Bars in a trading year at ``horizon`` minutes.

    Args:
        session_minutes: length of a regular session, close-open in minutes. When
            given, a session holds ``session_minutes / horizon + 1`` bars (the
            close time is the last bar's *open*). When ``None``, a 24h day is
            assumed, which is right for FX and wrong for an exchange session.
    """
    if session_minutes and session_minutes > 0:
        bars_per_day = session_minutes / horizon + 1
    else:
        bars_per_day = 1440 / horizon
    return bars_per_day * TRADING_DAYS_PER_YEAR


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute(
    equity: pd.Series,
    trades: pd.DataFrame,
    *,
    periods_per_year: float,
    exposure: float = float("nan"),
    turnover: float = float("nan"),
) -> Metrics:
    """Reduce an equity curve and a trade blotter to a :class:`Metrics`.

    Args:
        equity: portfolio equity indexed by event time, oldest first.
        trades: one row per closed round-trip; needs a ``return`` and a ``pnl``
            column. May be empty.
        exposure: fraction of bars with an open position; the engine measures it.
        turnover: traded notional over starting equity; the engine measures it.
    """
    equity = equity.astype(float)
    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    rets = equity.pct_change().dropna()

    total_return = _safe_div(end, start) - 1.0 if start else 0.0
    years = _safe_div(len(rets), periods_per_year)
    # Annualising a return measured over days rather than months explodes the
    # exponent into a meaningless number; report it only past a week of bars.
    if years >= 1.0 / 52.0 and start > 0 and end > 0:
        cagr = (end / start) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")

    std = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    ann_vol = std * math.sqrt(periods_per_year)
    mean = float(rets.mean()) if len(rets) else 0.0
    sharpe = _safe_div(mean, std) * math.sqrt(periods_per_year) if std else 0.0

    downside = rets[rets < 0]
    dstd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = _safe_div(mean, dstd) * math.sqrt(periods_per_year) if dstd else 0.0

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = _safe_div(cagr, abs(max_drawdown))

    hit_rate, avg_win, avg_loss, profit_factor, n_trades = _trade_stats(trades)

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        exposure=exposure,
        turnover=turnover,
        n_trades=n_trades,
    )


def _trade_stats(trades: pd.DataFrame) -> tuple[float, float, float, float, int]:
    if trades is None or trades.empty:
        return 0.0, 0.0, 0.0, 0.0, 0
    pnl = trades["pnl"].astype(float)
    ret = trades["return"].astype(float)
    wins, losses = ret[pnl > 0], ret[pnl < 0]
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return (
        _safe_div(float((pnl > 0).sum()), float(len(pnl))),
        float(wins.mean()) if len(wins) else 0.0,
        float(losses.mean()) if len(losses) else 0.0,
        _safe_div(gross_win, gross_loss) if gross_loss else float("inf") if gross_win else 0.0,
        int(len(pnl)),
    )


def instrument_stats(trades: pd.DataFrame, label: str, starting_cash: float) -> InstrumentStats:
    """Trade attribution for one instrument's slice of the blotter."""
    subset = trades[trades["label"] == label] if not trades.empty else trades
    hit_rate, avg_win, avg_loss, profit_factor, n_trades = _trade_stats(subset)
    pnl = float(subset["pnl"].sum()) if not subset.empty else 0.0
    return InstrumentStats(
        n_trades=n_trades,
        pnl=pnl,
        return_on_start=_safe_div(pnl, starting_cash),
        hit_rate=hit_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
    )
