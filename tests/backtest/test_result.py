"""``BacktestResult.to_dict`` -- the JSON report boundary."""

from __future__ import annotations

import json

import pandas as pd

from trader.backtest.metrics import InstrumentStats, Metrics
from trader.backtest.result import BacktestResult, _json_safe

_FILL_COLUMNS = ("time", "label", "side", "units", "price", "commission", "cash_after", "reason")


def _result(metrics: Metrics) -> BacktestResult:
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-03-01T14:30Z"), pd.Timestamp("2024-03-01T14:35Z")], name="time"
    )
    return BacktestResult(
        equity=pd.Series([100_000.0, 99_000.0], index=idx, name="equity"),
        fills=pd.DataFrame(columns=_FILL_COLUMNS),
        trades=pd.DataFrame(
            columns=("label", "entry_time", "exit_time", "pnl", "return", "exit_reason")
        ),
        metrics=metrics,
        by_instrument={"X": InstrumentStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, float("inf"))},
        config={"labels": ["X"], "horizon": 5},
    )


def test_json_safe_replaces_non_finite_floats():
    cleaned = _json_safe(
        {"a": float("nan"), "b": float("inf"), "c": [1.0, float("-inf")], "d": "ok"}
    )
    assert cleaned == {"a": None, "b": None, "c": [1.0, None], "d": "ok"}


def test_to_dict_is_strict_json_even_with_nan_metrics():
    metrics = Metrics(
        total_return=-0.01,
        cagr=float("nan"),
        ann_vol=0.0,
        sharpe=0.0,
        sortino=0.0,
        max_drawdown=-0.01,
        calmar=float("nan"),
        hit_rate=0.0,
        avg_win=0.0,
        avg_loss=0.0,
        profit_factor=float("inf"),
        exposure=0.0,
        turnover=float("nan"),
        n_trades=0,
    )
    report = _result(metrics).to_dict()

    # Strict parsers reject NaN/Infinity; this must round-trip.
    round_tripped = json.loads(json.dumps(report), parse_constant=_reject)
    assert round_tripped["metrics"]["cagr"] is None
    assert round_tripped["metrics"]["profit_factor"] is None
    assert round_tripped["by_instrument"]["X"]["profit_factor"] is None
    assert "fills" in report and report["n_fills"] == 0


def _reject(token: str):  # pragma: no cover - only runs if the report is dirty
    raise AssertionError(f"non-finite token in report: {token}")
