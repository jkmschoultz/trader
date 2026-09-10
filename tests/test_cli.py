"""CLI wiring: argument parsing, date handling, and error presentation.

Commands that talk to Saxo are covered by the module tests. What is checked
here is everything that happens before and after the network call, because that
is where a mistake shows up as a confusing message rather than a failed request.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trader.cli import UsageError, _build_parser, _parse_params, _parse_since, main

# ------------------------------------------------------------------- --since


def test_since_accepts_a_plain_date():
    assert _parse_since("2024-03-01") == datetime(2024, 3, 1, tzinfo=UTC)


def test_since_accepts_an_iso_timestamp_with_a_zone():
    assert _parse_since("2024-03-01T14:30:00+02:00") == datetime(2024, 3, 1, 12, 30, tzinfo=UTC)


def test_a_naive_timestamp_is_read_as_utc():
    """Guessing the local zone would silently shift a backfill window."""
    assert _parse_since("2024-03-01T14:30:00") == datetime(2024, 3, 1, 14, 30, tzinfo=UTC)


@pytest.mark.parametrize(("given", "days"), [("90d", 90), ("1.5d", 1.5), ("2y", 730.5)])
def test_since_accepts_a_lookback(given, days):
    delta = datetime.now(UTC) - _parse_since(given)
    assert abs(delta.total_seconds() - days * 86400) < 5


@pytest.mark.parametrize("given", [None, "", "all", "max"])
def test_no_lower_bound_means_walk_to_the_start_of_history(given):
    assert _parse_since(given) is None


@pytest.mark.parametrize("given", ["last tuesday", "2024-13-45", "soon"])
def test_an_unreadable_since_explains_the_accepted_formats(given):
    with pytest.raises(UsageError, match="YYYY-MM-DD"):
        _parse_since(given)


# ------------------------------------------------------------ argument surface


def test_backfill_defaults_to_one_minute_bars_and_resuming():
    args = _build_parser().parse_args(["data", "backfill", "AAPL:xnas"])
    assert args.horizon == "1m"
    assert args.since is None
    assert args.no_resume is False


def test_backfill_accepts_several_horizons():
    args = _build_parser().parse_args(
        ["data", "backfill", "AAPL:xnas", "--horizon", "1m,5m,1h", "--since", "90d"]
    )
    assert args.horizon == "1m,5m,1h"


def test_data_symbols_parses_the_refresh_flag():
    args = _build_parser().parse_args(["data", "symbols", "--refresh"])
    assert args.data_command == "symbols"
    assert args.refresh is True
    assert _build_parser().parse_args(["data", "symbols"]).refresh is False


def test_depth_has_a_conservative_default_page_cap():
    """The spike should not pull a month of 1-minute bars unless asked to."""
    args = _build_parser().parse_args(["data", "depth", "AAPL:xnas"])
    assert args.max_pages == 40
    assert args.horizons is None


def test_exchange_id_is_optional():
    assert _build_parser().parse_args(["exchange"]).exchange_id is None
    assert _build_parser().parse_args(["exchange", "NASDAQ"]).exchange_id == "NASDAQ"


def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["data"])


# -------------------------------------------------------------- error surface


def test_an_unreadable_since_exits_two_without_a_traceback(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    code = main(["data", "backfill", "AAPL:xnas", "--since", "whenever"])

    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_an_unsupported_horizon_exits_two_and_names_the_valid_ones(capsys):
    code = main(["data", "backfill", "AAPL:xnas", "--horizon", "7m"])

    assert code == 2
    err = capsys.readouterr().err
    assert "7 is not served by Saxo" in err
    assert "1440" in err


def test_coverage_of_an_empty_lake_exits_one_with_a_next_step(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    code = main(["data", "coverage"])

    assert code == 1
    assert "trader data backfill" in capsys.readouterr().out


def test_coverage_lists_stored_series(capsys, monkeypatch, tmp_path):
    from trader.data.lake import BarLake, SeriesKey, bars_to_frame
    from trader.saxo.charts import Bar

    lake = BarLake(tmp_path)
    lake.write(
        SeriesKey("Stock", 211, 1),
        bars_to_frame(
            [Bar(Time=datetime(2024, 3, 1, tzinfo=UTC), open=1.0, high=1.0, low=1.0, close=1.0)]
        ),
    )

    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    assert main(["data", "coverage"]) == 0
    out = capsys.readouterr().out
    assert "Stock:211@1m" in out
    # no symbol recorded yet -> the command points at the fix
    assert "trader data symbols" in out


def test_coverage_shows_the_symbol_once_the_registry_knows_it(capsys, monkeypatch, tmp_path):
    from trader.data.instruments import InstrumentRegistry
    from trader.data.lake import BarLake, SeriesKey, bars_to_frame
    from trader.saxo.charts import Bar

    BarLake(tmp_path).write(
        SeriesKey("CfdOnIndex", 4913, 1),
        bars_to_frame(
            [Bar(Time=datetime(2024, 3, 1, tzinfo=UTC), open=1.0, high=1.0, low=1.0, close=1.0)]
        ),
    )
    InstrumentRegistry(tmp_path).put("CfdOnIndex", 4913, symbol="US500.I")

    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    assert main(["data", "coverage"]) == 0
    assert "US500.I:CfdOnIndex:4913@1m" in capsys.readouterr().out


# --------------------------------------------------------------------- backtest


def _seed_5m_lake(root, uic=211, bars=60):
    """A gently trending 5-minute Stock series, enough for a warmup plus trades."""
    from datetime import timedelta

    from trader.data.lake import BarLake, SeriesKey, bars_to_frame
    from trader.saxo.charts import Bar

    start = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)
    prices = [100.0 + (i % 20) - 10 for i in range(bars)]  # sawtooth: real crossovers
    lake = BarLake(root)
    lake.write(
        SeriesKey("Stock", uic, 5),
        bars_to_frame(
            [
                Bar(
                    Time=start + timedelta(minutes=5 * i),
                    open=p,
                    high=p + 0.5,
                    low=p - 0.5,
                    close=p,
                    volume=1_000.0,
                )
                for i, p in enumerate(prices)
            ]
        ),
    )


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("fast=10", ("fast", 10)),
        ("stop=0.005", ("stop", 0.005)),
        ("long_only=true", ("long_only", True)),
        ("max_bars=none", ("max_bars", None)),
        ("label=orb", ("label", "orb")),
    ],
)
def test_parse_params_coerces_by_type(given, expected):
    key, value = expected
    assert _parse_params([given]) == {key: value}


def test_parse_params_rejects_a_bare_token():
    with pytest.raises(UsageError, match="KEY=VALUE"):
        _parse_params(["justakey"])


def test_backtest_takes_repeated_symbols_and_params():
    args = _build_parser().parse_args(
        [
            "backtest",
            "--symbol",
            "AAPL:xnas",
            "--symbol",
            "MSFT:xnas",
            "--strategy",
            "ma_cross",
            "--param",
            "fast=5",
            "--param",
            "slow=20",
        ]
    )
    assert args.symbol == ["AAPL:xnas", "MSFT:xnas"]
    assert args.param == ["fast=5", "slow=20"]
    assert args.horizon == "5m"
    assert args.allocator == "equal-weight"


def test_backtest_unknown_strategy_exits_two_and_lists_the_known_ones(capsys):
    code = main(["backtest", "--symbol", "X", "--uic", "211", "--strategy", "nope"])
    assert code == 2
    assert "ma_cross" in capsys.readouterr().err


def test_backtest_on_an_empty_lake_points_at_backfill(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    code = main(
        [
            "backtest",
            "--symbol",
            "X",
            "--uic",
            "211",
            "--asset-type",
            "Stock",
            "--strategy",
            "ma_cross",
            "--horizon",
            "5m",
        ]
    )
    assert code == 2
    assert "trader data backfill" in capsys.readouterr().err


def test_backtest_golden_run_prints_metrics_and_writes_a_report(capsys, monkeypatch, tmp_path):
    _seed_5m_lake(tmp_path)
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    out = tmp_path / "report.json"

    code = main(
        [
            "backtest",
            "--symbol",
            "X",
            "--uic",
            "211",
            "--asset-type",
            "Stock",
            "--strategy",
            "ma_cross",
            "--horizon",
            "5m",
            "--param",
            "fast=3",
            "--param",
            "slow=8",
            "--fee-bps",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "Total return" in printed and "Sharpe" in printed

    import json

    report = json.loads(out.read_text())
    assert report["config"]["labels"] == ["X"]
    assert "metrics" in report and "equity" in report


# ------------------------------------------------------------------- features / labels / train


def test_features_build_parses_context_and_flags():
    args = _build_parser().parse_args(
        ["features", "build", "--symbol", "AAPL", "--context", "15m,1h", "--persist"]
    )
    assert args.features_command == "build"
    assert args.context == "15m,1h"
    assert args.persist is True
    assert args.feature_set == "price_v1"


def test_labels_defaults():
    args = _build_parser().parse_args(["labels", "--symbol", "AAPL"])
    assert args.stop == 0.005 and args.take == 0.01 and args.max_bars == 24


def test_train_requires_the_split_dates():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["train", "--symbol", "X"])

    args = _build_parser().parse_args(
        [
            "train",
            "--symbol",
            "X",
            "--symbol",
            "Y",
            "--feature-set",
            "mtf_v1",
            "--context",
            "15m,1h",
            "--window",
            "48",
            "--train-end",
            "2025-04-01",
            "--val-end",
            "2025-06-01",
            "--sample-weights",
        ]
    )
    assert args.symbol == ["X", "Y"]
    assert args.window == 48
    assert args.sample_weights is True
    assert args.epochs == 40


def test_tune_parses_grid_and_cv_flags():
    from trader.cli import _parse_grid

    args = _build_parser().parse_args(
        [
            "tune",
            "--symbol",
            "US500",
            "--asset-type",
            "CfdOnIndex",
            "--horizon",
            "15m",
            "--feature-set",
            "mtf_v1",
            "--context",
            "1h,4h",
            "--grid",
            "window=16,32",
            "--grid",
            "stop=0.004,0.008",
            "--folds",
            "4",
            "--fee-bps",
            "0.2",
            "--out",
            "state/sweep.json",
        ]
    )
    assert args.grid == ["window=16,32", "stop=0.004,0.008"]
    assert args.folds == 4 and args.cv_mode == "rolling"
    assert args.workers == 0
    assert _parse_grid(args.grid) == {"window": [16, 32], "stop": [0.004, 0.008]}


def test_tune_requires_a_grid():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["tune", "--symbol", "X"])


def test_models_needs_a_subcommand():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["models"])
    assert _build_parser().parse_args(["models", "show", "abc"]).model_id == "abc"


def test_models_list_on_an_empty_registry_exits_one(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    code = main(["models", "list"])
    assert code == 1
    assert "trader train" in capsys.readouterr().out


@pytest.mark.slow
def test_train_golden_run_registers_a_model(capsys, monkeypatch, tmp_path, seed_trainable_lake):
    pytest.importorskip("torch")
    seed_trainable_lake(tmp_path, bars=1600)
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))

    code = main(
        [
            "train",
            "--symbol",
            "X",
            "--uic",
            "211",
            "--asset-type",
            "Stock",
            "--horizon",
            "5m",
            "--stop",
            "0.01",
            "--take",
            "0.01",
            "--max-bars",
            "6",
            "--window",
            "8",
            "--train-end",
            "2024-01-08",
            "--val-end",
            "2024-01-10",
            "--hidden",
            "8",
            "--layers",
            "1",
            "--epochs",
            "2",
            "--batch-size",
            "16",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "model_id: lstm-" in out

    assert main(["models", "list"]) == 0
    assert "lstm-" in capsys.readouterr().out


@pytest.mark.slow
def test_tune_golden_run_prints_a_ranked_table(capsys, monkeypatch, tmp_path, seed_trainable_lake):
    pytest.importorskip("torch")
    seed_trainable_lake(tmp_path, bars=4000)
    monkeypatch.setenv("TRADER_DATA_DIR", str(tmp_path))
    out_path = tmp_path / "sweep.json"

    code = main(
        [
            "tune",
            "--symbol",
            "X",
            "--uic",
            "211",
            "--asset-type",
            "Stock",
            "--horizon",
            "5m",
            "--stop",
            "0.01",
            "--take",
            "0.01",
            "--max-bars",
            "6",
            "--window",
            "8",
            "--hidden",
            "8",
            "--layers",
            "1",
            "--epochs",
            "2",
            "--batch-size",
            "16",
            "--folds",
            "2",
            "--train-days",
            "8",
            "--val-days",
            "2",
            "--test-days",
            "1.5",
            "--grid",
            "threshold=0.0,0.3",
            "--workers",
            "1",
            "--out",
            str(out_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "Tuning sweep [lstm]: 2 configs" in out
    assert out_path.is_file()
