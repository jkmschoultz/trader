"""CLI wiring: argument parsing, date handling, and error presentation.

Commands that talk to Saxo are covered by the module tests. What is checked
here is everything that happens before and after the network call, because that
is where a mistake shows up as a confusing message rather than a failed request.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from trader.cli import UsageError, _build_parser, _parse_since, main

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
    assert "Stock:211@1m" in capsys.readouterr().out
