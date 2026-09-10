"""The instrument registry: the symbol side table for the bar lake."""

from __future__ import annotations

import json

from trader.data.instruments import FILENAME, InstrumentRecord, InstrumentRegistry


def test_put_then_get_round_trips_through_the_file(tmp_path):
    reg = InstrumentRegistry(tmp_path)
    reg.put("CfdOnIndex", 4913, symbol="US500.I", description="S&P 500", currency="USD")

    # a fresh instance reads what the first one wrote
    again = InstrumentRegistry(tmp_path)
    record = again.get("CfdOnIndex", 4913)
    assert record is not None
    assert (record.symbol, record.description, record.currency) == ("US500.I", "S&P 500", "USD")
    assert record.updated is not None


def test_the_file_lands_where_expected_and_is_valid_json(tmp_path):
    InstrumentRegistry(tmp_path).put("FxSpot", 21, symbol="EURUSD")
    path = tmp_path / FILENAME
    assert path.is_file()
    assert json.loads(path.read_text())["FxSpot:21"]["symbol"] == "EURUSD"


def test_label_prefixes_the_symbol_when_known(tmp_path):
    reg = InstrumentRegistry(tmp_path)
    reg.put("CfdOnIndex", 4913, symbol="US500.I")
    assert reg.label("CfdOnIndex", 4913) == "US500.I:CfdOnIndex:4913"


def test_label_falls_back_to_the_bare_key_when_unknown(tmp_path):
    assert InstrumentRegistry(tmp_path).label("CfdOnIndex", 4910) == "CfdOnIndex:4910"
    assert InstrumentRegistry(tmp_path).symbol_for("CfdOnIndex", 4910) == ""


def test_put_replaces_rather_than_duplicates(tmp_path):
    reg = InstrumentRegistry(tmp_path)
    reg.put("Stock", 211, symbol="AAPL:xnas")
    reg.put("Stock", 211, symbol="AAPL:xmil")
    assert len(reg) == 1
    assert reg.symbol_for("Stock", 211) == "AAPL:xmil"


def test_records_are_sorted_by_key(tmp_path):
    reg = InstrumentRegistry(tmp_path)
    reg.put("Stock", 261, symbol="C")
    reg.put("CfdOnIndex", 4913, symbol="A")
    reg.put("Stock", 211, symbol="B")
    assert [r.symbol for r in reg.records()] == ["A", "B", "C"]


def test_a_corrupt_file_reads_as_empty_rather_than_raising(tmp_path):
    (tmp_path / FILENAME).write_text("{not json")
    reg = InstrumentRegistry(tmp_path)
    assert len(reg) == 0
    # and it can still be written to
    reg.put("Stock", 211, symbol="AAPL:xnas")
    assert reg.symbol_for("Stock", 211) == "AAPL:xnas"


def test_record_label_property_directly():
    assert InstrumentRecord("FxSpot", 21, symbol="EURUSD").label == "EURUSD:FxSpot:21"
    assert InstrumentRecord("FxSpot", 21).label == "FxSpot:21"
