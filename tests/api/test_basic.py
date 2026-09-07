"""Health, catalogue, auth status, and the lake read endpoints."""

from __future__ import annotations


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_catalog_strategies_carries_params(client):
    body = client.get("/api/catalog/strategies").json()
    ma = next(s for s in body if s["name"] == "ma_cross")
    params = {p["name"]: p for p in ma["params"]}
    assert params["fast"]["default"] == 10
    assert params["long_only"]["default"] is False


def test_catalog_allocators(client):
    names = {a["name"] for a in client.get("/api/catalog/allocators").json()}
    assert "equal-weight" in names and "vol-target" in names


def test_auth_status_reports_not_signed_in_on_a_clean_state_dir(client):
    body = client.get("/api/auth/status").json()
    assert body["authenticated"] is False
    assert body["hint"]


def test_lake_series_lists_the_seeded_series(client):
    series = client.get("/api/lake/series").json()
    assert len(series) == 1
    assert (series[0]["asset_type"], series[0]["uic"], series[0]["horizon"]) == (
        "Stock",
        211,
        5,
    )
    assert series[0]["rows"] == 80


def test_lake_bars_returns_chart_rows(client):
    body = client.get(
        "/api/lake/bars", params={"asset_type": "Stock", "uic": 211, "horizon": 5}
    ).json()
    assert body["rows"] == 80
    assert body["decimated"] is False
    assert set(body["bars"][0]) == {"time", "open", "high", "low", "close", "volume"}
    assert isinstance(body["bars"][0]["time"], int)


def test_lake_bars_decimates_to_the_cap(client):
    body = client.get(
        "/api/lake/bars",
        params={"asset_type": "Stock", "uic": 211, "horizon": 5, "max_points": 10},
    ).json()
    assert body["decimated"] is True
    assert body["returned"] <= 11


def test_lake_bars_missing_series_is_404(client):
    resp = client.get("/api/lake/bars", params={"asset_type": "Stock", "uic": 999, "horizon": 5})
    assert resp.status_code == 404
    assert "trader data backfill" in resp.json()["detail"]
