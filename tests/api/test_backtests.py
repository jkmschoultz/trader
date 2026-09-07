"""Submitting a backtest as a job, then polling it to completion."""

from __future__ import annotations

import time


def _wait(client, job_id, *, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def test_backtest_job_runs_and_returns_metrics(client):
    resp = client.post(
        "/api/backtests",
        json={
            "symbols": ["X"],
            "uics": [211],
            "asset_type": "Stock",
            "strategy": "ma_cross",
            "params": {"fast": 3, "slow": 8},
            "horizon": "5m",
        },
    )
    assert resp.status_code == 202
    job = _wait(client, resp.json()["job_id"])

    assert job["status"] == "done"
    assert job["progress"] == 1.0
    result = job["result"]
    assert result["config"]["labels"] == ["X"]
    assert "sharpe" in result["metrics"]
    assert "equity" in result and "fills" in result


def test_backtest_job_surfaces_an_unknown_strategy_as_an_error(client):
    resp = client.post(
        "/api/backtests",
        json={"symbols": ["X"], "uics": [211], "asset_type": "Stock", "strategy": "nope"},
    )
    assert resp.status_code == 202
    job = _wait(client, resp.json()["job_id"])
    assert job["status"] == "error"
    assert "ma_cross" in job["error"]


def test_backtest_on_a_missing_series_carries_error_data_for_a_fetch_button(client):
    resp = client.post(
        "/api/backtests",
        json={
            "symbols": ["MSFT"],
            "uics": [777],
            "asset_type": "Stock",
            "strategy": "ma_cross",
            "horizon": "5m",
        },
    )
    job = _wait(client, resp.json()["job_id"])
    assert job["status"] == "error"
    assert job["error_data"] == {
        "kind": "series_not_stored",
        "symbol": "MSFT",
        "horizon": "5m",
        "asset_type": "Stock",
        "uic": 777,
        "since": None,
    }


def test_backtest_rejects_a_bad_horizon_synchronously(client):
    resp = client.post(
        "/api/backtests",
        json={"symbols": ["X"], "uics": [211], "strategy": "ma_cross", "horizon": "7m"},
    )
    assert resp.status_code == 422


def test_jobs_list_shows_the_run(client):
    client.post(
        "/api/backtests",
        json={
            "symbols": ["X"],
            "uics": [211],
            "asset_type": "Stock",
            "strategy": "ma_cross",
        },
    )
    listing = client.get("/api/jobs").json()
    assert listing and listing[0]["kind"] == "backtest"
    assert "result" not in listing[0]
