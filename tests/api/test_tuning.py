"""``POST /api/tuning`` as a job: sweep runs, report comes back, file is written."""

from __future__ import annotations

import time

import pytest
from starlette.testclient import TestClient

from trader.api import create_app
from trader.config import Settings


def _wait(client, job_id, *, timeout=120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


_SPEC = {
    "base": {
        "symbols": ["X"],
        "uics": [211],
        "asset_type": "Stock",
        "horizon": "5m",
        "feature_set": "price_v1",
        "stop": 0.01,
        "take": 0.01,
        "max_bars": 6,
        "window": 8,
        "hidden": 8,
        "layers": 1,
        "epochs": 2,
        "batch_size": 16,
    },
    "cv": {"folds": 2, "train_days": 8, "val_days": 2, "test_days": 1.5, "fee_bps": 0.2},
    "grid": {"threshold": [0.0, 0.3]},
    "max_workers": 1,
}


@pytest.fixture
def sweepable(tmp_path, seed_trainable_lake):
    pytest.importorskip("torch")
    seed_trainable_lake(tmp_path, bars=4000)
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    with TestClient(create_app(settings)) as client:
        yield settings, client


@pytest.mark.slow
def test_tuning_job_sweeps_and_writes_a_report(sweepable):
    settings, client = sweepable

    resp = client.post("/api/tuning", json=_SPEC)
    assert resp.status_code == 202
    job = _wait(client, resp.json()["job_id"])

    assert job["status"] == "done", job.get("error")
    result = job["result"]
    assert result["n_configs"] == 2
    assert [r["rank"] for r in result["results"]] == [1, 2]

    report_path = settings.state_dir / result["report_path"].split("/")[-1]
    assert report_path.is_file()


def test_bad_grid_key_is_a_422(client):
    resp = client.post("/api/tuning", json={**_SPEC, "grid": {"bogus": [1, 2]}})
    assert resp.status_code == 422
