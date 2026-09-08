"""``POST /api/training`` as a job, the model registry routes, and the end-to-end."""

from __future__ import annotations

import json
import time

import pytest
from starlette.testclient import TestClient

from trader.api import create_app
from trader.config import Settings


def _wait(client, job_id, *, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


_SPEC = {
    "symbols": ["X"],
    "uics": [211],
    "asset_type": "Stock",
    "horizon": "5m",
    "feature_set": "price_v1",
    "stop": 0.01,
    "take": 0.01,
    "max_bars": 6,
    "window": 8,
    "train_end": "2024-01-08",
    "val_end": "2024-01-10",
    "hidden": 8,
    "layers": 1,
    "epochs": 2,
    "batch_size": 16,
}


@pytest.fixture
def trained(tmp_path, seed_trainable_lake):
    """Settings + TestClient over a trainable lake."""
    pytest.importorskip("torch")
    seed_trainable_lake(tmp_path, bars=1600)
    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    with TestClient(create_app(settings)) as client:
        yield settings, client


@pytest.mark.slow
def test_training_job_runs_and_registers_a_model(trained):
    _, client = trained

    resp = client.post("/api/training", json=_SPEC)
    assert resp.status_code == 202
    job = _wait(client, resp.json()["job_id"])

    assert job["status"] == "done", job.get("error")
    model_id = job["result"]["model_id"]
    assert "val_macro_f1" in job["result"]["metrics"]

    listed = client.get("/api/models").json()
    assert model_id in [m["id"] for m in listed]

    detail = client.get(f"/api/models/{model_id}").json()
    assert detail["feature_set"] == "price_v1"
    assert detail["window"] == 8

    assert client.get("/api/models/nope").status_code == 404


@pytest.mark.slow
def test_trained_model_backtests_through_the_backtests_route(trained):
    _, client = trained
    model_id = _wait(client, client.post("/api/training", json=_SPEC).json()["job_id"])["result"][
        "model_id"
    ]

    resp = client.post(
        "/api/backtests",
        json={
            "symbols": ["X"],
            "uics": [211],
            "asset_type": "Stock",
            "strategy": "lstm",
            "params": {"model": model_id, "threshold": 0.0},
            "horizon": "5m",
        },
    )
    assert resp.status_code == 202
    job = _wait(client, resp.json()["job_id"])
    assert job["status"] == "done", job.get("error")
    assert "equity" in job["result"]


def test_bad_spec_is_a_422(client):
    resp = client.post("/api/training", json={**_SPEC, "val_end": "2024-01-01"})
    assert resp.status_code == 422


def test_feature_sets_route(client):
    body = client.get("/api/catalog/feature-sets").json()
    assert {"price_v1", "mtf_v1"} <= {f["name"] for f in body}


def test_models_route_works_without_torch(tmp_path):
    """A pre-written manifest is listed and fetched with no torch import."""
    model_dir = tmp_path / "models" / "lstm-20240101-000000-abcd1234"
    model_dir.mkdir(parents=True)
    manifest = {
        "id": "lstm-20240101-000000-abcd1234",
        "name": "lstm",
        "created_at": "2024-01-01T00:00:00+00:00",
        "base_horizon": 5,
        "context_horizons": [],
        "feature_set": "price_v1",
        "feature_digest": "deadbeef",
        "window": 8,
        "barriers": {"stop": 0.01, "take": 0.01, "max_bars": 6, "min_return": 0.0},
        "split": {},
        "hyperparameters": {"hidden": 8},
        "metrics": {"val_macro_f1": 0.4},
        "class_distribution": {},
        "data": {"symbols": ["X"], "asset_type": "Stock"},
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))

    settings = Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
    with TestClient(create_app(settings)) as client:
        listed = client.get("/api/models").json()
        assert [m["id"] for m in listed] == ["lstm-20240101-000000-abcd1234"]
        detail = client.get("/api/models/lstm-20240101-000000-abcd1234").json()
        assert detail["metrics"]["val_macro_f1"] == 0.4
