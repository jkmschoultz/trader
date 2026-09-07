"""POST /api/auth/login runs the browser flow as a one-at-a-time job."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from trader.saxo.auth import SaxoAuth
from trader.saxo.tokens import TokenSet


def _wait(client, job_id, *, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")


@pytest.fixture
def fake_login(monkeypatch):
    """Replace the real browser flow; record whether the URL callback fired."""
    seen: dict[str, object] = {}

    async def fake(self, *, open_browser=True, on_auth_url=None):
        seen["open_browser"] = open_browser
        if on_auth_url is not None:
            on_auth_url("https://sim.logonvalidation.net/authorize?client_id=x&state=y")
            seen["url_reported"] = True
        now = datetime.now(UTC)
        return TokenSet(
            access_token="new-access",
            access_expires_at=now + timedelta(minutes=20),
            refresh_token="new-refresh",
            refresh_expires_at=now + timedelta(hours=1),
        )

    monkeypatch.setattr(SaxoAuth, "login_interactive", fake)
    return seen


def test_login_job_completes_and_reports_authenticated(client, fake_login):
    resp = client.post("/api/auth/login")
    assert resp.status_code == 202
    job = _wait(client, resp.json()["job_id"])

    assert job["status"] == "done"
    assert job["result"]["authenticated"] is True
    assert fake_login.get("url_reported") is True


def test_login_is_one_at_a_time(client, monkeypatch):
    import asyncio

    gate = asyncio.Event()
    calls = {"n": 0}

    async def blocking(self, *, open_browser=True, on_auth_url=None):
        calls["n"] += 1
        await gate.wait()
        now = datetime.now(UTC)
        return TokenSet(access_token="a", access_expires_at=now + timedelta(minutes=20))

    monkeypatch.setattr(SaxoAuth, "login_interactive", blocking)

    first = client.post("/api/auth/login").json()["job_id"]
    # give the first job a chance to start and block on the gate
    for _ in range(50):
        if client.get(f"/api/jobs/{first}").json()["status"] == "running":
            break
        time.sleep(0.02)
    second = client.post("/api/auth/login").json()["job_id"]

    assert second == first  # the pending job was handed back, not a new one
    gate.set()
    _wait(client, first)
    assert calls["n"] == 1


def test_auth_status_still_reports_not_signed_in_without_tokens(client):
    assert client.get("/api/auth/status").json()["authenticated"] is False
