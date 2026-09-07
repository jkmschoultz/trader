"""A TestClient over the app, with settings pointed at a seeded temp lake."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from trader.api import create_app
from trader.config import Settings


@pytest.fixture
def settings(tmp_path, seed_5m_lake) -> Settings:
    seed_5m_lake(tmp_path)
    return Settings(data_dir=tmp_path, state_dir=tmp_path / "state")


@pytest.fixture
def client(settings) -> TestClient:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
