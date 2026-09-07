"""A settings object pointed at a seeded temp lake, for the service helpers."""

from __future__ import annotations

import pytest

from trader.config import Settings


@pytest.fixture
def seeded_settings(tmp_path, seed_5m_lake) -> Settings:
    """Settings whose ``data_dir`` holds one 5-minute Stock series (uic 211)."""
    seed_5m_lake(tmp_path)
    return Settings(data_dir=tmp_path, state_dir=tmp_path / "state")
