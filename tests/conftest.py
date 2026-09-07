"""Test isolation from the developer's real configuration.

Without this, ``Settings()`` in a test reads the project's ``.env`` and picks up
live credentials -- which made an auth test attempt a real browser login and
bind the OAuth callback port. Tests must depend only on what they set
themselves.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

import trader.config
from trader.config import Settings, get_settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Detach Settings from the real ``.env`` and any TRADER_* environment."""
    for key in [k for k in os.environ if k.startswith("TRADER_")]:
        monkeypatch.delenv(key, raising=False)

    # DotEnvSettingsSource reads env_file from model_config at instantiation, so
    # pointing it at a nonexistent path disables it for the whole test session.
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / "absent.env")

    # get_settings() memoises for the life of the process. A CLI run is one
    # process, so that is right in production and wrong here: without clearing
    # it, the first test to call the CLI pins its tmp_path for the whole
    # session and later tests silently read the wrong data_dir.
    monkeypatch.setattr(trader.config, "_cached", None)
    yield
    get_settings(reload=True)


def _seed_5m_lake(root, *, uic: int = 211, asset_type: str = "Stock", bars: int = 80) -> None:
    """A gently sawtoothing 5-minute series, enough for a warmup plus real trades."""
    from trader.data.lake import BarLake, SeriesKey, bars_to_frame
    from trader.saxo.charts import Bar

    start = datetime(2024, 3, 1, 14, 30, tzinfo=UTC)
    prices = [100.0 + (i % 20) - 10 for i in range(bars)]
    BarLake(root).write(
        SeriesKey(asset_type, uic, 5),
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


@pytest.fixture
def seed_5m_lake():
    """Call ``seed_5m_lake(root, uic=..., bars=...)`` to write a series into a lake."""
    return _seed_5m_lake
