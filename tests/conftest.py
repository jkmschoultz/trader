"""Test isolation from the developer's real configuration.

Without this, ``Settings()`` in a test reads the project's ``.env`` and picks up
live credentials -- which made an auth test attempt a real browser login and
bind the OAuth callback port. Tests must depend only on what they set
themselves.
"""

from __future__ import annotations

import os

import pytest

from trader.config import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Detach Settings from the real ``.env`` and any TRADER_* environment."""
    for key in [k for k in os.environ if k.startswith("TRADER_")]:
        monkeypatch.delenv(key, raising=False)

    # DotEnvSettingsSource reads env_file from model_config at instantiation, so
    # pointing it at a nonexistent path disables it for the whole test session.
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / "absent.env")
    yield
