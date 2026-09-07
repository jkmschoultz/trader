"""Settings precedence and the live-trading gate."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from trader.config import PROJECT_ROOT, Settings
from trader.saxo.environments import Environment, hosts_for


def test_defaults_to_simulation():
    assert Settings().saxo.environment is Environment.SIM


def test_environment_variables_override_the_toml_file(monkeypatch):
    # settings.toml commits rate_limit_per_minute = 100; the environment must
    # win, otherwise a committed default could override a deployment's choice.
    monkeypatch.setenv("TRADER_SAXO__RATE_LIMIT_PER_MINUTE", "42")
    assert Settings().saxo.rate_limit_per_minute == 42


def test_nested_credentials_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("TRADER_SAXO__APP_KEY", "key-123")
    assert Settings().saxo.app_key == "key-123"


def test_secrets_are_not_exposed_by_repr(monkeypatch):
    monkeypatch.setenv("TRADER_SAXO__TOKEN_24H", "super-secret")
    settings = Settings()
    assert "super-secret" not in repr(settings)
    assert settings.saxo.token_24h.get_secret_value() == "super-secret"


def test_relative_directories_anchor_to_the_project_root():
    settings = Settings(state_dir="state", data_dir="data")
    assert settings.state_dir == PROJECT_ROOT / "state"
    assert settings.data_dir == PROJECT_ROOT / "data"


def test_non_loopback_redirect_is_rejected():
    with pytest.raises(ValidationError, match="loopback"):
        Settings(saxo={"redirect_uri": "https://example.com/callback"})


def test_token_file_is_per_environment():
    sim = Settings(saxo={"environment": "sim"})
    live = Settings(saxo={"environment": "live"})
    # A sim token must never be picked up by a live session, or vice versa.
    assert sim.token_file != live.token_file


@pytest.mark.parametrize(
    ("environment", "allow", "armed"),
    [("sim", False, False), ("sim", True, False), ("live", False, False), ("live", True, True)],
)
def test_live_trading_needs_both_the_environment_and_the_gate(environment, allow, armed):
    settings = Settings(saxo={"environment": environment}, allow_live_trading=allow)
    assert settings.live_trading_armed is armed


def test_sim_and_live_hosts_are_fully_disjoint():
    sim = hosts_for(Environment.SIM)
    live = hosts_for(Environment.LIVE)
    assert not set(sim) & set(live)
    assert "/sim/" in sim.gateway and "/sim/" not in live.gateway
