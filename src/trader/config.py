"""Application configuration.

Settings come from three layers. In increasing order of precedence:
``config/settings.toml`` (committed, non-secret), then ``.env`` (gitignored,
secrets), then real environment variables. Nested values use a ``__``
delimiter, so ``TRADER_SAXO__APP_KEY`` sets ``settings.saxo.app_key``.

The ordering matters: the TOML file is the weakest source, so committing a
default there can never silently override a secret supplied by the environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from trader.saxo.environments import Environment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.toml"


class SaxoSettings(BaseModel):
    """Credentials and environment selection for the Saxo OpenAPI."""

    app_key: str = ""
    app_secret: SecretStr = SecretStr("")
    redirect_uri: str = "http://localhost:8765/callback"
    environment: Environment = Environment.SIM

    # Developer-portal 24-hour token. SIM only. When set, it bypasses OAuth.
    token_24h: SecretStr = SecretStr("")

    # Which OAuth flow the registered app uses. "auto" picks "secret" when an
    # app_secret is present and "pkce" otherwise, which matches how the Saxo
    # Developer Portal issues credentials: public PKCE apps get no secret.
    # Sending PKCE parameters to a confidential app makes the token exchange
    # fail with an opaque 400, so this must match the app registration.
    oauth_flow: Literal["auto", "pkce", "secret"] = "auto"

    # Requests per minute per service group. Saxo's documented cap is 120; we
    # stay under it so a burst never trips a 429 that stalls a live session.
    rate_limit_per_minute: int = 100

    # Refresh the access token once this fraction of its lifetime has elapsed.
    # Measured against SIM: access tokens live 1200s (20 min) and refresh tokens
    # 3600s (60 min). Refreshing early is what keeps a long session alive.
    refresh_at_lifetime_fraction: float = 0.75

    # Exchange suffixes tried, in order, to break a tie when a bare ticker like
    # "AAPL" matches the same company on several venues. Matched against the
    # ``:xxxx`` suffix of the Saxo symbol; an explicit "AAPL:xmil" is never
    # overridden. Consulted by the shared resolver in ``trader.service`` (the UI
    # and ``trader backtest``), not by ``trader instruments``/``data``.
    preferred_exchanges: list[str] = ["xnas", "xnys", "arcx", "xlon", "xetr"]

    @field_validator("redirect_uri")
    @classmethod
    def _must_be_loopback(cls, v: str) -> str:
        if not v.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                "redirect_uri must be a loopback URL -- the local callback server "
                "can only receive redirects addressed to this machine"
            )
        return v


class Settings(BaseSettings):
    """Top-level settings object."""

    model_config = SettingsConfigDict(
        env_prefix="TRADER_",
        env_nested_delimiter="__",
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        toml_file=DEFAULT_SETTINGS_FILE,
        extra="ignore",
    )

    saxo: SaxoSettings = Field(default_factory=SaxoSettings)

    # Local state: encrypted tokens, instrument cache. Never committed.
    state_dir: Path = PROJECT_ROOT / "state"
    data_dir: Path = PROJECT_ROOT / "data"

    # Explicit opt-in required before any code path may place a real order.
    # Setting environment=live is deliberately not sufficient on its own.
    allow_live_trading: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Highest precedence first. TOML sits at the bottom on purpose.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        # Directories may be given relative in settings.toml; anchor them to the
        # project root so behaviour does not depend on the current directory.
        if not self.state_dir.is_absolute():
            self.state_dir = PROJECT_ROOT / self.state_dir
        if not self.data_dir.is_absolute():
            self.data_dir = PROJECT_ROOT / self.data_dir
        return self

    @property
    def token_file(self) -> Path:
        return self.state_dir / f"tokens-{self.saxo.environment.value}.json"

    @property
    def token_key_file(self) -> Path:
        return self.state_dir / "token.key"

    @property
    def live_trading_armed(self) -> bool:
        """True only when both the environment and the explicit gate say live."""
        return self.allow_live_trading and self.saxo.environment is Environment.LIVE


_cached: Settings | None = None


def get_settings(*, reload: bool = False) -> Settings:
    """Return the process-wide settings singleton."""
    global _cached
    if _cached is None or reload:
        _cached = Settings()
    return _cached
