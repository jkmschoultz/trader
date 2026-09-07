"""OAuth token model and encrypted at-rest storage.

Saxo's access tokens are short-lived and refresh tokens not much longer --
measured against SIM, 1200s (20 minutes) and 3600s (60 minutes) respectively.
Two consequences shape this module:

1. Expiry is stored as an absolute deadline, not the ``expires_in`` duration the
   API returns, so a token loaded from disk is judged against wall-clock time.
2. If the process is down longer than the refresh token's lifetime, the session
   is unrecoverable without a browser round-trip. ``TokenSet.refresh_expired``
   distinguishes that case so callers can report it precisely instead of
   failing with an opaque 401.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, Field

# Treat a token as expired slightly before it really is, to cover clock skew and
# the flight time of a request already in progress.
EXPIRY_MARGIN = timedelta(seconds=30)


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TokenSet(BaseModel):
    """A Saxo OAuth2 token pair with absolute expiry deadlines."""

    access_token: str
    access_expires_at: datetime
    refresh_token: str = ""
    refresh_expires_at: datetime | None = None
    token_type: str = "Bearer"

    # True for developer-portal 24h tokens, which cannot be refreshed.
    is_static: bool = False

    obtained_at: datetime = Field(default_factory=_utcnow)

    @classmethod
    def from_response(cls, payload: dict, *, obtained_at: datetime | None = None) -> TokenSet:
        """Build a token set from a ``/token`` endpoint response body."""
        now = obtained_at or _utcnow()
        refresh_expires_in = payload.get("refresh_token_expires_in")
        return cls(
            access_token=payload["access_token"],
            access_expires_at=now + timedelta(seconds=int(payload["expires_in"])),
            refresh_token=payload.get("refresh_token", ""),
            refresh_expires_at=(
                now + timedelta(seconds=int(refresh_expires_in)) if refresh_expires_in else None
            ),
            token_type=payload.get("token_type", "Bearer"),
            obtained_at=now,
        )

    @classmethod
    def static(cls, access_token: str, *, lifetime: timedelta = timedelta(hours=24)) -> TokenSet:
        """Wrap a developer-portal 24-hour token.

        The portal does not report an expiry, so ``lifetime`` is assumed. The
        token still works until Saxo rejects it; the assumed deadline only
        drives the warning about needing a fresh one.
        """
        now = _utcnow()
        return cls(
            access_token=access_token,
            access_expires_at=now + lifetime,
            is_static=True,
            obtained_at=now,
        )

    def access_expired(self, *, now: datetime | None = None) -> bool:
        return (now or _utcnow()) >= self.access_expires_at - EXPIRY_MARGIN

    def refresh_expired(self, *, now: datetime | None = None) -> bool:
        """True when the refresh token can no longer buy a new access token."""
        if self.is_static or not self.refresh_token:
            return True
        if self.refresh_expires_at is None:
            return False
        return (now or _utcnow()) >= self.refresh_expires_at - EXPIRY_MARGIN

    def seconds_until_refresh(self, fraction: float, *, now: datetime | None = None) -> float:
        """Seconds to wait before proactively refreshing.

        Refreshes once ``fraction`` of the access token's lifetime has elapsed,
        so a 20-minute token at 0.75 refreshes after 15 minutes. Never negative.
        """
        now = now or _utcnow()
        lifetime = (self.access_expires_at - self.obtained_at).total_seconds()
        deadline = self.obtained_at + timedelta(seconds=lifetime * fraction)
        return max(0.0, (deadline - now).total_seconds())


class TokenStore:
    """Persists a :class:`TokenSet` to disk, encrypted with Fernet.

    The encryption key comes from ``TRADER_TOKEN_KEY`` when set, otherwise a
    generated key file. Both the key file and the token file are written 0600.

    A local key file next to the ciphertext is not a defence against an attacker
    who already has your user account -- it protects against the token leaking
    through backups, syncs, or a stray ``cat`` of the state directory.
    """

    def __init__(self, token_file: Path, key_file: Path) -> None:
        self._token_file = token_file
        self._key_file = key_file

    def _fernet(self) -> Fernet:
        env_key = os.environ.get("TRADER_TOKEN_KEY")
        if env_key:
            return Fernet(env_key.encode())
        if not self._key_file.exists():
            self._key_file.parent.mkdir(parents=True, exist_ok=True)
            # Create with 0600 from the outset rather than chmod-ing after --
            # otherwise the key is briefly world-readable.
            fd = os.open(self._key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(Fernet.generate_key())
        return Fernet(self._key_file.read_bytes())

    def save(self, tokens: TokenSet) -> None:
        self._token_file.parent.mkdir(parents=True, exist_ok=True)
        blob = self._fernet().encrypt(tokens.model_dump_json().encode())
        tmp = self._token_file.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        # Atomic replace so a crash mid-write cannot leave a truncated file.
        tmp.replace(self._token_file)

    def load(self) -> TokenSet | None:
        """Return the stored tokens, or None if absent or unreadable.

        An unreadable file (rotated key, corruption) is treated as absent: the
        caller re-authenticates, which is always recoverable.
        """
        if not self._token_file.exists():
            return None
        try:
            raw = self._fernet().decrypt(self._token_file.read_bytes())
        except (InvalidToken, ValueError):
            return None
        try:
            return TokenSet.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValueError):
            return None

    def clear(self) -> None:
        self._token_file.unlink(missing_ok=True)
