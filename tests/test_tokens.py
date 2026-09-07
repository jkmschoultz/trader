"""Token expiry arithmetic and encrypted storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from trader.saxo.tokens import TokenSet, TokenStore

NOW = datetime(2026, 9, 7, 12, 0, 0, tzinfo=UTC)


def _tokens(**overrides) -> TokenSet:
    base = {
        "access_token": "access-abc",
        "access_expires_at": NOW + timedelta(seconds=1200),
        "refresh_token": "refresh-xyz",
        "refresh_expires_at": NOW + timedelta(seconds=2400),
        "obtained_at": NOW,
    }
    return TokenSet(**(base | overrides))


def test_from_response_converts_durations_to_deadlines():
    tokens = TokenSet.from_response(
        {
            "access_token": "a",
            "expires_in": 1200,
            "refresh_token": "r",
            "refresh_token_expires_in": 2400,
            "token_type": "Bearer",
        },
        obtained_at=NOW,
    )
    assert tokens.access_expires_at == NOW + timedelta(seconds=1200)
    assert tokens.refresh_expires_at == NOW + timedelta(seconds=2400)


def test_access_expiry_uses_a_safety_margin():
    tokens = _tokens()
    # Ten seconds before the nominal deadline the token is already treated as
    # expired, so a request in flight cannot outlive it.
    assert tokens.access_expired(now=NOW + timedelta(seconds=1190))
    assert not tokens.access_expired(now=NOW + timedelta(seconds=1100))


def test_refresh_expiry_is_reported_separately():
    tokens = _tokens()
    assert not tokens.refresh_expired(now=NOW + timedelta(seconds=1300))
    # Past the refresh deadline the session is unrecoverable without a browser.
    assert tokens.refresh_expired(now=NOW + timedelta(seconds=2399))


def test_static_token_is_never_refreshable():
    tokens = TokenSet.static("portal-token")
    assert tokens.refresh_expired()
    assert tokens.is_static


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.75, 900.0), (0.5, 600.0), (1.0, 1200.0)],
)
def test_seconds_until_refresh_scales_with_lifetime(fraction, expected):
    assert _tokens().seconds_until_refresh(fraction, now=NOW) == pytest.approx(expected)


def test_seconds_until_refresh_never_negative():
    assert _tokens().seconds_until_refresh(0.75, now=NOW + timedelta(seconds=1_000)) == 0.0


def test_store_roundtrip_and_permissions(tmp_path):
    store = TokenStore(tmp_path / "tokens.json", tmp_path / "token.key")
    store.save(_tokens())

    loaded = store.load()
    assert loaded is not None
    assert loaded.access_token == "access-abc"
    assert loaded.refresh_token == "refresh-xyz"

    # Tokens are a bearer credential; the files must not be group/world readable.
    assert (tmp_path / "tokens.json").stat().st_mode & 0o077 == 0
    assert (tmp_path / "token.key").stat().st_mode & 0o077 == 0


def test_stored_file_is_not_plaintext(tmp_path):
    store = TokenStore(tmp_path / "tokens.json", tmp_path / "token.key")
    store.save(_tokens())
    assert b"access-abc" not in (tmp_path / "tokens.json").read_bytes()


def test_unreadable_store_is_treated_as_absent(tmp_path):
    store = TokenStore(tmp_path / "tokens.json", tmp_path / "token.key")
    store.save(_tokens())
    # Simulate a rotated or corrupted key: recovery is re-authentication, so
    # load must return None rather than raising.
    (tmp_path / "tokens.json").write_bytes(b"not-a-fernet-blob")
    assert store.load() is None


def test_missing_store_returns_none(tmp_path):
    assert TokenStore(tmp_path / "absent.json", tmp_path / "token.key").load() is None
