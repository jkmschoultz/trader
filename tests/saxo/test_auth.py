"""Token acquisition, refresh, and the PKCE helper."""

from __future__ import annotations

import base64
import hashlib
import urllib.parse
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from trader.config import Settings
from trader.saxo.auth import (
    AuthError,
    OAuthFlow,
    ReauthRequired,
    SaxoAuth,
    _pkce_pair,
    _redact_tokens,
)
from trader.saxo.tokens import TokenSet, TokenStore


def _settings(tmp_path, **saxo) -> Settings:
    return Settings(
        state_dir=tmp_path,
        saxo={"app_key": "test-app-key", **saxo},
    )


def _auth(settings: Settings, handler) -> SaxoAuth:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SaxoAuth(settings, client=client)


def _stub_callback(auth: SaxoAuth, monkeypatch, captured: dict, code: str = "the-code"):
    """Bypass the browser and callback server, recording the authorize URL."""

    def fake(url: str, port: int, state: str, open_browser: bool) -> str:
        captured["url"] = url
        return code

    monkeypatch.setattr(auth, "_await_callback", fake)


def _stored(tmp_path, **overrides) -> TokenSet:
    now = datetime.now(UTC)
    fields = {
        "access_token": "old-access",
        "access_expires_at": now + timedelta(seconds=1200),
        "refresh_token": "old-refresh",
        "refresh_expires_at": now + timedelta(seconds=2400),
        "obtained_at": now,
    } | overrides
    tokens = TokenSet(**fields)
    settings = Settings(state_dir=tmp_path)
    TokenStore(settings.token_file, settings.token_key_file).save(tokens)
    return tokens


def test_pkce_challenge_is_the_s256_digest_of_the_verifier():
    verifier, challenge = _pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode()
        .rstrip("=")
    )
    assert challenge == expected
    # RFC 7636 requires a verifier of 43-128 characters.
    assert 43 <= len(verifier) <= 128
    assert "=" not in verifier and "=" not in challenge


def test_pkce_pairs_are_unique_per_login():
    assert _pkce_pair()[0] != _pkce_pair()[0]


async def test_static_token_short_circuits_oauth(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("the token endpoint must not be called for a static token")

    auth = _auth(_settings(tmp_path, token_24h="portal-token"), handler)
    assert await auth.get_access_token() == "portal-token"
    await auth.aclose()


async def test_static_token_is_refused_against_live(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    auth = _auth(_settings(tmp_path, token_24h="portal-token", environment="live"), handler)
    with pytest.raises(AuthError, match="only valid against SIM"):
        await auth.get_access_token()
    await auth.aclose()


async def test_valid_stored_token_is_reused(tmp_path):
    _stored(tmp_path)

    async def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a valid token must not trigger a refresh")

    auth = _auth(_settings(tmp_path), handler)
    assert await auth.get_access_token() == "old-access"
    await auth.aclose()


async def test_expired_token_is_refreshed_and_persisted(tmp_path):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    calls: list[dict[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "expires_in": 1200,
                "refresh_token": "new-refresh",
                "refresh_token_expires_in": 2400,
                "token_type": "Bearer",
            },
        )

    settings = _settings(tmp_path)
    auth = _auth(settings, handler)
    assert await auth.get_access_token() == "new-access"
    await auth.aclose()

    assert calls[0]["grant_type"] == "refresh_token"
    assert calls[0]["refresh_token"] == "old-refresh"

    # Saxo rotates refresh tokens, so the new one must be written to disk or the
    # next process start would present a spent token.
    reloaded = TokenStore(settings.token_file, settings.token_key_file).load()
    assert reloaded.access_token == "new-access"
    assert reloaded.refresh_token == "new-refresh"


async def test_concurrent_callers_trigger_a_single_refresh(tmp_path):
    import asyncio

    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    refreshes = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0.01)
        return httpx.Response(
            200,
            json={
                "access_token": "new-access",
                "expires_in": 1200,
                "refresh_token": "new-refresh",
                "refresh_token_expires_in": 2400,
            },
        )

    auth = _auth(_settings(tmp_path), handler)
    tokens = await asyncio.gather(*(auth.get_access_token() for _ in range(5)))
    await auth.aclose()

    # Parallel refreshes would spend and invalidate each other's refresh tokens.
    assert refreshes == 1
    assert set(tokens) == {"new-access"}


async def test_expired_refresh_token_demands_reauth(tmp_path):
    now = datetime.now(UTC)
    _stored(
        tmp_path,
        access_expires_at=now - timedelta(seconds=10),
        refresh_expires_at=now - timedelta(seconds=1),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("a spent refresh token must not be sent")

    auth = _auth(_settings(tmp_path), handler)
    with pytest.raises(ReauthRequired, match="refresh token has expired"):
        await auth.get_access_token()
    await auth.aclose()


async def test_missing_token_demands_reauth(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    auth = _auth(_settings(tmp_path), handler)
    with pytest.raises(ReauthRequired, match="No stored token"):
        await auth.get_access_token()
    await auth.aclose()


async def test_token_endpoint_failure_is_reported(tmp_path):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='{"error":"invalid_grant"}')

    auth = _auth(_settings(tmp_path), handler)
    with pytest.raises(AuthError, match="invalid_grant"):
        await auth.get_access_token()
    await auth.aclose()


async def test_login_requires_an_app_key(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    auth = _auth(Settings(state_dir=tmp_path), handler)
    with pytest.raises(AuthError, match="TRADER_SAXO__APP_KEY"):
        await auth.login_interactive(open_browser=False)
    await auth.aclose()


# --- Flow selection ---------------------------------------------------------
#
# Saxo apps are registered as either confidential (AppSecret, HTTP Basic) or
# public (PKCE). Using the wrong one fails with an opaque 400, so the choice is
# derived from configuration rather than guessed at request time.


def test_flow_is_secret_when_an_app_secret_is_present(tmp_path):
    auth = SaxoAuth(_settings(tmp_path, app_secret="s3cret"))
    assert auth.flow is OAuthFlow.SECRET


def test_flow_is_pkce_without_a_secret(tmp_path):
    assert SaxoAuth(_settings(tmp_path)).flow is OAuthFlow.PKCE


def test_flow_can_be_forced(tmp_path):
    auth = SaxoAuth(_settings(tmp_path, app_secret="s3cret", oauth_flow="pkce"))
    assert auth.flow is OAuthFlow.PKCE


async def test_secret_flow_refresh_uses_basic_auth(tmp_path):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(
            200, json={"access_token": "new", "expires_in": 1200, "refresh_token": "r2"}
        )

    auth = _auth(_settings(tmp_path, app_secret="s3cret"), handler)
    await auth.get_access_token()
    await auth.aclose()

    expected = base64.b64encode(b"test-app-key:s3cret").decode()
    assert captured["auth"] == f"Basic {expected}"
    # Credentials travel in the header, so client_id must not also be in the body.
    assert "client_id" not in captured["body"]


async def test_pkce_flow_refresh_sends_client_id_and_no_basic_auth(tmp_path):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(
            200, json={"access_token": "new", "expires_in": 1200, "refresh_token": "r2"}
        )

    auth = _auth(_settings(tmp_path), handler)
    await auth.get_access_token()
    await auth.aclose()

    assert captured["auth"] == ""
    assert captured["body"]["client_id"] == "test-app-key"


async def test_secret_flow_login_omits_pkce_and_uses_basic_auth(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("Authorization", "")
        captured["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(
            200, json={"access_token": "a", "expires_in": 1200, "refresh_token": "r"}
        )

    auth = _auth(_settings(tmp_path, app_secret="s3cret"), handler)
    _stub_callback(auth, monkeypatch, captured)

    await auth.login_interactive(open_browser=False)
    await auth.aclose()

    assert captured["auth"].startswith("Basic ")
    # A confidential app never issued a challenge, so a verifier would be rejected.
    assert "code_verifier" not in captured["body"]
    assert "code_challenge" not in str(captured["url"])
    assert captured["body"]["code"] == "the-code"


async def test_pkce_flow_login_sends_challenge_and_verifier(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = dict(httpx.QueryParams(request.content.decode()))
        return httpx.Response(
            200, json={"access_token": "a", "expires_in": 1200, "refresh_token": "r"}
        )

    auth = _auth(_settings(tmp_path), handler)
    _stub_callback(auth, monkeypatch, captured)

    await auth.login_interactive(open_browser=False)
    await auth.aclose()

    url = str(captured["url"])
    assert "code_challenge_method=S256" in url
    verifier = captured["body"]["code_verifier"]
    challenge = httpx.QueryParams(urllib.parse.urlparse(url).query)["code_challenge"]
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .decode()
        .rstrip("=")
    )
    # The verifier sent must be the preimage of the challenge shown in the browser.
    assert challenge == expected


async def test_201_created_is_treated_as_success(tmp_path):
    """Saxo answers the token endpoint with 201, not 200."""
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "access_token": "fresh-access",
                "expires_in": 1200,
                "refresh_token": "fresh-refresh",
                "refresh_token_expires_in": 2400,
            },
        )

    auth = _auth(_settings(tmp_path, app_secret="s3cret"), handler)
    assert await auth.get_access_token() == "fresh-access"
    await auth.aclose()


@pytest.mark.parametrize("status", [200, 201, 202])
async def test_any_2xx_is_accepted(tmp_path, status):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"access_token": "a", "expires_in": 1200})

    auth = _auth(_settings(tmp_path), handler)
    assert await auth.get_access_token() == "a"
    await auth.aclose()


def test_redaction_hides_credentials_in_echoed_bodies():
    body = (
        '{"access_token":"eyJhbGciOiJFUzI1NiJ9.secret-payload",'
        '"expires_in":1200,"refresh_token":"1f0e-refresh-uuid"}'
    )
    redacted = _redact_tokens(body)
    assert "secret-payload" not in redacted
    assert "1f0e-refresh-uuid" not in redacted
    # Non-credential fields stay visible so the message is still diagnostic.
    assert '"expires_in":1200' in redacted


async def test_error_bodies_never_leak_a_token(tmp_path):
    """An error path must not print a usable credential to the terminal."""
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    async def handler(request: httpx.Request) -> httpx.Response:
        # A 4xx that nonetheless carries a token-shaped body.
        return httpx.Response(400, text='{"access_token":"leaked-value","expires_in":1200}')

    auth = _auth(_settings(tmp_path), handler)
    with pytest.raises(AuthError) as exc:
        await auth.get_access_token()
    await auth.aclose()

    assert "leaked-value" not in str(exc.value)
    assert "<redacted>" in str(exc.value)


async def test_400_on_secret_flow_explains_the_likely_cause(tmp_path):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    auth = _auth(_settings(tmp_path, app_secret="s3cret"), handler)
    with pytest.raises(AuthError, match="OAUTH_FLOW=pkce"):
        await auth.get_access_token()
    await auth.aclose()


async def test_400_on_pkce_flow_suggests_the_secret_flow(tmp_path):
    now = datetime.now(UTC)
    _stored(tmp_path, access_expires_at=now - timedelta(seconds=1))

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    auth = _auth(_settings(tmp_path), handler)
    with pytest.raises(AuthError, match="APP_SECRET"):
        await auth.get_access_token()
    await auth.aclose()
