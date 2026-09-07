"""Saxo OAuth2 authentication: the authorization-code flow, plus token refresh.

Three ways to obtain a token, all behind :class:`SaxoAuth.get_access_token`:

* a developer-portal 24-hour token (``TRADER_SAXO__TOKEN_24H``), SIM only -- the
  fast path for early testing;
* a stored token pair, refreshed automatically as it ages;
* an interactive browser login, which is the only way to recover once the
  refresh token has itself expired.

Saxo registers apps as either *confidential* (issued an AppSecret, authenticating
at the token endpoint with HTTP Basic) or *public* (no secret, using PKCE). The
two are not interchangeable: sending a ``code_verifier`` to a confidential app,
or omitting Basic auth for one, fails with a bare ``400 Bad Request``. The flow
is therefore selected from whether an AppSecret is configured, overridable with
``TRADER_SAXO__OAUTH_FLOW``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
import secrets
import threading
import urllib.parse
import webbrowser
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

from trader.config import Settings
from trader.saxo.environments import Environment, hosts_for
from trader.saxo.tokens import TokenSet, TokenStore

log = logging.getLogger(__name__)

# How long to wait for the user to complete the browser login.
LOGIN_TIMEOUT_SECONDS = 300


class OAuthFlow(StrEnum):
    """How the registered Saxo app authenticates at the token endpoint."""

    # Public app: no secret, proves possession with a PKCE code_verifier.
    PKCE = "pkce"
    # Confidential app: authenticates with HTTP Basic client_id:client_secret.
    SECRET = "secret"


class AuthError(RuntimeError):
    """Authentication failed in a way the caller cannot retry around."""


class ReauthRequired(AuthError):
    """No usable token and no way to get one without a browser login."""


#: Matches the credential-bearing fields of a token response, so a body echoed
#: into an error message, a log, or a terminal cannot leak a usable token.
_TOKEN_FIELD = re.compile(
    r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]*(")', re.IGNORECASE
)


def _redact_tokens(text: str) -> str:
    """Replace token values in a JSON body with a placeholder."""
    return _TOKEN_FIELD.sub(r"\1<redacted>\2", text)


def _pkce_pair() -> tuple[str, str]:
    """Return a ``(verifier, challenge)`` pair for PKCE S256."""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).decode().rstrip("=")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    return verifier, challenge


class _CallbackHandler(BaseHTTPRequestHandler):
    """One-shot handler that captures the ``code`` from Saxo's redirect."""

    result: dict[str, str] = {}
    expected_state: str = ""

    def do_GET(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        state = query.get("state", [""])[0]

        if state != self.expected_state:
            # A mismatched state means this redirect did not originate from the
            # request we started. Refuse it rather than exchanging the code.
            self._respond(400, "State mismatch", "This response did not match the login request.")
            _CallbackHandler.result = {"error": "state_mismatch"}
            return

        if "error" in query:
            detail = query.get("error_description", [""])[0] or query["error"][0]
            self._respond(400, "Login failed", detail)
            _CallbackHandler.result = {"error": query["error"][0]}
            return

        code = query.get("code", [""])[0]
        if not code:
            self._respond(400, "Login failed", "No authorization code was returned.")
            _CallbackHandler.result = {"error": "no_code"}
            return

        self._respond(200, "Signed in", "You can close this tab and return to the terminal.")
        _CallbackHandler.result = {"code": code}

    def _respond(self, status: int, title: str, detail: str) -> None:
        body = (
            "<!doctype html><meta charset=utf-8>"
            "<style>body{font:16px system-ui;margin:4rem auto;max-width:32rem;"
            "color:#111}h1{font-size:1.25rem}</style>"
            f"<h1>{title}</h1><p>{detail}</p>"
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        """Silence the default stderr access log."""


class SaxoAuth:
    """Owns the token lifecycle for one Saxo environment."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._settings = settings
        self._hosts = hosts_for(settings.saxo.environment)
        self._store = TokenStore(settings.token_file, settings.token_key_file)
        self._client = client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = client is None
        self._tokens: TokenSet | None = None
        # Serialises refreshes so a burst of concurrent requests triggers one
        # token exchange, not one per caller. Saxo rotates refresh tokens, so
        # parallel refreshes would invalidate each other.
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def environment(self) -> Environment:
        return self._settings.saxo.environment

    async def get_access_token(self) -> str:
        """Return a valid access token, refreshing it if necessary.

        Raises:
            ReauthRequired: no stored token can be revived; run the interactive
                login (``trader auth login``).
        """
        static = self._settings.saxo.token_24h.get_secret_value()
        if static:
            if self.environment is Environment.LIVE:
                raise AuthError(
                    "A 24-hour developer token is only valid against SIM, but the "
                    "environment is set to live. Unset TRADER_SAXO__TOKEN_24H."
                )
            return static

        async with self._lock:
            if self._tokens is None:
                self._tokens = self._store.load()
            if self._tokens is None:
                raise ReauthRequired("No stored token. Run: trader auth login")

            if not self._tokens.access_expired():
                return self._tokens.access_token

            if self._tokens.refresh_expired():
                raise ReauthRequired(
                    "The refresh token has expired (Saxo refresh tokens last about "
                    "an hour). Run: trader auth login"
                )
            self._tokens = await self._exchange_refresh(self._tokens.refresh_token)
            self._store.save(self._tokens)
            return self._tokens.access_token

    async def refresh_now(self) -> TokenSet:
        """Force a refresh regardless of remaining lifetime."""
        async with self._lock:
            if self._tokens is None:
                self._tokens = self._store.load()
            if self._tokens is None or self._tokens.refresh_expired():
                raise ReauthRequired("No refreshable token. Run: trader auth login")
            self._tokens = await self._exchange_refresh(self._tokens.refresh_token)
            self._store.save(self._tokens)
            return self._tokens

    @property
    def flow(self) -> OAuthFlow:
        """Which OAuth flow this app is registered for.

        Saxo issues an AppSecret only to confidential apps; public apps use PKCE
        instead. Guessing wrong makes the token exchange fail with a bare 400,
        so the presence of a secret is the signal, overridable via config.
        """
        configured = self._settings.saxo.oauth_flow
        if configured != "auto":
            return OAuthFlow(configured)
        has_secret = bool(self._settings.saxo.app_secret.get_secret_value())
        return OAuthFlow.SECRET if has_secret else OAuthFlow.PKCE

    async def _exchange_refresh(self, refresh_token: str) -> TokenSet:
        log.debug("Refreshing Saxo access token (%s flow)", self.flow.value)
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "redirect_uri": self._settings.saxo.redirect_uri,
        }
        if self.flow is OAuthFlow.PKCE:
            data["client_id"] = self._settings.saxo.app_key
        return await self._post_token(data)

    def _client_auth(self) -> httpx.Auth | None:
        """HTTP Basic credentials for confidential apps, else None."""
        if self.flow is not OAuthFlow.SECRET:
            return None
        return httpx.BasicAuth(
            self._settings.saxo.app_key,
            self._settings.saxo.app_secret.get_secret_value(),
        )

    async def _post_token(self, data: dict[str, str]) -> TokenSet:
        response = await self._client.post(
            self._hosts.token,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            auth=self._client_auth() or httpx.USE_CLIENT_DEFAULT,
        )
        # Saxo answers the token endpoint with 201 Created, not 200 OK, so
        # success must be judged on the 2xx class rather than an exact code.
        if not response.is_success:
            raise AuthError(self._token_error(response))
        return TokenSet.from_response(response.json())

    def _token_error(self, response: httpx.Response) -> str:
        """Turn Saxo's terse token-endpoint failures into something actionable."""
        body = _redact_tokens(response.text)[:400].strip() or "(empty body)"
        message = f"Token endpoint returned {response.status_code}: {body}"

        if response.status_code == httpx.codes.BAD_REQUEST:
            if self.flow is OAuthFlow.PKCE:
                hint = (
                    "This app is being treated as a public PKCE app. If your Saxo app "
                    "has an AppSecret, it is confidential: set TRADER_SAXO__APP_SECRET "
                    "(or TRADER_SAXO__OAUTH_FLOW=secret)."
                )
            else:
                hint = (
                    "This app is being treated as a confidential app using HTTP Basic "
                    "auth. Check the AppSecret is correct, that redirect_uri exactly "
                    "matches a redirect URL registered on the app, and that the "
                    "authorization code has not already been used or expired. If the "
                    "app is registered for PKCE, set TRADER_SAXO__OAUTH_FLOW=pkce."
                )
            message = f"{message}\n\n{hint}"
        return message

    # --- Interactive login -------------------------------------------------

    async def login_interactive(self, *, open_browser: bool = True) -> TokenSet:
        """Run the browser authorization-code flow and store the result."""
        if not self._settings.saxo.app_key:
            raise AuthError(
                "TRADER_SAXO__APP_KEY is not set. Copy .env.example to .env and fill "
                "in the AppKey from your Saxo Developer Portal app."
            )
        if self.flow is OAuthFlow.SECRET and not self._settings.saxo.app_secret.get_secret_value():
            raise AuthError("TRADER_SAXO__OAUTH_FLOW=secret but no TRADER_SAXO__APP_SECRET is set.")

        verifier, challenge = _pkce_pair()
        state = secrets.token_urlsafe(24)

        redirect = urllib.parse.urlparse(self._settings.saxo.redirect_uri)
        port = redirect.port or 80

        params = {
            "response_type": "code",
            "client_id": self._settings.saxo.app_key,
            "redirect_uri": self._settings.saxo.redirect_uri,
            "state": state,
        }
        if self.flow is OAuthFlow.PKCE:
            params |= {"code_challenge": challenge, "code_challenge_method": "S256"}
        auth_url = f"{self._hosts.authorize}?{urllib.parse.urlencode(params)}"

        code = await asyncio.to_thread(self._await_callback, auth_url, port, state, open_browser)

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._settings.saxo.redirect_uri,
        }
        if self.flow is OAuthFlow.PKCE:
            # A confidential app authenticates with Basic auth instead; sending
            # a code_verifier it never issued a challenge for is rejected.
            data |= {"client_id": self._settings.saxo.app_key, "code_verifier": verifier}

        tokens = await self._post_token(data)
        self._store.save(tokens)
        self._tokens = tokens
        return tokens

    def _await_callback(self, auth_url: str, port: int, state: str, open_browser: bool) -> str:
        """Serve one request on ``port`` and return the authorization code."""
        _CallbackHandler.result = {}
        _CallbackHandler.expected_state = state

        try:
            server = HTTPServer(("127.0.0.1", port), _CallbackHandler)
        except OSError as exc:
            raise AuthError(
                f"Could not bind {port} for the OAuth callback: {exc}. "
                "Free the port, or point TRADER_SAXO__REDIRECT_URI at another one "
                "(it must also be registered on the Saxo app)."
            ) from exc

        server.timeout = LOGIN_TIMEOUT_SECONDS

        print(f"\nOpen this URL to sign in to Saxo ({self.environment.value}):\n\n{auth_url}\n")
        if open_browser:
            webbrowser.open(auth_url)

        # handle_request honours server.timeout and returns either way, so a
        # user who never completes the login fails with a clear message rather
        # than hanging forever.
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        thread.join(LOGIN_TIMEOUT_SECONDS + 5)
        server.server_close()

        result = _CallbackHandler.result
        if not result:
            raise AuthError(
                f"No callback received within {LOGIN_TIMEOUT_SECONDS}s. Check that "
                f"{self._settings.saxo.redirect_uri} is registered as a redirect URL "
                "on your Saxo app."
            )
        if "error" in result:
            raise AuthError(f"Authorization failed: {result['error']}")
        return result["code"]
