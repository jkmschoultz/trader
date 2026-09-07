"""Authenticated, rate-limited HTTP client for the Saxo OpenAPI.

Everything that talks to Saxo goes through :class:`SaxoClient`, so token
injection, rate limiting, retries, and error shaping are decided once.

Retry policy is deliberately asymmetric. Reads (GET) are safe to retry. Writes
(POST/PUT/PATCH/DELETE) are not retried on timeouts or 5xx, because a placed
order that times out in flight may well have reached the exchange -- retrying
could double it. Callers that write must reconcile instead, which is what
``execution/saxo_live.py`` does.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Literal

import httpx

from trader.config import Settings
from trader.saxo.auth import SaxoAuth
from trader.saxo.environments import hosts_for
from trader.saxo.ratelimit import ServiceGroupLimiter

log = logging.getLogger(__name__)

Method = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]

IDEMPOTENT_METHODS = frozenset({"GET"})
MAX_ATTEMPTS = 4
DEFAULT_RETRY_AFTER = 5.0


class SaxoAPIError(RuntimeError):
    """A non-retryable error response from the Saxo OpenAPI."""

    def __init__(self, status_code: int, message: str, body: Any = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class SaxoClient:
    """Thin async wrapper over the Saxo REST gateway."""

    def __init__(
        self,
        settings: Settings,
        auth: SaxoAuth | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._hosts = hosts_for(settings.saxo.environment)
        self._http = httpx.AsyncClient(
            base_url=self._hosts.gateway,
            timeout=httpx.Timeout(30.0, connect=10.0),
            transport=transport,
        )
        self._auth = auth or SaxoAuth(settings, client=self._http)
        self._limiter = ServiceGroupLimiter(settings.saxo.rate_limit_per_minute)

    async def __aenter__(self) -> SaxoClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    @property
    def auth(self) -> SaxoAuth:
        return self._auth

    @property
    def request_counts(self) -> dict[str, int]:
        return self._limiter.request_counts

    @staticmethod
    def _service_group(path: str) -> str:
        """First path segment, e.g. ``/port/v1/balances`` -> ``port``."""
        return path.lstrip("/").split("/", 1)[0] or "root"

    async def request(
        self,
        method: Method,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: Any = None,
        is_order: bool = False,
    ) -> Any:
        """Issue one request, returning the decoded JSON body (or None for 204).

        Raises:
            SaxoAPIError: the request failed in a way retrying will not fix.
        """
        group = self._service_group(path)
        retryable = method in IDEMPOTENT_METHODS
        last_error: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            await self._limiter.acquire(group, is_order=is_order)
            token = await self._auth.get_access_token()

            headers = {
                "Authorization": f"Bearer {token}",
                # Saxo rejects identical operations repeated within 15s with a
                # 409 unless they carry distinct request ids.
                "x-request-id": str(uuid.uuid4()),
            }

            try:
                response = await self._http.request(
                    method, path, params=params, json=json, headers=headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise SaxoAPIError(0, f"{method} {path} failed: {exc}") from exc
                await self._backoff(attempt)
                continue

            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                wait = self._retry_after(response)
                log.warning("Rate limited on %s; pausing %.1fs", group, wait)
                await self._limiter.penalise(group, wait)
                if attempt == MAX_ATTEMPTS:
                    raise SaxoAPIError(429, f"{method} {path}: rate limited after retries")
                continue

            if response.status_code == httpx.codes.UNAUTHORIZED and attempt == 1:
                # The token was accepted locally but rejected by the gateway;
                # force one refresh before giving up.
                log.info("401 from gateway; forcing token refresh")
                await self._auth.refresh_now()
                continue

            if response.status_code >= 500:
                last_error = SaxoAPIError(response.status_code, response.text[:400])
                if not retryable or attempt == MAX_ATTEMPTS:
                    raise SaxoAPIError(
                        response.status_code,
                        f"{method} {path}: server error {response.status_code}",
                        response.text[:400],
                    )
                await self._backoff(attempt)
                continue

            if response.status_code >= 400:
                raise SaxoAPIError(
                    response.status_code,
                    f"{method} {path}: {response.status_code} {self._error_message(response)}",
                    self._safe_json(response),
                )

            if response.status_code == httpx.codes.NO_CONTENT or not response.content:
                return None
            return response.json()

        raise SaxoAPIError(0, f"{method} {path}: exhausted retries ({last_error})")

    async def get(self, path: str, **params: Any) -> Any:
        """GET ``path``; keyword arguments become query parameters.

        ``None`` values are dropped so optional Saxo parameters (``Time`` on the
        first page of a chart request, say) can be passed unconditionally.
        """
        return await self.request(
            "GET", path, params={k: v for k, v in params.items() if v is not None}
        )

    async def post(self, path: str, json: Any = None, *, is_order: bool = False) -> Any:
        return await self.request("POST", path, json=json, is_order=is_order)

    async def delete(self, path: str, **params: Any) -> Any:
        return await self.request(
            "DELETE", path, params={k: v for k, v in params.items() if v is not None}
        )

    @staticmethod
    def _retry_after(response: httpx.Response) -> float:
        """Seconds to wait, from Saxo's rate-limit headers or a default."""
        for name, value in response.headers.items():
            lowered = name.lower()
            if lowered.startswith("x-ratelimit-") and lowered.endswith("-reset"):
                try:
                    return max(1.0, float(value))
                except ValueError:
                    continue
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                pass
        return DEFAULT_RETRY_AFTER

    @staticmethod
    def _safe_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return response.text[:400]

    @classmethod
    def _error_message(cls, response: httpx.Response) -> str:
        body = cls._safe_json(response)
        if isinstance(body, dict):
            # Saxo returns {"Message": ..., "ErrorCode": ...} on most failures.
            parts = [str(body[k]) for k in ("ErrorCode", "Message") if body.get(k)]
            if parts:
                return " ".join(parts)
        return str(body)[:400]

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(min(8.0, 0.5 * 2**attempt))
