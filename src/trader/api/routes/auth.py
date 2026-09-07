"""Saxo session: read the stored state, and start a browser sign-in.

``POST /auth/login`` runs the same authorization-code flow as ``trader auth
login`` -- it binds the local callback port and opens the system browser. That
is fine for the local single-user tool this is; it is not meant to be exposed
on a shared host (the callback redirect only reaches the machine the browser
runs on). Credentials are never handled in the page.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from trader.api.deps import get_jobs, get_settings
from trader.api.jobs import JobStore
from trader.config import Settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
def status(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    from trader.saxo.tokens import TokenStore

    env = settings.saxo.environment.value

    if settings.saxo.token_24h.get_secret_value():
        return {"authenticated": True, "kind": "static-24h", "environment": env}

    tokens = TokenStore(settings.token_file, settings.token_key_file).load()
    if tokens is None:
        return {
            "authenticated": False,
            "environment": env,
            "hint": "sign in to Saxo",
        }

    alive = not tokens.access_expired() or not tokens.refresh_expired()
    return {
        "authenticated": alive,
        "kind": "static-24h" if tokens.is_static else "oauth",
        "environment": env,
        "access_expired": tokens.access_expired(),
        "access_expires_at": tokens.access_expires_at.isoformat(),
        "refresh_expired": tokens.refresh_expired(),
        "refresh_expires_at": (
            tokens.refresh_expires_at.isoformat() if tokens.refresh_expires_at else None
        ),
        "hint": None if alive else "session expired -- sign in again",
    }


@router.post("/login", status_code=202)
async def login(
    settings: Settings = Depends(get_settings),
    jobs: JobStore = Depends(get_jobs),
) -> dict[str, str]:
    """Start the browser sign-in as a background job. One runs at a time."""
    for job in jobs.list():
        if job.kind == "auth-login" and not job.finished:
            return {"job_id": job.id}

    async def run(report: Any) -> Any:
        from trader.saxo.auth import SaxoAuth

        auth = SaxoAuth(settings)
        report(message="Opening the Saxo sign-in page…")
        try:
            tokens = await auth.login_interactive(
                on_auth_url=lambda url: report(
                    message=f"Waiting for sign-in. If the browser did not open, visit:\n{url}"
                ),
            )
        finally:
            await auth.aclose()
        return {
            "authenticated": True,
            "environment": settings.saxo.environment.value,
            "access_expires_at": tokens.access_expires_at.isoformat(),
        }

    job = jobs.submit("auth-login", run)
    return {"job_id": job.id}
