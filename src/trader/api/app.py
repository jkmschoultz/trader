"""The application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from trader.api.jobs import JobStore
from trader.api.routes import (
    auth,
    backtests,
    catalog,
    health,
    instruments,
    jobs,
    lake,
    training,
)
from trader.config import Settings, get_settings

# Where `npm run build` puts the compiled UI. Served at / when present.
FRONTEND_DIST = Path(__file__).resolve().parents[3] / "frontend" / "dist"

# The Vite dev server, which talks to this API cross-origin during development.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app. Pass ``settings`` in tests; otherwise the singleton."""
    settings = settings or get_settings()

    app = FastAPI(title="trader", version="0.1.0")
    app.state.settings = settings
    app.state.jobs = JobStore()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (health, auth, catalog, lake, instruments, backtests, training, jobs):
        app.include_router(module.router, prefix="/api")

    # Anything under /api that no router matched is a genuine 404. Without this,
    # the SPA mount below would catch it -- and StaticFiles answers every
    # non-GET with a misleading 405. Registered after the routers, so real
    # routes still win.
    @app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def _api_not_found(rest: str) -> None:
        raise HTTPException(status_code=404, detail=f"no API route: /api/{rest}")

    if FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")

    return app
