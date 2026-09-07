"""FastAPI dependencies: the shared settings and job store off ``app.state``."""

from __future__ import annotations

from fastapi import Request

from trader.api.jobs import JobStore
from trader.config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs
