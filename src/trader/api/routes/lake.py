"""The bar lake: what it holds, chart-ready slices, and backfilling into it."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from trader.api.deps import get_jobs, get_settings
from trader.api.jobs import JobStore
from trader.config import Settings
from trader.service.data import (
    BackfillSpec,
    SeriesInfo,
    lake_series,
    read_bars,
    refresh_symbols,
    run_backfill,
)
from trader.service.errors import SeriesNotStored

router = APIRouter(prefix="/lake", tags=["lake"])


@router.get("/series")
def series(settings: Settings = Depends(get_settings)) -> list[SeriesInfo]:
    return lake_series(settings)


@router.get("/bars")
def bars(
    asset_type: str,
    uic: int,
    horizon: int,
    start: datetime | None = None,
    end: datetime | None = None,
    max_points: int = Query(5000, ge=10, le=50_000),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    try:
        return read_bars(
            settings,
            asset_type=asset_type,
            uic=uic,
            horizon=horizon,
            start=start,
            end=end,
            max_points=max_points,
        )
    except SeriesNotStored as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/symbols", status_code=202)
async def submit_symbol_refresh(
    refresh: bool = False,
    settings: Settings = Depends(get_settings),
    jobs: JobStore = Depends(get_jobs),
) -> dict[str, str]:
    async def run(report: Any) -> Any:
        records = await refresh_symbols(
            settings, refresh=refresh, progress=lambda msg: report(message=f"fetched {msg}")
        )
        return [{"asset_type": r.asset_type, "uic": r.uic, "symbol": r.symbol} for r in records]

    job = jobs.submit("symbols", run)
    return {"job_id": job.id}


@router.post("/backfill", status_code=202)
async def submit_backfill(
    spec: BackfillSpec,
    settings: Settings = Depends(get_settings),
    jobs: JobStore = Depends(get_jobs),
) -> dict[str, str]:
    async def run(report: Any) -> Any:
        return await run_backfill(settings, spec, progress=lambda msg: report(message=msg))

    job = jobs.submit("backfill", run)
    return {"job_id": job.id}
