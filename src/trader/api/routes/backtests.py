"""Submit a backtest as a background job; the result is the job's ``result``.

Validation errors in the *shape* of the request (bad horizon, no symbols) come
back synchronously as 422. Errors that need the lake or the network -- unknown
strategy, nothing stored, not signed in -- surface as a failed job with the
message in ``job.error``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from trader.api.deps import get_jobs, get_settings
from trader.api.jobs import JobStore
from trader.config import Settings
from trader.service.backtest import BacktestSpec, run_backtest

router = APIRouter(prefix="/backtests", tags=["backtests"])


@router.post("", status_code=202)
async def submit(
    spec: BacktestSpec,
    settings: Settings = Depends(get_settings),
    jobs: JobStore = Depends(get_jobs),
) -> dict[str, str]:
    async def run(report: Any) -> Any:
        result = await run_backtest(settings, spec, progress=lambda p: report(progress=p))
        return result.to_dict()

    job = jobs.submit("backtest", run)
    return {"job_id": job.id}
