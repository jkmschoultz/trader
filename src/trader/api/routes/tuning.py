"""Submit a walk-forward grid sweep as a background job.

Same shape as ``/training``: a bad spec is a synchronous 422; the sweep itself
runs as ``kind="tuning"`` on the shared :class:`~trader.api.jobs.JobStore` and
its report is read back through ``GET /api/jobs/{id}``. A copy of the report is
also written under ``state/`` for the record.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends

from trader.api.deps import get_jobs, get_settings
from trader.api.jobs import JobStore
from trader.config import Settings
from trader.service.tuning import TuningSpec, run_tuning, save_report

router = APIRouter(tags=["tuning"])


@router.post("/tuning", status_code=202)
async def submit_tuning(
    spec: TuningSpec,
    settings: Settings = Depends(get_settings),
    jobs: JobStore = Depends(get_jobs),
) -> dict[str, str]:
    async def run(report: Any) -> Any:
        result = await run_tuning(
            settings,
            spec,
            progress=lambda p: report(progress=p),
            on_message=lambda m: report(message=m),
        )
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = save_report(result, settings.state_dir / f"tuning-{stamp}.json")
        return {**result, "report_path": str(path)}

    job = jobs.submit("tuning", run)
    return {"job_id": job.id}
