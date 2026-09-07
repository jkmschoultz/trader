"""Poll a job, or stream its progress as Server-Sent Events."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from trader.api.deps import get_jobs
from trader.api.jobs import JobStore

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
def list_jobs(jobs: JobStore = Depends(get_jobs)) -> list[dict[str, Any]]:
    return [job.snapshot(with_result=False) for job in jobs.list()]


@router.get("/{job_id}")
def get_job(job_id: str, jobs: JobStore = Depends(get_jobs)) -> dict[str, Any]:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")
    return job.snapshot()


@router.get("/{job_id}/events")
async def job_events(job_id: str, jobs: JobStore = Depends(get_jobs)) -> StreamingResponse:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="no such job")

    async def stream() -> Any:
        async for event in jobs.events(job):
            yield f"data: {json.dumps(event, default=str)}\n\n"
        yield f"event: done\ndata: {json.dumps(job.snapshot(), default=str)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
