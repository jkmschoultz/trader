"""An in-process registry of background jobs.

A job is one ``asyncio.Task`` running a coroutine that reports progress through a
``report`` callback. Callers poll ``GET /api/jobs/{id}`` or subscribe to its
event stream. State lives only in this process -- restart it and running jobs
are gone -- which is the right trade for a single-user local tool.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)

#: A job body: given a ``report`` callback, do the work and return a
#: JSON-serialisable result.
Reporter = Callable[..., None]
JobRun = Callable[[Reporter], Awaitable[Any]]


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    id: str
    kind: str
    status: JobStatus = JobStatus.QUEUED
    progress: float | None = None
    message: str = ""
    result: Any = None
    error: str | None = None
    error_data: dict | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    _task: asyncio.Task | None = field(default=None, repr=False)
    _subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)

    @property
    def finished(self) -> bool:
        return self.status in (JobStatus.DONE, JobStatus.ERROR)

    def snapshot(self, *, with_result: bool = True) -> dict[str, Any]:
        data = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "error_data": self.error_data,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }
        if with_result:
            data["result"] = self.result
        return data


class JobStore:
    """Holds every job this process has run, newest kept, oldest evicted."""

    def __init__(self, *, keep: int = 200) -> None:
        self._jobs: dict[str, Job] = {}
        self._keep = keep

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def submit(self, kind: str, run: JobRun) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        self._jobs[job.id] = job
        self._evict()
        job._task = asyncio.create_task(self._run(job, run))
        return job

    async def _run(self, job: Job, run: JobRun) -> None:
        job.status = JobStatus.RUNNING
        self._publish(job, {"type": "status", "status": job.status.value})

        def report(*, progress: float | None = None, message: str = "") -> None:
            if progress is not None:
                job.progress = max(0.0, min(1.0, float(progress)))
            if message:
                job.message = message
            self._publish(
                job,
                {"type": "progress", "progress": job.progress, "message": job.message},
            )

        try:
            job.result = await run(report)
            job.status = JobStatus.DONE
            if job.progress is not None:
                job.progress = 1.0
        except asyncio.CancelledError:
            job.status = JobStatus.ERROR
            job.error = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 - reported to the client, logged in full
            log.exception("job %s (%s) failed", job.id, job.kind)
            job.status = JobStatus.ERROR
            job.error = str(exc) or exc.__class__.__name__
            data = getattr(exc, "data", None)
            job.error_data = data if isinstance(data, dict) else None
        finally:
            job.finished_at = time.time()
            self._publish(job, {"type": "status", "status": job.status.value})
            for queue in list(job._subscribers):
                _offer(queue, None)

    def _publish(self, job: Job, event: dict[str, Any]) -> None:
        for queue in list(job._subscribers):
            _offer(queue, event)

    def _evict(self) -> None:
        if len(self._jobs) <= self._keep:
            return
        finished = [j for j in self.list() if j.finished][self._keep :]
        for job in finished:
            self._jobs.pop(job.id, None)

    async def events(self, job: Job) -> AsyncIterator[dict[str, Any]]:
        """Yield the job's current state, then every event until it finishes."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        job._subscribers.add(queue)
        try:
            yield {
                "type": "status",
                "status": job.status.value,
                "progress": job.progress,
                "message": job.message,
            }
            if job.finished:
                return
            while True:
                event = await queue.get()
                if event is None:
                    return
                yield event
        finally:
            job._subscribers.discard(queue)


def _offer(queue: asyncio.Queue, item: Any) -> None:
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:  # pragma: no cover - a stuck subscriber
        pass
