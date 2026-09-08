"""Submit a training run as a background job; browse the model registry.

Same shape as ``/backtests``: shape errors (missing dates, ``val_end`` before
``train_end``) are a synchronous 422; anything that needs the lake, the network,
or the ``[model]`` extra surfaces as a failed job with the message in
``job.error``. The trained model then flows through ``POST /api/backtests``
unchanged as ``{"strategy": "lstm", "params": {"model": "<id>"}}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from trader.api.deps import get_jobs, get_settings
from trader.api.jobs import JobStore
from trader.config import Settings
from trader.models.registry import ModelInfo, ModelNotFound, ModelRegistry
from trader.service.catalog import ModelSummary, list_models
from trader.service.training import TrainingSpec, run_training

router = APIRouter(tags=["training"])


@router.post("/training", status_code=202)
async def submit_training(
    spec: TrainingSpec,
    settings: Settings = Depends(get_settings),
    jobs: JobStore = Depends(get_jobs),
) -> dict[str, str]:
    async def run(report: Any) -> Any:
        return await run_training(settings, spec, progress=lambda p: report(progress=p))

    job = jobs.submit("training", run)
    return {"job_id": job.id}


@router.get("/models")
def models(settings: Settings = Depends(get_settings)) -> list[ModelSummary]:
    return list_models(settings)


@router.get("/models/{model_id}")
def model_detail(model_id: str, settings: Settings = Depends(get_settings)) -> ModelInfo:
    try:
        return ModelRegistry(settings.models_dir).get(model_id)
    except ModelNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
