"""What strategies and allocators the UI can offer, with their parameters."""

from __future__ import annotations

from fastapi import APIRouter

from trader.service import list_allocators, list_strategies
from trader.service.catalog import AllocatorInfo, StrategyInfo

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/strategies")
def strategies() -> list[StrategyInfo]:
    return list_strategies()


@router.get("/allocators")
def allocators() -> list[AllocatorInfo]:
    return list_allocators()
