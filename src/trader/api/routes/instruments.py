"""Ticker -> Uic search. Needs a live Saxo session."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from trader.api.deps import get_settings
from trader.config import Settings
from trader.saxo.auth import AuthError, ReauthRequired
from trader.saxo.client import SaxoAPIError
from trader.service.data import InstrumentHit, search_instruments

router = APIRouter(prefix="/instruments", tags=["instruments"])


@router.get("/search")
async def search(
    q: str = Query(min_length=1),
    asset_type: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    settings: Settings = Depends(get_settings),
) -> list[InstrumentHit]:
    try:
        return await search_instruments(settings, q, asset_type=asset_type, limit=limit)
    except (ReauthRequired, AuthError) as exc:
        raise HTTPException(
            status_code=409,
            detail=f"not signed in to Saxo ({exc}). Run: trader auth login",
        ) from exc
    except SaxoAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
