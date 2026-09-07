"""Run the API under uvicorn. Wired to ``trader serve``."""

from __future__ import annotations


def run(*, host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run(
        "trader.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )
