"""FastAPI service over the backtest engine and the bar lake.

Needs the ``[api]`` extra (and, for anything that touches data, ``[data]``)::

    pip install -e ".[data,api]"

``create_app`` is the factory; ``trader serve`` runs it under uvicorn. Every
route lives under ``/api``; when ``frontend/dist`` exists it is served at ``/``.
"""

from __future__ import annotations

from trader.api.app import create_app

__all__ = ["create_app"]
