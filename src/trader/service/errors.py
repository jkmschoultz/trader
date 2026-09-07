"""Errors the service layer raises, for a front-end to translate.

The CLI turns these into a ``UsageError`` (exit 2, no traceback); the API turns
them into a 4xx with the message as the detail, and carries any ``data`` payload
onto the failed job so the UI can offer a one-click fix.
"""

from __future__ import annotations

__all__ = ["ServiceError", "InvalidRequest", "SeriesNotStored"]


class ServiceError(Exception):
    """Base for anything the caller got wrong, as opposed to a bug.

    ``data`` is an optional JSON-safe dict a front-end can act on (e.g. which
    series to backfill).
    """

    data: dict | None = None


class InvalidRequest(ServiceError):
    """A malformed or unsatisfiable request: unknown strategy, bad params, ..."""


class SeriesNotStored(ServiceError):
    """The lake holds nothing for a series the request needs."""

    def __init__(
        self,
        symbol: str,
        horizon_label: str,
        *,
        asset_type: str | None = None,
        uic: int | None = None,
        since: str | None = None,
    ) -> None:
        super().__init__(
            f"nothing stored for {symbol!r} at {horizon_label}. "
            f"Fetch it first: trader data backfill {symbol} --horizon {horizon_label}"
        )
        self.symbol = symbol
        self.horizon_label = horizon_label
        self.data = {
            "kind": "series_not_stored",
            "symbol": symbol,
            "horizon": horizon_label,
            "asset_type": asset_type,
            "uic": uic,
            "since": since,
        }
