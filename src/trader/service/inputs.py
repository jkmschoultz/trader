"""Parsing of the loosely-typed inputs the CLI and the API both accept.

``--since 90d``, ``--param fast=10``, ``horizon=5m`` -- the same shorthands turn
up as query strings and JSON values on the API side. Each helper here raises a
plain :class:`ValueError` on bad input; the CLI turns that into its ``UsageError``
and FastAPI into a 422.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

__all__ = ["coerce_scalar", "parse_params", "parse_since"]


def parse_since(value: str | datetime | None) -> datetime | None:
    """Parse a lookback (``90d``, ``2y``), a date, an ISO timestamp, or ``all``.

    ``None`` and ``all``/``max`` mean "as far back as the data goes". A returned
    datetime is always timezone-aware UTC.
    """
    if value is None or isinstance(value, datetime):
        return _as_utc(value) if isinstance(value, datetime) else None

    text = value.strip()
    if not text or text.casefold() in {"all", "max"}:
        return None

    if text[-1:].casefold() in {"d", "y"} and text[:-1].replace(".", "", 1).isdigit():
        days = float(text[:-1]) * (365.25 if text[-1:].casefold() == "y" else 1)
        return datetime.now(UTC) - timedelta(days=days)

    try:
        from dateutil.parser import isoparse

        parsed = isoparse(text)
    except (ImportError, ValueError) as exc:
        raise ValueError(
            f"cannot read {value!r}; use YYYY-MM-DD, an ISO timestamp, "
            "a lookback like 90d or 2y, or 'all'"
        ) from exc
    return _as_utc(parsed)


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def coerce_scalar(text: str) -> object:
    """Coerce a string to ``None`` -> ``bool`` -> ``int`` -> ``float`` -> ``str``.

    In that order, so ``"none"`` becomes ``None``, ``"true"`` a bool, ``"10"`` an
    int and ``"0.005"`` a float, and anything else is left as text.
    """
    low = text.casefold()
    if low in {"none", "null", ""}:
        return None
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def parse_params(items: list[str]) -> dict[str, object]:
    """Turn ``["fast=10", "long_only=true"]`` into a coerced kwargs dict."""
    params: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        params[key.strip()] = coerce_scalar(raw.strip())
    return params
