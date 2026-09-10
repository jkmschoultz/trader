"""A symbol registry for the bar lake.

The lake addresses every series by a numeric ``Uic`` -- that is the stable key,
and partition paths stay numeric because a symbol can be reassigned while a Uic
cannot. This module is the side table that puts a human label back on a Uic, so
``coverage`` can read ``US500.I:CfdOnIndex:4913`` rather than ``CfdOnIndex:4913``.

It is a single JSON file, ``{data_dir}/instruments.json``, keyed by
``"{asset_type}:{uic}"``. It is written whenever a backfill resolves an
instrument, and can be rebuilt for an existing lake with ``trader data symbols``.
Nothing in the read or write path of the bars themselves depends on it: a
missing or stale registry only costs a prettier label.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)

FILENAME = "instruments.json"


@dataclass(frozen=True)
class InstrumentRecord:
    """What the registry knows about one ``(asset_type, uic)``."""

    asset_type: str
    uic: int
    symbol: str = ""
    description: str = ""
    currency: str = ""
    exchange_id: str = ""
    updated: datetime | None = None

    @property
    def label(self) -> str:
        """``"US500.I:CfdOnIndex:4913"``, or ``"CfdOnIndex:4913"`` with no symbol."""
        head = f"{self.symbol}:" if self.symbol else ""
        return f"{head}{self.asset_type}:{self.uic}"

    def to_json(self) -> dict[str, object]:
        return {
            "asset_type": self.asset_type,
            "uic": self.uic,
            "symbol": self.symbol,
            "description": self.description,
            "currency": self.currency,
            "exchange_id": self.exchange_id,
            "updated": self.updated.isoformat() if self.updated else None,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> InstrumentRecord:
        raw = data.get("updated")
        updated = None
        if isinstance(raw, str) and raw:
            try:
                updated = datetime.fromisoformat(raw)
            except ValueError:
                updated = None
        return cls(
            asset_type=str(data.get("asset_type", "")),
            uic=int(data.get("uic", 0)),  # type: ignore[arg-type]
            symbol=str(data.get("symbol", "")),
            description=str(data.get("description", "")),
            currency=str(data.get("currency", "")),
            exchange_id=str(data.get("exchange_id", "")),
            updated=updated,
        )


def _key(asset_type: str, uic: int) -> str:
    return f"{asset_type}:{int(uic)}"


class InstrumentRegistry:
    """Reads and writes the ``instruments.json`` side table under a data root."""

    def __init__(self, root: Path) -> None:
        self._path = Path(root) / FILENAME
        self._cache: dict[str, InstrumentRecord] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def _load(self) -> dict[str, InstrumentRecord]:
        if self._cache is not None:
            return self._cache
        records: dict[str, InstrumentRecord] = {}
        if self._path.is_file():
            try:
                raw = json.loads(self._path.read_text())
            except (OSError, ValueError):
                log.warning("could not read instrument registry at %s; treating it as empty",
                            self._path)
                raw = {}
            for key, value in (raw or {}).items():
                if isinstance(value, dict):
                    records[key] = InstrumentRecord.from_json(value)
        self._cache = records
        return records

    def get(self, asset_type: str, uic: int) -> InstrumentRecord | None:
        return self._load().get(_key(asset_type, uic))

    def symbol_for(self, asset_type: str, uic: int) -> str:
        record = self.get(asset_type, uic)
        return record.symbol if record else ""

    def label(self, asset_type: str, uic: int) -> str:
        """Human label for a series key, symbol-prefixed when one is known."""
        record = self.get(asset_type, uic)
        return record.label if record else f"{asset_type}:{int(uic)}"

    def records(self) -> list[InstrumentRecord]:
        return sorted(self._load().values(), key=lambda r: (r.asset_type, r.uic))

    def put(
        self,
        asset_type: str,
        uic: int,
        *,
        symbol: str = "",
        description: str = "",
        currency: str = "",
        exchange_id: str = "",
    ) -> InstrumentRecord:
        """Insert or replace one record, then rewrite the file atomically."""
        record = InstrumentRecord(
            asset_type=asset_type,
            uic=int(uic),
            symbol=symbol,
            description=description,
            currency=currency,
            exchange_id=exchange_id,
            updated=datetime.now(UTC),
        )
        records = dict(self._load())
        records[_key(asset_type, uic)] = record
        self._write(records)
        self._cache = records
        return record

    def _write(self, records: dict[str, InstrumentRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: records[key].to_json() for key in sorted(records)}
        tmp = self._path.with_suffix(f".{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, self._path)
        finally:
            tmp.unlink(missing_ok=True)

    def __iter__(self) -> Iterator[InstrumentRecord]:
        return iter(self.records())

    def __len__(self) -> int:
        return len(self._load())
