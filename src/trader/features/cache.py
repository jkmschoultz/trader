"""An optional on-disk feature cache, parallel to the bar lake.

Only used when a feature build is asked to ``persist``. The default path
recomputes features every time -- one code path, no staleness. This store exists
for when feature computation is profiled as the bottleneck of a training loop.

Layout::

    data/features/{asset_type}/{uic}/{base_horizon}/{set}@{digest}/{period}.parquet

The spec digest is in the path, so a changed spec writes to a new directory and
can never read a stale file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from trader.data.lake import _write_atomic, period_key
from trader.features.base import FeatureSpec

__all__ = ["FeatureLake"]


class FeatureLake:
    """Reads and writes computed feature frames under a root directory."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    def directory(self, key, spec: FeatureSpec) -> Path:
        return (
            self._root
            / "features"
            / key.asset_type
            / str(key.uic)
            / str(spec.base_horizon)
            / f"{spec.name}@{spec.digest()[:12]}"
        )

    def _paths(self, key, spec: FeatureSpec) -> list[Path]:
        directory = self.directory(key, spec)
        if not directory.is_dir():
            return []
        return sorted(directory.glob("*.parquet"))

    def write(self, key, spec: FeatureSpec, features: pd.DataFrame) -> int:
        """Merge ``features`` (indexed by ``time``) into the cache. Returns files touched."""
        if features.empty:
            return 0
        frame = features.reset_index()
        if "time" not in frame.columns:
            raise ValueError("feature frame must be indexed by 'time'")

        directory = self.directory(key, spec)
        directory.mkdir(parents=True, exist_ok=True)

        periods = frame["time"].map(lambda ts: period_key(ts, spec.base_horizon))
        touched = 0
        for period, chunk in frame.groupby(periods, sort=True):
            path = directory / f"{period}.parquet"
            existing = pd.read_parquet(path) if path.exists() else None
            merged = (
                chunk
                if existing is None
                else pd.concat([existing, chunk], ignore_index=True)
                .drop_duplicates(subset="time", keep="last")
                .sort_values("time")
            )
            _write_atomic(merged.reset_index(drop=True), path)
            touched += 1
        return touched

    def read(
        self,
        key,
        spec: FeatureSpec,
        *,
        start=None,
        end=None,
    ) -> pd.DataFrame:
        """Return cached features for ``key``/``spec`` in ``[start, end]``, indexed by ``time``."""
        paths = self._paths(key, spec)
        if not paths:
            return pd.DataFrame(columns=list(spec.columns))
        frame = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        frame["time"] = pd.to_datetime(frame["time"], utc=True)
        frame = frame.drop_duplicates(subset="time", keep="last").sort_values("time")
        if start is not None:
            frame = frame[frame["time"] >= pd.Timestamp(start)]
        if end is not None:
            frame = frame[frame["time"] <= pd.Timestamp(end)]
        return frame.set_index("time")[list(spec.columns)]
