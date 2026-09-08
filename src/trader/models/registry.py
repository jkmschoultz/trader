"""An on-disk registry of trained models, parallel to the bar lake.

One model is a directory under ``data/models/`` (``Settings.models_dir``)::

    {model_id}/
        manifest.json      -- everything needed to reproduce and describe the run
        weights.pt         -- torch state_dict
        scaler.json        -- the feature StandardScaler
        feature_spec.json  -- the frozen FeatureSpec (inference recomputes features)
        report.json        -- the full TrainReport, for the UI

``model_id`` is ``{name}-{YYYYMMDD-HHMMSS}-{digest8}`` where the digest fingerprints
the feature spec, barriers, window, and hyperparameters, so two genuinely
different models never share an id and a re-run of the same recipe is obvious.

Reading (``list`` / ``get``) needs only the ``[data]`` extra -- it parses JSON.
``save`` and ``load_torch`` need ``[model]``; they import torch lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trader.service.errors import ServiceError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trader.features.base import FeatureSpec
    from trader.labels.triple_barrier import LabelConfig
    from trader.models.dataset import SplitSpec
    from trader.models.lstm import LSTMClassifier, LSTMConfig
    from trader.models.scaler import StandardScaler
    from trader.models.training import TrainReport

__all__ = ["ModelInfo", "ModelNotFound", "ModelRegistry"]

_MANIFEST = "manifest.json"
_WEIGHTS = "weights.pt"
_SCALER = "scaler.json"
_FEATURE_SPEC = "feature_spec.json"
_REPORT = "report.json"


class ModelNotFound(ServiceError):
    """No model is registered under the given id."""


@dataclass(frozen=True)
class ModelInfo:
    """The typed, JSON-safe view of a model's manifest that the API returns."""

    id: str
    name: str
    created_at: str
    base_horizon: int
    context_horizons: list[int]
    feature_set: str
    feature_digest: str
    window: int
    barriers: dict[str, Any]
    split: dict[str, Any]
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]
    class_distribution: dict[str, Any]
    symbols: list[str]
    asset_type: str | None

    @classmethod
    def from_manifest(cls, manifest: dict) -> ModelInfo:
        data = manifest.get("data", {})
        return cls(
            id=manifest["id"],
            name=manifest["name"],
            created_at=manifest["created_at"],
            base_horizon=int(manifest["base_horizon"]),
            context_horizons=list(manifest.get("context_horizons", [])),
            feature_set=manifest["feature_set"],
            feature_digest=manifest["feature_digest"],
            window=int(manifest["window"]),
            barriers=dict(manifest.get("barriers", {})),
            split=dict(manifest.get("split", {})),
            hyperparameters=dict(manifest.get("hyperparameters", {})),
            metrics=dict(manifest.get("metrics", {})),
            class_distribution=dict(manifest.get("class_distribution", {})),
            symbols=list(data.get("symbols", [])),
            asset_type=data.get("asset_type"),
        )


class ModelRegistry:
    """Reads and writes model directories under a root."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def path(self, model_id: str) -> Path:
        return self._root / model_id

    def _manifest(self, model_id: str) -> dict:
        path = self.path(model_id) / _MANIFEST
        if not path.is_file():
            raise ModelNotFound(f"no model {model_id!r} under {self._root}")
        return json.loads(path.read_text(encoding="utf-8"))

    def get(self, model_id: str) -> ModelInfo:
        """The manifest of one model, typed.

        Raises:
            ModelNotFound: nothing registered under ``model_id``.
        """
        return ModelInfo.from_manifest(self._manifest(model_id))

    def manifest(self, model_id: str) -> dict:
        """The full raw manifest dict."""
        return self._manifest(model_id)

    def report(self, model_id: str) -> dict:
        """The full training report, or ``{}`` if none was written."""
        path = self.path(model_id) / _REPORT
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def list(self) -> list[ModelInfo]:
        """Every registered model, newest first. Parses manifests only."""
        if not self._root.is_dir():
            return []
        out: list[ModelInfo] = []
        for manifest_path in self._root.glob("*/" + _MANIFEST):
            if manifest_path.parent.name.startswith("."):
                continue
            try:
                out.append(ModelInfo.from_manifest(json.loads(manifest_path.read_text("utf-8"))))
            except (json.JSONDecodeError, KeyError):
                continue
        return sorted(out, key=lambda info: info.created_at, reverse=True)

    def save(
        self,
        *,
        name: str,
        model: LSTMClassifier,
        scaler: StandardScaler,
        feature_spec: FeatureSpec,
        label: LabelConfig,
        window: int,
        split: SplitSpec,
        config: LSTMConfig,
        report: TrainReport,
        data_spec: dict,
        optimiser: dict | None = None,
    ) -> ModelInfo:
        """Write a model directory atomically and return its :class:`ModelInfo`."""
        import numpy as np
        import torch

        self._root.mkdir(parents=True, exist_ok=True)

        hyperparameters = {**config.to_dict(), **(optimiser or {})}
        barriers = {
            "stop": label.stop,
            "take": label.take,
            "max_bars": label.max_bars,
            "min_return": label.min_return,
        }
        split_dict = {
            "train_end": split.train_end.isoformat(),
            "val_end": split.val_end.isoformat(),
            "embargo_bars": split.embargo_bars,
        }
        created_at = datetime.now(UTC).isoformat()
        digest = _digest(feature_spec.digest(), barriers, window, hyperparameters, split_dict)
        model_id = f"{name}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{digest[:8]}"

        manifest = {
            "id": model_id,
            "name": name,
            "created_at": created_at,
            "git_commit": _git_commit(),
            "data": data_spec,
            "base_horizon": feature_spec.base_horizon,
            "context_horizons": list(feature_spec.context_horizons),
            "feature_set": feature_spec.name,
            "feature_digest": feature_spec.digest(),
            "window": int(window),
            "barriers": barriers,
            "split": split_dict,
            "hyperparameters": hyperparameters,
            "metrics": report.metrics,
            "class_distribution": report.class_distribution,
            "framework": {"torch": torch.__version__, "numpy": np.__version__},
            "files": {
                "weights": _WEIGHTS,
                "scaler": _SCALER,
                "feature_spec": _FEATURE_SPEC,
                "report": _REPORT,
            },
        }

        target = self.path(model_id)
        tmp = self._root / f".{model_id}.{os.getpid()}.tmp"
        if tmp.exists():  # pragma: no cover - stale tmp from a killed process
            _rmtree(tmp)
        tmp.mkdir(parents=True)
        try:
            (tmp / _MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (tmp / _FEATURE_SPEC).write_text(json.dumps(feature_spec.to_dict(), indent=2), "utf-8")
            (tmp / _SCALER).write_text(json.dumps(scaler.to_dict()), encoding="utf-8")
            (tmp / _REPORT).write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            torch.save(model.state_dict(), tmp / _WEIGHTS)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                _rmtree(tmp)

        return ModelInfo.from_manifest(manifest)

    def load_torch(self, model_id: str) -> tuple[LSTMClassifier, StandardScaler, ModelInfo]:
        """Load one model's net (eval mode), scaler, and info.

        Raises:
            ModelNotFound: nothing registered under ``model_id``.
        """
        import torch

        from trader.models.lstm import LSTMClassifier, LSTMConfig
        from trader.models.scaler import StandardScaler

        manifest = self._manifest(model_id)
        info = ModelInfo.from_manifest(manifest)
        directory = self.path(model_id)

        config = LSTMConfig.from_dict(manifest["hyperparameters"])
        net = LSTMClassifier(config)
        state = torch.load(directory / _WEIGHTS, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()

        scaler = StandardScaler.from_dict(json.loads((directory / _SCALER).read_text("utf-8")))
        return net, scaler, info


def _digest(*parts: object) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - git absent
        return None


def _rmtree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
