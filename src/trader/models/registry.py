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

A model directory is not LSTM-specific: ``manifest["model_type"]`` is ``"lstm"``
(torch ``state_dict`` in ``weights.pt``), ``"gbm"`` (a LightGBM text model in
``model.txt``) or ``"xgb"`` (an XGBoost JSON model in ``model.json``), and
``manifest["layout"]`` records whether the features were framed as a
``"sequence"`` or flattened to ``"tabular"``. ``load_model``
dispatches on ``model_type``; ``load_torch`` / ``load_gbm`` / ``load_xgb`` are
the concrete loaders.

Reading (``list`` / ``get``) needs only the ``[data]`` extra -- it parses JSON.
``save`` and the loaders need ``[model]``; they import torch / lightgbm / xgboost lazily.
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
    from trader.models.lstm import LSTMClassifier
    from trader.models.report import TrainReport
    from trader.models.scaler import StandardScaler

__all__ = ["ModelInfo", "ModelNotFound", "ModelRegistry"]

_MANIFEST = "manifest.json"
_WEIGHTS = "weights.pt"  # torch state_dict (model_type="lstm")
_WEIGHTS_GBM = "model.txt"  # LightGBM native text model (model_type="gbm")
_WEIGHTS_XGB = "model.json"  # XGBoost native JSON model (model_type="xgb")
_SCALER = "scaler.json"
_FEATURE_SPEC = "feature_spec.json"
_REPORT = "report.json"

_WEIGHTS_FOR = {"lstm": _WEIGHTS, "gbm": _WEIGHTS_GBM, "xgb": _WEIGHTS_XGB}


class ModelNotFound(ServiceError):
    """No model is registered under the given id."""


@dataclass(frozen=True)
class ModelInfo:
    """The typed, JSON-safe view of a model's manifest that the API returns."""

    id: str
    name: str
    created_at: str
    model_type: str
    layout: str
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
            model_type=manifest.get("model_type", "lstm"),
            layout=manifest.get("layout", "sequence"),
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
        model: Any,
        scaler: StandardScaler,
        feature_spec: FeatureSpec,
        label: LabelConfig,
        window: int,
        split: SplitSpec,
        config: Any,
        report: TrainReport,
        data_spec: dict,
        optimiser: dict | None = None,
        model_type: str = "lstm",
        layout: str = "sequence",
    ) -> ModelInfo:
        """Write a model directory atomically and return its :class:`ModelInfo`.

        ``model_type`` picks the weights format: ``"lstm"`` writes a torch
        ``state_dict`` to ``weights.pt``; ``"gbm"`` writes a LightGBM text model
        to ``model.txt``. ``config`` only needs a ``to_dict()``.
        """
        import numpy as np

        if model_type not in _WEIGHTS_FOR:
            raise ValueError(
                f"unknown model_type {model_type!r}; choose one of {list(_WEIGHTS_FOR)}"
            )

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
            "test_end": split.test_end.isoformat() if split.test_end else None,
            "train_start": split.train_start.isoformat() if split.train_start else None,
        }
        created_at = datetime.now(UTC).isoformat()
        digest = _digest(
            model_type, feature_spec.digest(), barriers, window, hyperparameters, split_dict
        )
        model_id = f"{name}-{datetime.now(UTC):%Y%m%d-%H%M%S}-{digest[:8]}"
        weights_name = _WEIGHTS_FOR[model_type]

        framework: dict[str, str] = {"numpy": np.__version__}
        if model_type == "lstm":
            import torch

            framework["torch"] = torch.__version__
        elif model_type == "xgb":
            from trader.models._optional import require_xgboost

            framework["xgboost"] = require_xgboost().__version__
        else:
            from trader.models._optional import require_lightgbm

            framework["lightgbm"] = require_lightgbm().__version__

        manifest = {
            "id": model_id,
            "name": name,
            "created_at": created_at,
            "model_type": model_type,
            "layout": layout,
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
            "framework": framework,
            "files": {
                "weights": weights_name,
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
            _write_weights(model, model_type, tmp / weights_name)
            os.replace(tmp, target)
        finally:
            if tmp.exists():
                _rmtree(tmp)

        return ModelInfo.from_manifest(manifest)

    def load_model(self, model_id: str) -> tuple[Any, StandardScaler, ModelInfo]:
        """Load a model by id, dispatching on ``manifest["model_type"]``.

        Returns ``(predictor, scaler, info)`` where ``predictor`` is an eval-mode
        torch module (``lstm``), a LightGBM ``Booster`` (``gbm``) or an XGBoost
        ``Booster`` (``xgb``).
        """
        model_type = self._manifest(model_id).get("model_type", "lstm")
        if model_type == "gbm":
            return self.load_gbm(model_id)
        if model_type == "xgb":
            return self.load_xgb(model_id)
        return self.load_torch(model_id)

    def load_torch(self, model_id: str) -> tuple[LSTMClassifier, StandardScaler, ModelInfo]:
        """Load one LSTM model's net (eval mode), scaler, and info.

        Raises:
            ModelNotFound: nothing registered under ``model_id``.
        """
        import torch

        from trader.models.lstm import LSTMClassifier, LSTMConfig
        from trader.models.scaler import StandardScaler

        manifest = self._manifest(model_id)
        info = ModelInfo.from_manifest(manifest)
        directory = self.path(model_id)
        weights = manifest.get("files", {}).get("weights", _WEIGHTS)

        config = LSTMConfig.from_dict(manifest["hyperparameters"])
        net = LSTMClassifier(config)
        state = torch.load(directory / weights, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()

        scaler = StandardScaler.from_dict(json.loads((directory / _SCALER).read_text("utf-8")))
        return net, scaler, info

    def load_gbm(self, model_id: str) -> tuple[Any, StandardScaler, ModelInfo]:
        """Load one GBM model's LightGBM ``Booster``, scaler, and info.

        Raises:
            ModelNotFound: nothing registered under ``model_id``.
        """
        from trader.models._optional import require_lightgbm
        from trader.models.scaler import StandardScaler

        lgb = require_lightgbm()
        manifest = self._manifest(model_id)
        info = ModelInfo.from_manifest(manifest)
        directory = self.path(model_id)
        weights = manifest.get("files", {}).get("weights", _WEIGHTS_GBM)

        booster = lgb.Booster(model_file=str(directory / weights))
        scaler = StandardScaler.from_dict(json.loads((directory / _SCALER).read_text("utf-8")))
        return booster, scaler, info

    def load_xgb(self, model_id: str) -> tuple[Any, StandardScaler, ModelInfo]:
        """Load one XGBoost model's ``Booster`` (predicting on CPU), scaler, and info.

        A model trained on CUDA remembers its device; inference is one row at a
        time, where a GPU round trip costs more than it saves, and the trading
        box may have no GPU at all -- so prediction is pinned to the CPU.

        Raises:
            ModelNotFound: nothing registered under ``model_id``.
        """
        from trader.models._optional import require_xgboost
        from trader.models.scaler import StandardScaler

        xgb = require_xgboost()
        manifest = self._manifest(model_id)
        info = ModelInfo.from_manifest(manifest)
        directory = self.path(model_id)
        weights = manifest.get("files", {}).get("weights", _WEIGHTS_XGB)

        booster = xgb.Booster()
        booster.load_model(str(directory / weights))
        booster.set_param({"device": "cpu", "nthread": 1})
        scaler = StandardScaler.from_dict(json.loads((directory / _SCALER).read_text("utf-8")))
        return booster, scaler, info


def _write_weights(model: Any, model_type: str, path: Path) -> None:
    """Serialise ``model``'s weights to ``path`` in the format for ``model_type``."""
    if model_type == "lstm":
        import torch

        torch.save(model.state_dict(), path)
        return
    if model_type == "xgb":
        # XGBoost sklearn estimator -> its Booster's portable JSON format
        booster = model.get_booster() if hasattr(model, "get_booster") else model
        booster.save_model(str(path))
        return
    # LightGBM sklearn estimator -> its underlying Booster's portable text format.
    booster = getattr(model, "booster_", model)
    booster.save_model(str(path))


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
