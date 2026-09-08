"""``ModelRegistry``: save round-trips, reads need no torch, failures are clean."""

from __future__ import annotations

import json

import pytest

from trader.models.registry import ModelInfo, ModelNotFound, ModelRegistry

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.slow


def test_save_then_get_and_list_round_trip(tmp_path, trained_artifacts):
    registry = ModelRegistry(tmp_path)
    info = registry.save(**trained_artifacts())

    assert isinstance(info, ModelInfo)
    assert info.id.startswith("lstm-")
    assert info.feature_set == "price_v1"
    assert info.barriers["max_bars"] == 12
    assert info.window == 16
    assert info.symbols == ["X"]

    fetched = registry.get(info.id)
    assert fetched == info
    assert [m.id for m in registry.list()] == [info.id]


def test_saved_directory_has_every_file(tmp_path, trained_artifacts):
    registry = ModelRegistry(tmp_path)
    info = registry.save(**trained_artifacts())
    directory = registry.path(info.id)

    for name in ("manifest.json", "weights.pt", "scaler.json", "feature_spec.json", "report.json"):
        assert (directory / name).is_file(), name

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["id"] == info.id
    assert manifest["hyperparameters"]["hidden"] == 8
    assert manifest["hyperparameters"]["lr"] == 1e-3


def test_load_torch_returns_a_working_net(tmp_path, trained_artifacts):
    registry = ModelRegistry(tmp_path)
    info = registry.save(**trained_artifacts())

    net, scaler, loaded = registry.load_torch(info.id)
    assert loaded == info
    assert not net.training  # eval mode
    x = torch.zeros(2, info.window, net.config.n_features)
    assert net(x).shape == (2, 3)
    assert scaler.mean_ is not None


def test_get_unknown_model_raises_model_not_found(tmp_path):
    with pytest.raises(ModelNotFound, match="nope"):
        ModelRegistry(tmp_path).get("nope")


def test_list_is_empty_when_the_root_does_not_exist(tmp_path):
    assert ModelRegistry(tmp_path / "absent").list() == []


def test_reads_do_not_need_torch(tmp_path, trained_artifacts, monkeypatch):
    # save first (needs torch), then blind the import and prove list/get still work
    info = ModelRegistry(tmp_path).save(**trained_artifacts())

    import builtins

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch hidden for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    registry = ModelRegistry(tmp_path)
    assert registry.get(info.id).id == info.id
    assert [m.id for m in registry.list()] == [info.id]


def test_a_failed_save_leaves_no_directory_behind(tmp_path, trained_artifacts, monkeypatch):
    registry = ModelRegistry(tmp_path)
    args = trained_artifacts()

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        registry.save(**args)

    assert list(tmp_path.iterdir()) == []  # no partial dir, no leftover .tmp
