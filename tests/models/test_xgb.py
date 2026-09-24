"""``train_xgb`` + the XGB branch of ``ModelRegistry``: fit, save, reload, CPU inference."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from trader.labels.triple_barrier import LabelConfig  # noqa: E402
from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split  # noqa: E402
from trader.models.gbm import GBMConfig  # noqa: E402
from trader.models.registry import ModelRegistry  # noqa: E402
from trader.models.scaler import StandardScaler  # noqa: E402
from trader.models.xgb import _balanced_weights, cuda_available, train_xgb  # noqa: E402

pytestmark = pytest.mark.slow

WINDOW = 8
MAX_BARS = 10


@pytest.fixture
def tabular_fold(bars):
    label = LabelConfig(stop=0.01, take=0.01, max_bars=MAX_BARS)
    spec = WindowSpec(
        window=WINDOW,
        feature_set="price_v1",
        base_horizon=5,
        context_horizons=(),
        label=label,
        layout="tabular",
    )
    bundle = build_bundle(bars(2600, seed=3), spec=spec)
    t = bundle.t
    split = SplitSpec(
        train_end=pd.Timestamp(t[int(len(t) * 0.7)]).to_pydatetime(),
        val_end=pd.Timestamp(t[int(len(t) * 0.85)]).to_pydatetime(),
        embargo_bars=15,
    )
    tr, va, _ = time_split(bundle, split, base_horizon=5, max_bars=MAX_BARS)
    scaler = StandardScaler().fit(bundle.X[tr])
    return bundle, label, split, tr, va, scaler


def _fit(fold, *, device="cpu", **config):
    bundle, _, _, tr, va, scaler = fold
    return train_xgb(
        scaler.transform(bundle.X[tr]),
        bundle.y[tr],
        bundle.w[tr],
        scaler.transform(bundle.X[va]),
        bundle.y[va],
        config=GBMConfig(**{"n_estimators": 40, "num_leaves": 15, **config}),
        seed=0,
        device=device,
    )


def test_balanced_weights_match_sklearn():
    from sklearn.utils.class_weight import compute_sample_weight

    y = np.array([0, 0, 0, 0, 1, 2, 2, 2])
    np.testing.assert_allclose(_balanced_weights(y, 3), compute_sample_weight("balanced", y))


def test_trains_when_a_fold_has_no_flat_labels(tabular_fold):
    """min_return=0 leaves the flat class nearly empty; a fold may lack it entirely."""
    bundle, _, _, tr, va, scaler = tabular_fold
    keep_tr = bundle.y[tr] != 1
    keep_va = bundle.y[va] != 1
    booster, report = train_xgb(
        scaler.transform(bundle.X[tr][keep_tr]),
        bundle.y[tr][keep_tr],
        bundle.w[tr][keep_tr],
        scaler.transform(bundle.X[va][keep_va]),
        bundle.y[va][keep_va],
        config=GBMConfig(n_estimators=20, num_leaves=7),
        device="cpu",
    )
    assert booster.inplace_predict(scaler.transform(bundle.X[va][:3])).shape == (3, 3)
    assert report.class_distribution["train"]["flat"] == 0


def test_booster_is_trimmed_to_the_best_iteration(tabular_fold):
    booster, report = _fit(tabular_fold, n_estimators=400, learning_rate=0.3)
    assert booster.num_boosted_rounds() == report.best_epoch
    assert len(report.epochs) > report.best_epoch  # early stopping ran past the best


def test_balanced_weights_tolerate_a_missing_class():
    y = np.array([0, 0, 2, 2, 2])  # no flat labels, as with min_return=0
    weights = _balanced_weights(y, 3)
    assert np.isfinite(weights).all()
    assert weights[0] > weights[-1]  # the rarer class counts more


def test_train_xgb_reports_val_metrics(tabular_fold):
    bundle, _, _, _, va, scaler = tabular_fold
    model, report = _fit(tabular_fold)
    assert "val_macro_f1" in report.metrics
    assert len(report.val_confusion) == 3
    assert report.framework_version.startswith("xgboost") and "(cpu)" in report.framework_version
    assert report.epochs and report.best_epoch >= 1
    assert model.inplace_predict(scaler.transform(bundle.X[va][:5])).shape == (5, 3)


def test_train_xgb_is_deterministic_given_seed(tabular_fold):
    bundle, _, _, _, va, scaler = tabular_fold
    x = scaler.transform(bundle.X[va][:20])
    a, _ = _fit(tabular_fold)
    b, _ = _fit(tabular_fold)
    np.testing.assert_array_equal(a.inplace_predict(x), b.inplace_predict(x))


def test_save_and_load_xgb_round_trips(tmp_path, tabular_fold):
    bundle, label, split, _, va, scaler = tabular_fold
    config = GBMConfig(n_estimators=30, num_leaves=15)
    model, report = _fit(tabular_fold, n_estimators=30)

    registry = ModelRegistry(tmp_path)
    info = registry.save(
        name="xgb",
        model=model,
        scaler=scaler,
        feature_spec=bundle.feature_spec,
        label=label,
        window=WINDOW,
        split=split,
        config=config,
        report=report,
        model_type="xgb",
        layout="tabular",
        data_spec={"symbols": ["X"]},
    )
    assert info.model_type == "xgb"
    assert info.layout == "tabular"

    directory = registry.path(info.id)
    assert (directory / "model.json").is_file()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["files"]["weights"] == "model.json"
    assert "xgboost" in manifest["framework"]

    booster, loaded_scaler, loaded = registry.load_model(info.id)
    assert loaded == info
    x = loaded_scaler.transform(bundle.X[va][:4]).astype(np.float32)
    proba = booster.inplace_predict(x)
    assert proba.shape == (4, 3)
    np.testing.assert_allclose(proba, model.inplace_predict(x), rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not cuda_available(), reason="no CUDA device for xgboost")
def test_cuda_trained_model_loads_and_predicts_on_cpu(tmp_path, tabular_fold):
    bundle, label, split, _, va, scaler = tabular_fold
    model, report = _fit(tabular_fold, device="cuda")
    assert "(cuda)" in report.framework_version

    registry = ModelRegistry(tmp_path)
    info = registry.save(
        name="xgb",
        model=model,
        scaler=scaler,
        feature_spec=bundle.feature_spec,
        label=label,
        window=WINDOW,
        split=split,
        config=GBMConfig(n_estimators=40, num_leaves=15),
        report=report,
        model_type="xgb",
        layout="tabular",
        data_spec={"symbols": ["X"]},
    )
    booster, loaded_scaler, _ = registry.load_model(info.id)
    config = json.loads(booster.save_config())
    assert config["learner"]["generic_param"]["device"] == "cpu"
    x = loaded_scaler.transform(bundle.X[va][:4]).astype(np.float32)
    assert booster.inplace_predict(x).shape == (4, 3)
