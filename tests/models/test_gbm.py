"""``train_gbm`` + the GBM branch of ``ModelRegistry``: fit, save, reload."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from trader.labels.triple_barrier import LabelConfig  # noqa: E402
from trader.models.dataset import SplitSpec, WindowSpec, build_bundle, time_split  # noqa: E402
from trader.models.gbm import GBMConfig, train_gbm  # noqa: E402
from trader.models.registry import ModelRegistry  # noqa: E402
from trader.models.scaler import StandardScaler  # noqa: E402

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
    tr, va, te = time_split(bundle, split, base_horizon=5, max_bars=MAX_BARS)
    return bundle, label, split, tr, va, te


def test_build_bundle_tabular_is_two_dimensional(tabular_fold):
    bundle, *_ = tabular_fold
    assert bundle.X.ndim == 2
    # 17 price_v1 columns * WINDOW lags
    assert bundle.X.shape[1] == 17 * WINDOW
    assert len(bundle.feature_names) == bundle.X.shape[1]
    assert bundle.feature_names[0].endswith(f"__t-{WINDOW - 1}")


def test_train_gbm_reports_val_metrics(tabular_fold):
    bundle, _, _, tr, va, _ = tabular_fold
    scaler = StandardScaler().fit(bundle.X[tr])
    model, report = train_gbm(
        scaler.transform(bundle.X[tr]),
        bundle.y[tr],
        bundle.w[tr],
        scaler.transform(bundle.X[va]),
        bundle.y[va],
        config=GBMConfig(n_estimators=40, num_leaves=15),
        seed=0,
    )
    assert "val_macro_f1" in report.metrics
    assert len(report.val_confusion) == 3
    assert report.framework_version
    assert model.predict_proba(scaler.transform(bundle.X[va][:5])).shape == (5, 3)


def test_save_and_load_gbm_round_trips(tmp_path, tabular_fold):
    bundle, label, split, tr, va, _ = tabular_fold
    scaler = StandardScaler().fit(bundle.X[tr])
    config = GBMConfig(n_estimators=30, num_leaves=15)
    model, report = train_gbm(
        scaler.transform(bundle.X[tr]),
        bundle.y[tr],
        bundle.w[tr],
        scaler.transform(bundle.X[va]),
        bundle.y[va],
        config=config,
        seed=0,
    )

    registry = ModelRegistry(tmp_path)
    info = registry.save(
        name="gbm",
        model=model,
        scaler=scaler,
        feature_spec=bundle.feature_spec,
        label=label,
        window=WINDOW,
        split=split,
        config=config,
        report=report,
        model_type="gbm",
        layout="tabular",
        data_spec={"symbols": ["X"]},
    )
    assert info.model_type == "gbm"
    assert info.layout == "tabular"
    assert info.id.startswith("gbm-")

    directory = registry.path(info.id)
    assert (directory / "model.txt").is_file()
    assert not (directory / "weights.pt").exists()
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["model_type"] == "gbm"
    assert manifest["files"]["weights"] == "model.txt"
    assert "lightgbm" in manifest["framework"]

    booster, loaded_scaler, loaded = registry.load_gbm(info.id)
    assert loaded == info
    proba = booster.predict(loaded_scaler.transform(bundle.X[va][:4]))
    assert proba.shape == (4, 3)
    # load_model dispatches to the same thing
    assert registry.load_model(info.id)[2] == info
