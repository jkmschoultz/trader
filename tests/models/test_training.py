"""``train_model``: learns a separable set, reports, early-stops, is deterministic."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from trader.models.lstm import LSTMConfig  # noqa: E402
from trader.models.training import train_model  # noqa: E402

pytestmark = pytest.mark.slow


def _split(x, y, a=0.7, b=0.85):
    n = len(y)
    i, j = int(n * a), int(n * b)
    return (x[:i], y[:i]), (x[i:j], y[i:j]), (x[j:], y[j:])


def test_learns_a_separable_dataset(separable_sequences):
    x, y = separable_sequences(n_per_class=250, window=8, features=4, seed=0)
    (xtr, ytr), (xva, yva), (xte, yte) = _split(x, y)
    wtr = np.ones(len(ytr), dtype="float32")

    model, report = train_model(
        xtr,
        ytr,
        wtr,
        xva,
        yva,
        config=LSTMConfig(n_features=4, hidden=12, layers=1, dropout=0.0),
        epochs=8,
        Xte=xte,
        yte=yte,
        batch_size=64,
        patience=10,
        seed=0,
    )

    assert report.metrics["val_macro_f1"] > 0.9
    assert report.metrics["test_macro_f1"] > 0.85
    assert report.n_train == len(ytr)
    assert report.best_epoch >= 1
    assert len(report.val_confusion) == 3
    assert report.test_confusion is not None
    assert set(report.class_distribution) == {"train", "val", "test"}


def test_report_is_json_safe(separable_sequences):
    x, y = separable_sequences(n_per_class=80, window=6, features=3, seed=1)
    (xtr, ytr), (xva, yva), _ = _split(x, y)
    _, report = train_model(
        xtr,
        ytr,
        np.ones(len(ytr), "float32"),
        xva,
        yva,
        config=LSTMConfig(n_features=3, hidden=6, layers=1),
        epochs=3,
        seed=0,
    )
    json.dumps(report.to_dict())  # must not raise


def test_early_stops_on_a_flat_validation_curve(separable_sequences):
    # pure noise -> val F1 never improves -> stop after `patience` epochs
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(300, 6, 3)).astype("float32")
    y = rng.integers(0, 3, size=300).astype("int64")
    (xtr, ytr), (xva, yva), _ = _split(x, y)

    _, report = train_model(
        xtr,
        ytr,
        np.ones(len(ytr), "float32"),
        xva,
        yva,
        config=LSTMConfig(n_features=3, hidden=4, layers=1),
        epochs=50,
        patience=3,
        seed=0,
    )
    assert len(report.epochs) < 50


def test_two_runs_with_the_same_seed_match(separable_sequences):
    x, y = separable_sequences(n_per_class=120, window=6, features=3, seed=2)
    (xtr, ytr), (xva, yva), _ = _split(x, y)
    wtr = np.ones(len(ytr), dtype="float32")
    cfg = LSTMConfig(n_features=3, hidden=6, layers=1)

    _, r1 = train_model(xtr, ytr, wtr, xva, yva, config=cfg, epochs=5, seed=7, batch_size=32)
    _, r2 = train_model(xtr, ytr, wtr, xva, yva, config=cfg, epochs=5, seed=7, batch_size=32)
    assert r1.epochs == r2.epochs


def test_no_training_samples_raises():
    empty = np.zeros((0, 6, 3), dtype="float32")
    with pytest.raises(ValueError, match="no training samples"):
        train_model(
            empty,
            np.array([], "int64"),
            np.array([], "float32"),
            np.zeros((4, 6, 3), "float32"),
            np.zeros(4, "int64"),
            config=LSTMConfig(n_features=3, hidden=4, layers=1),
            epochs=2,
        )
