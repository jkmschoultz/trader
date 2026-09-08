"""The LSTM module: output shape and determinism."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from trader.models.lstm import LSTMClassifier, LSTMConfig  # noqa: E402


def test_forward_shape():
    config = LSTMConfig(n_features=5, hidden=8, layers=1)
    model = LSTMClassifier(config)
    out = model(torch.zeros(7, 12, 5))
    assert out.shape == (7, 3)


def test_mean_pool_also_works():
    model = LSTMClassifier(LSTMConfig(n_features=4, hidden=6, layers=2, pool="mean"))
    assert model(torch.zeros(3, 9, 4)).shape == (3, 3)


def test_deterministic_given_a_seed():
    x = torch.randn(4, 10, 5)

    torch.manual_seed(0)
    a = LSTMClassifier(LSTMConfig(n_features=5, hidden=8, layers=1))(x)
    torch.manual_seed(0)
    b = LSTMClassifier(LSTMConfig(n_features=5, hidden=8, layers=1))(x)

    assert torch.allclose(a, b)


def test_config_dict_round_trip():
    config = LSTMConfig(n_features=9, hidden=16, layers=3, dropout=0.1, bidirectional=True)
    assert LSTMConfig.from_dict(config.to_dict()) == config


def test_bidirectional_head_dimension():
    model = LSTMClassifier(LSTMConfig(n_features=3, hidden=5, layers=1, bidirectional=True))
    assert model(torch.zeros(2, 6, 3)).shape == (2, 3)
    # head input is hidden * 2
    linear = [m for m in model.head if isinstance(m, torch.nn.Linear)][0]
    assert linear.in_features == 10
