"""The LSTM classifier. Imports torch at module load -- import it lazily.

A plain stacked LSTM over the ``(batch, window, features)`` sequence, the last
(or mean-pooled) hidden state, dropout, and a linear head to three classes
(down / flat / up). Multi-timeframe context enters as extra feature columns
upstream, so there is one LSTM, not one per horizon.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import nn

__all__ = ["LSTMClassifier", "LSTMConfig"]


@dataclass(frozen=True)
class LSTMConfig:
    n_features: int
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    n_classes: int = 3
    pool: Literal["last", "mean"] = "last"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LSTMConfig:
        fields = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**fields)


class LSTMClassifier(nn.Module):
    """``(B, T, F) -> (B, n_classes)`` logits."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.n_features,
            hidden_size=config.hidden,
            num_layers=config.layers,
            batch_first=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        out_dim = config.hidden * (2 if config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(out_dim, config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled = out[:, -1, :] if self.config.pool == "last" else out.mean(dim=1)
        return self.head(pooled)
