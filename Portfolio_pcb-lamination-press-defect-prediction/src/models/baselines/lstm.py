"""LSTM baseline skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LSTMConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    bidirectional: bool = False

