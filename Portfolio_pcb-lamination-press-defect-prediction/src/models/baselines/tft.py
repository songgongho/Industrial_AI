"""Temporal Fusion Transformer baseline skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TFTConfig:
    hidden_size: int = 64
    attention_heads: int = 4

