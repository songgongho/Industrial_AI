"""Anomaly Transformer baseline skeleton."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AnomalyTransformerConfig:
    d_model: int = 128
    n_heads: int = 4

