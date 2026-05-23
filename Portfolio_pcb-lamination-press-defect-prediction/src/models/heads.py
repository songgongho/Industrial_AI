"""Task heads for the semiconductor PCB lamination models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class HeadConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    binary_out: int = 1
    defect_classes: int = 2
    anomaly_out: int = 1


class TaskHeads(nn.Module):
    """Shared multi-task prediction heads."""

    def __init__(self, input_dim: int, config: HeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or HeadConfig()
        hidden = self.config.hidden_dim
        dropout = self.config.dropout
        self.binary_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.config.binary_out),
        )
        self.defect_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.config.defect_classes),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.config.anomaly_out),
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {
            "binary_logits": self.binary_head(features),
            "defect_logits": self.defect_head(features),
            "anomaly_score": self.anomaly_head(features),
        }


