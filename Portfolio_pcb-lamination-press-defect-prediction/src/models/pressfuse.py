"""Cross-modal fusion model for the semiconductor PCB lamination project."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from src.models.heads import HeadConfig, TaskHeads


@dataclass(slots=True)
class CrossModalAttentionConfig:
    ts_input_dim: int = 19
    event_input_dim: int = 19
    d_model: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    head_hidden_dim: int = 128
    defect_classes: int = 2


class CrossModalAttention(nn.Module):
    """Multi-head attention with residual connection and LayerNorm."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        attn_output, attn_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        fused = self.norm(query + self.dropout(attn_output))
        return fused, attn_weights


class PressFuseModel(nn.Module):
    """Practical multimodal fusion model with a single-cycle fallback path."""

    def __init__(self, config: CrossModalAttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or CrossModalAttentionConfig()
        self.ts_encoder = nn.Sequential(
            nn.Linear(self.config.ts_input_dim, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(self.config.event_input_dim, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, self.config.d_model),
        )
        self.fusion = CrossModalAttention(
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
        )
        self.post_fusion = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.heads = TaskHeads(
            input_dim=self.config.d_model,
            config=HeadConfig(
                hidden_dim=self.config.head_hidden_dim,
                dropout=self.config.dropout,
                defect_classes=self.config.defect_classes,
            ),
        )

    def forward(
        self,
        ts: Tensor,
        events: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if ts.dim() != 3:
            raise ValueError(f"ts must be a 3D tensor of shape (B, T, F), got {tuple(ts.shape)}")

        ts_emb = self.ts_encoder(ts)
        ev_source = ts if events is None else events
        if ev_source.dim() != 3:
            raise ValueError(
                f"events must be a 3D tensor of shape (B, N, F), got {tuple(ev_source.shape)}"
            )
        ev_emb = self.event_encoder(ev_source)
        fused, attn_weights = self.fusion(ts_emb, ev_emb, key_padding_mask=key_padding_mask)
        pooled = self.post_fusion(fused.mean(dim=1))
        outputs = self.heads(pooled)
        outputs["attention"] = attn_weights
        outputs["pooled_embedding"] = pooled
        return outputs

    def summary(self) -> dict[str, float | int]:
        return {
            "ts_input_dim": self.config.ts_input_dim,
            "event_input_dim": self.config.event_input_dim,
            "d_model": self.config.d_model,
            "num_heads": self.config.num_heads,
            "dropout": self.config.dropout,
        }

