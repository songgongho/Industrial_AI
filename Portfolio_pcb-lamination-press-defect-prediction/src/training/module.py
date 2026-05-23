"""Training module for the semiconductor PCB lamination thesis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryRecall

from src.models.pressfuse import CrossModalAttentionConfig, PressFuseModel


@dataclass(slots=True)
class TrainingConfig:
    max_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    ts_input_dim: int = 19
    event_input_dim: int = 19
    d_model: int = 128
    num_heads: int = 4
    dropout: float = 0.1
    head_hidden_dim: int = 128
    defect_classes: int = 2
    weight_bce: float = 1.0
    weight_ce: float = 0.5
    weight_mse: float = 0.25
    recall_target: float = 0.9


def false_alarm_rate_at_recall(y_true: Tensor, y_score: Tensor, target_recall: float = 0.9) -> Tensor:
    """Compute the minimum false alarm rate at or above a recall target."""

    if y_true.numel() == 0:
        return torch.tensor(0.0, device=y_score.device)

    thresholds = torch.linspace(0.0, 1.0, steps=201, device=y_score.device)
    best_far = torch.tensor(1.0, device=y_score.device)
    for threshold in thresholds:
        pred = (y_score >= threshold).to(torch.int64)
        tp = torch.logical_and(pred == 1, y_true == 1).sum().float()
        fp = torch.logical_and(pred == 1, y_true == 0).sum().float()
        tn = torch.logical_and(pred == 0, y_true == 0).sum().float()
        fn = torch.logical_and(pred == 0, y_true == 1).sum().float()
        recall = tp / torch.clamp(tp + fn, min=1.0)
        far = fp / torch.clamp(fp + tn, min=1.0)
        if recall >= target_recall:
            best_far = torch.minimum(best_far, far)
    return best_far


class PCBStackLightningModule(pl.LightningModule):
    """LightningModule for Press cycle classification and anomaly scoring."""

    def __init__(self, config: TrainingConfig | None = None) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.config = config or TrainingConfig()
        self.model = PressFuseModel(
            CrossModalAttentionConfig(
                ts_input_dim=self.config.ts_input_dim,
                event_input_dim=self.config.event_input_dim,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                dropout=self.config.dropout,
                head_hidden_dim=self.config.head_hidden_dim,
                defect_classes=self.config.defect_classes,
            )
        )
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.train_f1 = BinaryF1Score()
        self.train_auroc = BinaryAUROC()
        self.train_acc = BinaryAccuracy()
        self.train_recall = BinaryRecall()

        self.val_f1 = BinaryF1Score()
        self.val_auroc = BinaryAUROC()
        self.val_acc = BinaryAccuracy()
        self.val_recall = BinaryRecall()

        self._val_targets: list[Tensor] = []
        self._val_scores: list[Tensor] = []

    def forward(self, ts: Tensor, events: Tensor | None = None) -> dict[str, Tensor]:
        return self.model(ts, events)

    @staticmethod
    def _extract_batch(batch: Any) -> tuple[Tensor, Tensor, Tensor | None]:
        if isinstance(batch, dict):
            ts_value = batch.get("ts") if "ts" in batch else batch.get("x")
            y_value = batch.get("y") if "y" in batch else batch.get("label")
            if ts_value is None or y_value is None:
                raise KeyError("Dictionary batch must contain ts/x and y/label keys")
            ts = torch.as_tensor(ts_value, dtype=torch.float32)
            y = torch.as_tensor(y_value, dtype=torch.long)
            events = batch.get("events") if "events" in batch else None
            if events is not None:
                events = torch.as_tensor(events, dtype=torch.float32)
            return ts, y, events

        if not isinstance(batch, (tuple, list)):
            raise TypeError("Batch must be a mapping or tuple/list")

        if len(batch) == 2:
            ts, y = batch
            events = None
        elif len(batch) == 3:
            ts, events, y = batch
        else:
            raise ValueError("Batch tuple must have length 2 or 3")

        ts = torch.as_tensor(ts, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.long)
        if events is not None:
            events = torch.as_tensor(events, dtype=torch.float32)
        return ts, y, events

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        ts, y, events = self._extract_batch(batch)
        outputs = self(ts, events)
        binary_logits = outputs["binary_logits"].squeeze(-1)
        defect_logits = outputs["defect_logits"]
        anomaly_score = outputs["anomaly_score"].squeeze(-1)

        y_float = y.float()
        loss_bce = self.bce_loss(binary_logits, y_float)
        loss_ce = self.ce_loss(defect_logits, y)
        loss_mse = self.mse_loss(torch.sigmoid(anomaly_score), y_float)
        loss = (
            self.config.weight_bce * loss_bce
            + self.config.weight_ce * loss_ce
            + self.config.weight_mse * loss_mse
        )

        probs = torch.sigmoid(binary_logits)
        preds = (probs >= 0.5).to(torch.int64)

        if stage == "train":
            self.train_f1.update(preds, y)
            self.train_auroc.update(probs, y)
            self.train_acc.update(preds, y)
            self.train_recall.update(preds, y)
        else:
            self.val_f1.update(preds, y)
            self.val_auroc.update(probs, y)
            self.val_acc.update(preds, y)
            self.val_recall.update(preds, y)
            self._val_targets.append(y.detach())
            self._val_scores.append(probs.detach())
        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage == "train"),
            on_step=(stage == "train"),
            on_epoch=True,
        )
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_end(self) -> None:
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=False)
        self.log("train_recall", self.train_recall.compute(), prog_bar=False)
        self.train_f1.reset()
        self.train_auroc.reset()
        self.train_acc.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self) -> None:
        if self._val_targets and self._val_scores:
            targets = torch.cat(self._val_targets)
            scores = torch.cat(self._val_scores)
            far = false_alarm_rate_at_recall(targets, scores, self.config.recall_target)
            self.log(f"val_far_at_recall_{self.config.recall_target:.1f}", far, prog_bar=True)
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=False)
        self.log("val_recall", self.val_recall.compute(), prog_bar=False)
        self._val_targets.clear()
        self._val_scores.clear()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_acc.reset()
        self.val_recall.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

