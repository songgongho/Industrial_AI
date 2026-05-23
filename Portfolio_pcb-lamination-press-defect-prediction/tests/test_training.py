import torch
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl

from src.training.module import PCBStackLightningModule, TrainingConfig


def test_press_fuse_forward_shapes() -> None:
    config = TrainingConfig(ts_input_dim=19, event_input_dim=19, d_model=32, head_hidden_dim=16)
    module = PCBStackLightningModule(config)
    x = torch.randn(2, 12, 19)
    outputs = module(x)
    assert outputs["binary_logits"].shape == (2, 1)
    assert outputs["defect_logits"].shape == (2, 2)
    assert outputs["anomaly_score"].shape == (2, 1)
    assert outputs["attention"].dim() == 4


def test_lightning_module_fast_dev_run() -> None:
    config = TrainingConfig(ts_input_dim=19, event_input_dim=19, d_model=32, head_hidden_dim=16, batch_size=2)
    module = PCBStackLightningModule(config)
    x = torch.randn(4, 10, 19)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    trainer = pl.Trainer(
        fast_dev_run=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(module, loader, loader)

