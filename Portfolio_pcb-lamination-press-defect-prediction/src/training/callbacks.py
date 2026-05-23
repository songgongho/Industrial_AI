"""Training callback skeletons."""

from __future__ import annotations


def callback_names() -> list[str]:
    return ["early_stopping", "model_checkpoint", "lr_monitor"]

