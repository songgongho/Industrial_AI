"""Baseline model scaffold."""

from __future__ import annotations

import pandas as pd


def train_baseline(df: pd.DataFrame, label_col: str = "state_label"):
    """Train a simple baseline model on engineered features."""
    raise NotImplementedError

