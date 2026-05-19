"""Feature engineering scaffold."""

from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling / lag / time-of-day features for modeling."""
    return df.copy()

