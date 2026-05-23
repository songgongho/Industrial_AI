"""Press data loading utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.data.schema import DatasetSplitSpec, PressFeatureSpec


@dataclass(frozen=True, slots=True)
class PressSample:
    cycle_id: int
    panel_id: int
    x: np.ndarray
    y: int


def _read_tabular(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def resample_cycle(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    *,
    sort_column: str = "t_ms",
    target_length: int = 1024,
) -> np.ndarray:
    """Resample one cycle frame to a fixed-length feature matrix."""

    if frame.empty:
        raise ValueError("Cannot resample an empty frame")
    ordered = frame.sort_values(sort_column).reset_index(drop=True)
    values = ordered.loc[:, feature_columns].to_numpy(dtype=float)
    if len(ordered) == 1:
        return np.repeat(values, target_length, axis=0)

    source_x = ordered[sort_column].to_numpy(dtype=float)
    if np.all(source_x == source_x[0]):
        source_x = np.linspace(0.0, 1.0, len(source_x), dtype=float)
    else:
        source_x = (source_x - source_x.min()) / (source_x.max() - source_x.min())

    target_x = np.linspace(0.0, 1.0, target_length, dtype=float)
    resampled = np.empty((target_length, values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        resampled[:, idx] = np.interp(target_x, source_x, values[:, idx])
    return resampled


def _label_lookup(
    labels: pd.DataFrame | None, label_column: str, panel_column: str
) -> dict[Any, int]:
    if labels is None or label_column not in labels.columns:
        return {}
    if panel_column not in labels.columns:
        raise KeyError(f"labels table must contain '{panel_column}'")
    grouped = labels[[panel_column, label_column]].dropna(subset=[panel_column]).copy()
    grouped[label_column] = grouped[label_column].astype(int)
    return dict(zip(grouped[panel_column], grouped[label_column], strict=False))


def load_press_samples(
    data_path: str | Path,
    *,
    labels_path: str | Path | None = None,
    spec: PressFeatureSpec | None = None,
    target_length: int = 1024,
    mapping_path: str | Path | None = None,
) -> list[PressSample]:
    """Load Press cycles from a tabular file and return fixed-length samples."""

    spec = spec or PressFeatureSpec()
    df = _read_tabular(data_path)
    labels_df = _read_tabular(labels_path) if labels_path is not None else None
    lookup = _label_lookup(labels_df, spec.label_column, spec.panel_column)

    # Optional mapping file to resolve cycle_id -> panel_id / lot_id
    mapping_df = None
    mapping_lookup: dict[Any, Any] = {}
    if mapping_path is not None:
        mapping_df = _read_tabular(mapping_path)
        # Expect columns: cycle_id, panel_id, lot_id (panel_id at least)
        if "cycle_id" in mapping_df.columns and "panel_id" in mapping_df.columns:
            mapping_lookup = dict(zip(mapping_df["cycle_id"], mapping_df["panel_id"]))

    required = {
        spec.group_column,
        spec.panel_column,
        spec.sort_column,
        *spec.feature_columns,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    samples: list[PressSample] = []
    for cycle_id, group in df.groupby(spec.group_column, sort=False):
        # Prefer mapping lookup (mapping file), then group panel column, then labels lookup
        if mapping_lookup and cycle_id in mapping_lookup:
            panel_id = int(mapping_lookup[cycle_id])
        else:
            panel_id = int(group[spec.panel_column].iloc[0])

        if spec.label_column in group.columns:
            label = int(group[spec.label_column].iloc[0])
        else:
            label = int(lookup.get(panel_id, 0))
        x = resample_cycle(
            group,
            spec.feature_columns,
            sort_column=spec.sort_column,
            target_length=target_length,
        )
        samples.append(
            PressSample(cycle_id=int(cycle_id), panel_id=panel_id, x=x, y=label)
        )
    return samples


class PressDataset(Dataset[tuple[np.ndarray, int]]):
    """PyTorch dataset wrapper around fixed-length Press samples."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        labels_path: str | Path | None = None,
        spec: PressFeatureSpec | None = None,
        target_length: int = 1024,
        mapping_path: str | Path | None = None,
    ) -> None:
        self.samples = load_press_samples(
            data_path,
            labels_path=labels_path,
            spec=spec,
            target_length=target_length,
            mapping_path=mapping_path,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        sample = self.samples[index]
        return sample.x, sample.y


def split_by_group(
    groups: Iterable[Any],
    split_spec: DatasetSplitSpec | None = None,
    *,
    seed: int = 42,
) -> dict[str, list[Any]]:
    """Split group identifiers into train/val/test buckets without leakage."""

    split_spec = split_spec or DatasetSplitSpec()
    train_ratio, val_ratio, test_ratio = split_spec.validated_ratios()
    series = pd.Series(list(groups))
    unique_groups = pd.Index(series.dropna().unique())
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.to_numpy(copy=True)
    rng.shuffle(shuffled)

    n_groups = len(shuffled)
    n_train = int(round(n_groups * train_ratio))
    n_val = int(round(n_groups * val_ratio))
    n_train = min(n_train, n_groups)
    n_val = min(n_val, max(n_groups - n_train, 0))
    n_test = n_groups - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n_groups - n_train

    train_groups = shuffled[:n_train].tolist()
    val_groups = shuffled[n_train : n_train + n_val].tolist()
    test_groups = shuffled[n_train + n_val : n_train + n_val + n_test].tolist()

    return {"train": train_groups, "val": val_groups, "test": test_groups}


def load_secom(
    data_dir: str | Path, target_length: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    """Load SECOM dataset from secom.data and secom_labels.data files.

    SECOM is a public semiconductor manufacturing dataset.
    - secom.data: 1500 samples × 590 features (tab-separated, no header)
    - secom_labels.data: 1500 labels (0=pass, 1=fail, -1=unannotated)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (X, y) where X is (N, target_length, 1) and y is (N,).
        Excludes unannotated samples (label == -1).
    """
    data_dir = Path(data_dir)

    # Load features
    x_path = data_dir / "secom.data"
    if not x_path.exists():
        raise FileNotFoundError(f"Feature file not found: {x_path}")
    # SECOM data is tab-separated with no header
    x_raw = pd.read_csv(x_path, sep=r"\s+", header=None).to_numpy(dtype=float)

    # Load labels (first column is label, second column is timestamp)
    y_path = data_dir / "secom_labels.data"
    if not y_path.exists():
        raise FileNotFoundError(f"Label file not found: {y_path}")
    # Parse label and timestamp columns
    labels_df = pd.read_csv(y_path, sep=r"\s+", header=None, quotechar='"')
    y_raw = labels_df.iloc[:, 0].to_numpy(dtype=int)

    # Filter out unannotated (-1) samples
    valid_mask = y_raw != -1
    x_valid = x_raw[valid_mask]
    y_valid = y_raw[valid_mask]

    # Resample to target_length (simple downsampling or interpolation)
    if x_valid.shape[1] > target_length:
        # Simple column selection (preserves first target_length features)
        x_resampled = x_valid[:, :target_length]
    else:
        # Pad with zeros if needed
        padded = np.zeros((x_valid.shape[0], target_length))
        padded[:, : x_valid.shape[1]] = x_valid
        x_resampled = padded

    # Handle NaN values (forward-fill or replace with 0)
    x_resampled = np.nan_to_num(x_resampled, nan=0.0)

    # Reshape to (N, target_length, 1) for 1D-CNN/1D-Transformer compatibility
    x_final = x_resampled[:, :, np.newaxis]

    return x_final, y_valid
