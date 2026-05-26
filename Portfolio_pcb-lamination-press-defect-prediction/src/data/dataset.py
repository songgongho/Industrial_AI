from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.synthpress import generate_press_cycle_multi, generate_press_cycle


class SyntheticPressDataset(Dataset):
    """In-memory synthetic dataset for Press cycles.

    Produces tensors of shape (T, D) and labels (0/1).
    """

    def __init__(
        self,
        n_cycles: int = 512,
        n_points: int = 192,
        anomaly_prob: float = 0.15,
        seed: Optional[int] = 42,
        features: Optional[List[str]] = None,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.n_cycles = n_cycles
        self.n_points = n_points
        self._raw: List[pd.DataFrame] = []
        self.labels: List[int] = []
        self.cycle_ids: List[int] = []

        # generate cycles
        for i in range(n_cycles):
            df, label, _ = generate_press_cycle_multi(
                cycle_id=i,
                panel_id=1000 + i % 10,
                n_points=n_points,
                anomaly_prob=anomaly_prob,
                multi_anomaly_prob=0.0,
                seed=int(rng.integers(0, 2 ** 31 - 1)),
            )
            self._raw.append(df)
            self.labels.append(int(label))
            self.cycle_ids.append(int(df["cycle_id"].iloc[0]))

        # feature selection: build feature list to produce D features
        if features is None:
            # default feature list (19 features): POINT1..POINT12 + HPPRESS_SV + HPPRESS_PV + FHPPRESS_PV + HPTEMP_PV + VACUUM + t_ms_norm + panel_id_norm
            base = [f"POINT{i}" for i in range(1, 13)]
            extra = [
                "HPPRESS_SV",
                "HPPRESS_PV",
                "FHPPRESS_PV",
                "HPTEMP_PV",
                "VACUUM",
            ]
            features = base + extra
            # add placeholders to reach 19
            # we'll add `t_ms` and `panel_id` normalized later
        self.features = features

    def __len__(self) -> int:
        return self.n_cycles

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        df = self._raw[idx]
        # build feature matrix (T, D)
        T = len(df)
        # base features
        arr_list = []
        for f in self.features:
            if f in df.columns:
                col = df[f].to_numpy(dtype=float)
                # replace NaN
                col = np.nan_to_num(col, nan=np.nanmean(col))
            else:
                col = np.zeros(T, dtype=float)
            arr_list.append(col.reshape(T, 1))

        # add t_ms normalized
        t_ms = df["t_ms"].to_numpy(dtype=float)
        t_ms = (t_ms - t_ms.min()) / max(1.0, (t_ms.max() - t_ms.min()))
        arr_list.append(t_ms.reshape(T, 1))

        # add panel_id normalized
        panel_id = df["panel_id"].to_numpy(dtype=float)
        panel_norm = (panel_id - panel_id.min()) / max(1.0, (panel_id.max() - panel_id.min()))
        arr_list.append(panel_norm.reshape(T, 1))

        X = np.concatenate(arr_list, axis=1)  # (T, D)
        # ensure float32
        X = X.astype(np.float32)
        y = int(self.labels[idx])
        cycle_id = int(self.cycle_ids[idx])
        return torch.from_numpy(X), y, cycle_id


def collate_fn(batch: List[Tuple[torch.Tensor, int, int]]):
    # batch: list of (T,D), y, cycle_id
    Xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    ids = [b[2] for b in batch]
    # assume all same T
    X = torch.stack(Xs, dim=0)  # (B, T, D)
    y = torch.tensor(ys, dtype=torch.float32).unsqueeze(1)
    return X, y, ids

