"""Pydantic schemas for semiconductor PCB lamination data tables and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PressCycleRow(BaseModel):
    """One row from a Press cycle table."""

    model_config = ConfigDict(extra="forbid")

    cycle_id: int
    panel_id: int
    t_ms: int
    label: int = Field(ge=0, le=1)
    anomaly_type: str | None = None

    POINT1: float
    POINT2: float
    POINT3: float
    POINT4: float
    POINT5: float
    POINT6: float
    POINT7: float
    POINT8: float
    POINT9: float
    POINT10: float
    POINT11: float
    POINT12: float
    HPPRESS_SV: float
    HPPRESS_PV: float
    FHPPRESS_SV: float
    FHPPRESS_PV: float
    HPTEMP_SV: float
    HPTEMP_PV: float
    VACUUM: float


class DatasetSplitSpec(BaseModel):
    """Group-aware split ratios."""

    model_config = ConfigDict(extra="forbid")

    train_ratio: float = Field(default=0.7, gt=0.0, lt=1.0)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    test_ratio: float | None = None

    def validated_ratios(self) -> tuple[float, float, float]:
        test_ratio = self.test_ratio
        if test_ratio is None:
            test_ratio = 1.0 - self.train_ratio - self.val_ratio
        total = self.train_ratio + self.val_ratio + test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        if test_ratio <= 0:
            raise ValueError("test_ratio must be positive")
        return self.train_ratio, self.val_ratio, test_ratio


@dataclass(frozen=True, slots=True)
class PressFeatureSpec:
    """Feature layout used by the Press dataset."""

    feature_columns: tuple[str, ...] = (
        "POINT1",
        "POINT2",
        "POINT3",
        "POINT4",
        "POINT5",
        "POINT6",
        "POINT7",
        "POINT8",
        "POINT9",
        "POINT10",
        "POINT11",
        "POINT12",
        "HPPRESS_SV",
        "HPPRESS_PV",
        "FHPPRESS_SV",
        "FHPPRESS_PV",
        "HPTEMP_SV",
        "HPTEMP_PV",
        "VACUUM",
    )
    group_column: str = "cycle_id"
    sort_column: str = "t_ms"
    label_column: str = "label"
    panel_column: str = "panel_id"


SplitName = Literal["train", "val", "test"]

