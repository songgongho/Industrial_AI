"""Synthetic Press cycle generator for early validation.

Generates synthetic Press cycles with 6 P013 anomaly types:
  - P013-001: temp_offset (온도 이상)
  - P013-002: pressure_drop (압력 이상)
  - P013-003: vacuum_leak (진공 이상)
  - P013-004: equipment_fault (설비 이상)
  - P013-005: power_loss (순간정전 / 결측)
  - P013-006: program_mismatch (Program 오적용)

Multi-anomaly patterns (현실적인 결함 조합):
  - Vacuum+Pressure: 진공 누수로 기압 저하 동반
  - Pressure+Temp: 가압 시스템 부분 고장으로 온도 저하
  - Equipment+All: 설비 전체 고장으로 모든 변수에 영향
  - Power+Missing: 순간정전으로 일시적 결측
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

AnomalyType = Literal[
    "temp_offset",
    "pressure_drop",
    "vacuum_leak",
    "equipment_fault",
    "power_loss",
    "program_mismatch",
]

MultiAnomalyType = Literal[
    "vacuum_pressure_cascade",  # 진공 누수 → 기압 저하
    "pressure_temp_coupling",    # 기압 이상 → 온도 저하
    "equipment_full_fault",      # 설비 고장 → 모든 변수 영향
    "power_vacuum_loss",         # 순간정전 → 결측 + 진공 불안정
]


@dataclass(frozen=True)
class PressCycleSpec:
    vacuum_s: int = 180
    hot_press_s: int = 9960
    cooling_s: int = 3000
    release_s: int = 60

    @property
    def total_s(self) -> int:
        return self.vacuum_s + self.hot_press_s + self.cooling_s + self.release_s


def _logistic(x: np.ndarray, x0: float, k: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def generate_press_cycle(
    cycle_id: int,
    panel_id: int,
    *,
    spec: PressCycleSpec | None = None,
    n_points: int = 1024,
    anomaly_prob: float = 0.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, int, str | None]:
    """Generate one synthetic Press cycle and its binary label.

    Parameters
    ----------
    cycle_id : int
        Cycle identifier.
    panel_id : int
        Panel identifier.
    spec : PressCycleSpec | None
        Press cycle time specification. Defaults to standard.
    n_points : int
        Number of time points in the cycle.
    anomaly_prob : float
        Probability of injecting one of 6 P013 anomalies.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, int, str | None]
        (DataFrame with cycle data, label (0/1), anomaly_type or None).
    """

    rng = np.random.default_rng(seed)
    spec = spec or PressCycleSpec()

    t_s = np.linspace(0.0, spec.total_s, n_points, dtype=float)
    t_ms = np.round(t_s * 1000.0).astype(int)

    vacuum_end = spec.vacuum_s
    hot_end = vacuum_end + spec.hot_press_s
    cooling_end = hot_end + spec.cooling_s

    pressure_sv = np.piecewise(
        t_s,
        [
            t_s <= vacuum_end,
            (t_s > vacuum_end) & (t_s <= hot_end),
            (t_s > hot_end) & (t_s <= cooling_end),
            t_s > cooling_end,
        ],
        [0.05, 18.0, 8.0, 0.0],
    ).astype(float)
    pressure_pv = pressure_sv + rng.normal(0.0, 0.35, size=n_points)

    hot_ramp = _logistic(t_s, x0=vacuum_end + spec.hot_press_s * 0.08, k=0.01)
    hot_decay = _logistic(t_s, x0=cooling_end - spec.cooling_s * 0.15, k=0.008)
    temp_sv = 25.0 + 235.0 * hot_ramp * (1.0 - 0.7 * hot_decay)
    temp_pv = temp_sv + rng.normal(0.0, 1.2, size=n_points)

    vacuum_sv = np.where(
        t_s <= vacuum_end, 0.98, np.where(t_s <= cooling_end, 0.12, 0.0)
    )
    vacuum_pv = np.clip(vacuum_sv + rng.normal(0.0, 0.02, size=n_points), 0.0, 1.0)

    anomaly_type: str | None = None
    label = 0
    if rng.random() < anomaly_prob:
        label = 1
        anomaly_type = rng.choice(
            [
                "temp_offset",
                "pressure_drop",
                "vacuum_leak",
                "equipment_fault",
                "power_loss",
                "program_mismatch",
            ]
        )

        if anomaly_type == "temp_offset":
            # P013-001: Temperature spike/offset
            temp_pv = temp_pv + 18.0

        elif anomaly_type == "pressure_drop":
            # P013-002: Pressure drop during hot press
            pressure_pv = pressure_pv - 3.5 * (t_s > vacuum_end)

        elif anomaly_type == "vacuum_leak":
            # P013-003: Vacuum loss
            vacuum_pv = np.clip(vacuum_pv + 0.18, 0.0, 1.0)

        elif anomaly_type == "equipment_fault":
            # P013-004: Equipment fault - abrupt pressure/temp drop
            fault_start_idx = max(n_points // 2, 100)
            pressure_pv[fault_start_idx:] = pressure_pv[fault_start_idx:] - 5.0
            temp_pv[fault_start_idx:] = temp_pv[fault_start_idx:] - 12.0

        elif anomaly_type == "power_loss":
            # P013-005: Power loss - missing values (NaN) in middle of cycle
            outage_start = int(n_points * 0.4)
            outage_end = int(n_points * 0.55)
            pressure_pv[outage_start:outage_end] = np.nan
            temp_pv[outage_start:outage_end] = np.nan
            vacuum_pv[outage_start:outage_end] = np.nan

        elif anomaly_type == "program_mismatch":
            # P013-006: Program mismatch - wrong SV profile (e.g., low pressure setting)
            pressure_sv = pressure_sv * 0.65

    frame = pd.DataFrame(
        {
            "cycle_id": cycle_id,
            "panel_id": panel_id,
            "t_ms": t_ms,
            **{f"POINT{i}": pressure_pv * (0.95 + 0.01 * i) for i in range(1, 13)},
            "HPPRESS_SV": pressure_sv,
            "HPPRESS_PV": pressure_pv,
            "FHPPRESS_SV": pressure_sv * 0.98,
            "FHPPRESS_PV": pressure_pv * 0.98,
            "HPTEMP_SV": temp_sv,
            "HPTEMP_PV": temp_pv,
            "VACUUM": vacuum_pv,
            "label": label,
            "anomaly_type": anomaly_type,
        }
    )
    return frame, label, anomaly_type


def apply_multi_anomaly(
    df: pd.DataFrame,
    t_s: np.ndarray,
    pressure_sv: np.ndarray,
    pressure_pv: np.ndarray,
    temp_pv: np.ndarray,
    vacuum_pv: np.ndarray,
    n_points: int,
    multi_type: MultiAnomalyType,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Apply realistic multi-anomaly patterns with causal coupling.

    Parameters
    ----------
    df : pd.DataFrame
        Reference dataframe (unused in this version, but kept for API consistency).
    t_s : np.ndarray
        Time array in seconds.
    pressure_sv, pressure_pv, temp_pv, vacuum_pv : np.ndarray
        Current signal arrays.
    n_points : int
        Number of points.
    multi_type : MultiAnomalyType
        Type of multi-anomaly pattern.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
        Modified (pressure_pv, temp_pv, vacuum_pv, pressure_sv, multi_type_name)
    """
    if multi_type == "vacuum_pressure_cascade":
        # Step 1: Vacuum leak (P013-003)
        vacuum_pv = np.clip(vacuum_pv + 0.22, 0.0, 1.0)  # Larger leak than single anomaly

        # Step 2: Pressure drops due to vacuum loss (P013-002 triggered by P013-003)
        pressure_pv = pressure_pv - 4.2 * (t_s > 180.0)

    elif multi_type == "pressure_temp_coupling":
        # Step 1: Pressure drop (P013-002)
        pressure_pv = pressure_pv - 3.8 * (t_s > 180.0)

        # Step 2: Temperature cannot stabilize due to pressure (P013-001 consequence)
        temp_delta = -15.0 * np.maximum(0.0, 1.0 - (pressure_pv + 3.8) / 18.0)
        temp_pv = temp_pv + temp_delta

    elif multi_type == "equipment_full_fault":
        # Step 1: Equipment fault (P013-004) triggers widespread failure
        fault_start_idx = max(n_points // 2, 100)

        # All signals degrade
        pressure_pv[fault_start_idx:] = pressure_pv[fault_start_idx:] - 5.5
        temp_pv[fault_start_idx:] = temp_pv[fault_start_idx:] - 14.0
        vacuum_pv[fault_start_idx:] = np.clip(
            vacuum_pv[fault_start_idx:] + 0.25, 0.0, 1.0
        )

    elif multi_type == "power_vacuum_loss":
        # Step 1: Power loss (P013-005) creates data gap
        outage_start = int(n_points * 0.4)
        outage_end = int(n_points * 0.55)
        pressure_pv[outage_start:outage_end] = np.nan
        temp_pv[outage_start:outage_end] = np.nan

        # Step 2: After recovery, vacuum cannot stabilize (P013-003 consequence)
        if outage_end < n_points:
            vacuum_pv[outage_end:] = np.clip(
                vacuum_pv[outage_end:] + 0.20, 0.0, 1.0
            )

    return pressure_pv, temp_pv, vacuum_pv, pressure_sv, multi_type


def generate_press_cycle_multi(
    cycle_id: int,
    panel_id: int,
    *,
    spec: PressCycleSpec | None = None,
    n_points: int = 1024,
    anomaly_prob: float = 0.0,
    multi_anomaly_prob: float = 0.0,
    seed: int | None = None,
) -> tuple[pd.DataFrame, int, str | None]:
    """Generate synthetic Press cycle with single or multi-anomalies (P013).

    This extends generate_press_cycle with realistic multi-anomaly patterns
    that model causal cascades observed in actual Press faults.

    Parameters
    ----------
    cycle_id : int
        Cycle identifier.
    panel_id : int
        Panel identifier.
    spec : PressCycleSpec | None
        Press cycle time specification.
    n_points : int
        Number of time points.
    anomaly_prob : float
        Probability of injecting a single anomaly (0, 1).
    multi_anomaly_prob : float
        Probability of injecting multi-anomalies instead (0, 1).
        If both anomaly_prob and multi_anomaly_prob trigger, multi takes precedence.
    seed : int | None
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, int, str | None]
        (DataFrame with cycle data, label (0/1), anomaly_type or None).
    """
    rng = np.random.default_rng(seed)
    spec = spec or PressCycleSpec()

    t_s = np.linspace(0.0, spec.total_s, n_points, dtype=float)
    t_ms = np.round(t_s * 1000.0).astype(int)

    vacuum_end = spec.vacuum_s
    hot_end = vacuum_end + spec.hot_press_s
    cooling_end = hot_end + spec.cooling_s

    pressure_sv = np.piecewise(
        t_s,
        [
            t_s <= vacuum_end,
            (t_s > vacuum_end) & (t_s <= hot_end),
            (t_s > hot_end) & (t_s <= cooling_end),
            t_s > cooling_end,
        ],
        [0.05, 18.0, 8.0, 0.0],
    ).astype(float)
    pressure_pv = pressure_sv + rng.normal(0.0, 0.35, size=n_points)

    hot_ramp = _logistic(t_s, x0=vacuum_end + spec.hot_press_s * 0.08, k=0.01)
    hot_decay = _logistic(t_s, x0=cooling_end - spec.cooling_s * 0.15, k=0.008)
    temp_sv = 25.0 + 235.0 * hot_ramp * (1.0 - 0.7 * hot_decay)
    temp_pv = temp_sv + rng.normal(0.0, 1.2, size=n_points)

    vacuum_sv = np.where(
        t_s <= vacuum_end, 0.98, np.where(t_s <= cooling_end, 0.12, 0.0)
    )
    vacuum_pv = np.clip(vacuum_sv + rng.normal(0.0, 0.02, size=n_points), 0.0, 1.0)

    anomaly_type: str | None = None
    label = 0

    # Multi-anomaly takes precedence over single anomaly
    if rng.random() < multi_anomaly_prob:
        label = 1
        multi_type = rng.choice(
            [
                "vacuum_pressure_cascade",
                "pressure_temp_coupling",
                "equipment_full_fault",
                "power_vacuum_loss",
            ]
        )
        pressure_pv, temp_pv, vacuum_pv, pressure_sv, anomaly_type = apply_multi_anomaly(
            pd.DataFrame(),
            t_s,
            pressure_sv,
            pressure_pv,
            temp_pv,
            vacuum_pv,
            n_points,
            multi_type,
            rng,
        )

    elif rng.random() < anomaly_prob:
        label = 1
        anomaly_type = rng.choice(
            [
                "temp_offset",
                "pressure_drop",
                "vacuum_leak",
                "equipment_fault",
                "power_loss",
                "program_mismatch",
            ]
        )

        if anomaly_type == "temp_offset":
            temp_pv = temp_pv + 18.0

        elif anomaly_type == "pressure_drop":
            pressure_pv = pressure_pv - 3.5 * (t_s > vacuum_end)

        elif anomaly_type == "vacuum_leak":
            vacuum_pv = np.clip(vacuum_pv + 0.18, 0.0, 1.0)

        elif anomaly_type == "equipment_fault":
            fault_start_idx = max(n_points // 2, 100)
            pressure_pv[fault_start_idx:] = pressure_pv[fault_start_idx:] - 5.0
            temp_pv[fault_start_idx:] = temp_pv[fault_start_idx:] - 12.0

        elif anomaly_type == "power_loss":
            outage_start = int(n_points * 0.4)
            outage_end = int(n_points * 0.55)
            pressure_pv[outage_start:outage_end] = np.nan
            temp_pv[outage_start:outage_end] = np.nan
            vacuum_pv[outage_start:outage_end] = np.nan

        elif anomaly_type == "program_mismatch":
            pressure_sv = pressure_sv * 0.65

    frame = pd.DataFrame(
        {
            "cycle_id": cycle_id,
            "panel_id": panel_id,
            "t_ms": t_ms,
            **{f"POINT{i}": pressure_pv * (0.95 + 0.01 * i) for i in range(1, 13)},
            "HPPRESS_SV": pressure_sv,
            "HPPRESS_PV": pressure_pv,
            "FHPPRESS_SV": pressure_sv * 0.98,
            "FHPPRESS_PV": pressure_pv * 0.98,
            "HPTEMP_SV": temp_sv,
            "HPTEMP_PV": temp_pv,
            "VACUUM": vacuum_pv,
            "label": label,
            "anomaly_type": anomaly_type,
        }
    )
    return frame, label, anomaly_type

