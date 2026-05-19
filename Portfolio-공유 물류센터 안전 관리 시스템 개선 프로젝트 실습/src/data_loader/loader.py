"""Load raw CSV and apply recommended headers.

This is a scaffold file for the portfolio project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd


RECOMMENDED_MAPPING: Dict[int, str] = {
    0: "rec_id",
    1: "device_id",
    2: "timestamp",
    3: "phase_a_current_a",
    4: "phase_b_current_a",
    5: "phase_c_current_a",
    6: "phase_a_voltage_v",
    7: "phase_b_voltage_v",
    8: "phase_c_voltage_v",
    9: "voltage_avg_v",
    10: "p_a_w",
    11: "p_b_w",
    12: "p_c_w",
    13: "power_factor",
    14: "angle_deg",
    15: "frequency_hz",
    16: "active_power_w",
    17: "reactive_power_var",
    18: "apparent_power_va",
    19: "energy_import_wh",
    20: "energy_export_wh",
    21: "energy_total_wh",
    22: "status_flag",
    23: "temp_c",
    24: "humidity_pct",
}


def map_headers(df: pd.DataFrame, mapping: Dict[int, str] = RECOMMENDED_MAPPING) -> pd.DataFrame:
    """Rename raw positional columns into domain-friendly headers."""
    return df.rename(columns={idx: name for idx, name in mapping.items() if idx in df.columns})


def load_with_headers(input_path: str | Path) -> pd.DataFrame:
    """Load raw CSV and attach recommended headers."""
    df = pd.read_csv(input_path, header=None, na_values=["\\N", "\\N.1"])
    return map_headers(df)

