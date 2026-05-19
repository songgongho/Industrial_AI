"""Rule-based state labeling utilities for shared logistics / factory data.

The module is intentionally simple and explainable so it can be used as a
portfolio MVP and later as weak labels / pseudo-labels for supervised learning.

Supported labels (priority order):
    1. alert_state
    2. suspected_sensor_fault
    3. abnormal_pattern
    4. idle
    5. normal_operation
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Optional
import re
import warnings

import numpy as np
import pandas as pd


DEFAULT_CONFIG: Dict[str, object] = {
    "low_power_threshold": 1.0,
    "high_power_threshold": 5.0,
    "low_current_threshold": 0.5,
    "high_current_threshold": 2.0,
    "abnormal_temp_threshold": 60.0,
    "humidity_alert_threshold": 90.0,
    "gas_alert_threshold": 1.0,
    "smoke_alert_threshold": 1.0,
    "fire_alert_threshold": 1.0,
    "sensor_stuck_min_period": 5,
    "sensor_nan_streak_min_period": 3,
    "spike_power_delta": 1000.0,
    "spike_current_delta": 10.0,
    "night_hours": [0, 1, 2, 3, 4, 5],
    "column_candidates": {
        "timestamp": ["timestamp", "time", "datetime"],
        "hour": ["hour"],
        "device_id": ["device_id", "eqp_id", "sensor_id", "machine_id"],
        "zone_id": ["zone_id", "area_id", "cell_id"],
        "line_id": ["line_id", "production_line_id", "line"],
        "power": ["power_kw", "power_w", "active_power_w", "active_power", "P", "kW"],
        "current": ["current_a", "current", "I"],
        "voltage": ["voltage_v", "voltage", "V"],
        "temp": ["temp_c", "temperature", "temp"],
        "humidity": ["humidity", "humidity_pct", "rh", "relative_humidity"],
        "gas": ["gas", "gas_ppm", "gas_value"],
        "smoke": ["smoke", "smoke_flag"],
        "fire": ["fire", "fire_flag"],
        "alarm_flag": ["alarm_flag", "alarm", "alert_flag"],
        "error_code": ["error_code", "err_code", "fault_code"],
        "event_flag": ["event_flag", "event", "event_code"],
        "eqp_status": ["eqp_status", "status", "state", "status_flag"],
        "mode": ["mode", "operating_mode"],
        "run_flag": ["run_flag", "run", "is_running", "operating_flag"],
    },
}

# Fallback when the CSV is loaded with header=None and positional integer columns.
RAW_POSITION_FALLBACK: Dict[str, int] = {
    "timestamp": 2,
    "device_id": 1,
    "power": 16,
    "current": 3,
    "voltage": 6,
    "temp": 23,
    "humidity": 24,
    "alarm_flag": 22,
}


def _merged_config(config: Optional[Dict]) -> Dict[str, object]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        for key, value in config.items():
            if key == "column_candidates" and isinstance(value, dict):
                merged["column_candidates"] = {**merged["column_candidates"], **value}
            else:
                merged[key] = value
    return merged


def _normalize_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_column(df: pd.DataFrame, candidates: Iterable[object]) -> Optional[object]:
    """Find the first matching column from candidate names or positional integers."""
    columns = list(df.columns)
    normalized_map = defaultdict(list)
    for col in columns:
        normalized_map[_normalize_name(col)].append(col)

    for candidate in candidates:
        if candidate in columns:
            return candidate
        normalized = _normalize_name(candidate)
        if normalized in normalized_map:
            return normalized_map[normalized][0]
        try:
            idx = int(candidate)
        except (TypeError, ValueError):
            idx = None
        if idx is not None and idx in columns:
            return idx
    return None


def _series(df: pd.DataFrame, col: Optional[object]) -> Optional[pd.Series]:
    if col is None or col not in df.columns:
        return None
    return df[col]


def _numeric(df: pd.DataFrame, col: Optional[object]) -> Optional[pd.Series]:
    series = _series(df, col)
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce")


def _truthy(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).ne(0)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "y", "yes", "on", "run", "running", "alarm", "alert", "error"})


def _off_like_status(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"0", "off", "stop", "stopped", "standby", "idle", "sleep", "inactive"})


def _on_like_status(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "on", "run", "running", "active", "operate", "operating"})


def _has_any(mask_list: Iterable[Optional[pd.Series]], index: pd.Index) -> pd.Series:
    out = pd.Series(False, index=index)
    for mask in mask_list:
        if mask is not None:
            out = out | mask.fillna(False)
    return out


def infer_available_columns(df: pd.DataFrame) -> Dict[str, Optional[object]]:
    """Infer the best available column for each logical field used by rules.

    Returns a dict mapping logical field names to actual DataFrame columns
    (or ``None`` if not found). If several essential fields are missing, a
    warning is emitted so the user can inspect the schema.
    """
    cfg = DEFAULT_CONFIG["column_candidates"]
    available: Dict[str, Optional[object]] = {}
    missing = []

    for field, candidates in cfg.items():
        col = _find_column(df, candidates)
        if col is None and field in RAW_POSITION_FALLBACK and RAW_POSITION_FALLBACK[field] in df.columns:
            col = RAW_POSITION_FALLBACK[field]
        available[field] = col
        if col is None:
            missing.append(field)

    essential_missing = [f for f in ["timestamp", "power", "current", "eqp_status", "alarm_flag"] if available.get(f) is None]
    if essential_missing:
        warnings.warn(
            "infer_available_columns: 일부 핵심 컬럼을 찾지 못했습니다 -> " + ", ".join(essential_missing),
            RuntimeWarning,
            stacklevel=2,
        )

    return available


def _get_available(df: pd.DataFrame, config: Dict[str, object]) -> Dict[str, Optional[object]]:
    merged = _merged_config(config)
    mapping = infer_available_columns(df)
    # Allow user-provided column_candidates to influence lookup order.
    for field, candidates in merged.get("column_candidates", {}).items():
        if mapping.get(field) is None:
            col = _find_column(df, candidates)
            if col is not None:
                mapping[field] = col
    return mapping


def _ordered_frame(df: pd.DataFrame, available: Dict[str, Optional[object]]) -> pd.DataFrame:
    """Return a copy sorted by device/timestamp when possible.

    The returned DataFrame preserves original indices so masks can be reindexed
    back to the input order.
    """
    work = df.copy()
    sort_cols = []
    for key in ["device_id", "zone_id", "line_id"]:
        col = available.get(key)
        if col is not None:
            sort_cols.append(col)
            break
    ts_col = available.get("timestamp")
    if ts_col is not None:
        work["__parsed_timestamp__"] = pd.to_datetime(work[ts_col], errors="coerce")
        sort_cols.append("__parsed_timestamp__")
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort")
    return work


def detect_idle(df: pd.DataFrame, config: Dict) -> pd.Series:
    """Return a boolean mask for idle state.

    Rule idea:
    - low power and/or low current
    - device status is off/standby/idle
    - no alarm / error / event flags
    """
    cfg = _merged_config(config)
    available = _get_available(df, cfg)

    power = _numeric(df, available.get("power"))
    current = _numeric(df, available.get("current"))
    alarm = _truthy(_series(df, available.get("alarm_flag")))
    error = _truthy(_series(df, available.get("error_code")))
    event = _truthy(_series(df, available.get("event_flag")))
    status = _series(df, available.get("eqp_status"))
    mode = _series(df, available.get("mode"))
    run_flag = _truthy(_series(df, available.get("run_flag")))

    low_power = power.le(cfg["low_power_threshold"]) if power is not None else None
    low_current = current.le(cfg["low_current_threshold"]) if current is not None else None
    off_like = _has_any(
        [
            _off_like_status(status),
            _off_like_status(mode),
            (~run_flag) if run_flag is not None else None,
        ],
        df.index,
    )

    if low_power is not None and low_current is not None:
        base_idle = low_power & low_current
    elif low_power is not None:
        base_idle = low_power
    elif low_current is not None:
        base_idle = low_current
    else:
        base_idle = off_like

    if off_like is not None:
        base_idle = base_idle | off_like

    no_alert_signals = ~_has_any([alarm, error, event], df.index)
    return (base_idle & no_alert_signals).fillna(False)


def detect_alert_state(df: pd.DataFrame, config: Dict) -> pd.Series:
    """Return a boolean mask for alert state.

    Alert is the highest-priority state and is triggered by clear safety or
    operational alarms, for example:
    - alarm / event / error flags
    - fire / smoke / gas signals
    - severe temperature or humidity thresholds
    - high power during night hours with any alarm signal
    """
    cfg = _merged_config(config)
    available = _get_available(df, cfg)
    work = _ordered_frame(df, available)

    power = _numeric(work, available.get("power"))
    temp = _numeric(work, available.get("temp"))
    humidity = _numeric(work, available.get("humidity"))
    gas = _numeric(work, available.get("gas"))
    smoke = _truthy(_series(work, available.get("smoke")))
    fire = _truthy(_series(work, available.get("fire")))
    alarm = _truthy(_series(work, available.get("alarm_flag")))
    error = _truthy(_series(work, available.get("error_code")))
    event = _truthy(_series(work, available.get("event_flag")))
    status = _series(work, available.get("eqp_status"))
    run_flag = _truthy(_series(work, available.get("run_flag")))

    any_alarm = _has_any([alarm, error, event, smoke, fire], work.index)
    severe_env = pd.Series(False, index=work.index)
    if temp is not None:
        severe_env = severe_env | temp.ge(cfg["abnormal_temp_threshold"])
    if humidity is not None:
        severe_env = severe_env | humidity.ge(cfg["humidity_alert_threshold"])
    if gas is not None:
        severe_env = severe_env | gas.ge(cfg["gas_alert_threshold"])

    night_hours = set(cfg["night_hours"])
    if available.get("timestamp") is not None:
        parsed = pd.to_datetime(work[available["timestamp"]], errors="coerce")
        hour = parsed.dt.hour
        night = hour.isin(night_hours)
    else:
        night = pd.Series(False, index=work.index)

    high_power = power.ge(cfg["high_power_threshold"]) if power is not None else pd.Series(False, index=work.index)
    on_like = _has_any([
        _on_like_status(status),
        run_flag,
    ], work.index)

    night_high_power_alarm = night & high_power & any_alarm

    alert = any_alarm | severe_env | night_high_power_alarm
    return alert.reindex(df.index).fillna(False)


def detect_sensor_fault(df: pd.DataFrame, config: Dict) -> pd.Series:
    """Return a boolean mask for suspected sensor fault.

    Typical weak-label rules:
    - a sensor value stays identical for a long streak (stuck sensor)
    - a critical signal is missing repeatedly
    - impossible values appear (negative power/current, zero voltage while the
      equipment is clearly running)
    """
    cfg = _merged_config(config)
    available = _get_available(df, cfg)
    work = _ordered_frame(df, available)

    min_period = int(cfg["sensor_stuck_min_period"])
    nan_period = int(cfg["sensor_nan_streak_min_period"])

    power = _numeric(work, available.get("power"))
    current = _numeric(work, available.get("current"))
    voltage = _numeric(work, available.get("voltage"))
    temp = _numeric(work, available.get("temp"))
    humidity = _numeric(work, available.get("humidity"))
    gas = _numeric(work, available.get("gas"))
    alarm = _truthy(_series(work, available.get("alarm_flag")))
    status = _series(work, available.get("eqp_status"))
    run_flag = _truthy(_series(work, available.get("run_flag")))

    active_signal = _has_any(
        [
            power.ge(cfg["low_power_threshold"]) if power is not None else None,
            current.ge(cfg["low_current_threshold"]) if current is not None else None,
            _on_like_status(status),
            run_flag,
            alarm,
        ],
        work.index,
    )

    monitored_cols = [c for c in [power, current, voltage, temp, humidity, gas] if c is not None]
    sensor_fault = pd.Series(False, index=work.index)

    for series in monitored_cols:
        filled = series.copy()
        as_text = filled.astype("string")
        same_run = as_text.ne(as_text.shift()).cumsum()
        run_length = same_run.groupby(same_run).transform("size")
        stuck = run_length.ge(min_period)

        nan_run = filled.isna().astype(int)
        nan_group = nan_run.ne(nan_run.shift()).cumsum()
        nan_length = nan_group.groupby(nan_group).transform("size")
        repeated_nan = filled.isna() & nan_length.ge(nan_period)

        if series is power or series is current or series is voltage:
            impossible = series.lt(0) | (series.le(0) & active_signal)
        else:
            impossible = pd.Series(False, index=work.index)

        sensor_fault = sensor_fault | (stuck & active_signal) | repeated_nan | impossible.fillna(False)

    # Special case: if a running device has zero voltage while active signal is on.
    if voltage is not None:
        sensor_fault = sensor_fault | (voltage.le(0) & active_signal)

    return sensor_fault.reindex(df.index).fillna(False)


def detect_abnormal_pattern(df: pd.DataFrame, config: Dict) -> pd.Series:
    """Return a boolean mask for abnormal operating pattern.

    This catches non-alarm but clearly odd behavior, such as:
    - sudden spikes/drops in power or current
    - status mismatch (OFF with high power, ON with near-zero power)
    - unusually high temperature that is not yet an alert
    """
    cfg = _merged_config(config)
    available = _get_available(df, cfg)
    work = _ordered_frame(df, available)

    power = _numeric(work, available.get("power"))
    current = _numeric(work, available.get("current"))
    temp = _numeric(work, available.get("temp"))
    status = _series(work, available.get("eqp_status"))
    mode = _series(work, available.get("mode"))
    run_flag = _truthy(_series(work, available.get("run_flag")))
    alarm = _truthy(_series(work, available.get("alarm_flag")))

    if available.get("device_id") is not None:
        group_key = available["device_id"]
        group_obj = work.groupby(group_key, dropna=False, sort=False)
    else:
        group_obj = [(None, work)]

    abnormal = pd.Series(False, index=work.index)

    for _, group in group_obj:
        idx = group.index
        p = _numeric(group, available.get("power"))
        c = _numeric(group, available.get("current"))
        t = _numeric(group, available.get("temp"))
        st = _series(group, available.get("eqp_status"))
        md = _series(group, available.get("mode"))
        rf = _truthy(_series(group, available.get("run_flag")))

        if p is not None:
            p_delta = p.diff().abs()
            abnormal.loc[idx] = abnormal.loc[idx] | p_delta.ge(cfg["spike_power_delta"]).fillna(False)
        if c is not None:
            c_delta = c.diff().abs()
            abnormal.loc[idx] = abnormal.loc[idx] | c_delta.ge(cfg["spike_current_delta"]).fillna(False)

        off_like = _has_any([_off_like_status(st), _off_like_status(md), (~rf) if rf is not None else None], idx)
        on_like = _has_any([_on_like_status(st), _on_like_status(md), rf], idx)

        if p is not None:
            high_power = p.ge(cfg["high_power_threshold"])
            low_power = p.le(cfg["low_power_threshold"])
            abnormal.loc[idx] = abnormal.loc[idx] | (off_like & high_power) | (on_like & low_power)
        if c is not None:
            high_current = c.ge(cfg["high_current_threshold"])
            low_current = c.le(cfg["low_current_threshold"])
            abnormal.loc[idx] = abnormal.loc[idx] | (off_like & high_current) | (on_like & low_current)

        if t is not None:
            abnormal.loc[idx] = abnormal.loc[idx] | t.ge(cfg["abnormal_temp_threshold"] * 0.9)

    # If any alert-like signal is already present, let alert_state win later.
    abnormal = abnormal & ~alarm.fillna(False)
    return abnormal.reindex(df.index).fillna(False)


def build_state_label(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Attach a `state_label` column using the configured priority order."""
    out = df.copy()
    cfg = _merged_config(config)

    alert_mask = detect_alert_state(out, cfg)
    sensor_fault_mask = detect_sensor_fault(out, cfg)
    abnormal_mask = detect_abnormal_pattern(out, cfg)
    idle_mask = detect_idle(out, cfg)

    labels = pd.Series("normal_operation", index=out.index, dtype="object")
    labels.loc[idle_mask] = "idle"
    labels.loc[abnormal_mask] = "abnormal_pattern"
    labels.loc[sensor_fault_mask] = "suspected_sensor_fault"
    labels.loc[alert_mask] = "alert_state"

    out["state_label"] = labels
    return out


def summarize_label_distribution(df: pd.DataFrame, label_col: str = "state_label") -> pd.DataFrame:
    """Summarize label counts and proportions."""
    if label_col not in df.columns:
        raise KeyError(f"`{label_col}` column not found")

    counts = df[label_col].value_counts(dropna=False).rename("count")
    summary = counts.to_frame()
    summary["ratio"] = summary["count"] / len(df) if len(df) else 0.0
    summary.index.name = label_col
    return summary.reset_index()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python labeling.py <csv_path>")
        sys.exit(0)

    path = sys.argv[1]
    frame = pd.read_csv(path, header=None, na_values=["\\N", "\\N.1"])
    labeled = build_state_label(frame, DEFAULT_CONFIG)
    print(labeled.head())
    print(summarize_label_distribution(labeled))

