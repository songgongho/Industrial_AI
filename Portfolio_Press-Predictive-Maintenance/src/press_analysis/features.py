from __future__ import annotations

import re

import pandas as pd

TIME_KEYS = ("time", "date", "timestamp", "created", "dt")
MACHINE_KEYS = ("machine", "line", "press", "equip", "tool", "die", "mc")
RESULT_KEYS = ("result", "judge", "status", "ok_ng", "pass_fail", "inspection")
DEFECT_KEYS = ("defect", "ng_code", "error", "fault", "reject", "alarm", "reason")

OK_TOKENS = {"OK", "PASS", "GOOD", "NORMAL", "Y", "0"}
NG_TOKENS = {"NG", "NOK", "FAIL", "DEFECT", "REJECT", "BAD", "X", "1"}

SENSOR_GROUPS: tuple[tuple[str, str], ...] = (
    ("PRESS", "PRESS"),
    ("VACUUM", "VACUUM"),
    ("TEMP", "TEMP"),
    ("PT", "PT"),
    ("POINT", "POINT"),
)


def _name_score(col_name: str, keys: tuple[str, ...]) -> int:
    col_low = str(col_name).lower()
    return sum(1 for key in keys if key in col_low)


def _infer_time_column(df: pd.DataFrame) -> str | None:
    best_col = None
    best_score = -1.0

    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
        parse_ratio = float(parsed.notna().mean())
        if parse_ratio < 0.6:
            continue

        score = _name_score(str(col), TIME_KEYS) * 2 + parse_ratio
        if score > best_score:
            best_col = str(col)
            best_score = score

    return best_col


def _infer_text_column(
    df: pd.DataFrame,
    keys: tuple[str, ...],
    max_unique_ratio: float,
    allow_single_value: bool = False,
    require_name_match: bool = True,
) -> str | None:
    best_col = None
    best_score = -1.0
    nrows = max(len(df), 1)

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            continue

        non_na = series.dropna().astype(str).str.strip()
        non_empty = non_na[non_na != ""]
        if non_empty.empty:
            continue

        nunique = int(non_empty.nunique())
        unique_ratio = nunique / nrows
        if unique_ratio > max_unique_ratio:
            continue
        if not allow_single_value and nunique < 2:
            continue

        name_score = _name_score(str(col), keys)
        if require_name_match and name_score == 0:
            continue

        score = name_score * 2
        score += min(1.0, 20.0 / max(nunique, 1))

        if score > best_score:
            best_col = str(col)
            best_score = score

    return best_col


def _to_norm_text(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().fillna("")


def _build_is_defect(df: pd.DataFrame, result_col: str | None, defect_col: str | None) -> pd.Series:
    is_defect = pd.Series(False, index=df.index)

    if result_col:
        rs = _to_norm_text(df[result_col])
        is_defect = is_defect | rs.isin(NG_TOKENS)
        is_defect = is_defect | rs.str.contains(r"NG|FAIL|DEFECT|REJECT|BAD|ERROR", regex=True, na=False)

    if defect_col:
        ds = _to_norm_text(df[defect_col])
        has_value = ds != ""
        not_ok = ~ds.isin(OK_TOKENS)
        is_defect = is_defect | (has_value & not_ok)

    return is_defect


def _normalize_label(text: str) -> str:
    # Keep labels compact for reporting and remove noisy punctuation.
    cleaned = re.sub(r"\s+", "_", text.strip().upper())
    cleaned = re.sub(r"[^A-Z0-9_\-]", "", cleaned)
    return cleaned or "UNKNOWN"


def _build_defect_type(
    df: pd.DataFrame,
    is_defect: pd.Series,
    result_col: str | None,
    defect_col: str | None,
) -> pd.Series:
    if defect_col:
        defect_type = df[defect_col].astype(str).map(_normalize_label)
    elif result_col:
        defect_type = df[result_col].astype(str).map(_normalize_label)
    else:
        defect_type = pd.Series("UNKNOWN", index=df.index)

    return defect_type.where(is_defect, "NORMAL")


def _extract_numeric_anomaly_score(df: pd.DataFrame) -> pd.Series:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.Series(0.0, index=df.index)

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = (q3 - q1).replace(0, pd.NA)

    z = ((numeric - q1) / iqr).abs()
    score = z.fillna(0).sum(axis=1)
    return score


def _extract_numeric_anomaly_components(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return pd.Series(0.0, index=df.index), pd.Series("UNKNOWN", index=df.index)

    median = numeric.median()
    mad = (numeric - median).abs().median().replace(0, pd.NA)
    robust_z = ((numeric - median).abs() / mad).fillna(0)

    score = robust_z.sum(axis=1)
    primary_col = robust_z.idxmax(axis=1)
    return score, primary_col


def _map_feature_to_anomaly_type(col_name: str) -> str:
    col_up = str(col_name).upper()
    for key, label in SENSOR_GROUPS:
        if key in col_up:
            return f"ANOMALY_{label}"
    return "ANOMALY_MISC"


def _add_predictive_maintenance_signals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Derive PM-oriented risk and alert timing signals from anomaly score timeline."""
    out = df.copy()

    risk_pct = out.groupby("machine_id", dropna=False)["anomaly_score"].rank(method="average", pct=True)
    out["pm_risk_score"] = (risk_pct * 100.0).round(4)

    alert_threshold_pct = 97.0
    out["pm_alert_flag"] = out["pm_risk_score"] >= alert_threshold_pct

    has_event_time = "event_time" in out.columns and out["event_time"].notna().any()
    out["pm_event_id"] = pd.NA
    out["pm_lead_time_min"] = pd.NA

    if not has_event_time:
        meta = {
            "pm_alert_threshold_pct": alert_threshold_pct,
            "pm_has_timeline": False,
            "pm_event_count": 0,
        }
        return out, meta

    sorted_df = out.sort_values(["machine_id", "event_time"]).copy()
    sorted_df["pm_alert_flag"] = sorted_df["pm_alert_flag"].fillna(False)

    # New event starts when alert turns on, machine changes, or time gap gets large.
    prev_alert = sorted_df.groupby("machine_id", dropna=False)["pm_alert_flag"].shift(1).fillna(False)
    prev_time = sorted_df.groupby("machine_id", dropna=False)["event_time"].shift(1)
    time_gap_min = (sorted_df["event_time"] - prev_time).dt.total_seconds().div(60)
    large_gap = time_gap_min.isna() | (time_gap_min > 10)
    event_start = sorted_df["pm_alert_flag"] & ((~prev_alert) | large_gap)

    event_counter = event_start.astype(int).cumsum()
    sorted_df["pm_event_id"] = event_counter.where(sorted_df["pm_alert_flag"], pd.NA)

    next_defect_time = (
        sorted_df["event_time"].where(sorted_df["is_defect"])
        .groupby(sorted_df["machine_id"], dropna=False)
        .transform(lambda s: s.bfill())
    )
    lead_min = (next_defect_time - sorted_df["event_time"]).dt.total_seconds().div(60)
    sorted_df["pm_lead_time_min"] = lead_min.where(sorted_df["pm_alert_flag"])

    out = sorted_df.sort_index()
    event_count = int(out["pm_event_id"].nunique(dropna=True))

    meta = {
        "pm_alert_threshold_pct": alert_threshold_pct,
        "pm_has_timeline": True,
        "pm_event_count": event_count,
    }
    return out, meta


def prepare_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Infer core columns and derive standardized labels for defect analysis."""
    time_col = _infer_time_column(df)
    result_col = _infer_text_column(df, RESULT_KEYS, max_unique_ratio=1.0)
    defect_col = _infer_text_column(df, DEFECT_KEYS, max_unique_ratio=0.8, allow_single_value=True)
    machine_col = _infer_text_column(df, MACHINE_KEYS, max_unique_ratio=1.0)

    out = df.copy()
    out["is_defect"] = _build_is_defect(out, result_col, defect_col)
    out["defect_type"] = _build_defect_type(out, out["is_defect"], result_col, defect_col)

    if time_col:
        out["event_time"] = pd.to_datetime(out[time_col], errors="coerce")
        out["event_hour"] = out["event_time"].dt.hour
    else:
        out["event_time"] = pd.NaT
        out["event_hour"] = pd.NA

    if machine_col:
        out["machine_id"] = out[machine_col].astype(str).str.strip().replace("", "UNKNOWN")
    else:
        out["machine_id"] = "UNKNOWN"

    anomaly_score, primary_anomaly_feature = _extract_numeric_anomaly_components(out)
    out["anomaly_score"] = anomaly_score
    out["primary_anomaly_feature"] = primary_anomaly_feature.astype(str)

    label_strategy = "explicit_labels"
    anomaly_threshold = None
    if result_col is None and defect_col is None:
        # For signal-only logs, fallback to high-anomaly rows as suspect defects.
        quantile = 0.97 if len(out) >= 100 else 0.95
        anomaly_threshold = float(out["anomaly_score"].quantile(quantile))
        out["is_defect"] = out["anomaly_score"] >= anomaly_threshold
        out["defect_type"] = out["primary_anomaly_feature"].map(_map_feature_to_anomaly_type)
        out["defect_type"] = out["defect_type"].where(out["is_defect"], "NORMAL")
        label_strategy = "anomaly_fallback"

    out, pm_meta = _add_predictive_maintenance_signals(out)

    meta = {
        "result_col": result_col,
        "defect_col": defect_col,
        "time_col": time_col,
        "machine_col": machine_col,
        "label_strategy": label_strategy,
        "anomaly_threshold": anomaly_threshold,
    }
    meta.update(pm_meta)
    return out, meta

