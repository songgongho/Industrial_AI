from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
try:
    import pyodbc
except ImportError:  # pragma: no cover
    pyodbc = None

from tools import (
    AnomalyInput,
    analyze_history,
    assess_severity,
    build_markdown_report,
    infer_causes,
    suggest_actions,
)


BASE_DIR = Path(r"E:\2026-1학기\이상탐지 해석 에이전트 프롬프트 실습.260406\학습대상\Press Profile log")
OUTPUT_PATH = Path("press_mdb_analysis.md")


def connect_mdb(mdb_path: Path):
    if pyodbc is None:
        raise ImportError("pyodbc가 설치되어 있지 않습니다. requirements.txt를 설치하세요.")
    driver = pyodbc
    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={mdb_path};"
    return driver.connect(conn_str)


def get_tables(conn) -> List[str]:
    cur = conn.cursor()
    return [row.table_name for row in cur.tables(tableType="TABLE")]


def pick_main_table(conn, tables: List[str]) -> str:
    cur = conn.cursor()
    best_table = ""
    best_count = -1
    for table in tables:
        try:
            count = cur.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
            if count > best_count:
                best_count = int(count)
                best_table = table
        except Exception:
            continue
    if not best_table:
        raise ValueError("분석 가능한 테이블을 찾지 못했습니다.")
    return best_table


def load_table_df(conn, table_name: str) -> pd.DataFrame:
    query = f"SELECT * FROM [{table_name}]"
    return pd.read_sql(query, conn)


def find_numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() >= 0.9:
            numeric_cols.append(col)
    return numeric_cols


def detect_top_anomalies(df: pd.DataFrame, numeric_cols: List[str], top_k: int = 3) -> Dict[str, Any]:
    z_abs: Dict[str, float] = {}
    latest_row = df.iloc[-1]

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        mean = series.mean()
        std = series.std(ddof=0)
        last_val = pd.to_numeric(pd.Series([latest_row[col]]), errors="coerce").iloc[0]

        if pd.isna(last_val) or pd.isna(std) or std == 0:
            z = 0.0
        else:
            z = abs((float(last_val) - float(mean)) / float(std))
        z_abs[col] = float(z)

    top = sorted(z_abs.items(), key=lambda item: item[1], reverse=True)[:top_k]
    total = sum(score for _, score in top) or 1.0
    feature_importance = {name: score / total for name, score in top}

    max_z = top[0][1] if top else 0.0
    anomaly_score = min(max_z / 3.0, 1.0)

    return {
        "anomaly_score": anomaly_score,
        "feature_importance": feature_importance,
        "top_z": top,
    }


def find_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [
        col
        for col in df.columns
        if any(key in str(col).lower() for key in ["time", "date", "timestamp"])
    ]
    if not candidates:
        return ""

    for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.7:
            return col
    return ""


def make_report_for_mdb(mdb_path: Path) -> str:
    conn = connect_mdb(mdb_path)
    try:
        tables = get_tables(conn)
        table_name = pick_main_table(conn, tables)
        df = load_table_df(conn, table_name)
    finally:
        conn.close()

    if df.empty:
        return f"## {mdb_path.name}\n\n- 데이터가 비어 있어 분석할 수 없습니다.\n"

    numeric_cols = find_numeric_columns(df)
    if len(numeric_cols) < 3:
        return (
            f"## {mdb_path.name}\n\n"
            f"- 숫자형 컬럼이 충분하지 않아 이상탐지를 수행하지 못했습니다.\n"
        )

    anomaly = detect_top_anomalies(df, numeric_cols)

    recent_scores = []
    window = min(len(df), 10)
    start_idx = len(df) - window
    for idx in range(start_idx, len(df)):
        row = df.iloc[idx]
        row_z = []
        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce")
            mean = s.mean()
            std = s.std(ddof=0)
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.isna(val) or pd.isna(std) or std == 0:
                row_z.append(0.0)
            else:
                row_z.append(abs((float(val) - float(mean)) / float(std)))
        recent_scores.append(min((max(row_z) if row_z else 0.0) / 3.0, 1.0))

    timestamp_col = find_timestamp_column(df)
    timestamp_val = "N/A"
    if timestamp_col:
        timestamp_val = str(df.iloc[-1][timestamp_col])

    stats = analyze_history(recent_scores)
    severity = assess_severity(anomaly["anomaly_score"], stats["change_rate"])
    causes = infer_causes(anomaly["feature_importance"], stats)
    actions = suggest_actions(severity)

    input_data = AnomalyInput(
        timestamp=timestamp_val,
        anomaly_score=anomaly["anomaly_score"],
        feature_importance=anomaly["feature_importance"],
        historical_data=recent_scores,
        sensor_id=f"{mdb_path.stem}:{table_name}",
    )

    report = build_markdown_report(
        anomaly_input=input_data,
        stats=stats,
        severity=severity,
        causes=causes,
        actions=actions,
        llm_text=None,
    )

    profile_lines = [
        "",
        "### 데이터 프로파일",
        f"- 파일: {mdb_path.name}",
        f"- 테이블: {table_name}",
        f"- 행 수: {len(df)}",
        f"- 숫자형 컬럼 수: {len(numeric_cols)}",
        f"- 상위 z-score feature: {json.dumps(anomaly['top_z'], ensure_ascii=False)}",
        "",
    ]

    return report + "\n" + "\n".join(profile_lines)


def main() -> None:
    mdb_files = sorted(BASE_DIR.glob("*.mdb"))
    if not mdb_files:
        raise FileNotFoundError(f"MDB 파일이 없습니다: {BASE_DIR}")

    reports = ["# Press MDB 이상탐지 분석 결과", ""]
    for mdb in mdb_files:
        reports.append(make_report_for_mdb(mdb))

    OUTPUT_PATH.write_text("\n".join(reports), encoding="utf-8")
    print(f"분석 완료: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

