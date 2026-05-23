"""데이터셋 전처리 유틸리티.

분석된 또는 업로드된 tabular 데이터에서 기본적인 정리 작업을 수행하고
학습용 파일과 메타데이터를 저장한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".data", ".parquet", ".pq", ".xlsx", ".xls"}


@dataclass(frozen=True)
class PreprocessResult:
    source: Path
    input_path: Path
    output_dir: Path
    cleaned_parquet: Path
    cleaned_csv: Path
    metadata_json: Path
    report_md: Path
    rows_before: int
    rows_after: int
    columns: list[str]
    notes: list[str]


def preprocess_tabular_dataset(source: Path | str, output_dir: Path | str) -> PreprocessResult:
    """Preprocess a tabular file or a folder containing at least one tabular file."""

    source_path = Path(source)
    input_path = _resolve_input_path(source_path)
    frame = _read_tabular(input_path)
    original_rows = int(len(frame))

    cleaned = _clean_frame(frame)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = _safe_stem(source_path.stem or input_path.stem or "dataset")
    cleaned_parquet = output_path / f"{stem}.parquet"
    cleaned_csv = output_path / f"{stem}.csv"
    metadata_json = output_path / f"{stem}.json"
    report_md = output_path / f"{stem}.md"

    cleaned.to_parquet(cleaned_parquet, index=False)
    cleaned.to_csv(cleaned_csv, index=False)

    metadata: dict[str, Any] = {
        "source": str(source_path),
        "input_path": str(input_path),
        "rows_before": original_rows,
        "rows_after": int(len(cleaned)),
        "columns": list(cleaned.columns),
        "dtypes": {col: str(dtype) for col, dtype in cleaned.dtypes.items()},
        "missing_after": int(cleaned.isna().sum().sum()),
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_make_report(metadata), encoding="utf-8")

    notes = [
        "컬럼명을 공백 제거 형태로 정리함",
        "중복 행 제거",
        "숫자형 결측치는 중앙값으로 대체",
        "범주형 결측치는 최빈값 또는 'missing'으로 대체",
    ]
    return PreprocessResult(
        source=source_path,
        input_path=input_path,
        output_dir=output_path,
        cleaned_parquet=cleaned_parquet,
        cleaned_csv=cleaned_csv,
        metadata_json=metadata_json,
        report_md=report_md,
        rows_before=original_rows,
        rows_after=int(len(cleaned)),
        columns=list(cleaned.columns),
        notes=notes,
    )


def _resolve_input_path(source_path: Path) -> Path:
    if source_path.is_file():
        return source_path
    if not source_path.is_dir():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {source_path}")
    for candidate in sorted(source_path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in TABULAR_SUFFIXES:
            return candidate
    raise FileNotFoundError(f"tabular 파일을 찾지 못했습니다: {source_path}")


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python", na_values=["?", "NA", "N/A", ""])


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip().replace(" ", "_") for column in cleaned.columns]
    cleaned = cleaned.replace({"": None, " ": None, "NA": None, "N/A": None, "?": None})
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    for column in cleaned.columns:
        series = cleaned[column]
        if series.dtype.kind in {"i", "u", "f"}:
            if series.isna().any():
                median = series.dropna().median()
                cleaned[column] = series.fillna(median)
        else:
            series = series.astype("string").str.strip()
            non_null = series.dropna()
            fill_value = non_null.mode().iloc[0] if not non_null.empty and not non_null.mode().empty else "missing"
            cleaned[column] = series.fillna(fill_value)
    return cleaned


def _make_report(metadata: dict[str, Any]) -> str:
    lines = [
        "# 전처리 결과",
        "",
        f"- 원본 경로: `{metadata['source']}`",
        f"- 입력 파일: `{metadata['input_path']}`",
        f"- 행 수: `{metadata['rows_before']}` → `{metadata['rows_after']}`",
        f"- 컬럼 수: `{len(metadata['columns'])}`",
        f"- 남은 결측치: `{metadata['missing_after']}`",
        "",
        "## 컬럼",
        "",
    ]
    lines.extend(f"- `{column}` ({metadata['dtypes'][column]})" for column in metadata["columns"])
    return "\n".join(lines).strip() + "\n"


def _safe_stem(value: str) -> str:
    keep = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    stem = "".join(keep).strip("_")
    return stem or "dataset"

