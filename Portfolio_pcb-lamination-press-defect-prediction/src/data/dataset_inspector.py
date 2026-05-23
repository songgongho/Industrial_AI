"""데이터셋 업로드/폴더 분석 유틸리티.

이 모듈은 Streamlit UI와 테스트에서 재사용할 수 있는
가벼운 프로파일링 기능을 제공한다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import tempfile
from pathlib import Path
from typing import Any
import zipfile

from pandas import DataFrame, read_csv, read_excel, read_parquet


TABULAR_SUFFIXES = {".csv", ".tsv", ".txt", ".data", ".parquet", ".pq", ".xlsx", ".xls"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetAnalysis:
    source: Path
    source_type: str
    exists: bool
    total_files: int = 0
    total_dirs: int = 0
    total_size_bytes: int = 0
    extension_counts: dict[str, int] = field(default_factory=dict)
    sample_files: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    preview_frame: DataFrame | None = None
    dtype_frame: DataFrame | None = None
    numeric_summary: DataFrame | None = None
    tabular_shape: tuple[int, int] | None = None
    column_names: list[str] = field(default_factory=list)
    missing_cells: int | None = None
    image_info: dict[str, Any] | None = None
    archive_members: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"- 경로: `{self.source}`",
            f"- 유형: `{self.source_type}`",
            f"- 존재 여부: `{self.exists}`",
            f"- 파일 수: `{self.total_files}`",
            f"- 폴더 수: `{self.total_dirs}`",
            f"- 총 용량(byte): `{self.total_size_bytes}`",
        ]
        if self.tabular_shape is not None:
            lines.append(f"- 표 형태 크기: `{self.tabular_shape[0]} x {self.tabular_shape[1]}`")
        if self.missing_cells is not None:
            lines.append(f"- 결측 셀 수: `{self.missing_cells}`")
        if self.column_names:
            lines.append(f"- 컬럼 수: `{len(self.column_names)}`")
        if self.image_info:
            lines.append(
                "- 이미지 정보: "
                + ", ".join(f"{key}={value}" for key, value in self.image_info.items())
            )
        if self.notes:
            lines.append("- 메모:")
            lines.extend(f"  - {note}" for note in self.notes)
        if self.extension_counts:
            lines.append("- 확장자 분포:")
            lines.extend(
                f"  - {ext}: {count}" for ext, count in sorted(self.extension_counts.items(), key=lambda x: (-x[1], x[0]))
            )
        if self.sample_files:
            lines.append("- 샘플 파일:")
            lines.extend(f"  - `{path}`" for path in self.sample_files)
        if self.archive_members:
            lines.append("- 압축 파일 내부 항목:")
            lines.extend(f"  - `{member}`" for member in self.archive_members)
        return "\n".join(lines)


def analyze_dataset_source(source: Path | str, max_preview_rows: int = 10, max_samples: int = 10) -> DatasetAnalysis:
    source_path = Path(source)
    if not source_path.exists():
        return DatasetAnalysis(source=source_path, source_type="missing", exists=False, notes=["경로가 존재하지 않습니다."])

    if source_path.is_dir():
        return _analyze_folder(source_path, max_preview_rows=max_preview_rows, max_samples=max_samples)

    suffix = source_path.suffix.lower()
    if suffix == ".zip":
        return _analyze_zip(source_path, max_preview_rows=max_preview_rows, max_samples=max_samples)
    if suffix in TABULAR_SUFFIXES:
        return _analyze_tabular_file(source_path, max_preview_rows=max_preview_rows)
    if suffix in IMAGE_SUFFIXES:
        return _analyze_image_file(source_path)
    return _analyze_generic_file(source_path)


def save_analysis_report(
    analysis: DatasetAnalysis,
    output_dir: Path | str,
    report_name: str | None = None,
) -> dict[str, Path]:
    """Save a markdown report and optional preview CSV for an analysis."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stem = report_name or _safe_stem(analysis.source.name or analysis.source_type)
    report_path = output_path / f"{stem}.md"
    report_path.write_text(_analysis_markdown(analysis), encoding="utf-8")

    saved: dict[str, Path] = {"report": report_path}
    if analysis.preview_frame is not None:
        preview_path = output_path / f"{stem}_preview.csv"
        analysis.preview_frame.to_csv(preview_path, index=False)
        saved["preview"] = preview_path
    if analysis.dtype_frame is not None:
        dtype_path = output_path / f"{stem}_dtypes.csv"
        analysis.dtype_frame.to_csv(dtype_path, index=False)
        saved["dtypes"] = dtype_path
    if analysis.numeric_summary is not None:
        numeric_path = output_path / f"{stem}_numeric_summary.csv"
        analysis.numeric_summary.to_csv(numeric_path)
        saved["numeric_summary"] = numeric_path
    return saved


def _analyze_folder(folder: Path, max_preview_rows: int, max_samples: int) -> DatasetAnalysis:
    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    dirs = [p for p in sorted(folder.rglob("*")) if p.is_dir()]
    ext_counter = Counter((p.suffix.lower() or "[no_ext]") for p in files)
    total_size = sum(p.stat().st_size for p in files)
    sample_files = files[:max_samples]
    notes: list[str] = []
    analysis = DatasetAnalysis(
        source=folder,
        source_type="folder",
        exists=True,
        total_files=len(files),
        total_dirs=len(dirs),
        total_size_bytes=total_size,
        extension_counts=dict(ext_counter),
        sample_files=sample_files,
        notes=notes,
    )

    tabular_candidate = _first_match(files, TABULAR_SUFFIXES)
    if tabular_candidate is not None:
        nested = _analyze_tabular_file(tabular_candidate, max_preview_rows=max_preview_rows)
        analysis.source_type = f"folder+{nested.source_type}"
        analysis.tabular_shape = nested.tabular_shape
        analysis.column_names = nested.column_names
        analysis.missing_cells = nested.missing_cells
        analysis.preview_frame = nested.preview_frame
        analysis.dtype_frame = nested.dtype_frame
        analysis.numeric_summary = nested.numeric_summary
        analysis.notes.extend(nested.notes)

    image_candidate = _first_match(files, IMAGE_SUFFIXES)
    if image_candidate is not None:
        image_info = _image_info(image_candidate)
        if image_info:
            analysis.image_info = image_info
            analysis.notes.append(f"대표 이미지: {image_candidate.name}")

    if not files:
        analysis.notes.append("폴더 내부에 파일이 없습니다.")

    return analysis


def _analyze_zip(zip_path: Path, max_preview_rows: int, max_samples: int) -> DatasetAnalysis:
    with tempfile.TemporaryDirectory() as tmp_dir:
        extract_root = Path(tmp_dir)
        members: list[str] = []
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [name for name in zf.namelist() if not name.endswith("/")][: max_samples * 5]
            zf.extractall(extract_root)
        extracted_files = [p for p in sorted(extract_root.rglob("*")) if p.is_file()]
        folder_analysis = _analyze_folder(extract_root, max_preview_rows=max_preview_rows, max_samples=max_samples)
        folder_analysis.source = zip_path
        folder_analysis.source_type = "zip"
        folder_analysis.archive_members = members
        folder_analysis.notes.insert(0, f"압축 해제 후 내부 파일 수: {len(extracted_files)}")
        return folder_analysis


def _analyze_tabular_file(path: Path, max_preview_rows: int) -> DatasetAnalysis:
    frame = _read_tabular(path)
    missing_cells = int(frame.isna().sum().sum())
    dtype_frame = DataFrame({"dtype": frame.dtypes.astype(str)})
    numeric = frame.select_dtypes(include="number")
    numeric_summary = numeric.describe().T if not numeric.empty else None
    return DatasetAnalysis(
        source=path,
        source_type="tabular",
        exists=True,
        total_files=1,
        total_size_bytes=path.stat().st_size,
        extension_counts={path.suffix.lower() or "[no_ext]": 1},
        sample_files=[path],
        notes=[f"표 데이터로 인식됨: {frame.shape[0]}행 x {frame.shape[1]}열"],
        preview_frame=frame.head(max_preview_rows),
        dtype_frame=dtype_frame,
        numeric_summary=numeric_summary,
        tabular_shape=frame.shape,
        column_names=list(map(str, frame.columns.tolist())),
        missing_cells=missing_cells,
    )


def _analyze_image_file(path: Path) -> DatasetAnalysis:
    info = _image_info(path)
    notes = [f"이미지 파일로 인식됨: {path.name}"]
    return DatasetAnalysis(
        source=path,
        source_type="image",
        exists=True,
        total_files=1,
        total_size_bytes=path.stat().st_size,
        extension_counts={path.suffix.lower() or "[no_ext]": 1},
        sample_files=[path],
        notes=notes,
        image_info=info,
    )


def _analyze_generic_file(path: Path) -> DatasetAnalysis:
    return DatasetAnalysis(
        source=path,
        source_type="file",
        exists=True,
        total_files=1,
        total_size_bytes=path.stat().st_size,
        extension_counts={path.suffix.lower() or "[no_ext]": 1},
        sample_files=[path],
        notes=["자동 프로파일링이 어려운 형식입니다. 폴더 단위로 다시 분석하면 더 많은 정보를 볼 수 있습니다."],
    )


def _analysis_markdown(analysis: DatasetAnalysis) -> str:
    lines = [
        "# 데이터셋 분석 리포트",
        "",
        analysis.to_markdown(),
    ]
    if analysis.preview_frame is not None:
        lines.extend([
            "",
            "## 미리보기",
            "",
            analysis.preview_frame.head(10).to_markdown(index=False),
        ])
    if analysis.dtype_frame is not None:
        lines.extend([
            "",
            "## 컬럼 자료형",
            "",
            analysis.dtype_frame.to_markdown(),
        ])
    if analysis.numeric_summary is not None:
        lines.extend([
            "",
            "## 수치형 요약 통계",
            "",
            analysis.numeric_summary.to_markdown(),
        ])
    if analysis.image_info is not None:
        lines.extend([
            "",
            "## 이미지 정보",
            "",
            "```json",
            str(analysis.image_info),
            "```",
        ])
    if analysis.archive_members:
        lines.extend([
            "",
            "## 압축 파일 내부 항목",
            "",
            "\n".join(f"- `{member}`" for member in analysis.archive_members),
        ])
    return "\n".join(lines).strip() + "\n"


def _safe_stem(value: str) -> str:
    keep = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    stem = "".join(keep).strip("_")
    return stem or "dataset_analysis"


def _first_match(files: list[Path], suffixes: set[str]) -> Path | None:
    for path in files:
        if path.suffix.lower() in suffixes:
            return path
    return None


def _read_tabular(path: Path) -> DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet" or suffix == ".pq":
        return read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return read_excel(path)
    if suffix == ".tsv":
        return read_csv(path, sep="\t")
    try:
        return read_csv(path)
    except Exception:
        return read_csv(path, sep=None, engine="python", na_values=["?", "NA", "N/A", ""])


def _image_info(path: Path) -> dict[str, Any] | None:
    try:
        from PIL import Image as PILImage
    except Exception:  # pragma: no cover - pillow may be missing in constrained envs
        return None
    try:
        with PILImage.open(path) as image:
            return {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
            }
    except Exception as ex:  # pragma: no cover - image corruption path
        return {"error": str(ex)}

