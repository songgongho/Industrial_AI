from pathlib import Path
import zipfile
from base64 import b64decode

from src.data.dataset_inspector import analyze_dataset_source, save_analysis_report


def test_analyze_dataset_source_for_csv_folder(tmp_path: Path) -> None:
    folder = tmp_path / "dataset"
    folder.mkdir()
    csv_path = folder / "sample.csv"
    csv_path.write_text("a,b\n1,3\n2,4\n,5\n", encoding="utf-8")

    analysis = analyze_dataset_source(folder)

    assert analysis.exists is True
    assert analysis.total_files == 1
    assert analysis.tabular_shape == (3, 2)
    assert analysis.missing_cells == 1
    assert analysis.preview_frame is not None
    assert list(analysis.preview_frame.columns) == ["a", "b"]
    assert analysis.dtype_frame is not None
    assert "sample.csv" in "\n".join(str(path) for path in analysis.sample_files)


def test_analyze_dataset_source_for_image_folder(tmp_path: Path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    image_path = folder / "sample.png"
    image_path.write_bytes(
        b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7+X9kAAAAASUVORK5CYII="
        )
    )

    analysis = analyze_dataset_source(folder)

    assert analysis.total_files == 1
    assert analysis.image_info is not None
    assert analysis.image_info["width"] == 1
    assert analysis.image_info["height"] == 1


def test_analyze_dataset_source_for_zip_archive(tmp_path: Path) -> None:
    folder = tmp_path / "archive_source"
    folder.mkdir()
    csv_path = folder / "sample.csv"
    csv_path.write_text("x,y\n1,3\n2,4\n", encoding="utf-8")
    zip_path = tmp_path / "dataset.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname="sample.csv")

    analysis = analyze_dataset_source(zip_path)

    assert analysis.source_type == "zip"
    assert analysis.archive_members == ["sample.csv"]
    assert analysis.tabular_shape == (2, 2)
    assert analysis.preview_frame is not None


def test_save_analysis_report_writes_markdown_and_preview(tmp_path: Path) -> None:
    folder = tmp_path / "dataset"
    folder.mkdir()
    csv_path = folder / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    analysis = analyze_dataset_source(folder)
    output_dir = tmp_path / "reports"

    outputs = save_analysis_report(analysis, output_dir, report_name="sample_dataset")

    assert outputs["report"].exists()
    assert outputs["preview"].exists()
    assert "데이터셋 분석 리포트" in outputs["report"].read_text(encoding="utf-8")


