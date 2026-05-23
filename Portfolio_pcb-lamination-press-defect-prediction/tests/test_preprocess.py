from pathlib import Path

from src.data.preprocess import preprocess_tabular_dataset


def test_preprocess_tabular_dataset_creates_clean_artifacts(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    source_dir.mkdir()
    source_csv = source_dir / "sample.csv"
    source_csv.write_text(
        "a,b,c\n1,x,10\n1,x,10\n, y,20\n3,z,30\n",
        encoding="utf-8",
    )

    result = preprocess_tabular_dataset(source_dir, tmp_path / "processed")

    assert result.cleaned_csv.exists()
    assert result.cleaned_parquet.exists()
    assert result.report_md.exists()
    assert result.metadata_json.exists()
    assert result.rows_before == 4
    assert result.rows_after == 3

    cleaned_text = result.cleaned_csv.read_text(encoding="utf-8")
    assert cleaned_text.startswith("a,b,c")
    assert "1.0,x,10" in cleaned_text
    assert "2.0,y,20" in cleaned_text
    assert "3.0,z,30" in cleaned_text
    assert cleaned_text.count("\n") == 4



