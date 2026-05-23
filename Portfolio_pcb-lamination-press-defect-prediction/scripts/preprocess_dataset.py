"""데이터셋 전처리 CLI.

예시:
  python scripts/preprocess_dataset.py --source data/raw/secom --output-dir data/processed/secom
  python scripts/preprocess_dataset.py --source C:\\path\\to\\uploaded.csv --output-dir data/processed/uploaded
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocess import preprocess_tabular_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="preprocess_dataset.py")
    parser.add_argument("--source", required=True, help="전처리할 파일 또는 폴더 경로")
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="전처리 결과를 저장할 디렉토리",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    result = preprocess_tabular_dataset(Path(args.source), Path(args.output_dir))
    print("전처리 완료")
    print(f"- 입력 파일: {result.input_path}")
    print(f"- 저장 폴더: {result.output_dir}")
    print(f"- 행 수: {result.rows_before} -> {result.rows_after}")
    print(f"- CSV: {result.cleaned_csv}")
    print(f"- Parquet: {result.cleaned_parquet}")
    print(f"- 리포트: {result.report_md}")
    print(f"- 메타데이터: {result.metadata_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

