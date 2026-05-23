"""데이터셋 분석 CLI.

예시:
  python scripts/analyze_dataset.py --source data/raw/secom
  python scripts/analyze_dataset.py --source C:\\path\\to\\dataset.zip --output-dir reports\analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_inspector import analyze_dataset_source, save_analysis_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="analyze_dataset.py")
    parser.add_argument("--source", required=True, help="분석할 파일 또는 폴더 경로")
    parser.add_argument(
        "--output-dir",
        default="reports/analysis",
        help="리포트와 미리보기 CSV를 저장할 디렉토리",
    )
    parser.add_argument("--preview-rows", type=int, default=10, help="표 데이터 미리보기 행 수")
    parser.add_argument("--max-samples", type=int, default=10, help="표시할 샘플 파일 수")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    source = Path(args.source)
    analysis = analyze_dataset_source(source, max_preview_rows=args.preview_rows, max_samples=args.max_samples)
    outputs = save_analysis_report(analysis, Path(args.output_dir), report_name=source.stem or source.name)

    print(analysis.to_markdown())
    print("\n[Saved files]")
    for key, path in outputs.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

