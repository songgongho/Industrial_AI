"""Quick EDA script for SECOM dataset using the project's dataset inspector.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_inspector import analyze_dataset_source, save_analysis_report


def main() -> int:
    data_dir = ROOT / "data" / "raw" / "secom"
    if not data_dir.exists():
        print(f"SECOM data directory not found: {data_dir}")
        return 1
    analysis = analyze_dataset_source(data_dir)
    out = save_analysis_report(analysis, ROOT / "reports", report_name="eda_secom")
    print("SECOM EDA saved:")
    for k, p in out.items():
        print(f" - {k}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

