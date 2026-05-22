# scripts/run_labeling_and_save.py
# Usage: python scripts/run_labeling_and_save.py "<csv_path>" [out_dir]
import sys
from pathlib import Path

import pandas as pd

from src.labeling.labeling import build_state_label, DEFAULT_CONFIG, summarize_label_distribution

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_labeling_and_save.py <csv_path> [out_dir]")
        return
    csv_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 읽기: 원래 모듈은 header=None으로 읽도록 작성되어 있으므로 동일하게 읽습니다.
    frame = pd.read_csv(csv_path, header=None, na_values=["\\N", "\\N.1"])
    labeled = build_state_label(frame, DEFAULT_CONFIG)

    out_labeled = out_dir / "labeled_data.csv"
    out_summary = out_dir / "label_distribution.csv"

    # 저장 (index=False로 저장하면 깔끔합니다)
    labeled.to_csv(out_labeled, index=False, encoding="utf-8")
    summary = summarize_label_distribution(labeled)
    summary.to_csv(out_summary, index=False, encoding="utf-8")

    # 콘솔 출력(일부)
    print("Saved labeled data to:", out_labeled)
    print("Saved label distribution summary to:", out_summary)
    print("\n---- Head of labeled dataframe ----")
    print(labeled.head().to_string())
    print("\n---- Label distribution summary ----")
    print(summary.to_string())

if __name__ == "__main__":
    main()
