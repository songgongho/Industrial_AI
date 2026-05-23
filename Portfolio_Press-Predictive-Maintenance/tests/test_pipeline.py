from __future__ import annotations

from pathlib import Path
import tempfile
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_pipeline_from_dataframe


def test_pipeline_from_dataframe() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2026-03-08 09:00:00",
                "2026-03-08 09:01:00",
                "2026-03-08 10:00:00",
                "2026-03-08 10:01:00",
            ],
            "machine_id": ["P1", "P1", "P2", "P2"],
            "result": ["OK", "NG", "FAIL", "OK"],
            "defect_reason": ["", "BURR", "CRACK", ""],
            "max_force": [10.2, 12.0, 13.5, 9.9],
        }
    )

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "outputs"
        report = run_pipeline_from_dataframe(frame, out_dir)

        assert report["summary"]["total_count"] == 4
        assert report["summary"]["defect_count"] == 2
        assert report["meta"]["label_strategy"] == "explicit_labels"
        assert "pm_event_count" in report["meta"]
        assert (out_dir / "tables" / "summary.csv").exists()
        assert (out_dir / "tables" / "combined_dataset.csv").exists()
        assert (out_dir / "tables" / "defect_type_pareto.csv").exists()
        assert (out_dir / "tables" / "defect_rate_by_source_file.csv").exists()
        assert (out_dir / "tables" / "pm_machine_risk_summary.csv").exists()
        assert (out_dir / "tables" / "pm_event_timeline.csv").exists()
        assert (out_dir / "tables" / "pm_alert_leadtime_summary.csv").exists()
        assert (out_dir / "report.md").exists()
        report_text = (out_dir / "report.md").read_text(encoding="utf-8")
        assert "## Executive Summary" in report_text
        assert "## 7. 권고 사항" in report_text
        assert "| 지표 | 값 |" in report_text
        assert "![PM 위험도 타임라인](figures/pm_risk_timeline.png)" in report_text


def test_pipeline_anomaly_fallback() -> None:
    frame = pd.DataFrame(
        {
            "Data Time": [
                "2026-03-08 09:00:00",
                "2026-03-08 09:01:00",
                "2026-03-08 09:02:00",
                "2026-03-08 09:03:00",
                "2026-03-08 09:04:00",
            ],
            "6HPPRESSPV": [10.0, 10.2, 10.1, 55.0, 10.0],
            "6VACUUM": [2.1, 2.0, 2.1, 1.9, 2.0],
            "6POINT1": [5.0, 5.1, 5.0, 4.9, 5.0],
        }
    )

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "outputs"
        report = run_pipeline_from_dataframe(frame, out_dir)

        assert report["summary"]["total_count"] == 5
        assert report["summary"]["defect_count"] >= 1
        assert report["meta"]["label_strategy"] == "anomaly_fallback"
        assert (out_dir / "tables" / "defect_type_counts.csv").exists()
        assert (out_dir / "tables" / "primary_anomaly_feature_contrib.csv").exists()
        assert report["meta"]["pm_event_count"] >= 1


def run_smoke() -> None:
    test_pipeline_from_dataframe()
    test_pipeline_anomaly_fallback()
    print("smoke test passed")


if __name__ == "__main__":
    run_smoke()

