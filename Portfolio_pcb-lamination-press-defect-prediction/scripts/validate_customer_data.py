#!/usr/bin/env python
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer', 'raw')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'customer')
os.makedirs(OUT_DIR, exist_ok=True)

files_to_validate = [
    ('quality_daily.csv', ['date', 'shift', 'good_qty', 'scrap_qty', 'defect_code']),
    ('equipment_state.csv', ['machine_id', 'timestamp', 'state', 'downtime_min']),
    ('press_alarms.csv', ['alarm_time', 'alarm_code', 'severity', 'duration_sec']),
    ('recipe_changes.csv', ['change_timestamp', 'parameter_name', 'old_value', 'new_value']),
    ('lot_panel_mapping.csv', ['lot_id', 'panel_id', 'cycle_id', 'timestamp']),
    ('maintenance_logs.csv', ['maintenance_date', 'component', 'work_type']),
    ('operator_info.csv', ['shift', 'operator_id', 'experience_level']),
]


def validate_customer_data(data_dir=DATA_DIR, out_dir=OUT_DIR):
    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "files": {},
        "summary": {}
    }
    total_issues = 0
    for filename, required_cols in files_to_validate:
        filepath = os.path.join(data_dir, filename)
        file_report = {"file": filename, "status": "PASS", "issues": []}
        try:
            df = pd.read_csv(filepath)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                file_report["issues"].append(f"Missing columns: {missing_cols}")
                file_report["status"] = "FAIL"
            missing_pct = (df.isnull().sum() / max(1, len(df)) * 100).to_dict()
            for col, pct in missing_pct.items():
                if pct > 5:
                    file_report["issues"].append(f"Column '{col}' has {pct:.1f}% missing values")
            duplicate_rows = int(df.duplicated().sum())
            if duplicate_rows > 0:
                file_report["issues"].append(f"Found {duplicate_rows} duplicate rows")
                file_report["status"] = "FAIL"
            timestamp_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
            for ts_col in timestamp_cols:
                try:
                    ts_series = pd.to_datetime(df[ts_col])
                    if not ts_series.is_monotonic_increasing:
                        file_report["issues"].append(f"Column '{ts_col}' is not monotonic increasing")
                except Exception as e:
                    file_report["issues"].append(f"Column '{ts_col}' datetime parsing failed: {str(e)}")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if ('qty' in col.lower() or 'time' in col.lower()) and (df[col] < 0).any():
                    file_report["issues"].append(f"Column '{col}' contains negative values")
                    file_report["status"] = "FAIL"
            file_report["summary"] = {"rows": int(len(df)), "columns": int(len(df.columns))}
            if file_report["status"] == "FAIL":
                total_issues += len(file_report["issues"])
        except FileNotFoundError:
            file_report["status"] = "FILE_NOT_FOUND"
            file_report["issues"].append(f"File not found: {filepath}")
            total_issues += 1
        except Exception as e:
            file_report["status"] = "ERROR"
            file_report["issues"].append(f"Error processing file: {str(e)}")
            total_issues += 1
        report["files"][filename] = file_report
    report["summary"] = {"total_files": len(files_to_validate), "files_passed": sum(1 for f in report["files"].values() if f["status"] == "PASS"), "files_failed": sum(1 for f in report["files"].values() if f["status"] != "PASS"), "total_issues": total_issues, "recommendation": "Data ready for analysis" if total_issues == 0 else "Data needs cleaning before analysis"}
    out_path = os.path.join(out_dir, 'validation_report.json')
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"Validation report written to: {out_path}")
    return report


if __name__ == '__main__':
    validate_customer_data()

