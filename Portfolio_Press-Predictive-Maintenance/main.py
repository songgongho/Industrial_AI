from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.press_analysis.classify import export_reports
from src.press_analysis.features import prepare_dataset
from src.press_analysis.io_mdb import discover_mdb_files, load_best_table, resolve_access_driver


def run_pipeline(input_dir: Path, output_dir: Path, driver: str | None = None) -> dict:
    """Load all MDB files, infer defect labels/types, and export analysis reports."""
    mdb_files = discover_mdb_files(input_dir)
    if not mdb_files:
        raise FileNotFoundError(f"No .mdb files found in: {input_dir}")

    driver_name = resolve_access_driver(driver)

    frames: list[pd.DataFrame] = []
    load_log: list[tuple[str, str, int]] = []
    for mdb_path in mdb_files:
        table_name, frame = load_best_table(mdb_path, driver_name)
        frame["source_file"] = mdb_path.name
        frames.append(frame)
        load_log.append((mdb_path.name, table_name, len(frame)))

    merged = pd.concat(frames, ignore_index=True)
    return run_pipeline_from_dataframe(merged, output_dir, load_log)


def run_pipeline_from_dataframe(
    merged: pd.DataFrame,
    output_dir: Path,
    load_log: list[tuple[str, str, int]] | None = None,
) -> dict:
    """Run analysis using an already loaded dataframe (used by tests and notebooks)."""
    prepared, meta = prepare_dataset(merged)
    report = export_reports(prepared, output_dir, meta)

    if load_log:
        print("=== MDB load summary ===")
        for name, table_name, rows in load_log:
            print(f"{name} -> table={table_name}, rows={rows}")

    print("\n=== inferred columns ===")
    for k, v in meta.items():
        print(f"{k}: {v}")

    print("\n=== result ===")
    for k, v in report["summary"].items():
        print(f"{k}: {v}")

    print(f"\nOutputs saved to: {output_dir}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Press MDB defect analysis")
    parser.add_argument("--input-dir", type=Path, default=Path("Press Profile log"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--driver", type=str, default=None, help="Optional ODBC driver name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input_dir, args.output_dir, args.driver)


if __name__ == "__main__":
    main()
