"""Header mapping utility for data_R2.csv

Reads the raw CSV (which lacks a header), applies a recommended header mapping
suitable for the shared-facility domain, and writes an interim CSV with headers.

Usage:
    python src/data_loader/loader.py  # uses default paths
    python src/data_loader/loader.py --input data/raw/data_R2.csv --output data/interim/data_with_header.csv
"""

from pathlib import Path
import argparse
import pandas as pd
import os
import sys


RECOMMENDED_MAPPING = {
    0: 'rec_id',
    1: 'device_id',
    2: 'timestamp',
    3: 'phase_a_current_a',
    4: 'phase_b_current_a',
    5: 'phase_c_current_a',
    6: 'phase_a_voltage_v',
    7: 'phase_b_voltage_v',
    8: 'phase_c_voltage_v',
    9: 'voltage_avg_v',
    10: 'p_a_w',
    11: 'p_b_w',
    12: 'p_c_w',
    13: 'power_factor',
    14: 'angle_deg',
    15: 'frequency_hz',
    16: 'active_power_w',
    17: 'reactive_power_var',
    18: 'apparent_power_va',
    19: 'energy_import_wh',
    20: 'energy_export_wh',
    21: 'energy_total_wh',
    22: 'status_flag',
    23: 'temp_c',
    24: 'humidity_pct',
}


def map_headers(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Apply header mapping to a DataFrame read with header=None.

    mapping: dict where keys are integer original column positions and values
    are recommended column names.
    """
    # Only rename columns that exist in df
    available_map = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=available_map)
    # For any remaining unnamed columns, give placeholder names
    for col in df.columns:
        if isinstance(col, int):
            df = df.rename(columns={col: f'col_{col}'})
    return df


def main(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        print(f'Error: input file not found: {input_path}', file=sys.stderr)
        sys.exit(2)

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # read raw CSV without header; treat '\\N' as NA
    df = pd.read_csv(input_path, header=None, na_values=['\\N', '\\N.1'])

    df = map_headers(df, RECOMMENDED_MAPPING)

    # basic normalization: strip whitespace in device_id and timestamp
    if 'device_id' in df.columns:
        df['device_id'] = df['device_id'].astype(str).str.strip()
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(str).str.strip()

    # write interim CSV with header
    df.to_csv(output_path, index=False)

    print(f'Wrote mapped CSV to: {output_path}')
    print('\nSample header and first rows:')
    print(df.head().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Map header for data_R2.csv and write interim file')
    parser.add_argument('--input', '-i', default='data/data_R2.csv', help='path to raw CSV')
    parser.add_argument('--output', '-o', default='data/interim/data_with_header.csv', help='path to write mapped CSV')
    args = parser.parse_args()
    main(args.input, args.output)

