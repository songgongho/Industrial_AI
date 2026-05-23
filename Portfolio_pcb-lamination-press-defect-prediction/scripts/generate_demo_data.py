"""Generate synthetic demo data for quick testing.

Usage:
    python scripts/generate_demo_data.py --output data/demo/sample.parquet --num-cycles 100

Generates realistic synthetic Press cycles with known anomalies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.synthpress import generate_press_cycle, AnomalyType


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo dataset")
    parser.add_argument(
        "--output",
        type=str,
        default="data/demo/sample.parquet",
        help="Output file path (default: data/demo/sample.parquet)",
    )
    parser.add_argument(
        "--num-cycles",
        type=int,
        default=100,
        help="Number of synthetic cycles (default: 100)",
    )
    parser.add_argument(
        "--anomaly-prob",
        type=float,
        default=0.3,
        help="Probability of anomaly injection (default: 0.3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="parquet",
        choices=["parquet", "csv"],
        help="Output format (default: parquet)",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_cycles} synthetic Press cycles...")
    print(f"  Anomaly probability: {args.anomaly_prob:.1%}")
    print(f"  Seed: {args.seed}")

    frames = []
    labels = []
    metadata = []

    anomaly_types: list[AnomalyType | None] = [
        "temp_offset",
        "pressure_drop",
        "vacuum_leak",
        "equipment_fault",
        "power_loss",
        "program_mismatch",
        None,  # Normal cycles
    ]

    for cycle_id in range(args.num_cycles):
        if (cycle_id + 1) % max(1, args.num_cycles // 10) == 0:
            print(f"  {cycle_id + 1}/{args.num_cycles}")

        # Cycle through anomaly types
        anomaly_type = anomaly_types[cycle_id % len(anomaly_types)]

        frame, label, meta = generate_press_cycle(
            cycle_id=cycle_id + 1,
            panel_id=1000 + cycle_id,
            anomaly_type=anomaly_type,
            anomaly_prob=args.anomaly_prob,
            seed=args.seed + cycle_id,
        )

        frames.append(frame)
        labels.append(label)
        metadata.append({**meta, "cycle_id": cycle_id + 1, "label": label})

    # Concatenate all frames
    df = pd.concat(frames, ignore_index=False)
    df["label"] = labels

    # Add metadata columns
    df["anomaly_type"] = [m.get("anomaly_type") for m in metadata]
    df["cycle_id"] = [m["cycle_id"] for m in metadata]

    print(f"\nDataset shape: {df.shape}")
    print(f"Label distribution:")
    print(df["label"].value_counts().to_string())
    print(f"Anomaly types:")
    print(df["anomaly_type"].value_counts().to_string())

    # Save
    if args.output_format == "parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()

