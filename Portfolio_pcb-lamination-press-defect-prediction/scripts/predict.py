"""Inference CLI for Press defect prediction.

Usage:
    python scripts/predict.py --data data/demo/sample.parquet --checkpoint outputs/model.ckpt

Outputs predictions as JSON with defect probabilities and explanations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loaders import PressDataset
from src.data.schema import PressFeatureSpec


def main():
    parser = argparse.ArgumentParser(description="Make predictions on Press data")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to data (.parquet or .csv file)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (*.ckpt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions.json",
        help="Output file path for predictions (default: outputs/predictions.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "parquet"],
        help="Output format (default: json)",
    )

    args = parser.parse_args()

    # Normalize paths
    data_path = Path(args.data)
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    # Load data
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        print(f"Error: Unsupported file format {data_path.suffix}")
        sys.exit(1)

    print(f"Loaded {len(df)} samples")

    # Load checkpoint
    print(f"Loading model from {checkpoint_path}...")
    try:
        from src.training.module import PCBStackLightningModule

        module = PCBStackLightningModule.load_from_checkpoint(
            checkpoint_path, map_location=args.device
        )
        model = module.model
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    model.eval()
    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device)
    )
    model = model.to(device)

    # Prepare predictions
    predictions = []
    feature_spec = PressFeatureSpec()
    feature_columns = feature_spec.feature_columns

    print(f"Running inference on {len(df)} samples...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx % max(1, len(df) // 10) == 0:
                print(f"  {idx}/{len(df)}")

            # Extract features
            features = torch.tensor(
                row[feature_columns].values, dtype=torch.float32
            ).unsqueeze(0)
            features = features.to(device)

            # Forward pass
            outputs = model(features, None, None)
            defect_prob = (
                outputs["defect_prob"].squeeze().cpu().item()
                if "defect_prob" in outputs
                else None
            )
            defect_type = (
                outputs["defect_type"].argmax(dim=-1).squeeze().cpu().item()
                if "defect_type" in outputs
                else None
            )
            anomaly_conf = (
                outputs["anomaly_conf"].squeeze().cpu().item()
                if "anomaly_conf" in outputs
                else None
            )

            pred_dict = {
                "sample_id": int(idx),
                "defect_probability": float(defect_prob) if defect_prob else None,
                "defect_type_id": int(defect_type) if defect_type else None,
                "anomaly_confidence": float(anomaly_conf) if anomaly_conf else None,
                "prediction": "DEFECTIVE" if defect_prob and defect_prob > 0.5 else "NORMAL",
            }
            predictions.append(pred_dict)

    # Save predictions
    print(f"Saving predictions to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "json":
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
    elif args.output_format == "csv":
        df_pred = pd.DataFrame(predictions)
        df_pred.to_csv(output_path, index=False)
    elif args.output_format == "parquet":
        df_pred = pd.DataFrame(predictions)
        df_pred.to_parquet(output_path, index=False)

    print(f"✓ Predictions saved to {output_path}")

    # Summary
    defects = sum(1 for p in predictions if p["prediction"] == "DEFECTIVE")
    print(f"\nSummary:")
    print(f"  Total samples: {len(predictions)}")
    print(f"  Defects detected: {defects} ({100 * defects / len(predictions):.1f}%)")
    print(f"  Normal samples: {len(predictions) - defects}")


if __name__ == "__main__":
    main()

