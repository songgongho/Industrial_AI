"""Training entry point for the semiconductor PCB lamination scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loaders import PressDataset
from src.data.schema import PressFeatureSpec
from src.data.synthpress import generate_press_cycle
from src.training.module import PCBStackLightningModule, TrainingConfig
from pytorch_lightning.loggers import MLFlowLogger
import mlflow


def _build_synthetic_dataset(num_cycles: int, seed: int) -> TensorDataset:
    feature_columns = PressFeatureSpec().feature_columns
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for index in range(num_cycles):
        frame, label, _ = generate_press_cycle(
            cycle_id=index + 1,
            panel_id=1000 + index,
            anomaly_prob=0.5,
            seed=seed + index,
        )
        xs.append(torch.tensor(frame.loc[:, feature_columns].to_numpy(), dtype=torch.float32))
        ys.append(int(label))
    return TensorDataset(torch.stack(xs), torch.tensor(ys, dtype=torch.long))


def _build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    if args.data_path is None:
        dataset = _build_synthetic_dataset(args.synthetic_cycles, args.seed)
    else:
        dataset = PressDataset(
            args.data_path,
            labels_path=args.labels_path,
            target_length=args.target_length,
            mapping_path=getattr(args, "mapping_path", None),
        )

    if len(dataset) < 2:
        return dataset, dataset

    val_size = max(1, int(round(len(dataset) * args.val_ratio)))
    train_size = max(1, len(dataset) - val_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    return train_dataset, val_dataset


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the semiconductor PCB lamination baseline model")
    parser.add_argument("--data-path", type=Path, help="CSV/parquet file with Press cycles")
    parser.add_argument("--labels-path", type=Path, help="Optional label table path")
    parser.add_argument("--mapping-path", type=Path, help="Optional cycle->panel mapping CSV/parquet to resolve IDs")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-length", type=int, default=1024)
    parser.add_argument("--synthetic-cycles", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--use-mlflow", action="store_true", help="Enable MLflow experiment logging")
    parser.add_argument("--mlflow-experiment", type=str, default="ms_cdpnet_baseline", help="MLflow experiment name")
    parser.add_argument("--mlflow-uri", type=str, default=None, help="MLflow tracking URI (defaults to ./mlruns)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    train_dataset, val_dataset = _build_datasets(args)
    train_loader = _make_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, args.batch_size, shuffle=False)

    feature_dim = len(PressFeatureSpec().feature_columns)
    module = PCBStackLightningModule(
        TrainingConfig(
            max_epochs=args.epochs,
            batch_size=args.batch_size,
            ts_input_dim=feature_dim,
            event_input_dim=feature_dim,
        )
    )
    # Configure MLflow logger if requested
    trainer_logger = False
    if args.use_mlflow:
        tracking_uri = args.mlflow_uri or f"file:{Path.cwd() / 'mlruns'}"
        mlflow.set_tracking_uri(tracking_uri)
        mlf_logger = MLFlowLogger(experiment_name=args.mlflow_experiment, tracking_uri=tracking_uri)
        trainer_logger = mlf_logger
        print(f"MLflow logging enabled. Tracking URI: {tracking_uri}, experiment: {args.mlflow_experiment}")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        logger=trainer_logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(module, train_loader, val_loader)
    print("Training finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

