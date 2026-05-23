from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from audio_ae_utils import (
    AudioConfig,
    CNNAutoEncoder,
    ReconstructionLoss,
    build_dataset_bundle,
    build_confidence_calibration,
    find_best_threshold_by_pr_curve,
    make_loader,
    reconstruction_scores,
    save_checkpoint,
    set_seed,
    train_autoencoder,
    write_manifest_csv,
)


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    base_dir = _base_dir()
    config = AudioConfig(latent_dim=96, base_channels=32, dropout=0.25, epochs=25, batch_size=32, lr=1e-3, ssim_weight=0.2, mse_weight=0.8)
    set_seed(config.seed)

    bundle = build_dataset_bundle(base_dir, config)
    if len(bundle.records_train) == 0:
        raise RuntimeError("학습용 정상 데이터가 없습니다")

    train_records, val_records = train_test_split(
        bundle.records_train,
        test_size=min(config.val_size, 0.3),
        random_state=config.seed,
        shuffle=True,
    ) if len(bundle.records_train) > 1 else (bundle.records_train, [])

    if len(bundle.records_test) > 1:
        try:
            threshold_records, final_test_records = train_test_split(
                bundle.records_test,
                test_size=0.5,
                random_state=config.seed,
                shuffle=True,
                stratify=[r.label for r in bundle.records_test],
            )
        except ValueError:
            threshold_records, final_test_records = train_test_split(
                bundle.records_test,
                test_size=0.5,
                random_state=config.seed,
                shuffle=True,
            )
    else:
        threshold_records = bundle.records_test
        final_test_records = bundle.records_test

    artifacts_dir = base_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    write_manifest_csv(train_records, artifacts_dir / "train_manifest.csv")
    write_manifest_csv(val_records, artifacts_dir / "val_manifest.csv")
    write_manifest_csv(threshold_records, artifacts_dir / "threshold_manifest.csv")
    write_manifest_csv(final_test_records, artifacts_dir / "test_manifest.csv")

    train_loader = make_loader(train_records, config, shuffle=True, augment=False)
    val_loader = make_loader(val_records, config, shuffle=False, augment=False) if len(val_records) > 0 else None
    threshold_loader = make_loader(threshold_records, config, shuffle=False, augment=False) if len(threshold_records) > 0 else None

    model = CNNAutoEncoder(latent_dim=config.latent_dim, base_channels=config.base_channels, dropout=config.dropout)
    criterion = ReconstructionLoss(mse_weight=config.mse_weight, ssim_weight=config.ssim_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    print(f"데이터 소스 모드: {bundle.source_mode}")
    print(f"학습 샘플 수: {len(train_records)}")
    print(f"검증 샘플 수: {len(val_records)}")
    print(f"threshold 후보 샘플 수: {len(threshold_records)}")
    print(f"사용 장치: {config.device}")

    result = train_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config.device,
        epochs=config.epochs,
        early_stopping_patience=7,
    )

    threshold = 0.0
    threshold_metrics = {}
    score_calibration = {}
    if threshold_loader is not None:
        threshold_scores, threshold_labels, _ = reconstruction_scores(model, threshold_loader, config.device)
        threshold, threshold_metrics = find_best_threshold_by_pr_curve(threshold_scores, threshold_labels)
        score_calibration = build_confidence_calibration(threshold_scores)
    else:
        threshold = float(result.best_val_loss)

    model_path = artifacts_dir / "cnn_autoencoder.pt"
    save_checkpoint(
        model_path,
        model,
        config,
        threshold,
        result.history,
        bundle.source_mode,
        extra_metadata={"score_calibration": score_calibration, "threshold_report": threshold_metrics},
    )

    history_df = pd.DataFrame(result.history)
    history_df.to_csv(artifacts_dir / "train_history.csv", index=False, encoding="utf-8-sig")

    serializable_threshold_metrics = {}
    if threshold_metrics:
        serializable_threshold_metrics = {
            "threshold": float(threshold_metrics.get("threshold", threshold)),
            "accuracy": float(threshold_metrics.get("accuracy", 0.0)),
            "precision": float(threshold_metrics.get("precision", 0.0)),
            "recall": float(threshold_metrics.get("recall", 0.0)),
            "f1": float(threshold_metrics.get("f1", 0.0)),
            "roc_auc": float(threshold_metrics.get("roc_auc", float("nan"))),
            "confusion_matrix": threshold_metrics.get("confusion_matrix").tolist(),
        }

    summary = {
        "threshold": float(threshold),
        "best_val_loss": float(result.best_val_loss),
        "threshold_metrics": serializable_threshold_metrics,
        "score_calibration": score_calibration,
        "config": config.__dict__,
    }
    (artifacts_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"모델 저장 완료: {model_path}")
    print(f"선택된 threshold: {threshold:.6f}")
    if threshold_metrics:
        print(f"threshold 선택용 F1: {threshold_metrics['f1']:.4f}")
        print(f"threshold 선택용 Confusion Matrix:\n{threshold_metrics['confusion_matrix']}")
    print(f"best_val_loss: {result.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
