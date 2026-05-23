from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from audio_ae_utils import (
    AudioConfig,
    CNNAutoEncoder,
    build_inference_decision,
    build_dataset_bundle,
    evaluate_predictions,
    load_checkpoint,
    load_manifest_csv,
    make_loader,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_score_distribution,
    reconstruction_scores,
    save_reconstruction_examples,
)


def _base_dir() -> Path:
    return Path(__file__).resolve().parent


def _factory(config: AudioConfig) -> CNNAutoEncoder:
    return CNNAutoEncoder(latent_dim=config.latent_dim, base_channels=config.base_channels, dropout=config.dropout)


def main() -> None:
    base_dir = _base_dir()
    artifacts_dir = base_dir / "artifacts"
    model_path = artifacts_dir / "cnn_autoencoder.pt"
    manifest_path = artifacts_dir / "test_manifest.csv"

    if not model_path.exists():
        raise FileNotFoundError("학습된 모델이 없습니다. 먼저 `step2_train.py`를 실행해주세요")

    model, payload = load_checkpoint(model_path, _factory)
    config = AudioConfig(**payload["config"])
    threshold = float(payload.get("threshold", 0.0))
    calibration = payload.get("score_calibration", {})

    if manifest_path.exists():
        records = load_manifest_csv(manifest_path)
        test_loader = make_loader(records, config, shuffle=False, augment=False)
        labels_source = "saved_manifest"
    else:
        bundle = build_dataset_bundle(base_dir, config)
        test_loader = bundle.test_loader
        labels_source = "auto_discovery"

    scores, labels, paths = reconstruction_scores(model, test_loader, config.device)
    metrics_dict = evaluate_predictions(labels, scores, threshold)
    decisions = [build_inference_decision(float(score), threshold, calibration) for score in scores]
    band_counts = {}
    for item in decisions:
        band_counts[item["confidence_band"]] = band_counts.get(item["confidence_band"], 0) + 1

    eval_dir = artifacts_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(metrics_dict["confusion_matrix"], "CNN AutoEncoder Confusion Matrix", eval_dir / "confusion_matrix.png")
    plot_score_distribution(scores, labels, threshold, eval_dir / "score_distribution.png")
    plot_precision_recall_curve(scores, labels, threshold, eval_dir / "precision_recall_curve.png")
    save_reconstruction_examples(model, test_loader, config.device, eval_dir / "reconstruction_examples", max_items=4)

    summary = {
        "labels_source": labels_source,
        "threshold": threshold,
        "accuracy": metrics_dict["accuracy"],
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "f1": metrics_dict["f1"],
        "roc_auc": metrics_dict["roc_auc"],
        "confusion_matrix": metrics_dict["confusion_matrix"].tolist(),
        "band_counts": band_counts,
        "paths": paths,
    }
    (eval_dir / "evaluation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(
        {
            "filepath": paths,
            "label": labels,
            "score": scores,
            "prediction": metrics_dict["predictions"],
            "normal_probability": [item["normal_probability"] for item in decisions],
            "anomaly_probability": [item["anomaly_probability"] for item in decisions],
            "confidence": [item["confidence"] for item in decisions],
            "confidence_band": [item["confidence_band"] for item in decisions],
            "business_action": [item["business_action"] for item in decisions],
        }
    )
    df.to_csv(eval_dir / "evaluation_scores.csv", index=False, encoding="utf-8-sig")

    print("=== CNN AutoEncoder Evaluation ===")
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall: {metrics_dict['recall']:.4f}")
    print(f"F1-Score: {metrics_dict['f1']:.4f}")
    print(f"ROC-AUC: {metrics_dict['roc_auc']:.4f}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Confidence band counts: {band_counts}")
    print(f"Confusion Matrix:\n{metrics_dict['confusion_matrix']}")
    print(f"결과 저장 위치: {eval_dir}")


if __name__ == "__main__":
    main()
