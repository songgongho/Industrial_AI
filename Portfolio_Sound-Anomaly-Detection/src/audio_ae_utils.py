from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 512
    fixed_frames: int = 128
    batch_size: int = 32
    num_workers: int = 0
    latent_dim: int = 64
    base_channels: int = 32
    dropout: float = 0.25
    ssim_weight: float = 0.2
    mse_weight: float = 0.8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    max_threshold_points: int = 200
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class AudioRecord:
    filepath: str
    label: int
    split: str


@dataclass
class DatasetBundle:
    train_dataset: "AudioSpectrogramDataset"
    test_dataset: "AudioSpectrogramDataset"
    train_loader: DataLoader
    test_loader: DataLoader
    records_train: List[AudioRecord]
    records_test: List[AudioRecord]
    label_counts: dict
    source_mode: str


@dataclass
class TrainingResult:
    history: List[dict]
    best_val_loss: float
    best_state_dict: dict


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sorted_wavs(folder: Path) -> List[str]:
    if not folder.exists():
        return []
    return sorted(str(p) for p in folder.glob("*.wav"))


def discover_audio_records(base_dir: str | Path, seed: int = 42, test_size: float = 0.2) -> Tuple[List[AudioRecord], List[AudioRecord], str]:
    base_path = Path(base_dir)
    modern_train = base_path / "data" / "train" / "normal"
    modern_test_normal = base_path / "data" / "test" / "normal"
    modern_test_abnormal = base_path / "data" / "test" / "abnormal"

    if modern_train.exists() or modern_test_normal.exists() or modern_test_abnormal.exists():
        train_paths = _sorted_wavs(modern_train)
        test_paths = _sorted_wavs(modern_test_normal) + _sorted_wavs(modern_test_abnormal)
        records_train = [AudioRecord(filepath=p, label=0, split="train") for p in train_paths]
        records_test = [AudioRecord(filepath=p, label=0, split="test") for p in _sorted_wavs(modern_test_normal)] + [
            AudioRecord(filepath=p, label=1, split="test") for p in _sorted_wavs(modern_test_abnormal)
        ]
        return records_train, records_test, "modern"

    ok_paths = _sorted_wavs(base_path / "FAN_sound_OK")
    err_paths = _sorted_wavs(base_path / "FAN_sound_error")
    if not ok_paths and not err_paths:
        raise FileNotFoundError(
            "오디오 파일을 찾을 수 없습니다. 'data/train/normal' 또는 'FAN_sound_OK' 구조를 확인해주세요"
        )

    if len(ok_paths) > 1:
        train_ok, test_ok = train_test_split(ok_paths, test_size=test_size, random_state=seed, shuffle=True)
    else:
        train_ok, test_ok = ok_paths, []

    records_train = [AudioRecord(filepath=p, label=0, split="train") for p in train_ok]
    records_test = [AudioRecord(filepath=p, label=0, split="test") for p in test_ok] + [
        AudioRecord(filepath=p, label=1, split="test") for p in err_paths
    ]
    return records_train, records_test, "legacy"


def _pad_or_crop_time(spec: np.ndarray, fixed_frames: int) -> np.ndarray:
    current = spec.shape[-1]
    if current == fixed_frames:
        return spec
    if current > fixed_frames:
        start = (current - fixed_frames) // 2
        return spec[..., start : start + fixed_frames]
    pad_left = (fixed_frames - current) // 2
    pad_right = fixed_frames - current - pad_left
    return np.pad(spec, ((0, 0), (pad_left, pad_right)), mode="constant")


def _minmax_scale(spec: np.ndarray) -> np.ndarray:
    min_v = spec.min()
    max_v = spec.max()
    denom = max(max_v - min_v, 1e-8)
    return (spec - min_v) / denom


def load_mel_spectrogram(path: str | Path, config: AudioConfig, augment: bool = False) -> np.ndarray:
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    if y.size == 0:
        y = np.zeros(config.sample_rate, dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=2.0,
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = _pad_or_crop_time(mel, config.fixed_frames)
    mel = _minmax_scale(mel)

    if augment:
        mel = _apply_spec_augmentation(mel)

    mel = mel.astype(np.float32)
    return mel[np.newaxis, :, :]


def _apply_spec_augmentation(spec: np.ndarray) -> np.ndarray:
    aug = spec.copy()
    if np.random.rand() < 0.5:
        aug = np.clip(aug + np.random.normal(0.0, 0.02, size=aug.shape), 0.0, 1.0)
    if np.random.rand() < 0.3:
        width = np.random.randint(5, 15)
        start = np.random.randint(0, max(1, aug.shape[1] - width))
        aug[:, start : start + width] = 0.0
    if np.random.rand() < 0.3:
        width = np.random.randint(5, 15)
        start = np.random.randint(0, max(1, aug.shape[2] - width))
        aug[:, :, start : start + width] = 0.0
    return np.clip(aug, 0.0, 1.0)


class AudioSpectrogramDataset(Dataset):
    def __init__(self, records: Sequence[AudioRecord], config: AudioConfig, augment: bool = False):
        self.records = list(records)
        self.config = config
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        spec = load_mel_spectrogram(record.filepath, self.config, augment=self.augment)
        tensor = torch.from_numpy(spec)
        label = torch.tensor(record.label, dtype=torch.float32)
        return tensor, label, record.filepath


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float, stride: int = 2, final: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=1 if stride == 2 else 0,
                bias=False,
            )
        ]
        if not final:
            layers.extend([nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Dropout2d(dropout)])
        else:
            layers.append(nn.Sigmoid())
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, dropout: float = 0.25, input_size: Tuple[int, int, int] = (1, 128, 128)):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.dropout = dropout
        self.input_size = input_size

        c = base_channels
        self.encoder = nn.Sequential(
            ConvBlock(1, c, dropout, stride=2),
            ConvBlock(c, c * 2, dropout, stride=2),
            ConvBlock(c * 2, c * 4, dropout, stride=2),
            ConvBlock(c * 4, c * 8, dropout, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            encoded = self.encoder(dummy)
            self._encoded_shape = encoded.shape[1:]
            self._flat_dim = int(np.prod(self._encoded_shape))

        self.fc_enc = nn.Linear(self._flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self._flat_dim)

        self.decoder = nn.Sequential(
            DeconvBlock(c * 8, c * 4, dropout, stride=2),
            DeconvBlock(c * 4, c * 2, dropout, stride=2),
            DeconvBlock(c * 2, c, dropout, stride=2),
            DeconvBlock(c, 1, dropout, stride=2, final=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.size(0), *self._encoded_shape)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 7, c1: float = 0.01 ** 2, c2: float = 0.03 ** 2):
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {pred.shape} and target shape {target.shape} must match")

        mu_x = F.avg_pool2d(pred, self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(target, self.window_size, stride=1, padding=self.window_size // 2)

        sigma_x = F.avg_pool2d(pred * pred, self.window_size, stride=1, padding=self.window_size // 2) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(target * target, self.window_size, stride=1, padding=self.window_size // 2) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(pred * target, self.window_size, stride=1, padding=self.window_size // 2) - mu_x * mu_y

        ssim_map = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + self.c1) * (sigma_x + sigma_y + self.c2)
        )
        return 1.0 - ssim_map.mean()


class ReconstructionLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.8, ssim_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


def build_dataset_bundle(base_dir: str | Path, config: Optional[AudioConfig] = None) -> DatasetBundle:
    config = config or AudioConfig()
    set_seed(config.seed)
    records_train, records_test, source_mode = discover_audio_records(base_dir, seed=config.seed, test_size=config.test_size)

    train_dataset = AudioSpectrogramDataset(records_train, config, augment=False)
    test_dataset = AudioSpectrogramDataset(records_test, config, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    label_counts = {
        "train_normal": sum(r.label == 0 for r in records_train),
        "test_normal": sum(r.label == 0 for r in records_test),
        "test_abnormal": sum(r.label == 1 for r in records_test),
    }
    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        records_train=records_train,
        records_test=records_test,
        label_counts=label_counts,
        source_mode=source_mode,
    )


def split_train_validation(records: Sequence[AudioRecord], config: AudioConfig, val_ratio: float = 0.2) -> Tuple[List[AudioRecord], List[AudioRecord]]:
    if len(records) < 2:
        return list(records), []
    train_records, val_records = train_test_split(list(records), test_size=val_ratio, random_state=config.seed, shuffle=True)
    return list(train_records), list(val_records)


def make_loader(records: Sequence[AudioRecord], config: AudioConfig, shuffle: bool, augment: bool = False) -> DataLoader:
    dataset = AudioSpectrogramDataset(records, config, augment=augment)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers, pin_memory=torch.cuda.is_available())


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 30,
    early_stopping_patience: int = 7,
) -> TrainingResult:
    history: List[dict] = []
    best_val_loss = math.inf
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= max(len(train_loader.dataset), 1)

        val_loss = train_loss
        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(device)
                    recon = model(x)
                    loss = criterion(recon, x)
                    total += loss.item() * x.size(0)
            val_loss = total / max(len(val_loader.dataset), 1)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                break

    model.load_state_dict(best_state)
    return TrainingResult(history=history, best_val_loss=best_val_loss, best_state_dict=best_state)


@torch.no_grad()
def reconstruction_scores(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    model.to(device)
    scores: List[float] = []
    labels: List[int] = []
    paths: List[str] = []

    for batch in loader:
        x, y, batch_paths = batch
        x = x.to(device)
        recon = model(x)
        batch_scores = torch.mean((recon - x) ** 2, dim=(1, 2, 3)).detach().cpu().numpy()
        scores.extend(batch_scores.tolist())
        labels.extend(y.numpy().astype(int).tolist())
        paths.extend(list(batch_paths))

    return np.asarray(scores), np.asarray(labels), paths


def find_best_threshold(scores: np.ndarray, labels: np.ndarray, max_points: int = 200) -> Tuple[float, dict]:
    if scores.size == 0:
        raise ValueError("threshold를 찾기 위한 점수가 비어 있습니다")

    lo, hi = float(scores.min()), float(scores.max())
    if math.isclose(lo, hi):
        threshold = lo
        preds = (scores > threshold).astype(int)
        return threshold, evaluate_predictions(labels, scores, threshold)

    candidates = np.linspace(lo, hi, num=min(max_points, max(2, scores.size * 2)))
    best_threshold = candidates[0]
    best_metrics = None
    best_f1 = -1.0
    for threshold in candidates:
        preds = (scores > threshold).astype(int)
        f1 = metrics.f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
            best_metrics = evaluate_predictions(labels, scores, best_threshold)
    return best_threshold, best_metrics or evaluate_predictions(labels, scores, best_threshold)


def evaluate_predictions(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores > threshold).astype(int)
    cm = metrics.confusion_matrix(labels, preds, labels=[0, 1])
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)
    accuracy = metrics.accuracy_score(labels, preds)
    roc_auc = metrics.roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "confusion_matrix": cm,
        "predictions": preds,
    }


def find_best_threshold_by_pr_curve(scores: np.ndarray, labels: np.ndarray, beta: float = 1.0) -> Tuple[float, dict]:
    if scores.size == 0:
        raise ValueError("threshold를 찾기 위한 점수가 비어 있습니다")

    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    if thresholds.size == 0:
        threshold = float(scores.mean())
        report = evaluate_predictions(labels, scores, threshold)
        report.update(
            {
                "precision_curve": precision.tolist(),
                "recall_curve": recall.tolist(),
                "threshold_curve": [],
                "fbeta_curve": [],
                "best_index": 0,
            }
        )
        return threshold, report

    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    fbeta = np.where(denom > 0, (1 + beta_sq) * precision * recall / denom, 0.0)
    best_index = int(np.argmax(fbeta))
    threshold = float(thresholds[best_index])
    report = evaluate_predictions(labels, scores, threshold)
    report.update(
        {
            "precision_curve": precision.tolist(),
            "recall_curve": recall.tolist(),
            "threshold_curve": thresholds.tolist(),
            "fbeta_curve": fbeta.tolist(),
            "best_index": best_index,
        }
    )
    return threshold, report


def build_confidence_calibration(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {"mean": 0.0, "std": 1e-3, "q10": 0.0, "q25": 0.0, "q75": 0.0, "q90": 0.0, "scale": 1e-3}

    q10, q25, q75, q90 = np.percentile(scores, [10, 25, 75, 90]).astype(float)
    std = float(np.std(scores))
    iqr = float(max(q75 - q25, 1e-6))
    scale = max(std, iqr / 1.349, 1e-3)
    return {
        "mean": float(np.mean(scores)),
        "std": float(std),
        "q10": float(q10),
        "q25": float(q25),
        "q75": float(q75),
        "q90": float(q90),
        "scale": float(scale),
    }


def score_to_probabilities(score: float, threshold: float, calibration: Optional[dict] = None) -> np.ndarray:
    scale = 1e-3
    if calibration:
        scale = float(calibration.get("scale") or calibration.get("std") or scale)
        if scale <= 0:
            scale = 1e-3
    else:
        scale = max(abs(threshold) * 0.15, 1e-3)

    anomaly_logit = (float(score) - float(threshold)) / scale
    anomaly_prob = 1.0 / (1.0 + math.exp(-anomaly_logit))
    normal_prob = 1.0 - anomaly_prob
    return np.asarray([normal_prob, anomaly_prob], dtype=np.float32)


def assign_confidence_band(confidence: float, high_threshold: float = 0.95, uncertain_threshold: float = 0.70) -> str:
    if confidence >= high_threshold:
        return "high_confidence"
    if confidence >= uncertain_threshold:
        return "uncertain"
    return "low_confidence"


def build_inference_decision(score: float, threshold: float, calibration: Optional[dict] = None) -> dict:
    probabilities = score_to_probabilities(score, threshold, calibration)
    prediction = int(np.argmax(probabilities))
    confidence = float(probabilities[prediction])
    anomaly_probability = float(probabilities[1])
    band = assign_confidence_band(confidence)
    if band == "high_confidence":
        action = "auto_alert" if prediction == 1 else "auto_accept"
    else:
        action = "manual_review"
    return {
        "prediction": prediction,
        "normal_probability": float(probabilities[0]),
        "anomaly_probability": anomaly_probability,
        "confidence": confidence,
        "confidence_band": band,
        "business_action": action,
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str | Path) -> None:
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Normal", "Abnormal"], yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_score_distribution(scores: np.ndarray, labels: np.ndarray, threshold: float, save_path: str | Path) -> None:
    plt.figure(figsize=(7.5, 4.5))
    sns.histplot(scores[labels == 0], color="royalblue", label="Normal", kde=True, stat="density", bins=24, alpha=0.45)
    if np.any(labels == 1):
        sns.histplot(scores[labels == 1], color="crimson", label="Abnormal", kde=True, stat="density", bins=24, alpha=0.45)
    plt.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold:.4f}")
    plt.xlabel("Reconstruction error")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_precision_recall_curve(scores: np.ndarray, labels: np.ndarray, threshold: float, save_path: str | Path) -> None:
    precision, recall, thresholds = metrics.precision_recall_curve(labels, scores)
    plt.figure(figsize=(6.5, 5))
    plt.plot(recall, precision, color="darkorange", linewidth=2, label="PR curve")
    if thresholds.size > 0:
        best_precision, best_recall, _ = precision[:-1], recall[:-1], thresholds
        f1 = np.where((best_precision + best_recall) > 0, 2 * best_precision * best_recall / (best_precision + best_recall), 0.0)
        best_idx = int(np.argmax(f1))
        plt.scatter([best_recall[best_idx]], [best_precision[best_idx]], color="red", s=70, label=f"Best F1 threshold={threshold:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()


@torch.no_grad()
def save_reconstruction_examples(model: nn.Module, loader: DataLoader, device: str, save_dir: str | Path, max_items: int = 4) -> None:
    model.eval()
    model.to(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in loader:
        x, y, paths = batch
        x = x.to(device)
        recon = model(x)
        x_np = x.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()
        for i in range(x_np.shape[0]):
            if count >= max_items:
                return
            fig, axes = plt.subplots(1, 3, figsize=(10, 3))
            axes[0].imshow(x_np[i, 0], aspect="auto", origin="lower", cmap="magma")
            axes[0].set_title(f"Input {int(y[i].item())}")
            axes[1].imshow(recon_np[i, 0], aspect="auto", origin="lower", cmap="magma")
            axes[1].set_title("Reconstruction")
            diff = np.abs(x_np[i, 0] - recon_np[i, 0])
            axes[2].imshow(diff, aspect="auto", origin="lower", cmap="viridis")
            axes[2].set_title("Abs Diff")
            for ax in axes:
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(save_dir / f"reconstruction_{count}.png", dpi=160)
            plt.close()
            count += 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: AudioConfig,
    threshold: float,
    history: List[dict],
    source_mode: str,
    extra_metadata: Optional[dict] = None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
        "config": asdict(config),
        "threshold": float(threshold),
        "history": history,
        "source_mode": source_mode,
    }
    if extra_metadata:
        payload.update(extra_metadata)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model_factory) -> Tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = AudioConfig(**payload["config"])
    model = model_factory(config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def write_manifest_csv(records: Sequence[AudioRecord], csv_path: str | Path) -> None:
    import pandas as pd

    df = pd.DataFrame([{"filepath": r.filepath, "label": r.label, "split": r.split} for r in records])
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def load_manifest_csv(csv_path: str | Path) -> List[AudioRecord]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return [AudioRecord(filepath=row["filepath"], label=int(row["label"]), split=row.get("split", "unknown")) for _, row in df.iterrows()]

