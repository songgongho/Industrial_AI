from __future__ import annotations

import warnings
from pathlib import Path

import librosa.display
import matplotlib.pyplot as plt

from audio_ae_utils import AudioConfig, build_dataset_bundle, load_mel_spectrogram


warnings.filterwarnings("ignore")
plt.style.use("ggplot")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def _find_base_dir() -> Path:
    return Path(__file__).resolve().parent


def _plot_example_spectrograms(bundle, config: AudioConfig) -> None:
    figures_dir = Path("visualizations") / "step1"
    figures_dir.mkdir(parents=True, exist_ok=True)

    normal_record = next((r for r in bundle.records_test if r.label == 0), None)
    abnormal_record = next((r for r in bundle.records_test if r.label == 1), None)

    if normal_record is None:
        normal_record = bundle.records_train[0] if bundle.records_train else None
    if normal_record is not None:
        normal_spec = load_mel_spectrogram(normal_record.filepath, config)
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(normal_spec[0], x_axis="time", y_axis="mel", sr=config.sample_rate, hop_length=config.hop_length, cmap="magma")
        plt.colorbar()
        plt.title("Normal Mel-Spectrogram")
        plt.tight_layout()
        plt.savefig(figures_dir / "normal_example.png", dpi=160)
        plt.close()

    if abnormal_record is not None:
        abnormal_spec = load_mel_spectrogram(abnormal_record.filepath, config)
        plt.figure(figsize=(6, 4))
        librosa.display.specshow(abnormal_spec[0], x_axis="time", y_axis="mel", sr=config.sample_rate, hop_length=config.hop_length, cmap="magma")
        plt.colorbar()
        plt.title("Abnormal Mel-Spectrogram")
        plt.tight_layout()
        plt.savefig(figures_dir / "abnormal_example.png", dpi=160)
        plt.close()


def main() -> None:
    base_dir = _find_base_dir()
    config = AudioConfig(batch_size=32, seed=42)
    bundle = build_dataset_bundle(base_dir, config)

    train_dataset = bundle.train_dataset
    test_dataset = bundle.test_dataset
    train_loader = bundle.train_loader
    test_loader = bundle.test_loader

    train_total = len(train_dataset)
    test_total = len(test_dataset)
    total_normal = bundle.label_counts["train_normal"] + bundle.label_counts["test_normal"]
    total_abnormal = bundle.label_counts["test_abnormal"]
    test_normal = bundle.label_counts["test_normal"]
    test_abnormal = bundle.label_counts["test_abnormal"]

    print(f"데이터 소스 모드: {bundle.source_mode}")
    print(f"train_dataset 크기: {train_total}")
    print(f"test_dataset 크기: {test_total}")
    print(f"전체 정상:비정상 비율 = {total_normal}:{total_abnormal}")
    print(f"test 정상:비정상 비율 = {test_normal}:{test_abnormal}")
    print(f"train_loader batch_size: {train_loader.batch_size}")
    print(f"test_loader batch_size: {test_loader.batch_size}")

    if train_total > 0:
        print(f"학습 샘플 예시 경로: {bundle.records_train[0].filepath}")
    if test_total > 0:
        print(f"평가 샘플 예시 경로: {bundle.records_test[0].filepath}")

    _plot_example_spectrograms(bundle, config)

    globals()["train_dataset"] = train_dataset
    globals()["test_dataset"] = test_dataset
    globals()["train_loader"] = train_loader
    globals()["test_loader"] = test_loader


if __name__ == "__main__":
    main()
