from pathlib import Path

from src.data.loaders import PressDataset, load_press_samples, resample_cycle, split_by_group
from src.data.schema import DatasetSplitSpec, PressFeatureSpec
from src.data.synthpress import generate_press_cycle


def _write_cycle_csv(tmp_path: Path, label: int = 1) -> Path:
    frame, _, _ = generate_press_cycle(1, 10, anomaly_prob=0.0, seed=7)
    frame = frame.copy()
    frame["label"] = label
    path = tmp_path / "press.csv"
    frame.to_csv(path, index=False)
    return path


def test_resample_cycle_fixed_length() -> None:
    frame = generate_press_cycle(99, 77, anomaly_prob=0.0, seed=11)[0][["t_ms", "POINT1", "POINT2"]]
    out = resample_cycle(frame, ("POINT1", "POINT2"), target_length=5)
    assert out.shape == (5, 2)
    assert out[0, 0] != out[-1, 0]


def test_load_press_samples_and_dataset(tmp_path: Path) -> None:
    path = _write_cycle_csv(tmp_path, label=1)
    samples = load_press_samples(path, target_length=128)
    assert len(samples) == 1
    assert samples[0].x.shape == (128, len(PressFeatureSpec().feature_columns))
    dataset = PressDataset(path, target_length=64)
    x, y = dataset[0]
    assert x.shape == (64, len(PressFeatureSpec().feature_columns))
    assert y == 1


def test_split_by_group_is_leak_free() -> None:
    groups = [1, 1, 2, 2, 3, 3, 4, 4]
    split = split_by_group(groups, DatasetSplitSpec(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25), seed=1)
    all_groups = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert all_groups == {1, 2, 3, 4}
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])

