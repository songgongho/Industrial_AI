"""Tests for the sklearn-based SECOM baseline."""

from __future__ import annotations

import numpy as np
import pytest

import src.models.baselines.secom as secom_module
from src.models.baselines.secom import (
    SecomBaselineResult,
    evaluate_secom_baseline,
    fit_secom_baseline,
    flatten_secom_features,
    format_secom_baseline_report,
)


def _make_synthetic_secom_like_data(n_samples: int = 60, n_features: int = 16) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 1.0, size=(n_samples, n_features, 1))
    y = np.zeros(n_samples, dtype=int)
    y[n_samples // 2 :] = 1
    x[n_samples // 2 :] += 2.5
    return x, y


def test_flatten_secom_features_accepts_3d_and_2d() -> None:
    x3d = np.zeros((4, 3, 1), dtype=float)
    x2d = np.zeros((4, 3), dtype=float)

    flat3d = flatten_secom_features(x3d)
    flat2d = flatten_secom_features(x2d)

    assert flat3d.shape == (4, 3)
    assert flat2d.shape == (4, 3)


def test_flatten_secom_features_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError):
        flatten_secom_features(np.zeros((4,), dtype=float))


def test_fit_secom_baseline_returns_useful_report() -> None:
    x, y = _make_synthetic_secom_like_data()

    result = fit_secom_baseline(x, y, test_size=0.25, seed=7)

    assert isinstance(result, SecomBaselineResult)
    assert result.train_size == 45
    assert result.test_size == 15
    assert result.positive_train + result.positive_test == int(y.sum())
    assert result.report["n_samples"] == 15
    assert result.report["auroc"] is not None
    assert result.report["auroc"] > 0.95
    assert result.report["far_at_recall_0_9"] is not None


def test_format_secom_baseline_report_contains_key_fields() -> None:
    x, y = _make_synthetic_secom_like_data()
    result = fit_secom_baseline(x, y, test_size=0.25, seed=7)

    text = format_secom_baseline_report(result, data_dir="data/raw/secom", target_length=128)

    assert "# SECOM Baseline Report" in text
    assert "data_dir:" in text
    assert "target_length: 128" in text
    assert "auroc" in text


def test_evaluate_secom_baseline_rejects_single_class() -> None:
    x = np.zeros((10, 8, 1), dtype=float)
    y = np.zeros(10, dtype=int)

    with pytest.raises(ValueError):
        fit_secom_baseline(x, y)


def test_evaluate_secom_baseline_uses_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = _make_synthetic_secom_like_data()

    def fake_loader(data_dir: str | bytes | object, target_length: int = 128) -> tuple[np.ndarray, np.ndarray]:
        assert target_length == 128
        return x, y

    monkeypatch.setattr(secom_module, "load_secom", fake_loader)

    result = evaluate_secom_baseline("data/raw/secom", target_length=128, seed=7)

    assert isinstance(result, SecomBaselineResult)
    assert result.report["n_samples"] == result.test_size


