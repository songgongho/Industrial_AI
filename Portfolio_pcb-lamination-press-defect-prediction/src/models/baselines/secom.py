"""SECOM baseline classifier utilities.

This module provides a fast, sklearn-based baseline for the public SECOM
semiconductor defect dataset. It is intentionally lightweight so it can be used
both from a CLI script and from unit tests with synthetic SECOM-like tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.loaders import load_secom
from src.eval.metrics import classification_report_dict


@dataclass(frozen=True, slots=True)
class SecomBaselineResult:
    """Container for a fitted SECOM baseline and its evaluation report."""

    model: Pipeline
    report: dict[str, Any]
    train_size: int
    test_size: int
    positive_train: int
    positive_test: int


def flatten_secom_features(x: np.ndarray) -> np.ndarray:
    """Flatten SECOM tensors to a 2D feature matrix.

    Parameters
    ----------
    x:
        Array with shape ``(n_samples, n_features, 1)`` from ``load_secom`` or
        a 2D matrix with shape ``(n_samples, n_features)``.
    """

    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 3:
        return x_arr.reshape(x_arr.shape[0], -1)
    if x_arr.ndim == 2:
        return x_arr
    raise ValueError("SECOM features must have shape (N, T, 1) or (N, T)")


def build_secom_baseline_pipeline(seed: int = 42, max_iter: int = 1000) -> Pipeline:
    """Create the sklearn pipeline used for the SECOM baseline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=max_iter,
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def _validate_binary_labels(y: np.ndarray) -> None:
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("SECOM baseline requires at least two classes in y")


def fit_secom_baseline(
    x: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    cost_fn: float = 100.0,
    cost_fp: float = 5.0,
    max_iter: int = 1000,
) -> SecomBaselineResult:
    """Fit a sklearn baseline on SECOM features and evaluate on a held-out split."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same first dimension")
    _validate_binary_labels(y_arr)

    x_flat = flatten_secom_features(x_arr)
    x_train, x_test, y_train, y_test = train_test_split(
        x_flat,
        y_arr,
        test_size=test_size,
        random_state=seed,
        stratify=y_arr,
    )

    model = build_secom_baseline_pipeline(seed=seed, max_iter=max_iter)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]
    report = classification_report_dict(
        y_test,
        y_pred,
        y_score,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
    )
    report.update(
        {
            "train_size": int(x_train.shape[0]),
            "test_size": int(x_test.shape[0]),
            "positive_train": int(np.sum(y_train)),
            "positive_test": int(np.sum(y_test)),
            "model": "StandardScaler + LogisticRegression(class_weight=balanced)",
        }
    )

    return SecomBaselineResult(
        model=model,
        report=report,
        train_size=int(x_train.shape[0]),
        test_size=int(x_test.shape[0]),
        positive_train=int(np.sum(y_train)),
        positive_test=int(np.sum(y_test)),
    )


def evaluate_secom_baseline(
    data_dir: str | Path,
    *,
    target_length: int = 128,
    test_size: float = 0.2,
    seed: int = 42,
    cost_fn: float = 100.0,
    cost_fp: float = 5.0,
    max_iter: int = 1000,
) -> SecomBaselineResult:
    """Load SECOM files from disk, fit the baseline, and return the result."""

    x, y = load_secom(data_dir, target_length=target_length)
    return fit_secom_baseline(
        x,
        y,
        test_size=test_size,
        seed=seed,
        cost_fn=cost_fn,
        cost_fp=cost_fp,
        max_iter=max_iter,
    )


def format_secom_baseline_report(
    result: SecomBaselineResult,
    *,
    data_dir: str | Path | None = None,
    target_length: int | None = None,
) -> str:
    """Render a compact markdown report for the SECOM baseline."""

    header = ["# SECOM Baseline Report", ""]
    if data_dir is not None:
        header.append(f"- data_dir: {Path(data_dir)}")
    if target_length is not None:
        header.append(f"- target_length: {target_length}")
    header.extend(
        [
            f"- train_size: {result.train_size}",
            f"- test_size: {result.test_size}",
            f"- positive_train: {result.positive_train}",
            f"- positive_test: {result.positive_test}",
            "",
            "## Metrics",
            "",
        ]
    )
    for key, value in result.report.items():
        header.append(f"- {key}: {value}")
    header.append("")
    return "\n".join(header)

