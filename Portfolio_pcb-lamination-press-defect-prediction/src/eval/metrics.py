"""Evaluation metrics for the thesis project."""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from typing import Any

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def cost_aware_detection_score(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    cost_fn: float,
    cost_fp: float,
) -> float:
    """Return mean misclassification cost per sample."""

    y_true_arr = [int(value) for value in y_true]
    y_pred_arr = [int(value) for value in y_pred]
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same shape")

    false_negative = sum(
        1 for true, pred in zip(y_true_arr, y_pred_arr) if true == 1 and pred == 0
    )
    false_positive = sum(
        1 for true, pred in zip(y_true_arr, y_pred_arr) if true == 0 and pred == 1
    )
    total_cost = false_negative * float(cost_fn) + false_positive * float(cost_fp)
    return float(total_cost / max(len(y_true_arr), 1))


def far_at_recall(
    y_true: Iterable[int],
    y_score: Iterable[float],
    target_recall: float = 0.9,
) -> float | None:
    """False Alarm Rate at the smallest threshold satisfying recall >= target_recall.

    FAR = FP / (FP + TN) = 1 - specificity, evaluated at the operating point
    that minimises FAR among all thresholds reaching at least ``target_recall``.

    Returns ``None`` if positives or negatives are missing, or if no threshold
    achieves the target recall.

    Parameters
    ----------
    y_true:
        Iterable of binary ground-truth labels (0/1).
    y_score:
        Iterable of model scores (higher = more positive).
    target_recall:
        Recall (TPR) constraint in the half-open interval ``(0, 1]``.
    """

    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    if y_true_arr.shape != y_score_arr.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if not (0.0 < target_recall <= 1.0):
        raise ValueError("target_recall must be in (0, 1]")
    n_pos = int(y_true_arr.sum())
    if n_pos == 0 or n_pos == y_true_arr.size:
        return None

    # roc_curve returns FPR (=FAR), TPR (=recall), thresholds sorted by descending threshold
    fpr, tpr, _ = roc_curve(y_true_arr, y_score_arr)
    mask = tpr >= target_recall
    if not mask.any():
        return None
    return float(fpr[mask].min())


def _binary_counts(y_true: list[int], y_pred: list[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for true, pred in zip(y_true, y_pred, strict=False):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 1 and pred == 0:
            fn += 1
    return tp, fp, tn, fn


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    """Return Area Under ROC Curve using sklearn.metrics.roc_auc_score.

    Returns None if positives or negatives are missing.
    """
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    return float(roc_auc_score(y_true, y_score))


def classification_report_dict(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    y_score: Iterable[float] | None = None,
    *,
    cost_fn: float = 100.0,
    cost_fp: float = 5.0,
) -> dict[str, Any]:
    """Return a compact dictionary of common binary-classification metrics."""

    y_true_arr = [int(value) for value in y_true]
    y_pred_arr = [int(value) for value in y_pred]
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true and y_pred must have the same shape")

    y_score_arr = None if y_score is None else [float(value) for value in y_score]
    if y_score_arr is not None and len(y_score_arr) != len(y_true_arr):
        raise ValueError("y_score must have the same shape as y_true")

    tp, fp, tn, fn = _binary_counts(y_true_arr, y_pred_arr)
    accuracy = _safe_div(tp + tn, len(y_true_arr))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    report: dict[str, Any] = {
        "n_samples": int(len(y_true_arr)),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "cost_aware_score": float(
            cost_aware_detection_score(
                y_true_arr, y_pred_arr, cost_fn=cost_fn, cost_fp=cost_fp
            )
        ),
    }

    if y_score_arr is not None:
        report["auroc"] = _binary_auroc(y_true_arr, y_score_arr)
        report["far_at_recall_0_9"] = far_at_recall(y_true_arr, y_score_arr, 0.9)
    else:
        report["auroc"] = None
        report["far_at_recall_0_9"] = None

    if isinstance(report["auroc"], float) and not isfinite(report["auroc"]):
        report["auroc"] = None

    return report
