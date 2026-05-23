"""Tests for far_at_recall in src.eval.metrics."""

import pytest

from src.eval.metrics import far_at_recall


def test_far_at_recall_perfect_separation_returns_zero() -> None:
    # 양성 점수 > 음성 점수: recall=1.0에서도 FAR=0
    assert far_at_recall([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.9) == 0.0


def test_far_at_recall_target_recall_attainable() -> None:
    # 양성 4개, 음성 4개. 가장 낮은 양성 점수(0.6) 임계로 recall=1.0,
    # 그 임계에서 음성 1개(0.7) 통과 → FAR = 1/4 = 0.25
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.7, 0.6, 0.7, 0.4, 0.3, 0.2]
    val = far_at_recall(y_true, y_score, 0.9)
    assert val is not None
    assert val == pytest.approx(0.25)


def test_far_at_recall_partial_recall_target() -> None:
    # target_recall=0.5 → 양성 절반만 잡으면 됨. 상위 점수 양성 2개만으로 달성, FAR=0
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_score = [0.95, 0.9, 0.3, 0.2, 0.5, 0.4, 0.3, 0.1]
    assert far_at_recall(y_true, y_score, 0.5) == 0.0


def test_far_at_recall_all_positive_returns_none() -> None:
    assert far_at_recall([1, 1, 1], [0.1, 0.5, 0.9], 0.9) is None


def test_far_at_recall_no_positive_returns_none() -> None:
    assert far_at_recall([0, 0, 0], [0.1, 0.5, 0.9], 0.9) is None


def test_far_at_recall_inverted_scores_high_far() -> None:
    # 점수가 라벨과 반대로 매겨진 최악 케이스: recall 0.9 위해 거의 모든 음성 통과
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_score = [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7, 0.6]
    val = far_at_recall(y_true, y_score, 0.9)
    assert val is not None
    assert val == pytest.approx(1.0)


def test_far_at_recall_invalid_target_recall_raises() -> None:
    with pytest.raises(ValueError):
        far_at_recall([0, 1], [0.1, 0.9], 1.5)
    with pytest.raises(ValueError):
        far_at_recall([0, 1], [0.1, 0.9], 0.0)


def test_far_at_recall_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        far_at_recall([0, 1, 1], [0.1, 0.9], 0.9)


def test_classification_report_includes_far_key() -> None:
    from src.eval.metrics import classification_report_dict

    report = classification_report_dict(
        [0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2]
    )
    assert "far_at_recall_0_9" in report
    assert report["far_at_recall_0_9"] is not None


def test_classification_report_far_none_without_scores() -> None:
    from src.eval.metrics import classification_report_dict

    report = classification_report_dict([0, 1, 1, 0], [0, 1, 0, 0])
    assert report["far_at_recall_0_9"] is None
