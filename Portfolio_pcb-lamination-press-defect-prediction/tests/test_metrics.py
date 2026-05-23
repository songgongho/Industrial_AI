from src.eval.metrics import cost_aware_detection_score


def test_cost_aware_detection_score_perfect_predictions() -> None:
    assert cost_aware_detection_score([0, 1, 1], [0, 1, 1], 100, 5) == 0.0


def test_cost_aware_detection_score_counts_fn_and_fp() -> None:
    score = cost_aware_detection_score([1, 0, 1, 0], [0, 1, 1, 0], 100, 5)
    assert score == (100 + 5) / 4

