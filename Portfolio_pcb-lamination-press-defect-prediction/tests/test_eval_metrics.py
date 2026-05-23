from src.eval.metrics import classification_report_dict, cost_aware_detection_score


def test_classification_report_dict_keys() -> None:
    report = classification_report_dict([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    assert report["n_samples"] == 4
    assert report["accuracy"] == 0.75
    assert report["f1"] == 0.6666666666666666
    assert report["precision"] == 1.0
    assert report["recall"] == 0.5
    assert report["cost_aware_score"] == cost_aware_detection_score([0, 1, 1, 0], [0, 1, 0, 0], 100, 5)
    assert report["auroc"] is not None


def test_classification_report_dict_shape_validation() -> None:
    try:
        classification_report_dict([0, 1], [0], [0.2, 0.8])
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")

