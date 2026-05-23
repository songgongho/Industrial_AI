"""Tests for SECOM dataset loader."""

import numpy as np
import pytest

from src.data.loaders import load_secom


def test_load_secom_returns_correct_shapes() -> None:
    """Test that load_secom returns expected shapes."""
    x, y = load_secom("data/raw/secom", target_length=128)

    # X should be (N, 128, 1)
    assert x.ndim == 3
    assert x.shape[1] == 128
    assert x.shape[2] == 1

    # y should be 1D
    assert y.ndim == 1
    # y and x should have matching sample count
    assert x.shape[0] == y.shape[0]


def test_load_secom_filters_unannotated() -> None:
    """Test that unannotated samples (label=-1) are filtered out."""
    x, y = load_secom("data/raw/secom", target_length=128)

    # Should have no -1 labels
    assert np.all(y != -1)
    # SECOM dataset should have at least some positive samples
    assert np.any(y == 1)


def test_load_secom_no_nan_values() -> None:
    """Test that output contains no NaN values."""
    x, y = load_secom("data/raw/secom", target_length=128)

    assert not np.isnan(x).any()
    assert not np.isnan(y).any()


def test_load_secom_large_target_length() -> None:
    """Test resampling with large target_length (padding case)."""
    x, y = load_secom("data/raw/secom", target_length=600)

    assert x.shape[1] == 600
    assert x.shape[0] == y.shape[0]


def test_load_secom_small_target_length() -> None:
    """Test resampling with small target_length (downsampling case)."""
    x, y = load_secom("data/raw/secom", target_length=32)

    assert x.shape[1] == 32
    assert x.shape[0] == y.shape[0]


def test_load_secom_reproducible() -> None:
    """Test that loading twice gives identical results."""
    x1, y1 = load_secom("data/raw/secom", target_length=128)
    x2, y2 = load_secom("data/raw/secom", target_length=128)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_load_secom_file_not_found() -> None:
    """Test that missing files raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_secom("data/raw/nonexistent", target_length=128)
