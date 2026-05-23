"""Tests for synthetic Press cycle generation with P013 anomalies."""

import numpy as np
import pytest

from src.data.synthpress import (
    AnomalyType,
    MultiAnomalyType,
    PressCycleSpec,
    generate_press_cycle,
    generate_press_cycle_multi,
)


def test_generate_press_cycle_baseline_no_anomaly() -> None:
    """Test baseline cycle generation without anomalies."""
    frame, label, anomaly_type = generate_press_cycle(1, 10, anomaly_prob=0.0, seed=42)

    assert label == 0
    assert anomaly_type is None
    assert frame.shape[0] == 1024
    assert "HPPRESS_PV" in frame.columns
    assert "HPTEMP_PV" in frame.columns
    assert "VACUUM" in frame.columns
    assert not frame["HPPRESS_SV"].isna().any()
    assert not frame["HPTEMP_SV"].isna().any()


def test_generate_press_cycle_with_anomaly_probability() -> None:
    """Test that anomaly_prob=1.0 guarantees label=1."""
    frame, label, anomaly_type = generate_press_cycle(1, 10, anomaly_prob=1.0, seed=42)

    assert label == 1
    assert anomaly_type is not None
    # Check that anomaly_type is one of the 6 types
    valid_types: list[AnomalyType] = [
        "temp_offset",
        "pressure_drop",
        "vacuum_leak",
        "equipment_fault",
        "power_loss",
        "program_mismatch",
    ]
    assert anomaly_type in valid_types


def test_p013_001_temp_offset() -> None:
    """Test P013-001: Temperature offset anomaly."""
    # Force seed to get consistent cycle, then generate multiple samples
    # to find one with temp_offset
    for seed in range(100):
        _, label, anomaly_type = generate_press_cycle(
            1, 10, anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "temp_offset":
            assert label == 1
            break
    else:
        pytest.fail("Could not generate temp_offset anomaly in 100 attempts")


def test_p013_002_pressure_drop() -> None:
    """Test P013-002: Pressure drop anomaly."""
    for seed in range(100):
        _, label, anomaly_type = generate_press_cycle(
            1, 10, anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "pressure_drop":
            assert label == 1
            break
    else:
        pytest.fail("Could not generate pressure_drop anomaly in 100 attempts")


def test_p013_003_vacuum_leak() -> None:
    """Test P013-003: Vacuum leak anomaly."""
    for seed in range(100):
        _, label, anomaly_type = generate_press_cycle(
            1, 10, anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "vacuum_leak":
            assert label == 1
            break
    else:
        pytest.fail("Could not generate vacuum_leak anomaly in 100 attempts")


def test_p013_004_equipment_fault() -> None:
    """Test P013-004: Equipment fault anomaly."""
    for seed in range(100):
        _, label, anomaly_type = generate_press_cycle(
            1, 10, anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "equipment_fault":
            assert label == 1
            break
    else:
        pytest.fail("Could not generate equipment_fault anomaly in 100 attempts")


def test_p013_005_power_loss() -> None:
    """Test P013-005: Power loss anomaly (creates NaN values)."""
    for seed in range(100):
        frame, label, anomaly_type = generate_press_cycle(
            1, 10, anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "power_loss":
            assert label == 1
            # Power loss should have NaN values in time ranges
            assert frame["HPPRESS_PV"].isna().any()
            assert frame["HPTEMP_PV"].isna().any()
            assert frame["VACUUM"].isna().any()
            break
    else:
        pytest.fail("Could not generate power_loss anomaly in 100 attempts")


def test_p013_006_program_mismatch() -> None:
    """Test P013-006: Program mismatch anomaly (wrong SV profile)."""
    for seed in range(100):
        _, label, anomaly_type = generate_press_cycle(
            1, 10, anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "program_mismatch":
            assert label == 1
            break
    else:
        pytest.fail("Could not generate program_mismatch anomaly in 100 attempts")


def test_press_cycle_spec_properties() -> None:
    """Test PressCycleSpec time calculations."""
    spec = PressCycleSpec(vacuum_s=180, hot_press_s=9960, cooling_s=3000, release_s=60)
    assert spec.total_s == 180 + 9960 + 3000 + 60


def test_generate_press_cycle_with_custom_spec() -> None:
    """Test cycle generation with custom spec."""
    custom_spec = PressCycleSpec(
        vacuum_s=100, hot_press_s=5000, cooling_s=1500, release_s=30
    )
    frame, _, _ = generate_press_cycle(
        1, 10, spec=custom_spec, n_points=500, anomaly_prob=0.0, seed=42
    )

    assert frame.shape[0] == 500
    assert "HPPRESS_SV" in frame.columns


def test_generate_press_cycle_reproducibility() -> None:
    """Test that same seed produces same cycle."""
    frame1, label1, anom1 = generate_press_cycle(1, 10, anomaly_prob=0.5, seed=999)
    frame2, label2, anom2 = generate_press_cycle(1, 10, anomaly_prob=0.5, seed=999)

    assert label1 == label2
    assert anom1 == anom2
    np.testing.assert_array_equal(
        frame1["HPPRESS_PV"].values, frame2["HPPRESS_PV"].values
    )


def test_generate_press_cycle_columns_present() -> None:
    """Test that all required columns exist."""
    frame, _, _ = generate_press_cycle(1, 10, anomaly_prob=0.0, seed=42)

    required_cols = [
        "cycle_id",
        "panel_id",
        "t_ms",
        "HPPRESS_SV",
        "HPPRESS_PV",
        "FHPPRESS_SV",
        "FHPPRESS_PV",
        "HPTEMP_SV",
        "HPTEMP_PV",
        "VACUUM",
        "label",
        "anomaly_type",
    ]
    for col in required_cols:
        assert col in frame.columns
    # Also check POINT1-POINT12 channels
    for i in range(1, 13):
        assert f"POINT{i}" in frame.columns


# ============================================================================
# Multi-anomaly tests (P013 causal patterns)
# ============================================================================


def test_generate_press_cycle_multi_no_anomaly() -> None:
    """Test multi-cycle generation without anomalies."""
    frame, label, anomaly_type = generate_press_cycle_multi(
        1, 10, anomaly_prob=0.0, multi_anomaly_prob=0.0, seed=42
    )

    assert label == 0
    assert anomaly_type is None
    assert frame.shape[0] == 1024
    assert "HPPRESS_PV" in frame.columns


def test_generate_press_cycle_multi_single_anomaly() -> None:
    """Test that single anomaly still works with multi function."""
    frame, label, anomaly_type = generate_press_cycle_multi(
        1, 10, anomaly_prob=1.0, multi_anomaly_prob=0.0, seed=42
    )

    assert label == 1
    assert anomaly_type is not None
    valid_types: list[AnomalyType] = [
        "temp_offset",
        "pressure_drop",
        "vacuum_leak",
        "equipment_fault",
        "power_loss",
        "program_mismatch",
    ]
    assert anomaly_type in valid_types


def test_generate_press_cycle_multi_takes_precedence() -> None:
    """Test that multi_anomaly_prob takes precedence over anomaly_prob."""
    frame, label, anomaly_type = generate_press_cycle_multi(
        1, 10, anomaly_prob=1.0, multi_anomaly_prob=1.0, seed=42
    )

    assert label == 1
    assert anomaly_type is not None
    # Should be a multi-anomaly type, not a single anomaly
    valid_multi: list[MultiAnomalyType] = [
        "vacuum_pressure_cascade",
        "pressure_temp_coupling",
        "equipment_full_fault",
        "power_vacuum_loss",
    ]
    assert anomaly_type in valid_multi


def test_multi_anomaly_vacuum_pressure_cascade() -> None:
    """Test vacuum + pressure causal pattern (P013-003 → P013-002)."""
    for seed in range(100):
        frame, label, anomaly_type = generate_press_cycle_multi(
            1, 10, anomaly_prob=0.0, multi_anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "vacuum_pressure_cascade":
            assert label == 1
            # Vacuum should be elevated (leak)
            assert frame["VACUUM"].mean() > 0.15  # Normal ~0.12, elevated state
            # Pressure should be reduced compared to normal (pressure_sv avg ~10)
            # Due to anomaly injection, PV mean should show degradation
            break
    else:
        pytest.fail("Could not generate vacuum_pressure_cascade in 100 attempts")


def test_multi_anomaly_pressure_temp_coupling() -> None:
    """Test pressure + temperature causal pattern (P013-002 → P013-001)."""
    for seed in range(100):
        frame, label, anomaly_type = generate_press_cycle_multi(
            1, 10, anomaly_prob=0.0, multi_anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "pressure_temp_coupling":
            assert label == 1
            # Both pressure and temperature should show reduction
            break
    else:
        pytest.fail("Could not generate pressure_temp_coupling in 100 attempts")


def test_multi_anomaly_equipment_full_fault() -> None:
    """Test equipment fault affecting all signals (P013-004 cascade)."""
    for seed in range(100):
        frame, label, anomaly_type = generate_press_cycle_multi(
            1, 10, anomaly_prob=0.0, multi_anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "equipment_full_fault":
            assert label == 1
            # After ~50% of cycle, all signals degrade significantly
            mid = frame.shape[0] // 2
            pressure_before = frame["HPPRESS_PV"].iloc[:mid].mean()
            pressure_after = frame["HPPRESS_PV"].iloc[mid:].mean()
            # After fault, pressure should drop
            assert pressure_after < pressure_before
            break
    else:
        pytest.fail("Could not generate equipment_full_fault in 100 attempts")


def test_multi_anomaly_power_vacuum_loss() -> None:
    """Test power loss + vacuum instability (P013-005 → P013-003)."""
    for seed in range(100):
        frame, label, anomaly_type = generate_press_cycle_multi(
            1, 10, anomaly_prob=0.0, multi_anomaly_prob=1.0, seed=seed
        )
        if anomaly_type == "power_vacuum_loss":
            assert label == 1
            # Should have NaN values (power loss gap)
            assert frame["HPPRESS_PV"].isna().any()
            break
    else:
        pytest.fail("Could not generate power_vacuum_loss in 100 attempts")


def test_generate_press_cycle_multi_reproducibility() -> None:
    """Test multi-cycle reproducibility with seed."""
    frame1, label1, anom1 = generate_press_cycle_multi(
        1, 10, anomaly_prob=0.5, multi_anomaly_prob=0.3, seed=12345
    )
    frame2, label2, anom2 = generate_press_cycle_multi(
        1, 10, anomaly_prob=0.5, multi_anomaly_prob=0.3, seed=12345
    )

    assert label1 == label2
    assert anom1 == anom2
    # Check that matching columns have same values (accounting for potential NaN)
    for col in ["HPPRESS_SV", "HPTEMP_SV"]:
        if not frame1[col].isna().any() and not frame2[col].isna().any():
            np.testing.assert_array_almost_equal(
                frame1[col].values, frame2[col].values
            )


def test_generate_press_cycle_multi_columns_present() -> None:
    """Test that multi-cycle generation has all required columns."""
    frame, _, _ = generate_press_cycle_multi(
        1, 10, anomaly_prob=0.0, multi_anomaly_prob=0.0, seed=42
    )

    required_cols = [
        "cycle_id",
        "panel_id",
        "t_ms",
        "HPPRESS_SV",
        "HPPRESS_PV",
        "FHPPRESS_SV",
        "FHPPRESS_PV",
        "HPTEMP_SV",
        "HPTEMP_PV",
        "VACUUM",
        "label",
        "anomaly_type",
    ]
    for col in required_cols:
        assert col in frame.columns

