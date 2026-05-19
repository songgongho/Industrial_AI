"""Tests for rule-based state labeling.

Three core scenarios:
1. Idle: low power + no alarm + off status
2. Alert: night + high power + alarm
3. Sensor fault: sensor value stuck for long period
"""

import pandas as pd
import pytest

from src.labeling.labeling import (
    DEFAULT_CONFIG,
    build_state_label,
    summarize_label_distribution,
    infer_available_columns,
)


class TestScenario1Idle:
    """Scenario 1: 낮은 전력 + 알람 없음 + 비가동 상태 → idle"""

    def test_idle_with_low_power_and_off_status(self):
        """낮은 전력과 OFF 상태 → idle로 판정되는지 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 00:00:00", "2026-03-01 00:00:01"],
                "device_id": ["PM01", "PM01"],
                "power_kw": [0.0, 0.2],
                "current_a": [0.0, 0.1],
                "eqp_status": ["OFF", "OFF"],
                "alarm_flag": [0, 0],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        assert (labeled["state_label"] == "idle").all()

    def test_idle_with_low_current_and_standby_status(self):
        """낮은 전류와 standby 상태 → idle로 판정되는지 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 01:00:00", "2026-03-01 01:00:01", "2026-03-01 01:00:02"],
                "device_id": ["PM02", "PM02", "PM02"],
                "power_kw": [0.5, 0.5, 0.8],
                "current_a": [0.1, 0.2, 0.3],
                "eqp_status": ["STANDBY", "STANDBY", "SLEEP"],
                "alarm_flag": [0, 0, 0],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        assert (labeled["state_label"] == "idle").all()

    def test_idle_is_not_set_when_alarm_present(self):
        """알람이 있으면 idle이 아님을 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 02:00:00"],
                "device_id": ["PM03"],
                "power_kw": [0.3],
                "current_a": [0.1],
                "eqp_status": ["OFF"],
                "alarm_flag": [1],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        assert labeled.loc[0, "state_label"] != "idle"


class TestScenario2Alert:
    """Scenario 2: 야간 + 높은 전력 + 알람 발생 → alert_state"""

    def test_alert_with_alarm_flag_set(self):
        """명확한 알람 플래그 → alert_state로 판정되는지 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 01:00:00", "2026-03-01 01:00:01"],
                "device_id": ["PM02", "PM02"],
                "power_kw": [0.2, 10.0],
                "current_a": [0.1, 5.0],
                "temp_c": [25.0, 72.0],
                "alarm_flag": [0, 1],
                "eqp_status": ["ON", "ON"],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        assert labeled.loc[1, "state_label"] == "alert_state"

    def test_alert_with_high_temperature(self):
        """고온(> 60°C) → alert_state로 판정되는지 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 03:00:00"],
                "device_id": ["TR01"],
                "power_kw": [2.0],
                "current_a": [0.5],
                "temp_c": [65.0],
                "alarm_flag": [0],
                "eqp_status": ["ON"],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        assert labeled.loc[0, "state_label"] == "alert_state"

    def test_alert_night_high_power_with_alarm(self):
        """야간(00:00~05:59) + 높은 전력 + 알람 → alert_state 우선순위 확인"""
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-01 02:00:00",
                    "2026-03-01 02:30:00",
                    "2026-03-01 03:45:00",
                    "2026-03-01 04:15:00",
                ],
                "device_id": ["PM04", "PM04", "PM04", "PM04"],
                "power_kw": [6.0, 6.5, 7.0, 7.2],
                "current_a": [1.0, 1.1, 1.2, 1.3],
                "eqp_status": ["ON", "ON", "ON", "ON"],
                "alarm_flag": [0, 0, 1, 0],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        # 야간 + 고전력 + 알람 → alert_state
        assert labeled.loc[2, "state_label"] == "alert_state"


class TestScenario3SensorFault:
    """Scenario 3: 센서값이 장시간 동일 → suspected_sensor_fault"""

    def test_sensor_stuck_for_long_period(self):
        """전력값이 5개 이상 연속으로 고정 → suspected_sensor_fault 판정 확인"""
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-01 05:00:00",
                    "2026-03-01 05:00:01",
                    "2026-03-01 05:00:02",
                    "2026-03-01 05:00:03",
                    "2026-03-01 05:00:04",
                    "2026-03-01 05:00:05",
                ],
                "device_id": ["TR02"] * 6,
                "power_kw": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                "current_a": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                "temp_c": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                "eqp_status": ["ON"] * 6,
                "alarm_flag": [0] * 6,
            }
        )
        config = dict(DEFAULT_CONFIG)
        config["sensor_stuck_min_period"] = 5
        labeled = build_state_label(df, config)
        # 센서값이 고정되어 있으면 suspected_sensor_fault
        assert any(labeled["state_label"] == "suspected_sensor_fault")

    def test_sensor_stuck_detector_respects_config(self):
        """config의 sensor_stuck_min_period를 반영하는지 확인"""
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-01 06:00:00",
                    "2026-03-01 06:00:01",
                    "2026-03-01 06:00:02",
                    "2026-03-01 06:00:03",
                ],
                "device_id": ["TR03"] * 4,
                "power_kw": [0.5, 0.5, 0.5, 0.5],
                "current_a": [0.1, 0.1, 0.1, 0.1],
                "eqp_status": ["ON"] * 4,
                "alarm_flag": [0] * 4,
            }
        )
        # 최소 기간을 3으로 설정 → 4개 연속 고정 값은 fault 판정
        config = dict(DEFAULT_CONFIG)
        config["sensor_stuck_min_period"] = 3
        labeled = build_state_label(df, config)
        assert any(labeled["state_label"] == "suspected_sensor_fault")

    def test_sensor_spike_detection(self):
        """전력값 급증 → abnormal_pattern 또는 anomaly 판정 확인"""
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2026-03-01 07:00:00",
                    "2026-03-01 07:00:01",
                    "2026-03-01 07:00:02",
                    "2026-03-01 07:00:03",
                ],
                "device_id": ["PM05"] * 4,
                "power_kw": [0.8, 0.8, 0.9, 2000.0],
                "current_a": [0.2, 0.2, 0.2, 100.0],
                "eqp_status": ["ON"] * 4,
                "alarm_flag": [0] * 4,
            }
        )
        config = dict(DEFAULT_CONFIG)
        config["spike_power_delta"] = 1000.0
        labeled = build_state_label(df, config)
        # 전력이 갑자기 2000으로 증가 → abnormal/fault 중 하나
        assert labeled.loc[3, "state_label"] in {"abnormal_pattern", "suspected_sensor_fault"}


class TestLabelingUtilities:
    """Labeling helper functions and utilities"""

    def test_infer_available_columns_with_standard_headers(self):
        """표준 컬럼명으로 이용 가능 컬럼 추론 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 08:00:00"],
                "device_id": ["PM06"],
                "power_kw": [1.0],
                "current_a": [0.3],
                "eqp_status": ["ON"],
                "alarm_flag": [0],
            }
        )
        available = infer_available_columns(df)
        assert available["timestamp"] == "timestamp"
        assert available["device_id"] == "device_id"
        assert available["power"] == "power_kw"
        assert available["current"] == "current_a"

    def test_label_distribution_summary(self):
        """라벨 분포 요약 함수 확인"""
        df = pd.DataFrame({"state_label": ["idle", "idle", "alert_state", "abnormal_pattern"]})
        summary = summarize_label_distribution(df)
        assert set(summary["state_label"]) == {"idle", "alert_state", "abnormal_pattern"}
        assert summary["count"].sum() == 4
        assert (summary["ratio"] > 0).all()
        assert summary["ratio"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_build_state_label_outputs_dataframe(self):
        """build_state_label이 state_label 컬럼을 가진 DataFrame 반환하는지 확인"""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-03-01 09:00:00"],
                "device_id": ["PM07"],
                "power_kw": [1.5],
                "current_a": [0.4],
                "eqp_status": ["ON"],
                "alarm_flag": [0],
            }
        )
        labeled = build_state_label(df, DEFAULT_CONFIG)
        assert isinstance(labeled, pd.DataFrame)
        assert "state_label" in labeled.columns
        assert len(labeled) == len(df)

