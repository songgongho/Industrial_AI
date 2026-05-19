"""labeling.py

초안: data_R2.csv 기반 상태 라벨 생성 유틸리티

함수:
 - detect_idle(df, config): 전력/전류 기준으로 유휴(idle) 판정
 - detect_alert_state(df, config): 알람/과온/명백한 이상 신호 기반 경보판정
 - detect_abnormal_pattern(df, config): 통계적/시계열 이상 패턴(과부하/급변 등)
 - build_state_label(df, config): 우선순위 기반으로 최종 상태 레이블을 생성

설계 원칙:
 - 하드코드된 임계값은 config dict로 관리
 - 컬럼명이 없거나 다를 경우 후보 이름 목록에서 자동 탐색
 - 컬럼이 아예 없으면 graceful fallback(해당 체크 건너뜀)
 - 입력: pandas.DataFrame, 출력: 같은 DataFrame에 'state_label' 컬럼 추가 후 반환
"""

from typing import Optional, Dict, List
import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """DataFrame에서 후보 컬럼명 리스트 중 존재하는 첫 컬럼명을 반환.
    없으면 None 반환한다. 대소문자/언더스코어를 유연하게 매칭한다.
    """
    cols = list(df.columns)
    lowered = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    # 약간의 유사 매칭(공백/-,.) 제거
    trans = {c.replace(' ', '').replace('-', '').replace('_', '').lower(): c for c in cols}
    for cand in candidates:
        key = cand.replace(' ', '').replace('-', '').replace('_', '').lower()
        if key in trans:
            return trans[key]
    return None


def _get_series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    col = _find_col(df, candidates)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors='coerce')


def detect_idle(df: pd.DataFrame, config: Dict) -> pd.Series:
    """유휴(idle) 판정.

    기준(우선순위 적용):
    - 전력(active_power) 또는 전류(current) 값이 낮음 (<= threshold)
    - 알람(alarm_flag) 없음
    - 상태(status) 가 OFF/0/standby 계열

    컬럼 후보는 config에 지정하거나, 자동탐색 후보 목록에서 찾는다.
    """
    # 후보 컬럼 목록
    active_p = _get_series(df, config.get('active_power_cols', ['active_power', 'P', 'power_active', 'activePower', 'kW']))
    current = _get_series(df, config.get('current_cols', ['current', 'I', 'ampere', 'current_a']))
    alarm = _find_col(df, config.get('alarm_cols', ['alarm', 'alarm_flag', 'alarm_flag']))
    status_col = _find_col(df, config.get('status_cols', ['status', 'mode', 'state']))

    idle_mask = pd.Series(False, index=df.index)

    # 전력/전류 기준
    if active_p is not None:
        idle_mask = idle_mask | (active_p.fillna(0) <= config.get('idle_active_power_threshold', 1.0))
    if current is not None:
        idle_mask = idle_mask | (current.fillna(0) <= config.get('idle_current_threshold', 0.5))

    # 알람 없음인 경우에만 idle로 인정
    if alarm is not None:
        alarm_series = df[alarm].astype(str).str.lower()
        alarm_present = alarm_series.isin(['1', 'true', 't', 'y', 'yes'])
        idle_mask = idle_mask & (~alarm_present)

    # status가 OFF/0/standby 계열일 경우 강화
    if status_col is not None:
        st = df[status_col].astype(str).str.lower()
        off_mask = st.isin(['off', '0', 'standby', 'idle'])
        idle_mask = idle_mask & (off_mask | pd.Series(True, index=df.index))

    return idle_mask.fillna(False)


def detect_alert_state(df: pd.DataFrame, config: Dict) -> pd.Series:
    """명백한 경보(alert) 판정.

    기준 예시:
    - 알람 플래그가 설정되었을 경우
    - 온도/습도 등이 임계치를 초과했을 경우
    - 역률(power factor) 또는 주파수(frequency) 이상
    - 강한 음수/비정상 전력(예: 역방향 에너지 수치)
    """
    alarm_col = _find_col(df, config.get('alarm_cols', ['alarm', 'alarm_flag']))
    temp = _get_series(df, config.get('temp_cols', ['temp', 'temperature']))
    pf = _get_series(df, config.get('pf_cols', ['power_factor', 'pf']))
    freq = _get_series(df, config.get('freq_cols', ['frequency', 'freq']))
    active_p = _get_series(df, config.get('active_power_cols', ['active_power', 'P', 'power_active']))

    alert = pd.Series(False, index=df.index)

    if alarm_col is not None:
        a = df[alarm_col].astype(str).str.lower()
        alert = alert | a.isin(['1', 'true', 't', 'y', 'yes'])

    if temp is not None:
        alert = alert | (temp > config.get('temp_alert_threshold', 60.0))

    if pf is not None:
        alert = alert | (pf < config.get('pf_alert_threshold', 0.6))

    if freq is not None:
        alert = alert | ((freq < config.get('freq_low', 49.0)) | (freq > config.get('freq_high', 61.0)))

    if active_p is not None:
        # 역방향 전력(음수) 또는 비정상 큰 변동
        alert = alert | (active_p < config.get('active_power_negative_threshold', -1e-3))

    return alert.fillna(False)


def detect_abnormal_pattern(df: pd.DataFrame, config: Dict) -> pd.Series:
    """통계/시계열 기반 비정상(abnormal pattern) 판정.

    기준 예시:
    - 전류/전력의 평균 대비 k*std를 초과하는 과부하
    - 짧은 구간 내 급변(롤링 차이) 발생
    - 야간(무인 시간) 중 비정상 동작(전력 사용 등)
    """
    active_p = _get_series(df, config.get('active_power_cols', ['active_power', 'P', 'power_active']))
    current = _get_series(df, config.get('current_cols', ['current', 'I']))
    timestamp_col = _find_col(df, config.get('timestamp_cols', ['timestamp', 'time', 'datetime']))

    abnormal = pd.Series(False, index=df.index)

    # 과부하: 평균 + k*std 초과
    if current is not None:
        mean = current.mean(skipna=True)
        std = current.std(skipna=True)
        k = config.get('overload_k', 3.0)
        abnormal = abnormal | (current > (mean + k * std))
    elif active_p is not None:
        mean = active_p.mean(skipna=True)
        std = active_p.std(skipna=True)
        k = config.get('overload_k', 3.0)
        abnormal = abnormal | (active_p > (mean + k * std))

    # 급변: 짧은 윈도우에서 차이
    if active_p is not None and len(df) >= config.get('spike_window', 3):
        diff = active_p.fillna(method='ffill').diff().abs()
        spike_mask = diff > (config.get('spike_threshold', 1000.0))
        abnormal = abnormal | spike_mask.fillna(False)

    # 야간 비정상(선택적): timestamp가 있고 야간 시간에 전력이 높으면 이상
    if timestamp_col is not None and active_p is not None:
        ts = pd.to_datetime(df[timestamp_col], errors='coerce')
        hour = ts.dt.hour
        night = hour.isin(config.get('night_hours', list(range(0, 6))))
        night_mask = night & (active_p > config.get('night_active_power_threshold', 5.0))
        abnormal = abnormal | night_mask.fillna(False)

    return abnormal.fillna(False)


def build_state_label(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """우선순위 기반으로 상태 레이블을 생성해 DataFrame에 'state_label' 컬럼을 추가.

    우선순위(상위->하위): alert_state > abnormal_pattern > idle > normal_operation
    """
    df = df.copy()
    alert = detect_alert_state(df, config)
    abnormal = detect_abnormal_pattern(df, config)
    idle = detect_idle(df, config)

    labels = pd.Series('unknown', index=df.index)
    labels[alert] = 'alert_state'
    labels[~alert & abnormal] = 'abnormal_pattern'
    labels[~alert & ~abnormal & idle] = 'idle'
    labels[(labels == 'unknown')] = 'normal_operation'

    df['state_label'] = labels
    return df


if __name__ == '__main__':
    # 간단한 사용 예시
    import sys
    if len(sys.argv) < 2:
        print('Usage: python labeling.py <path_to_csv>')
        sys.exit(0)
    path = sys.argv[1]
    cfg = {
        # 후보 컬럼 이름(우선순위)
        'timestamp_cols': ['timestamp', 'time', 'datetime'],
        'active_power_cols': ['active_power', 'P', 'power_active', 'activepower'],
        'current_cols': ['current', 'I', 'ampere'],
        'alarm_cols': ['alarm', 'alarm_flag'],
        'status_cols': ['status', 'mode', 'state'],
        'temp_cols': ['temp', 'temperature'],

        # thresholds (포트폴리오 초안값—프로젝트에서 튜닝)
        'idle_active_power_threshold': 1.0,
        'idle_current_threshold': 0.5,
        'temp_alert_threshold': 60.0,
        'pf_alert_threshold': 0.6,
        'freq_low': 49.0,
        'freq_high': 61.0,
        'active_power_negative_threshold': -1.0,
        'overload_k': 3.0,
        'spike_window': 3,
        'spike_threshold': 1000.0,
        'night_hours': list(range(0, 6)),
        'night_active_power_threshold': 5.0,
    }

    df = pd.read_csv(path, header=0)
    out = build_state_label(df, cfg)
    print(out[['state_label']].head(20))

