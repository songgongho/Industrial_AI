# 데이터 분석 리포트

- 입력 파일: `data_R2.csv`
- 총 행 수: **24,105,658**
- 최빈 라벨: **alert_state** (21,427,137건, 88.89%)
- `alert_state` 수: **21,427,137**
- `suspected_sensor_fault` 수: **2,678,517**

## 1) 라벨 분포
- alert_state: 21,427,137건 (88.89%)
- suspected_sensor_fault: 2,678,517건 (11.11%)
- abnormal_pattern: 2건 (0.00%)
- idle: 2건 (0.00%)

## 2) 컬럼 매핑 결과
- timestamp: 2
- hour: None
- device_id: 1
- zone_id: None
- line_id: None
- power: 16
- current: 3
- voltage: 6
- temp: 23
- humidity: 24
- gas: None
- smoke: None
- fire: None
- alarm_flag: 22
- error_code: None
- event_flag: None
- eqp_status: None
- mode: None
- run_flag: None

## 3) 핵심 해석
- 한 개의 라벨이 전체의 대부분을 차지하여 분포 편향이 큽니다.
- `alert_state` 비율이 높아 안전 알람 또는 이상 신호가 자주 감지되는 패턴입니다.
- `suspected_sensor_fault` 비율도 높아 센서 고착, 0값 지속, 결측 반복 여부를 추가 점검할 필요가 있습니다.
- `eqp_status`가 자동 인식되지 않아 장비 상태 기반의 `idle` 판정이 충분히 반영되지 않았을 가능성이 있습니다.
- 보고서에는 라벨 분포 막대그래프와 power-current 산점도를 함께 넣으면 해석력이 높아집니다.

## 4) 시각화 파일
- `results/plots/label_distribution.png`
- `results/plots/power_current_scatter.png`
- `results/plots/device_*_timeline.png`
