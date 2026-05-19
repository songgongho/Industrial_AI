# 공유 물류센터 안전 관리 이상 탐지 시스템

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/Testing-pytest-brightgreen)](#test)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project Status: MVP](https://img.shields.io/badge/Status-MVP-orange)](#현재-구현-범위)

## 핵심 개요

공유 물류센터와 공유공장에서 **실제 센서 데이터(`data_R2.csv`)를 기반으로** 복합 운영 패턴을 탐지하고 안전 이상을 자동 분류하는 rule-based 시스템입니다.

- ✓ **5개 상태 자동 분류**: idle, normal_operation, abnormal_pattern, alert_state, suspected_sensor_fault
- ✓ **Rule-based + 약 레이블**: 휴리스틱 기반 초기 라벨링 후 supervised learning 확장 가능
- ✓ **Jupyter 기반 재현성**: 전체 분석 파이프라인을 노트북으로 확인 가능
- ✓ **테스트 완성**: 12개 pytest 전수 통과 ✓

## 문제정의

공유 물류센터와 공유공장은 다수의 운영 주체가 동일한 공간, 설비, 구역을 시간 단위로 공유합니다. 이 환경에서는 단일 룰 기반 감시만으로는 다음과 같은 복합 상황을 안정적으로 포착하기 어렵습니다.

- 테넌트별 정상 가동 패턴의 차이
- 시간대별(주간/야간) 운영 강도 변화
- 전력/전류/주파수/온도 등 이종 센서 간 상호 불일치
- 센서 자체 이상과 실제 운영 이상의 구분 어려움

## 🚀 Quick Start

```bash
# 1. 환경 설정
python -m venv .venv
.venv\Scripts\activate  # or: source .venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 테스트 실행 (모두 통과)
pytest tests/ -v

# 4. Jupyter 노트북 실행
jupyter notebook notebooks/02_labeling_logic.ipynb
```

### 예시: Rule-based 라벨 생성

```python
from src.labeling.labeling import build_state_label, DEFAULT_CONFIG
import pandas as pd

# 데이터 로드 및 라벨 생성
df = pd.read_csv('data/interim/data_with_header.csv')
labeled_df = build_state_label(df, DEFAULT_CONFIG)

# 결과 확인
print(labeled_df[['timestamp', 'device_id', 'power_kw', 'state_label']].head(10))
```

**라벨 분포 예시**:
```
state_label            count    ratio
idle              45,321     42.1%
normal_operation  38,902     36.0%
abnormal_pattern   18,456     17.1%
alert_state        4,123      3.8%
suspected_sensor_fault  1,298  1.2%
```

## `data_R2.csv` 설명
이 저장소의 핵심 데이터는 `data_R2.csv`입니다. 샘플 기준으로는 다음과 같은 요소가 관측됩니다.
- 장비/계측기 식별자(`device_id`로 매핑 가능)
- 시각 정보(`timestamp`)
- 전압/전류/전력/역률/주파수 계열 수치
- 누적 에너지 및 상태 플래그
- 일부 환경 센서값(예: 온도/습도 추정)

즉, 이 데이터는 "공유 운영 환경에서 설비가 언제, 어떤 상태로, 얼마나 비정상적으로 움직였는가"를 다루는 출발점으로 적합합니다.

## 상태 레이블 방향
이 프로젝트는 우선 CSV만으로 1차 정의 가능한 상태를 사용합니다.
- `idle` : 유휴 상태
- `normal_operation` : 정상 가동
- `abnormal_pattern` : 과부하/급변/야간 이상 패턴
- `alert_state` : 알람 또는 임계치 초과 상태
- `sensor_fault_suspected` : 센서 결함 의심

이후 CCTV, 환경센서, 이벤트 로그가 추가되면 `human_activity`, `equipment_operation`, `combined_operation` 같은 멀티모달 상태로 확장할 수 있습니다.


