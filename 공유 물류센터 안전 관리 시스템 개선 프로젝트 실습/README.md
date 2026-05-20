# 공유 물류센터 안전 관리 이상 탐지 시스템

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![pytest](https://img.shields.io/badge/Testing-pytest-brightgreen)](#test)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Project Status: MVP](https://img.shields.io/badge/Status-MVP-orange)](#current-implementation)

## 핵심 개요

공유 물류센터와 공유공장에서 **실제 센서 데이터(`data_R2.csv`)를 기반으로** 복합 운영 패턴을 탐지하고 안전 이상을 자동 분류하는 rule-based 시스템입니다.

- ✓ **5개 상태 자동 분류**: idle, normal_operation, abnormal_pattern, alert_state, suspected_sensor_fault
- ✓ **Rule-based + 약 레이블**: 휴리스틱 기반 초기 라벨링 후 supervised learning 확장 가능
- ✓ **Jupyter 기반 재현성**: 전체 분석 파이프라인을 노트북으로 확인 가능
- ✓ **테스트 완성**: 12개 pytest 전수 통과 ✓

---

## 문제정의

공유 물류센터와 공유공장은 다수의 운영 주체가 동일한 공간, 설비, 구역을 시간 단위로 공유합니다. 이 환경에서는 단일 룰 기반 감시만으로는 다음과 같은 복합 상황을 안정적으로 포착하기 어렵습니다.

- 테넌트별 정상 가동 패턴의 차이
- 시간대별(주간/야간) 운영 강도 변화
- 전력/전류/주파수/온도 등 이종 센서 간 상호 불일치
- 센서 자체 이상과 실제 운영 이상의 구분 어려움

---

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

---

## 데이터: `data_R2.csv`

이 저장소의 핵심 데이터는 `data_R2.csv`입니다. 샘플 기준으로는 다음과 같은 요소가 관측됩니다.
- 장비/계측기 식별자(`device_id`로 매핑 가능)
- 시각 정보(`timestamp`)
- 전압/전류/전력/역률/주파수 계열 수치
- 누적 에너지 및 상태 플래그
- 일부 환경 센서값(예: 온도/습도 추정)

즉, 이 데이터는 "공유 운영 환경에서 설비가 언제, 어떤 상태로, 얼마나 비정상적으로 움직였는가"를 다루는 출발점으로 적합합니다.

---

## 상태 레이블 체계

### 현재 구현 (5가지 상태)
| 상태 | 의미 | 우선순위 |
|------|------|---------|
| `alert_state` | 알람 또는 임계치 초과 상태 | 1 (최고) |
| `suspected_sensor_fault` | 센서 결함 의심 | 2 |
| `abnormal_pattern` | 과부하/급변/야간 이상 패턴 | 3 |
| `idle` | 유휴 상태 | 4 |
| `normal_operation` | 정상 가동 | 5 (기준선) |

### 향후 확장 (Multimodal)
CCTV, 환경센서, 이벤트 로그가 추가되면:
- `human_activity` : 작업자 존재/활동
- `equipment_operation` : 설비 정상 가동
- `combined_operation` : 인적 + 설비 복합 가동

---

## 📋 현재 구현 범위 (MVP)

| 요소 | 상태 | 비고 |
|------|------|------|
| **Rule-based 라벨링** | ✓ 완성 | 7개 함수, 5개 상태 분류 |
| **휴리스틱 규칙** | ✓ 완성 | 전력/전류/온도/알람 기반 |
| **데이터 로더** | ✓ 완성 | CSV 헤더 자동 매핑 |
| **EDA 노트북** | ✓ 완성 | 기초 데이터 탐색 |
| **라벨링 로직 노트북** | ✓ 완성 | 실제 데이터 적용 예시 |
| **단위 테스트** | ✓ 통과 | 12개 pytest, 전수 통과 |
| **의존성 관리** | ✓ 완성 | requirements.txt + pyproject.toml |
| **문서화** | ✓ 완성 | README + architecture |

---

## 🔄 향후 추가 예정 (Roadmap)

| 우선순위 | 기능 | 설명 |
|---------|------|------|
| Phase 2 | Tree-based 분류기 | Random Forest / XGBoost baseline |
| Phase 2 | Anomaly Detection | Isolation Forest, One-Class SVM |
| Phase 3 | CCTV 분석 | 사람 감지, 움직임 추적 |
| Phase 3 | 환경센서 융합 | 온도/습도/가스 센서 통합 |
| Phase 3 | Multi-modal Fusion | 센서+영상+로그 결합 모델 |
| Phase 4 | 실시간 모니터링 | API 서빙, 대시보드 UI |

---

## 📂 프로젝트 구조

```text
Portfolio-공유 물류센터 안전 관리 시스템 개선 프로젝트 실습/
├── README.md                           # 프로젝트 개요
├── pyproject.toml                      # 의존성 관리
├── requirements.txt                    # 핀된 패키지
│
├── configs/
│   └── default.yaml                    # 라벨링 threshold 설정
│
├── data/
│   ├── raw/                            # 원본 data_R2.csv
│   ├── interim/                        # 헤더 매핑 버전
│   └── processed/                      # 라벨링 결과
│
├── notebooks/
│   ├── 01_eda.ipynb                    # 기초 데이터 탐색
│   └── 02_labeling_logic.ipynb         # Rule-based 라벨링 실행 ★
│
├── src/
│   ├── data_loader/loader.py           # CSV 헤더 매핑
│   ├── labeling/
│   │   ├── labeling.py                 # ★ 핵심 라벨링 모듈
│   │   └── __init__.py
│   ├── preprocessing/clean.py
│   ├── features/feature_engineering.py
│   ├── models/baseline.py
│   ├── evaluation/metrics.py
│   └── visualization/plots.py
│
├── tests/
│   └── test_labeling.py                # ★ 12개 pytest 전수 통과
│
├── docs/
│   ├── architecture.md                 # 시스템 아키텍처 설명
│   └── images/                         # 스크린샷 자리표시자
│       ├── architecture-diagram.png
│       ├── label-distribution.png
│       ├── timeseries-patterns.png
│       ├── anomaly-timeline.png
│       └── notebook-execution.png
│
└── .github/
    └── workflows/
        └── test.yml                    # CI/CD 자동 테스트
```

---

## 📊 시각 자료 (Screenshots & Diagrams)

### 1. 시스템 아키텍처
![System Architecture](docs/images/architecture-diagram.png)
*Rule-based labeling → Weak labels → Supervised learning → Multi-modal fusion*

### 2. EDA 결과: 시간대별 패턴
![Time-series Patterns](docs/images/timeseries-patterns.png)
*시간대별 전력 사용량 및 장비 상태 분포*

### 3. 라벨 분포
![Label Distribution](docs/images/label-distribution.png)
*5가지 상태 레이블의 비율 및 시간대별 빈도*

### 4. 이상 탐지 예시 타임라인
![Anomaly Timeline](docs/images/anomaly-timeline.png)
*alert_state, abnormal_pattern, suspected_sensor_fault 사례*

### 5. 노트북 실행 화면
![Notebook Screenshot](docs/images/notebook-execution.png)
*jupyter 기반 대화형 분석 파이프라인*

---

## 📈 분석 파이프라인

```
Raw CSV (data_R2.csv)
    ↓
데이터 로딩 & 헤더 매핑 (loader.py)
    ↓
EDA (01_eda.ipynb)
    ├─ 컬럼 분포 확인
    ├─ 시간대별 패턴 분석
    └─ 기본 통계
    ↓
Rule-based Labeling (labeling.py)
    ├─ idle 탐지 (저전력 + 비가동)
    ├─ alert_state 탐지 (알람 + 고온)
    ├─ abnormal_pattern 탐지 (급변 + 불일치)
    └─ suspected_sensor_fault 탐지 (고착 + 결측)
    ↓
Weak Labels 생성 (02_labeling_logic.ipynb)
    ├─ 레이블 분포 시각화
    ├─ 시간대별 빈도
    └─ 이상 사례 샘플링
    ↓
[향후] Supervised Learning
    └─ 라벨 정제 → 모델 학습 → 평가
```

---

## 🧪 테스트 및 검증

### 테스트 현황
```bash
$ pytest tests/ -v
tests/test_labeling.py::TestScenario1Idle::test_idle_with_low_power_and_off_status PASSED
tests/test_labeling.py::TestScenario1Idle::test_idle_with_low_current_and_standby_status PASSED
tests/test_labeling.py::TestScenario1Idle::test_idle_is_not_set_when_alarm_present PASSED
tests/test_labeling.py::TestScenario2Alert::test_alert_with_alarm_flag_set PASSED
tests/test_labeling.py::TestScenario2Alert::test_alert_with_high_temperature PASSED
tests/test_labeling.py::TestScenario2Alert::test_alert_night_high_power_with_alarm PASSED
tests/test_labeling.py::TestScenario3SensorFault::test_sensor_stuck_for_long_period PASSED
tests/test_labeling.py::TestScenario3SensorFault::test_sensor_stuck_detector_respects_config PASSED
tests/test_labeling.py::TestScenario3SensorFault::test_sensor_spike_detection PASSED
tests/test_labeling.py::TestLabelingUtilities::test_infer_available_columns_with_standard_headers PASSED
tests/test_labeling.py::TestLabelingUtilities::test_label_distribution_summary PASSED
tests/test_labeling.py::TestLabelingUtilities::test_build_state_label_outputs_dataframe PASSED

✓ 12/12 passed in 0.45s
```

### 테스트 커버리지 항목
- ✓ idle 상태 탐지 (저전력, 알람 없음, OFF 상태)
- ✓ alert_state 탐지 (알람, 고온, 야간 고전력+알람)
- ✓ abnormal_pattern 탐지 (전력 급증, 상태 불일치)
- ✓ sensor_fault 탐지 (값 고착, 급변, 설정값 반영)
- ✓ 유틸리티 함수 (컬럼 자동 추론, 분포 요약)

---

## 📚 주요 문서

- **[Architecture](docs/architecture.md)** - 시스템 설계 및 확장 전략
- **[notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)** - 기초 데이터 탐색
- **[notebooks/02_labeling_logic.ipynb](notebooks/02_labeling_logic.ipynb)** - Rule-based 라벨링 실행 예시

---

## 🔐 라이선스

MIT License - 자유롭게 사용, 수정, 분배 가능합니다.

---

## 🤝 포트폴리오 의도

이 프로젝트는 다음을 시연합니다:

1. **실제 데이터 기반 문제 정의**
   - "데이터가 주어졌으니 모델을 만든다"가 아닌, "운영 환경의 실제 문제를 데이터로 재정의하고 풀어낸다"

2. **Rule-based에서 ML로의 진화 설계**
   - 휴리스틱 규칙 (현재) → 약 레이블 (Phase 2) → 지도학습 (Phase 3) → 멀티모달 융합 (Phase 4)

3. **재현성과 테스트 주도 개발**
   - Jupyter 노트북으로 모든 분석이 재현 가능
   - pytest로 핵심 로직 보호

4. **포트폴리오 수준의 코드 품질**
   - 명확한 폴더 구조
   - 완벽한 의존성 관리
   - 자동 테스트 CI/CD 준비

---

## 📞 문의 및 피드백

GitHub Issues에서 버그 보고나 기능 제안을 환영합니다.

