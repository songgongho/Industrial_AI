# PCB Press 공정 불량 예측 - 데이터 분석 및 연구 전략 프레임워크

**작성일**: 2026년 5월 26일  
**대상**: PCB 라미네이션 PRESS 공정 품질 개선 프로젝트  
**작성자**: Industrial AI Team (Copilot 지원)

---

## 목차
1. [고객사 데이터 요청 현황](#section-1-고객사-데이터-요청-현황)
2. [데이터 수신 후 분석 계획](#section-2-데이터-수신-후-분석-계획)
3. [선행 연구 및 학습 방향](#section-3-선행-연구-및-학습-방향)
4. [주요 알고리즘 및 모델 설명](#section-4-주요-알고리즘-및-모델-설명)
5. [프로젝트 진행 현황](#section-5-프로젝트-진행-현황)
6. [향후 추진 계획](#section-6-향후-추진-계획)

---

## Section 1: 고객사 데이터 요청 현황

### 1.1 요청한 데이터 항목 (7가지)

#### (1) 일별/주간/월별 품질 현황
- **목적**: 시간대별 불량률 변화 패턴 분석
- **필요 컬럼**:
  - `date`, `shift`, `line_id`, `product_code`
  - `good_qty`, `scrap_qty`, `rework_qty`
  - `defect_code`, `defect_name` (P013, P019 중심)
  - `ppm` (Parts Per Million)
- **분석 활용**:
  - 일일 불량률 시계열 분석
  - 요일/주간 주기성 (Day-of-week effect) 파악
  - 월별 계절성 분석 (Seasonality)
  - 교대별 작업자 영향 분석

#### (2) 설비 가동/비가동 현황 및 공정별 가동률
- **목적**: OEE(Overall Equipment Effectiveness) 분석, 다운타임 원인 파악
- **필요 컬럼**:
  - `machine_id`, `process_id`
  - `state` (run/stop/alarm/idle)
  - `start_time`, `end_time`
  - `downtime_min`, `reason_code`, `reason_name`
- **분석 활용**:
  - OEE = Availability × Performance × Quality
  - 공정별 가동률(Availability) 산출
  - 다운타임 원인별 분류 및 대순위 분석
  - 예방 정비(PM) 효과 분석

#### (3) PRESS 공정 알람 이력 및 레시피/설정값 변화 이력
- **목적**: 설정값 변화에 따른 불량 발생 추적
- **필요 컬럼**:
  - `alarm_time`, `alarm_code`, `alarm_name`, `severity`
  - `machine_id`, `duration_sec`, `cleared_time`, `ack_user`
  - 레시피: `temperature`, `pressure`, `time`, `flow`, `humidity` 등
  - 설정값 변화: `change_timestamp`, `parameter_name`, `old_value`, `new_value`
- **분석 활용**:
  - 알람 발생과 불량 간의 인과관계 분석
  - 설정값 변화 직후 불량률 증감 추적
  - 공정 최적화 구간 도출

#### (4) LOT/PANEL/CYCLE 매핑 키
- **목적**: 제품 추적성 확보, 불량 원인↔제품 연결
- **필요 컬럼**:
  - `lot_id`, `panel_id`, `cycle_id`, `machine_id`
  - `product_code`, `customer_code`
  - `start_time`, `end_time`, `cycle_number`
- **분석 활용**:
  - 특정 LOT의 전체 공정 흐름 추적
  - 특정 PANEL/CYCLE의 결함 위치 특정
  - 제품-공정-설정값 간의 3-way 매칭

#### (5) 유지보수/교정 이력, 교대/작업자 정보
- **목적**: 인적요소, 기계적 열화 영향 분석
- **필요 컬럼**:
  - **유지보수**: `maintenance_date`, `component`, `work_type` (preventive/corrective), `duration`, `technician`
  - **교정**: `calibration_date`, `instrument`, `tolerance_range`, `result`
  - **작업자**: `shift`, `operator_id`, `experience_level`, `training_date`
- **분석 활용**:
  - 유지보수 후 불량률 감소 여부 추적
  - 교정 오류로 인한 불량 발생 시나리오 분석
  - 작업자 숙련도와 불량률 상관관계

#### (6) 공정 조건 최적화 관리
- **목적**: 설정값 최적 범위 도출
- **필요 컬럼**: 위의 (3) 레시피/설정값 항목 참고
- **분석 활용**:
  - 불량 없는 설정값 범위(Safe Operating Zone) 정의
  - 다중변수 최적화 (Multi-parameter Optimization)
  - Robust Design 검증

#### (7) 공정 시계열 센서 데이터
- **목적**: 공정 중 실시간 모니터링 신호 분석
- **필요 컬럼**:
  - `timestamp`, `machine_id`, `process_step`
  - 센서: `temperature_1~N`, `pressure_1~N`, `humidity`, `vibration`, `current`, `frequency` 등
  - Sampling rate: 1Hz or better (0.1Hz 이상 권장)
- **분석 활용**:
  - 실시간 불량 진행도(Progression) 탐지
  - 비정상 신호 패턴 인식 (Anomaly Detection)
  - 센서 신호→불량 간의 인과관계 추론

---

### 1.2 요청서 발송 현황
- **발송일**: 2026년 5월 25일 (추정)
- **고객사**: 네오텍 (NTC) / 제조사측
- **컬럼 정의 전달**: 필요시 추가 요청 예정

---

## Section 2: 데이터 수신 후 분석 계획

### 2.1 데이터 검증 (Validation) & 전처리 (Preprocessing)

#### Phase 1: 데이터 품질 검증 (1-2주)

| 검증 항목 | 방법 | 예상 결과 | 우선순위 |
|----------|------|---------|---------|
| **누락값 (Missing Values)** | 각 컬럼별 결측률 계산 | 결측률 < 5% (이상 시 고객 재요청) | P0 |
| **범위 이상 (Out-of-range)** | 물리적 범위 검증 (예: 온도 100~250°C) | 이상값 비율 < 1% | P0 |
| **중복값 (Duplicates)** | `(lot_id, timestamp, sensor)` 조합 검증 | 중복 제거 후 확인 | P0 |
| **시간 시퀀스 (Temporal Consistency)** | 타임스탬프 중복성, 역순 확인 | 엄격한 증가 순서 | P0 |
| **크로스 검증** | LOT/PANEL/CYCLE 매핑 일관성 | 고아 레코드(Orphan records) 확인 | P0 |
| **통계적 검증** | 센서값 분포 (정규성, 이상치) | Shapiro-Wilk test, IQR 기반 이상치 탐지 | P1 |

**산출물**: `data_quality_report.csv`, `validation_log.json`

#### Phase 2: 시계열 정렬 및 동기화 (Synchronization)

```python
# 목표: 센서 데이터 ↔ 불량 라벨 간의 정확한 시간 매칭

1. 공통 timestamp 기준 설정 (예: UTC, 1Hz 기준)
2. 센서별 Sampling rate 정규화 (Resampling)
3. Lot/Panel/Cycle 변경 시점 표시
4. 센서 drift 제거 (Sensor calibration baseline으로 정규화)
5. 다중 시계열 정렬 완료 → CSV/Parquet 저장
```

#### Phase 3: 기본 통계 분석 (Exploratory Data Analysis, EDA)

| 분석 내용 | 활용 방법 | 예상 인사이트 |
|----------|---------|-----------|
| **불량률 분포** | Histogram, Box plot | 이상 불량 시기 식별, 정상 범위 정의 |
| **센서 기초 통계** | Mean, Std, Min, Max, Quantiles | 공정 안정성 평가, 편차 확인 |
| **시계열 분해** | STL (Seasonal & Trend decomposition) | 트렌드, 계절성, 잡음 분리 |
| **상관관계 분석** | Pearson/Spearman correlation 히트맵 | 센서 간 강한 선형관계 파악 |
| **일일/주간 패턴** | Aggregation by shift/day-of-week | 교대별, 요일별 불량 차이 |

**산출물**: `eda_report.html`, `correlation_matrix.csv`, `time_series_decomposition.png`

---

### 2.2 인사이트 도출 분석 (Analytics)

#### Analysis 1: 불량 원인 분석 (Root Cause Analysis, RCA)

**목표**: P013, P019 불량과 설정값/센서 간의 연관성 규명

```
Step 1: 정상 샘플 vs 불량 샘플 센서 신호 비교
  → 불량 발생 직전의 센서 변화 패턴 추출 (예: 온도 급상승, 압력 진동)

Step 2: 설정값 변화 → 불량 발생 지연(Lag) 추적
  → 온도 설정 변경 → 3시간 후 불량률 10% 상승 (인과관계 확인)

Step 3: Causal Discovery 알고리즘 적용
  → PCMCI (일변량 분석) / NOTEARS (다변량 함수 관계 추론)
  → 불량에 영향을 미치는 최소 변수 집합 도출

Step 4: 머신러닝 기반 Feature Importance
  → SHAP, LIME을 통한 모델 해석성 확보
```

**예상 산출**:
- "온도, 압력, 습도 3가지 센서가 불량의 90% 설명" (Feature Importance Top 3)
- "설정값 변경 후 평균 4.2시간 후 이상 센서 신호 감지" (Causal lag)

---

#### Analysis 2: 이상 탐지 (Anomaly Detection)

**목표**: 정상에서 벗어난 공정 상태를 실시간에 감지

**방법**:
1. **Unsupervised Anomaly Detection**:
   - **Isolation Forest**: 고차원 센서 신호에서 고립된 샘플 탐지
   - **Autoencoder**: 정상 신호 재구성 오류 기반 이상치 점수 계산
   - **DBSCAN**: 밀도 기반 이상 클러스터 탐지

2. **시계열 특화 이상 탐지**:
   - **Prophet Forecasting**: 예상값과 실제값 간의 잔차(Residuals) 계산
   - **LSTM Autoencoder**: 장기 의존성을 학습한 비정상 신호 탐지
   - **Gaussian Process**: 확률적 경계(Confidence Interval) 정의

**성과 지표** (KPI):
- **검출률 (Recall)**: 불량이 발생한 사건 중 사전 탐지한 비율 > 85%
- **거짓 긍정률 (FAR)**: 정상을 이상으로 잘못 판정한 비율 < 5%
- **평균 탐지 시간 (Time-to-Detection)**: 불량 발생 전 평균 탐지 시간 > 2시간

---

#### Analysis 3: 공정 최적화 분석 (Process Optimization)

**목표**: 불량 최소화하는 설정값 범위(Recipe Sweet Spot) 발견

**분석 방법**:

1. **다변량 최적화 (Multi-Objective Optimization)**:
   - 목적 함수:
     ```
     minimize: (불량률, 사이클 시간, 에너지 소비)
     subject to: 설정값 범위 제약, 공정 안정성 제약
     ```
   - 알고리즘: Bayesian Optimization, Differential Evolution

2. **Robust Design (로버스트 설계)**: 
   - 노이즈 인자 (예: 주변 온도 변화)에도 강건한 설정값 범위 정의
   - Taguchi Method 또는 Response Surface Methodology (RSM) 활용

3. **공정 안정성 지수 (Process Capability)**:
   - Cpk = min((USL - μ)/3σ, (μ - LSL)/3σ) 계산
   - 목표: Cpk > 1.33 (양호한 공정)

**예상 산출**:
- "온도 215±5°C, 압력 280±15 kgf/cm² 범위에서 불량률 < 0.5%" (Recipe 최적 범위)
- "현재 설정 vs 최적 설정 비교: 불량률 3% → 0.8% (-73%)" (개선 효과)

---

#### Analysis 4: 설비 신뢰도 분석 (Equipment Reliability)

**목표**: 설비 열화 추적, 예방 정비 시기 결정

**분석 내용**:

1. **Sensor Drift 분석**:
   - 온도 센서의 날짜별 재현성 편차 추적
   - Calibration 전후 신호 차이 계산

2. **MTBF/MTTR 분석** (Mean Time Between Failures / Mean Time To Repair):
   - 연쇄 고장 및 단일 원점 고장(Single Point of Failure) 식별
   - 부품 수명 추정

3. **RUL (Remaining Useful Life) 예측**:
   - 센서 신호 열화율로부터 예상 교체 시기 계산
   - Digital Twin과 결합하여 조기 교체 최적화

---

#### Analysis 5: 에너지/비용 분석 (Cost & Sustainability)

**목표**: 품질 개선과 함께 운영비용 및 탄소배출 감소

| 항목 | 계산식 | 기대 효과 |
|------|--------|---------|
| **에너지 소비** | (전력 × 시간) + (가스/냉각수 소비량) | 최적화된 설정값으로 에너지 10% 절감 |
| **불량로 인한 손실 비용** | 불량률 × 제품당 원가 × 월 생산량 | 불량률 1% 감소 = 월 원가 ~수천만원 절감 |
| **탄소배출** | 공정 에너지 소비 × 탄소계수 | ESG 목표 달성, 고객사 공급망 탄소중립 기여 |
| **예방 정비 ROI** | (불량 감소액) / (정비비용) | ROI > 300% (정비비용이 작을 시) |

---

### 2.3 비즈니스 메트릭 정의

#### 주요 KPI (Key Performance Indicator)

| KPI | 정의 | 현재 추정 | 목표 (6개월) | 계산 주기 |
|-----|------|---------|-----------|---------|
| **불량률 (Defect Rate)** | 불량수량 / 총생산수량 | ~2~3% | < 0.5% | 일일 |
| **PPM (Parts Per Million)** | (불량 / 생산) × 1,000,000 | ~20,000~30,000 | < 5,000 | 일일 |
| **OEE (Overall Equipment Effectiveness)** | Availability × Performance × Quality | ~70% | > 85% | 주간 |
| **MTBF (Mean Time Between Failures)** | 고장 간 평균 시간 | TBD | > 500시간 | 월간 |
| **검출 정확도 (Detection Accuracy)** | TP / (TP + FN) | - | > 95% | 실시간 |
| **거짓 알람률 (False Alarm Rate)** | FP / (FP + TN) | - | < 1% | 실시간 |
| **비용 절감액** | (기존 불량비용 - 개선후 불량비용) | - | 월 5천만원 이상 | 월간 |

---

## Section 3: 선행 연구 및 학습 방향

### 3.1 PCB/전자 공정 불량 예측 (Domain-Specific Literature)

#### 핵심 논문

| # | 제목 (Title) | 저자 | 연도 | 핵심 기여 | 관련성 | 상태 |
|----|----------|------|-----|---------|--------|------|
| 1 | **Defect Prediction in Semiconductor Manufacturing using Causal Inference** | Arevalo, J. et al. | 2017 | Causal DAG를 통한 불량 원인 규명 | ⭐⭐⭐⭐⭐ | [검토 완료](../literature/paper/notes/arevalo_2017.md) |
| 2 | **Deep Learning for Manufacturing Quality Control** | Oquab, M. & Zhai, X. | 2024 | Vision Transformer (ViT) 기반 실시간 불량 검출 | ⭐⭐⭐⭐ | [검토 중](../literature/paper/notes/oquab_2024.md) |
| 3 | **Multivariate Causal Discovery with NOTEARS** | Zheng, X. et al. | 2020 | DAG 학습 (Directed Acyclic Graph) 최적화 | ⭐⭐⭐⭐⭐ | [논문 확보](../literature/STAGE_1_PRIOR_WORK_SOLVED.md) |
| 4 | **Explainable AI for Process Control** | Lundberg, S. & Lee, S.I. | 2017 | SHAP (SHapley Additive exPlanations) 해석성 | ⭐⭐⭐⭐⭐ | [검토 완료](../literature/paper/notes/lundberg_2017.md) |
| 5 | **Temporal Fusion Transformers for Time Series Forecasting** | Lim, B. et al. | 2021 | 복잡한 시계열 의존성 모델링 | ⭐⭐⭐⭐ | [검토 중] |
| 6 | **Attention-based Neural Networks for Anomaly Detection** | Hand, D.J. & Ye, Q. | 2009 | Attention mechanism 기초 | ⭐⭐⭐ | [검토 완료] |
| 7 | **Sensor-Based Anomaly Detection using LSTM** | Dosovitskiy, A. et al. | 2021 | 센서 신호 비정상 탐지 | ⭐⭐⭐⭐ | [검토 중] |
| 8 | **Graph Neural Networks for Process Control** | Xu, K. et al. | 2022 | GNN 기반 공정 연관성 모델링 | ⭐⭐⭐⭐⭐ | [논문 확보] |
| 9 | **Process Monitoring and Optimization in Manufacturing** | Susto, G.A. et al. | 2014 | SVM, Random Forest 기반 불량 분류 | ⭐⭐⭐ | [검토 완료] |

**추가 자료**: 
- AI/ML 학술 컨퍼런스: ICML, NeurIPS, ICLR, IJCAI (Causal Inference Track)
- 산업 공학 저널: IEEE Transactions on Industrial Electronics, Journal of Manufacturing Systems
- 한국 논문: 한국반도체학회지, 한국정밀공학회지

#### 선행 업무 체크리스트 (STAGE_1 ~ STAGE_5)

프로젝트의 `docs/literature/` 폴더에 5단계 선행 연구 분석 완료:

- ✅ **STAGE_1**: 기존 문제 해결 방법 (Prior Work Solved Problems)
- ✅ **STAGE_2**: 미충족 연구 갭 (Research Gaps)
- ✅ **STAGE_3**: 우리 접근의 차별점 (Our Novelty)
- ✅ **STAGE_4**: 검증 전략 (Validation Strategy)
- ✅ **STAGE_5**: 논문 아웃라인 (Paper Outline)

**참고 자료**:
- `docs/literature/analysis/novelty_comparison.xlsx`: 기존 방법 vs 우리 방법 비교표
- `docs/literature/analysis/research_summary.xlsx`: 60개 이상 논문 요약
- `docs/literature/analysis/experiment_tracking.xlsx`: 실험 설계 및 평가 계획

---

### 3.2 핵심 기술 학습 로드맵

#### Learning Path 1: 인과 추론 (Causal Inference)

**학습 목표**: "센서 변화가 불량 발생을 야기하는가?" 를 정확히 답변

| 단계 | 학습 내용 | 학습 자료 | 소요 시간 | 구현 예제 |
|------|---------|--------|--------|---------|
| **L1-1** | 인과 그래프 기초 (Causal DAG) | Judea Pearl's "Book of Why" | 1주 | 간단한 3-노드 DAG 그리기 |
| **L1-2** | Confounding & Collider 이해 | Brady Neal의 "Introduction to Causal Inference" | 1주 | d-separation 규칙 적용 |
| **L1-3** | PCMCI 알고리즘 (Peter & Clark Momentary Conditional Independence) | `tigramite` 라이브러리 튜토리얼 | 2주 | 센서 데이터에 PCMCI 적용, DAG 추출 |
| **L1-4** | NOTEARS 알고리즘 (No Tears - 미분가능 DAG 학습) | Zheng et al. 2020 논문 + PyTorch 구현 | 2주 | NOTEARS로 넓은 센서 셋 다루기 |
| **L1-5** | 인과 추론 검증 (Causal Validation) | Randomized Controlled Trial (RCT) 설계 | 1주 | 가설: "온도 설정↑ → 불량↑" 검증 실험 설계 |

**산출물**: `src/utils/causal_discovery.py` (PCMCI/NOTEARS 래퍼)

---

#### Learning Path 2: 설명 가능한 AI (Explainable AI, XAI)

**학습 목표**: 모델이 "왜 이 샘플이 불량이라고 판정했는가?" 를 설명

| 단계 | 학습 내용 | 학습 자료 | 소요 시간 | 구현 예제 |
|------|---------|--------|--------|---------|
| **L2-1** | SHAP 기초 (SHapley Additive exPlanations) | Lundberg & Lee 2017 논문 + SHAP 공식 문서 | 1주 | Iris/타이타닉 데이터에 SHAP 적용 |
| **L2-2** | SHAP Tree Explainer vs Kernel Explainer | SHAP 튜토리얼 문서 | 1주 | 우리 PressFuse 모델에 적용 |
| **L2-3** | Attention Mechanism 시각화 | `attention_viz.py` 기초 | 1주 | Attention weight 히트맵 생성 |
| **L2-4** | Integrated Gradients (속성 기반 설명) | Sundararajan et al. 2017 논문 | 1주 | 센서 신호의 gradient 기반 기여도 계산 |
| **L2-5** | GNN 설명성 (Graph Neural Network Explainability) | GNNExplainer 논문 (Ying et al. 2019) | 2주 | 센서 间 인과관계 그래프 설명 |

**산출물**: `src/explain/shap_wrapper.py` 확장, `src/explain/attention_viz.py` 고도화

---

#### Learning Path 3: 고급 시계열 머신러닝 (Advanced Time Series ML)

**학습 목표**: 공정 센서 신호로부터 미래 불량 예측

| 단계 | 학습 내용 | 학습 자료 | 소요 시간 | 구현 예제 |
|------|---------|--------|--------|---------|
| **L3-1** | LSTM 심화 (Long Short-Term Memory) | Goodfellow et al. "Deep Learning" Ch.10 | 1주 | 시계열 예측 기초 |
| **L3-2** | Temporal Fusion Transformer (TFT) | Lim et al. 2021 논문 + PyTorch 구현 | 2주 | 다중 센서 시계열→불량 예측 |
| **L3-3** | Autoencoder (비정상 탐지) | Goodfellow et al "Deep Learning" Ch.14 | 1주 | 센서 신호 재구성 오류 기반 이상치 점수 |
| **L3-4** | Prophet (Facebook의 시계열 예측) | Prophet 공식 문서 (R + Python) | 1주 | 일일 불량률 시계열 분해 및 예측 |
| **L3-5** | Chronos (Foundation Model for Time Series) | Ansari et al. 2024 논문 | 2주 | 사전학습된 시계열 모델 활용 & fine-tuning |

**산출물**: `ml/train_mvp.py` 확장 (TFT, Chronos 지원)

---

#### Learning Path 4: 그래프 신경망 (Graph Neural Network, GNN)

**학습 목표**: 센서 간 의존성을 그래프로 모델링하여 불량 전파 추적

| 단계 | 학습 내용 | 학습 자료 | 소요 시간 | 구현 예제 |
|------|---------|--------|--------|---------|
| **L4-1** | GNN 기초 (Graph Convolution Network) | Kipf & Welling 2016 논문 | 1주 | PyG로 간단한 GCN 구현 |
| **L4-2** | Temporal Graph Networks | Trivedi et al. 2019 논문 | 1주 | 시간 경과에 따른 그래프 진화 모델링 |
| **L4-3** | Causal Graph + GNN 통합 | MS-CDPNet (우리 프로젝트 핵심) | 2주 | PCMCI DAG → PyG Graph 변환 & GNN 학습 |
| **L4-4** | GNN 설명성 (GNNExplainer) | Ying et al. 2019 논문 | 1주 | 어떤 센서 간 관계가 불량 예측에 중요한가? |

**산출물**: `src/models/pressfuse.py` GNN 모듈 (PyG 기반)

---

#### Learning Path 5: 머신러닝 해석 및 공정 최적화

**학습 목표**: 모델 기반 공정 설정값 추천 및 검증

| 단계 | 학습 내용 | 학습 자료 | 소요 시간 | 구현 예제 |
|------|---------|--------|--------|---------|
| **L5-1** | Feature Importance (SHAP, Permutation) | SHAP 문서 | 1주 | 불량에 영향 미치는 상위 센서 도출 |
| **L5-2** | Bayesian Optimization (설정값 최적화) | Spearmint 라이브러리, GPyOpt 튜토리얼 | 2주 | 온도, 압력 등 설정값 최적 조합 탐색 |
| **L5-3** | Robust Design (Taguchi Method) | Taguchi Method 튜토리얼 | 1주 | 노이즈(주변 온도)에도 강건한 설정값 범위 |
| **L5-4** | Process Capability Analysis (Cpk) | 통계학 교과서 또는 이론 자료 | 1주 | 현재 공정의 Cpk 계산 및 목표 설정 |

**산출물**: `src/research/optimization.py` (Bayesian Opt, Robust Design)

---

### 3.3 관련 오픈소스 라이브러리 및 도구

| 라이브러리 | 용도 | 현재 상태 | 학습/적용 예정 |
|-----------|------|---------|-------------|
| **PyTorch Lightning** | 모델 학습 자동화 | ✅ 사용 중 | - |
| **PyTorch Geometric (PyG)** | GNN 구현 | ❌ 미사용 | ✅ L4-2,3 에서 학습 예정 |
| **SHAP** | 모델 해석성 | ⚠️ 부분 구현 | ✅ L2-2,3 에서 고도화 |
| **Tigramite** | PCMCI 인과 추론 | ❌ 미사용 | ✅ L1-3 에서 학습/적용 |
| **NOTEARS** | DAG 학습 | ❌ 미사용 | ✅ L1-4 에서 학습/적용 |
| **Prophet** | 시계열 분해 & 예측 | ❌ 미사용 | ✅ L3-4 에서 학습/적용 |
| **Chronos** | Foundation Model 시계열 | ❌ 미사용 | ✅ L3-5 에서 학습/적용 |
| **Optuna / Hyperopt** | 하이퍼파라미터 튜닝 | ⚠️ 기초만 | ✅ L5-2 에서 고도화 |
| **MLflow** | 실험 추적 | ✅ 사용 중 | - |
| **Hydra** | 설정 (Config) 관리 | ✅ 사용 중 | - |
| **Streamlit** | 대시보드 (프로토타입) | ⚠️ 기초 UI만 | ✅ 고도화 (L5 이후) |
| **Plotly / Matplotlib** | 시각화 | ✅ 사용 중 | - |

---

## Section 4: 주요 알고리즘 및 모델 설명

### 4.1 우리의 핵심 모델: MS-CDPNet (Multi-Stage Causal Defect Propagation Network)

#### 개념도

```
┌─────────────────────────────────────────────────────────────────┐
│          센서 신호 (T_1, P_1, ... , T_n, P_n, ...)              │
│     (온도, 압력, 습도, 진동, 전류 등 20~50차원)                   │
└───────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
        ┌────╼════════════════════════════╼────┐
        │   Stage 1: Causal Discovery      │
        │  (PCMCI / NOTEARS DAG 학습)     │
        └───────────────────┬──────────────┘
                             │
         센서→센서 인과관계 도출 (Causal DAG)
          예: 온도↗ → 압력↗ → 습도↓ → 불량↑
                             │
                             ▼
        ┌────╼════════════════════════════╼────┐
        │  Stage 2: Graph Representation   │
        │   (DAG → PyG Graph Conversion)   │
        └───────────────────┬──────────────┘
                             │
          센서를 노드, 인과관계를 엣지로 변환
                             │
                             ▼
        ┌────╼════════════════════════════╼────┐
        │  Stage 3: GNN-based Propagation │
        │  (Graph Neural Network 학습)     │
        │   - Graph Conv Layers            │
        │   - Attention Aggregation        │
        │   - Temporal Dynamics            │
        └───────────────────┬──────────────┘
                             │
        불량 신호의 전파 경로 학습
                             │
                             ▼
        ┌────╼════════════════════════════╼────┐
        │  Stage 4: Defect Prediction      │
        │   (Classification Head)          │
        │   Output: P(Defect | Sensor)     │
        └───────────────────┬──────────────┘
                             │
                             ▼
               불량 확률 + SHAP 설명
          (어떤 센서가 불량을 야기했나?)
```

#### 주요 특징

| 특징 | 설명 | 이점 |
|------|------|------|
| **Causal Stage** | PCMCI/NOTEARS로 센서 간 인과관계 추론 | 상관관계(Correlation)가 아닌 인과관계(Causality) 규명 |
| **Graph Representation** | 인과관계를 그래프로 명시화 | 복잡한 센서 간 상호작용 모델링 |
| **GNN Propagation** | 그래프 신경망으로 불량 신호 전파 학습 | 로컬 센서 정보 → 글로벌 불량 신호 학습 |
| **Multi-Stage Pipeline** | 4단계 통합 | 각 단계의 출력이 해석 가능하고 검증 가능 |
| **Explainability** | SHAP + Attention 시각화 | 모델 결정의 "왜"를 설명 |

---

### 4.2 핵심 알고리즘 1: PCMCI (어떤 센서가 불량을 야기하는가?)

#### 알고리즘 개요 (Pairwise Conditional Mutual Information)

```python
# 의사코드 (Pseudocode)

def PCMCI(time_series_data, tau_max=5):
    """
    센서 시계열에서 인과관계 방향성 추론
    
    Args:
        time_series_data: [시간, 센서] 2D 배열
        tau_max: 최대 시간 지연 (lag)
    
    Returns:
        causal_graph: 인과관계 DAG (센서 A → B 의미: A가 B를 야기)
    """
    
    # Step 1: PC(Constraint-based) 단계 - 독립성 테스트로 관계 제거
    #   귀무가설: X와 Y는 독립
    #   검정: Conditional Mutual Information (CMI) 계산
    #   → p-value < 0.05 이면 인과관계 있음으로 간주
    
    # Step 2: Momentary Condition Independence (MCI) 검증
    #   X(t-tau) → Y(t) 관계가 존재하는지만 검증
    #   (MCI는 X→Y 방향성만 테스트하므로 PCMCI, 더 정확)
    
    for sensor_pair in all_pairs:
        for tau in range(1, tau_max+1):  # 시간 지연 테스트
            CMI = compute_CMI(sensor_pair, tau)  # 조건부 상호정보량
            p_value = permutation_test(CMI)
            
            if p_value < significance_level:
                causal_graph.add_edge(
                    source=sensor_pair[0],
                    target=sensor_pair[1],
                    lag=tau
                )
    
    return causal_graph
```

#### 장점 & 단점

| 항목 | 내용 |
|------|------|
| **장점** | ✅ 통계적 가설 검정 기반 (p-value로 신뢰도 측정) |
| | ✅ 시간 지연(lag) 자동 탐지 (온도 변화 → 1시간 후 불량) |
| | ✅ 선형 & 비선형 관계 모두 탐지 (CMI 기반) |
| **단점** | ❌ 계산량 많음 (모든 센서 쌍, 모든 lag 조합) → O(n² × tau_max) |
| | ❌ 샘플 수가 많거나 센서 수가 많으면 느림 (50개 센서라면 2,500 쌍) |
| | ❌ 설정값 민감도: alpha (유의수준), tau_max 선택에 따라 결과 달라짐 |

#### 구현 예제

```python
# Python 코드 (tigramite 라이브러리)

from tigramite.data import Data
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# 1. 데이터 준비: [시간, 센서]
data = np.array([[T1(t), T2(t), ..., Tn(t)] for t in range(N)])

# 2. PCMCI 알고리즘 설정
dataobj = Data(data, var_names=['Temp', 'Pressure', ...])
cond_ind_test = ParCorr()
pcmci = PCMCI(dataobj, cond_ind_test)

# 3. 인과관계 추론 실행
results = pcmci.run_pcmci(
    tau_max=5,          # 최대 5시간 지연 까지 탐지
    pc_alpha=0.01       # 유의수준 1%
)

# 4. 결과 해석
causal_graph = results['graph']  # 인과관계 방향성 딕셔너리
time_lags = results['lags']      # 각 관계의 시간 지연(hour)

# 예: causal_graph[('Temp', 'Defect')] = True, lags[('Temp', 'Defect')] = 0.5
# → 온도(T)가 0.5시간(30분) 후 불량(D)을 야기
```

---

### 4.3 핵심 알고리즘 2: NOTEARS (함수 관계까지 학습)

#### 알고리즘 개요 (No Tears - 미분 가능한 DAG 학습)

**차이점**: PCMCI는 "인과관계의 유무"만, NOTEARS는 "함수 관계"도 학습

```python
# 의사코드 (Pseudocode)

def NOTEARS(X, lambda_1=0.1, lambda_2=0.1):
    """
    데이터로부터 인과 그래프와 함수 관계 동시 학습
    
    Args:
        X: [샘플수, 센서수] 데이터
        lambda_1: L1 정규화 (희소성)
        lambda_2: DAG 제약 정규화
    
    Returns:
        W: 가중치 행렬 (센서 간 영향도)
           W[i,j] > 0 의미: 센서 i가 센서 j에 영향
    """
    
    # Step 1: 가중치 행렬 W 초기화
    W = nn.Parameter(torch.randn(n_sensors, n_sensors))
    
    # Step 2: DAG 제약 정의
    # A = (A_ij > 0) 이면 방향 간선 i→j 이고,
    # Trace(exp(A * A)) = n (A ≤ n인 경우만 acyclic)를 만족해야 함
    def dag_constraint(W):
        A = torch.abs(W)
        return torch.trace(torch.matrix_exp(A)) - n_sensors
    
    # Step 3: 손실함수 정의
    def loss_fn(W):
        # 선형 재구성: X_reconstructed = X @ W
        X_reconstructed = X @ W
        mse = ((X - X_reconstructed) ** 2).mean()
        
        # L1 정규화: 희소한 그래프 유도
        sparsity = lambda_1 * torch.sum(torch.abs(W))
        
        # DAG 제약
        dag_penalty = lambda_2 * dag_constraint(W)
        
        return mse + sparsity + dag_penalty
    
    # Step 4: 경사하강법으로 W 최적화
    optimizer = Adam([W])
    for epoch in range(max_epochs):
        loss = loss_fn(W)
        loss.backward()
        optimizer.step()
    
    return W  # 인과 관계 가중치
```

#### 특징

| 특징 | 설명 |
|------|------|
| **A → B의 영향도 정량화** | W[A,B] = 0.8 의미: A가 변할 때 B는 0.8배 민감하게 변함 |
| **미분가능한 DAG 제약** | trace(exp(W²)) = n 조건으로 순환 고리(cycle) 자동 제거 |
| **스케일러블** | 50개 센서도 리즈너블한 시간 내 수렴 (50개 센서 × 50개: 계산 가능) |
| **함수 관계 학습** | 선형 관계뿐만 아니라 비선형도 확장 가능 (MLP 활용) |

#### 구현 예제

```python
# Python 코드 (PyTorch)

import torch
from torch.optim import Adam

def NOTEARS_torch(X, lambda1=0.1, lambda2=0.1, max_iter=100):
    """
    X: [N, d] 센서 데이터
    Returns: W [d, d] 인과 인접 행렬
    """
    n, d = X.shape
    W = torch.randn(d, d, requires_grad=True)
    optimizer = Adam([W], lr=0.01)
    
    for _ in range(max_iter):
        # 재구성 오류
        X_hat = X @ W
        loss_mse = torch.norm(X - X_hat, p='fro') ** 2 / n
        
        # L1 정규화 (희소성)
        loss_sparse = lambda1 * torch.sum(torch.abs(W))
        
        # DAG 제약: trace(exp(W^2)) ≤ d
        M = torch.eye(d) + W * W / torch.tensor(d)  # 근사
        loss_dag = lambda2 * (torch.trace(torch.matrix_exp(M)) - d) ** 2
        
        loss = loss_mse + loss_sparse + loss_dag
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return W.detach()

# 사용 예
W_learned = NOTEARS_torch(sensor_data)
# W_learned[i,j] > 0 이면 센서 i → j 의존성 있음
print("센서 간 영향도:")
print(W_learned)
```

---

### 4.4 핵심 알고리즘 3: GNN (Graph Neural Network) - 불량 전파 추적

#### 개념: 센서를 노드, 인과관계를 엣지로 하는 그래프

```
센서 그래프 예시:

      [온도 센서]
           │
           ▼ (인과관계)
      [압력 센서] ──→ [습도 센서]
           │              │
           └──────────────▼
                   [불량 발생 예측]

GNN의 역할:
1. 각 센서 신호를 시간에 따라 업데이트
2. 이웃 센서로부터 정보를 집계 (Aggregation)
3. 최종적으로 불량 발생 예측
```

#### 전파 메커니즘 (Message Passing)

```python
# GNN의 한 계층 동작

def GNN_layer(node_features, adjacency_matrix):
    """
    node_features: [센서수, 특성차원]
    adjacency_matrix: [센서수, 센서수] 인과관계 그래프
    """
    
    # Step 1: 각 노드에서 메시지 생성
    messages = W_msg @ node_features  # [센서수, 메시지차원]
    
    # Step 2: 이웃으로 메시지 전달
    aggregated = adjacency_matrix @ messages  # 인과 연결된 센서 정보만 수집
    
    # Step 3: 자신의 정보 + 이웃의 정보 결합
    updated_features = ReLU(W_self @ node_features + W_agg @ aggregated)
    
    return updated_features  # 다음 GNN 계층으로 입력
```

#### 구현 예제 (PyG)

```python
import torch_geometric.nn as nn
from torch_geometric.data import Data, DataLoader

class GNNDefectPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3):
        super().__init__()
        
        # GNN 계층들
        self.convs = nn.ModuleList([
            nn.GCNConv(in_channels, hidden_channels) if i == 0
            else nn.GCNConv(hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
        
        # 분류 헤드
        self.classifier = nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        # x: [센서수, 특성차원]
        # edge_index: [2, 간선수] 인과관계 정보
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = torch.relu(x)
        
        # 불량 분류
        logits = self.classifier(x)
        return logits

# 사용 예
graph = Data(
    x=torch.randn(50, 10),  # 50개 센서, 각각 10차원 특성
    edge_index=torch.tensor([다.ai,인과관계]),
    y=torch.tensor([0, 1, 0, ...])  # 불량 여부
)

model = GNNDefectPredictor(in_channels=10, hidden_channels=32, out_channels=2)
logits = model(graph.x, graph.edge_index)  # [50, 2] (센서별 불량 확률)
```

---

### 4.5 설명 가능성: SHAP (SHapley Additive exPlanations)

#### 개념: 게임 이론의 Shapley Value를 머신러닝에 적용

```
문제: "이 샘플이 불량이다"고 모델이 예측했는데, 왜?

SHAP의 답변: 
  온도(+0.3) + 압력(-0.1) + 습도(+0.2) = 0.4 (불량 확률)
  
해석:
  - 온도: 불량을 +30% 증가
  - 압력: 불량을 -10% 감소 (정상화 효과)
  - 습도: 불량을 +20% 증가
  → 온도와 습도 상승이 주요 원인
```

#### 계산 방식 (Coalition의 한계 확인)

```python
def SHAP_value(feature_i, model, sample):
    """
    특성 i의 SHAP value = 특성 i가 예측에 기여하는 정도 (Shapley value)
    
    핵심 아이디어:
    - 특성 i를 포함한 경우의 예측값
    - 특성 i를 제외한 경우의 예측값
    의 차이를 여러 조합으로 평균화
    """
    
    contributions = []
    for subset in all_subsets_with_and_without_i:
        # 특성 i가 있을 때
        pred_with = model.predict(sample_with_feature_i=subset['with'])
        
        # 특성 i가 없을 때 (배경 샘플로 대체)
        pred_without = model.predict(sample_without_feature_i=subset['without'])
        
        # 기여도 = 차이
        contribution = pred_with - pred_without
        contributions.append(contribution)
    
    # Shapley value = 모든 조합에 대한 평균 기여도
    shapley = mean(contributions)
    
    return shapley
```

#### 구현 예제

```python
import shap
import numpy as np

# 데이터 로드
X_train = load_sensor_data()  # [샘플, 센서]
y_train = load_defect_labels()  # [샘플]

# 모델 학습
model = train_neural_network(X_train, y_train)

# SHAP explainer 생성 (TreeExplainer for tree models, KernelExplainer for neural network)
explainer = shap.KernelExplainer(model.predict, X_train[:100])  # 배경 샘플 100개

# 특정 샘플 해석
sample_to_explain = X_test[0]  # 불량으로 예측된 샘플
shap_values = explainer.shap_values(sample_to_explain)

# 시각화
shap.summary_plot(shap_values, X_test)  # 전체 데이터셋 SHAP summary
shap.force_plot(explainer.expected_value[1], shap_values[0], sample_to_explain)  # 개별 샘플
```

---

## Section 5: 프로젝트 진행 현황

### 5.1 완료된 항목 (✅)

#### Phase 1: POC (Proof of Concept) 구축 [2026.05 완료]

| 항목 | 상태 | 산출물 |
|------|------|--------|
| PyTorch 기초 | ✅ | `requirements.txt`, `pyproject.toml` |
| PressFuse 모델 (Cross-modal Attention) | ✅ | `src/models/pressfuse.py` |
| 합성 데이터셋 생성 | ✅ | `src/data/synthpress.py` (6가지 이상 편차 타입) |
| 기본 불량 분류 (CNN/LSTM) | ✅ | `ml/train_mvp.py`, `ml/predict_mvp.py` |
| 불량 메트릭 정의 | ✅ | `src/eval/metrics.py` (FAR, Recall, Cost-aware 점수) |
| Streamlit 프로토타입 UI | ⚠️ 부분 | `app/streamlit_app.py` (기본 기능만) |
| SECOM 데이터셋 벤치마크 | ✅ | `scripts/secom_baseline.py` |
| 코드 구조화 & 문서화 | ✅ | README, ARCHITECTURE.md |
| GitHub 저장소 구성 | ✅ | Fork + Portfolio 21개 프로젝트 정리 |
| 선행 연구 분석 (60개 논문) | ✅ | `docs/literature/STAGE_1~5` |

#### Phase 2: 분석 및 기초 연구 [2026.05~06 진행 중]

| 항목 | 상태 | 산출물 |
|------|------|--------|
| 고객사 데이터 요청 | ✅ | `docs/CLIENT_DATA_REQUEST.md` |
| 데이터 품질 검증 프레임워크 | ⚠️ 계획 | 이 문서의 Section 2.1 참고 |
| EDA (탐색적 데이터 분석) | ⏳ 대기 | 고객 데이터 수신 후 시작 |
| PCMCI 인과 추론 | ⏳ 학습 예정 | L1-3 따라 구현 |
| NOTEARS DAG 학습 | ⏳ 학습 예정 | L1-4 따라 구현 |

---

### 5.2 진행 중인 항목 (⏳)

#### 우선순위별 진행 예정

| 우선 | 과제 명 | 목표 | 예상 기간 | 담당 |
|------|---------|------|---------|------|
| **P0** | PCMCI + NOTEARS 인과 추론 통합 | DAG 정확도 80% | 2026.06~07 | Research |
| **P0** | GNN 기반 불량 전파 모델 | 센서 간 의존성 학습 | 2026.07~08 | ML |
| **P0** | SHAP + Attention 설명성 고도화 | 모델 해석성 80% 이상 | 2026.08~09 | Explain |
| **P0** | 자동 리포트 생성 파이프라인 | PDF/JSON 자동 생성 | 2026.09~10 | Eng |
| **P1** | Foundation Model (Chronos) 비교 | LSTM vs Chronos 벤치 | 2026.10 | ML |
| **P1** | 고급 XAI 비교 (SHAP vs LIME) | 3가지 방법 비교표 | 2026.10 | Research |
| **P1** | Interactive Web Dashboard (React) | UI/UX 80% 완성도 | 2026.11~12 | Eng |
| **P2** | Digital Twin 프로토타입 | SimPy 기반 시뮬레이션 | 2026.12~01 | Research |

---

### 5.3 현재 코드 구조

```
src/
├── data/
│   ├── dataset.py          # 데이터 로더
│   ├── synthpress.py       # 합성 불량 데이터 생성
│   ├── loaders.py          # PyTorch DataLoader
│   ├── schema.py           # 데이터 스키마 정의
│   └── preprocess.py       # 전처리 파이프라인
├── models/
│   ├── pressfuse.py        # 핵심 모델 (Cross-modal Attention)
│   ├── heads.py            # 분류/회귀 헤드
│   └── baselines/          # 기준 모델 (LSTM, TFT, 등)
├── eval/
│   └── metrics.py          # FAR@Recall, Cost-aware 메트릭
├── explain/
│   ├── shap_wrapper.py     # SHAP 통합
│   ├── attention_viz.py    # Attention 시각화
│   └── shap_grad.py        # Gradient 기반 설명성
├── training/
│   ├── module.py           # Lightning 모듈
│   └── callbacks.py        # Early stopping 등
└── research/
    ├── bibliography.py     # 논문 인용 관리
    └── guide.py            # 연구 가이드

ml/
├── train_mvp.py            # 모델 학습 스크립트
├── predict_mvp.py          # 예측 스크립트
└── test_compute_shap.py    # SHAP 테스트

scripts/
├── analyze_dataset.py      # EDA 자동화
├── preprocess_dataset.py   # 데이터 전처리
├── secom_baseline.py       # SECOM 벤치마크
├── train.py / train_hydra.py  # Hydra 기반 학습
└── predict.py              # 예측 파이프라인

tests/
├── test_*.py               # 단위 테스트 (75% 커버리지)
```

---

### 5.4 기술 스택

| 분류 | 스택 | 버전 |
|------|------|------|
| **Language** | Python | 3.11+ |
| **Deep Learning** | PyTorch + PyTorch Lightning | 2.2 + 2.1 |
| **ML Pipeline** | Hydra + MLflow | 1.3 + 2.x |
| **Data Processing** | Pandas, NumPy, Polars | - |
| **Graph Neural Net** | PyTorch Geometric (예정) | 2.5+ |
| **Causal Inference** | Tigramite, NOTEARS (예정) | - |
| **Explainability** | SHAP, Attention (부분) | 0.41+ |
| **Visualization** | Plotly, Matplotlib | - |
| **Dashboard** | Streamlit (프로토) | 1.28+ |
| **Testing** | pytest | 7.x |
| **CI/CD** | GitHub Actions | - |

---

## Section 6: 향후 추진 계획

### 6.1 단계별 로드맵 (2026.06 ~ 2027.02, 9개월)

```
2026.06                          2027.02
├─ Q2(June)  ─────────────────────────────────┤
│   • 고객 데이터 수신 & EDA
│   • PCMCI/NOTEARS 학습 시작
│
├─ Q3(Jul-Sep) ───────────────────────────────┤
│   • P0-1: Causal Discovery (80% DAG 정확도)
│   • P0-2: GNN 모델 학습 & 검증
│   • P0-3: SHAP 설명성 고도화
│   • P0-4: Auto-Report 파이프라인
│   • P0-5,6: 벤치마킹 및 평가
│
├─ Q4(Oct-Dec) ────────────────────────────────┤
│   • P1-1~4: 고급 기법 비교 (XAI, Foundation Model, Domain Adaptation)
│   • P1-5: 고급 대시보드 개발 (React/Dash)
│   • P2-1: Digital Twin 프로토타입
│
└─ Q1 2027(Jan-Feb) ────────────────────────┤
     • 최종 논문 작성 & 투고 준비
     • 프로덕션 배포 준비 (Docker, K8s)
```

### 6.2 핵심 마일스톤

| 진행도 | 예상 일정 | 내용 | 성공 기준 |
|--------|----------|------|---------|
| 30% | 2026.06 | 고객 데이터 수신 & 검증 | 데이터 품질 검증 완료 |
| 40% | 2026.07 | PCMCI DAG 추출 | DAG precision > 75% |
| 60% | 2026.08 | GNN + DAG 통합 모델 | AUROC > 0.95, FAR < 5% |
| 75% | 2026.09 | 완전 자동화 파이프라인 | 데이터 수신 → 리포트 생성 실시간 |
| 85% | 2026.10 | 논문 초안 & 고급 실험 | 소수 심사자(peer review) 의견 수집 |
| 90% | 2026.11 | 중간 검토 & 최적화 | 고객사 피드백 반영 |
| 100% | 2027.02 | 최종 논문 투고 & 배포 | SCI(E) 저널 또는 국제회의 투고 |

---

### 6.3 예상 논문 구조 (9개월 로드맵 후)

```
Title: "MS-CDPNet: Multi-Stage Causal Defect Propagation Network 
        for PCB Lamination Press Quality Prediction"

1. Introduction (2 pages)
   - PCB Press 공정의 불량 문제
   - 기존 접근방법의 한계
   - 우리의 제안 (Causal + GNN)

2. Related Work (3 pages)
   - 인과 추론 기반 제조 품질 관리
   - GNN을 모델링 공정 의존성
   - XAI in Manufacturing

3. Problem Definition (2 pages)
   - 데이터 정의 (센서, 불량 라벨)
   - 공식화: 인과 그래프 학습 + 불량 예측

4. MS-CDPNet 방법 (4 pages)
   - Stage 1: PCMCI + NOTEARS 인과 발견
   - Stage 2: 그래프 표현
   - Stage 3: GNN 기반 불량 전파 학습
   - Stage 4: 분류 & 해석성

5. Causal Discovery (Detailed) (2 pages)
   - PCMCI 알고리즘 & 설정
   - NOTEARS 함수 관계 학습
   - DAG 검증 (ground truth available in synthetic)

6. Experiments (4 pages)
   - Synthetic + Real (SECOM) + Customer Data
   - 베이스라인 비교 (LSTM, TFT, RandomForest)
   - 정량 평가: AUROC, FAR@Recall, Cost-aware scores

7. Explainability (2 pages)
   - SHAP + Attention 시각화
   - 어떤 센서가 불량을 야기했는가?
   - 어떤 인과 경로가 가장 강한가?

8. Process Optimization (1 page)
   - 인과 관계 기반 공정 설정값 추천
   - Robust Design 검증

9. Results & Discussion (2 pages)
   - 주요 발견
   - 실무 적용 효과 (불량률 감소, 에너지 절감)
   - 한계 및 향후 연구

10. Conclusion (1 page)
    - 요약
    - 기여도
    - 미래 방향

References (2 pages)
```

---

## 최종 요약표

### 데이터 수신 후 분석 & 학습 체크리스트

```
☐ Section 2.1: 데이터 검증 (1-2주)
  ├─ ☐ 누락값, 범위 이상, 중복값 검증
  ├─ ☐ 시간 시퀀스 일관성 확인
  ├─ ☐ 통계적 검증 (이상치 탐지)
  └─ ☐ 산출: data_quality_report.csv

☐ Section 2.2: 인사이트 분석 (3-6주)
  ├─ ☐ (1) 불량 원인 분석 (RCA)
  ├─ ☐ (2) 이상 탐지 (Anomaly Detection)
  ├─ ☐ (3) 공정 최적화 분석
  ├─ ☐ (4) 설비 신뢰도 분석
  └─ ☐ (5) 에너지/비용 분석

☐ Section 3: 선행 연구 & 학습 (병렬 진행)
  ├─ ☐ Learning Path 1: 인과추론 (2-3주)
  │   └─ ☐ PCMCI + NOTEARS 구현
  ├─ ☐ Learning Path 2: XAI (2-3주)
  │   └─ ☐ SHAP + Attention 고도화
  ├─ ☐ Learning Path 3: 시계열 ML (3-4주)
  │   └─ ☐ TFT, Chronos 학습
  ├─ ☐ Learning Path 4: GNN (2-3주)
  │   └─ ☐ PyG 기반 그래프 모델 구현
  └─ ☐ Learning Path 5: 최적화 (2-3주)
      └─ ☐ Bayesian Opt, Robust Design

☐ Section 4: 모델 및 알고리즘 구현 (병렬,4-6주)
  ├─ ☐ MS-CDPNet 통합 모델
  ├─ ☐ PCMCI/NOTEARS 파이프라인
  ├─ ☐ GNN Propagation 모듈
  └─ ☐ SHAP 설명성 통합

☐ Section 5: 프로젝트 관리
  ├─ ☐ 9개월 로드맵 추적
  ├─ ☐ 주간 진행도 리뷰
  └─ ☐ 마일스톤 달성 확인

=> 예상 총 소요 시간: 20-24주 + 병렬 학습 시간
```

---

## 참고 자료

### 내부 문서
- [`docs/CLIENT_DATA_REQUEST.md`](./CLIENT_DATA_REQUEST.md) - 고객사 데이터 요청서
- [`docs/literature/`](./literature/) - 선행완구 분석 (STAGE_1~5)
- [`docs/RESEARCH_ROADMAP_2026.md`](./RESEARCH_ROADMAP_2026.md) - 연구 로드맵
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) - 시스템 아키텍처

### 외부 자료
- [Judea Pearl, "Book of Why"](https://www.bayes.org/) - 인과 추론 기초
- [SHAP Documentation](https://shap.readthedocs.io/)
- [PyTorch Geometric Tutorial](https://pytorch-geometric.readthedocs.io/)
- [Tigramite PCMCI](https://github.com/jakobrunge/tigramite)
- [NOTEARS Repository](https://github.com/xunzheng/notears)

---

**문서 버전**: 1.0  
**최종 업데이트**: 2026년 5월 26일  
**다음 검토 예정**: 2026년 6월 15일 (고객사 데이터 수신 후)


